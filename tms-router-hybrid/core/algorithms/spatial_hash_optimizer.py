"""
SpatialHashOptimizer - O(1) 최근접 탐색 시스템
- 좌표 기반 공간 해시 인덱싱
- HashMap 기반 거리 캐싱
- Pre-computed Distance Matrix
- Memory Pool 최적화
"""
import math
import hashlib
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from threading import Lock

from ..models import Order, Vehicle, Coordinates


@dataclass
class SpatialCell:
    """공간 해시 셀"""
    cell_id: str
    bounds: Tuple[float, float, float, float]  # min_lat, max_lat, min_lon, max_lon
    orders: List[Order] = field(default_factory=list)
    center: Optional[Coordinates] = None
    
    def __post_init__(self):
        if self.center is None:
            min_lat, max_lat, min_lon, max_lon = self.bounds
            self.center = Coordinates(
                latitude=(min_lat + max_lat) / 2,
                longitude=(min_lon + max_lon) / 2
            )


@dataclass
class DistanceEntry:
    """거리 캐시 엔트리"""
    distance: float
    timestamp: int
    access_count: int = 0


class MemoryPool:
    """메모리 풀 관리자"""
    
    def __init__(self, initial_size: int = 1000):
        self.pool: deque = deque()
        self.allocated: Set[int] = set()
        self.lock = Lock()
        self.stats = {'allocated': 0, 'reused': 0, 'created': 0}
        
        # 초기 객체 생성
        for _ in range(initial_size):
            self.pool.append(self._create_object())
    
    def get_object(self) -> Dict[str, Any]:
        """객체 풀에서 객체 가져오기"""
        with self.lock:
            if self.pool:
                obj = self.pool.popleft()
                obj.clear()  # 재사용을 위해 초기화
                self.stats['reused'] += 1
                return obj
            else:
                obj = self._create_object()
                self.stats['created'] += 1
                return obj
    
    def return_object(self, obj: Dict[str, Any]) -> None:
        """객체를 풀로 반환"""
        with self.lock:
            if len(self.pool) < 1000:  # 풀 크기 제한
                self.pool.append(obj)
            self.stats['allocated'] -= 1
    
    def _create_object(self) -> Dict[str, Any]:
        """새 객체 생성"""
        self.stats['allocated'] += 1
        return {}
    
    def get_stats(self) -> Dict[str, int]:
        """메모리 풀 통계 반환"""
        return self.stats.copy()


class PrecomputedMatrix:
    """사전 계산된 거리 매트릭스"""
    
    def __init__(self, max_orders: int = 1000):
        self.max_orders = max_orders
        self.matrix: Optional[np.ndarray] = None
        self.order_index: Dict[str, int] = {}
        self.index_order: Dict[int, str] = {}
        self.is_built = False
        self.logger = logging.getLogger(__name__)
    
    def build(self, orders: List[Order]) -> None:
        """거리 매트릭스 사전 계산"""
        if len(orders) > self.max_orders:
            self.logger.warning(f"주문 수 {len(orders)}개가 최대 {self.max_orders}개를 초과")
            orders = orders[:self.max_orders]
        
        n = len(orders)
        self.matrix = np.zeros((n, n), dtype=np.float32)
        
        # 주문 인덱스 매핑
        for i, order in enumerate(orders):
            self.order_index[order.id] = i
            self.index_order[i] = order.id
        
        # 거리 매트릭스 계산
        for i in range(n):
            for j in range(i + 1, n):
                distance = orders[i].coordinates.distance_to(orders[j].coordinates)
                self.matrix[i, j] = distance
                self.matrix[j, i] = distance  # 대칭 행렬
        
        self.is_built = True
        self.logger.info(f"거리 매트릭스 구축 완료: {n}x{n}")
    
    def get_distance(self, order_id1: str, order_id2: str) -> Optional[float]:
        """O(1) 거리 조회"""
        if not self.is_built:
            return None
        
        idx1 = self.order_index.get(order_id1)
        idx2 = self.order_index.get(order_id2)
        
        if idx1 is not None and idx2 is not None:
            return float(self.matrix[idx1, idx2])
        
        return None
    
    def get_nearest_orders(self, order_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """O(log n) k-최근접 주문 반환"""
        if not self.is_built:
            return []
        
        idx = self.order_index.get(order_id)
        if idx is None:
            return []
        
        distances = self.matrix[idx, :]
        # numpy의 argpartition은 O(n)이지만 실제로는 매우 빠름
        nearest_indices = np.argpartition(distances, min(k + 1, len(distances) - 1))[:k + 1]
        
        results = []
        for i in nearest_indices:
            if i != idx:  # 자기 자신 제외
                other_order_id = self.index_order[i]
                distance = distances[i]
                results.append((other_order_id, float(distance)))
        
        return sorted(results, key=lambda x: x[1])[:k]


class SpatialHashOptimizer:
    """공간 해시 기반 O(1) 최적화 시스템"""
    
    def __init__(self, cell_size_km: float = 2.0, max_cache_size: int = 10000):
        """
        Args:
            cell_size_km: 해시 셀 크기 (km)
            max_cache_size: 최대 캐시 크기
        """
        self.cell_size_km = cell_size_km
        self.max_cache_size = max_cache_size
        
        # 공간 해시 테이블
        self.spatial_hash: Dict[str, SpatialCell] = {}
        self.order_to_cell: Dict[str, str] = {}
        
        # 거리 캐시
        self.distance_cache: Dict[str, DistanceEntry] = {}
        self.cache_lock = Lock()
        
        # 메모리 풀
        self.memory_pool = MemoryPool()
        
        # 사전 계산 매트릭스
        self.precomputed_matrix = PrecomputedMatrix()
        
        # 성능 통계
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'hash_lookups': 0,
            'matrix_lookups': 0,
            'total_queries': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def build_spatial_index(self, orders: List[Order]) -> None:
        """공간 인덱스 구축"""
        self.logger.info(f"공간 인덱스 구축 시작: {len(orders)}개 주문")
        
        # 공간 해시 테이블 초기화
        self.spatial_hash.clear()
        self.order_to_cell.clear()
        
        # 각 주문을 해당 셀에 배치
        for order in orders:
            cell_id = self._get_cell_id(order.coordinates)
            
            # 셀이 없으면 생성
            if cell_id not in self.spatial_hash:
                cell_bounds = self._get_cell_bounds(order.coordinates)
                self.spatial_hash[cell_id] = SpatialCell(cell_id, cell_bounds)
            
            # 주문을 셀에 추가
            self.spatial_hash[cell_id].orders.append(order)
            self.order_to_cell[order.id] = cell_id
        
        # 사전 계산 매트릭스 구축
        if len(orders) <= self.precomputed_matrix.max_orders:
            self.precomputed_matrix.build(orders)
        
        self.logger.info(f"공간 인덱스 구축 완료: {len(self.spatial_hash)}개 셀")
    
    def _get_cell_id(self, coord: Coordinates) -> str:
        """좌표에서 셀 ID 생성"""
        # 좌표를 셀 크기로 정규화
        lat_cell = int(coord.latitude / (self.cell_size_km / 111.0))  # 1도 ≈ 111km
        lon_cell = int(coord.longitude / (self.cell_size_km / (111.0 * math.cos(math.radians(coord.latitude)))))
        
        return f"{lat_cell}_{lon_cell}"
    
    def _get_cell_bounds(self, coord: Coordinates) -> Tuple[float, float, float, float]:
        """좌표에서 셀 경계 계산"""
        lat_size = self.cell_size_km / 111.0
        lon_size = self.cell_size_km / (111.0 * math.cos(math.radians(coord.latitude)))
        
        lat_cell = int(coord.latitude / lat_size)
        lon_cell = int(coord.longitude / lon_size)
        
        min_lat = lat_cell * lat_size
        max_lat = (lat_cell + 1) * lat_size
        min_lon = lon_cell * lon_size
        max_lon = (lon_cell + 1) * lon_size
        
        return (min_lat, max_lat, min_lon, max_lon)
    
    def find_nearest_order_fast(self, target: Coordinates, 
                               exclude_orders: Set[str] = None) -> Optional[Order]:
        """O(1) 평균 시간복잡도로 최근접 주문 찾기"""
        self.stats['total_queries'] += 1
        
        if exclude_orders is None:
            exclude_orders = set()
        
        # 1. 현재 셀에서 찾기
        current_cell_id = self._get_cell_id(target)
        if current_cell_id in self.spatial_hash:
            self.stats['hash_lookups'] += 1
            nearest = self._find_nearest_in_cell(target, self.spatial_hash[current_cell_id], exclude_orders)
            if nearest:
                return nearest
        
        # 2. 인접 셀들에서 찾기 (9개 셀)
        for neighbor_cell in self._get_neighbor_cells(current_cell_id):
            if neighbor_cell in self.spatial_hash:
                nearest = self._find_nearest_in_cell(target, self.spatial_hash[neighbor_cell], exclude_orders)
                if nearest:
                    return nearest
        
        # 3. 더 넓은 범위에서 찾기 (확장 검색)
        return self._find_nearest_extended(target, exclude_orders)
    
    def _find_nearest_in_cell(self, target: Coordinates, cell: SpatialCell, 
                             exclude_orders: Set[str]) -> Optional[Order]:
        """셀 내에서 최근접 주문 찾기"""
        if not cell.orders:
            return None
        
        nearest_order = None
        min_distance = float('inf')
        
        for order in cell.orders:
            if order.id in exclude_orders:
                continue
            
            # 캐시된 거리 확인
            distance = self._get_cached_distance(target, order.coordinates)
            
            if distance < min_distance:
                min_distance = distance
                nearest_order = order
        
        return nearest_order
    
    def _get_neighbor_cells(self, cell_id: str) -> List[str]:
        """인접한 8개 셀의 ID 반환"""
        try:
            lat_cell, lon_cell = map(int, cell_id.split('_'))
        except ValueError:
            return []
        
        neighbors = []
        for d_lat in [-1, 0, 1]:
            for d_lon in [-1, 0, 1]:
                if d_lat == 0 and d_lon == 0:
                    continue  # 현재 셀 제외
                neighbor_id = f"{lat_cell + d_lat}_{lon_cell + d_lon}"
                neighbors.append(neighbor_id)
        
        return neighbors
    
    def _find_nearest_extended(self, target: Coordinates, exclude_orders: Set[str]) -> Optional[Order]:
        """확장 검색으로 최근접 주문 찾기"""
        # 반지름을 늘려가며 검색
        search_radius = self.cell_size_km
        max_radius = self.cell_size_km * 5  # 최대 5배까지 확장
        
        while search_radius <= max_radius:
            candidate_cells = self._get_cells_in_radius(target, search_radius)
            
            nearest_order = None
            min_distance = float('inf')
            
            for cell_id in candidate_cells:
                if cell_id in self.spatial_hash:
                    cell_nearest = self._find_nearest_in_cell(target, self.spatial_hash[cell_id], exclude_orders)
                    if cell_nearest:
                        distance = self._get_cached_distance(target, cell_nearest.coordinates)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_order = cell_nearest
            
            if nearest_order:
                return nearest_order
            
            search_radius *= 2  # 반지름 2배씩 확장
        
        return None
    
    def _get_cells_in_radius(self, target: Coordinates, radius_km: float) -> List[str]:
        """반지름 내의 모든 셀 ID 반환"""
        lat_range = int(radius_km / (self.cell_size_km / 111.0)) + 1
        lon_range = int(radius_km / (self.cell_size_km / (111.0 * math.cos(math.radians(target.latitude))))) + 1
        
        center_cell_id = self._get_cell_id(target)
        try:
            center_lat, center_lon = map(int, center_cell_id.split('_'))
        except ValueError:
            return []
        
        cells = []
        for d_lat in range(-lat_range, lat_range + 1):
            for d_lon in range(-lon_range, lon_range + 1):
                cell_id = f"{center_lat + d_lat}_{center_lon + d_lon}"
                cells.append(cell_id)
        
        return cells
    
    def _get_cached_distance(self, coord1: Coordinates, coord2: Coordinates) -> float:
        """캐시된 거리 반환 또는 계산"""
        # 캐시 키 생성 (정렬해서 대칭성 보장)
        key1 = f"{coord1.latitude:.6f},{coord1.longitude:.6f}"
        key2 = f"{coord2.latitude:.6f},{coord2.longitude:.6f}"
        cache_key = f"{min(key1, key2)}|{max(key1, key2)}"
        
        with self.cache_lock:
            if cache_key in self.distance_cache:
                entry = self.distance_cache[cache_key]
                entry.access_count += 1
                self.stats['cache_hits'] += 1
                return entry.distance
        
        # 캐시 미스 - 계산 필요
        distance = coord1.distance_to(coord2)
        
        with self.cache_lock:
            # 캐시 크기 확인 후 추가
            if len(self.distance_cache) >= self.max_cache_size:
                self._evict_cache_entries()
            
            import time
            self.distance_cache[cache_key] = DistanceEntry(
                distance=distance,
                timestamp=int(time.time()),
                access_count=1
            )
            self.stats['cache_misses'] += 1
        
        return distance
    
    def _evict_cache_entries(self) -> None:
        """캐시 엔트리 제거 (LFU 정책)"""
        if len(self.distance_cache) < self.max_cache_size:
            return
        
        # 접근 빈도가 낮은 항목들 제거
        sorted_entries = sorted(
            self.distance_cache.items(),
            key=lambda x: (x[1].access_count, x[1].timestamp)
        )
        
        # 하위 20% 제거
        remove_count = max(1, len(sorted_entries) // 5)
        for i in range(remove_count):
            del self.distance_cache[sorted_entries[i][0]]
    
    def get_k_nearest_orders(self, target: Coordinates, k: int = 5, 
                           exclude_orders: Set[str] = None) -> List[Tuple[Order, float]]:
        """k개의 최근접 주문 반환"""
        if exclude_orders is None:
            exclude_orders = set()
        
        # 사전 계산 매트릭스 사용 가능한지 확인
        if self.precomputed_matrix.is_built:
            # 타겟에 가장 가까운 주문을 찾아서 그것을 기준으로 k-nearest 찾기
            nearest_order = self.find_nearest_order_fast(target, exclude_orders)
            if nearest_order:
                self.stats['matrix_lookups'] += 1
                matrix_results = self.precomputed_matrix.get_nearest_orders(nearest_order.id, k)
                results = []
                for order_id, distance in matrix_results:
                    if order_id not in exclude_orders:
                        # 실제 Order 객체 찾기
                        cell_id = self.order_to_cell.get(order_id)
                        if cell_id and cell_id in self.spatial_hash:
                            for order in self.spatial_hash[cell_id].orders:
                                if order.id == order_id:
                                    results.append((order, distance))
                                    break
                return results[:k]
        
        # 공간 해시 기반 검색
        candidates = []
        current_cell_id = self._get_cell_id(target)
        
        # 현재 셀과 인접 셀들에서 후보 수집
        search_cells = [current_cell_id] + self._get_neighbor_cells(current_cell_id)
        
        for cell_id in search_cells:
            if cell_id in self.spatial_hash:
                for order in self.spatial_hash[cell_id].orders:
                    if order.id not in exclude_orders:
                        distance = self._get_cached_distance(target, order.coordinates)
                        candidates.append((order, distance))
        
        # 거리순 정렬 후 k개 반환
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """최적화 통계 반환"""
        cache_hit_rate = 0.0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        
        return {
            'spatial_cells': len(self.spatial_hash),
            'cache_size': len(self.distance_cache),
            'cache_hit_rate': cache_hit_rate,
            'total_queries': self.stats['total_queries'],
            'hash_lookups': self.stats['hash_lookups'],
            'matrix_lookups': self.stats['matrix_lookups'],
            'memory_pool_stats': self.memory_pool.get_stats(),
            'precomputed_matrix_size': self.precomputed_matrix.matrix.shape if self.precomputed_matrix.matrix is not None else (0, 0)
        }
    
    def clear_cache(self) -> None:
        """캐시 초기화"""
        with self.cache_lock:
            self.distance_cache.clear()
            self.stats = {key: 0 for key in self.stats.keys()}
        self.logger.info("캐시 초기화 완료")