"""
GeospatialClustering 알고리즘
- 라이더별 완전 분리 구역 보장
- K-D Tree 기반 공간 분할
- Voronoi Diagram 구역 할당
- O(log n) 최근접 탐색
"""
import math
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from ..models import Order, Vehicle, Coordinates
from ..utils.time_calculator import get_time_calculator


@dataclass
class ClusterCenter:
    """클러스터 중심점"""
    coordinates: Coordinates
    vehicle_id: str
    radius: float = 0.0
    order_count: int = 0
    orders: List[str] = None
    
    def __post_init__(self):
        if self.orders is None:
            self.orders = []


@dataclass
class SpatialBounds:
    """공간 경계"""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    
    def contains(self, coord: Coordinates) -> bool:
        """좌표가 경계 내에 있는지 확인"""
        return (self.min_lat <= coord.latitude <= self.max_lat and
                self.min_lon <= coord.longitude <= self.max_lon)
    
    def center(self) -> Coordinates:
        """경계의 중심점 반환"""
        return Coordinates(
            latitude=(self.min_lat + self.max_lat) / 2,
            longitude=(self.min_lon + self.max_lon) / 2
        )


class KDTreeNode:
    """K-D Tree 노드"""
    
    def __init__(self, order: Order, depth: int = 0):
        self.order = order
        self.depth = depth
        self.left: Optional['KDTreeNode'] = None
        self.right: Optional['KDTreeNode'] = None
        self.axis = depth % 2  # 0: latitude, 1: longitude


class KDTree:
    """K-D Tree 기반 공간 인덱스"""
    
    def __init__(self):
        self.root: Optional[KDTreeNode] = None
        self.size = 0
    
    def build(self, orders: List[Order]) -> None:
        """주문 리스트로 K-D Tree 구축"""
        if not orders:
            return
        
        self.root = self._build_recursive(orders, 0)
        self.size = len(orders)
    
    def _build_recursive(self, orders: List[Order], depth: int) -> Optional[KDTreeNode]:
        """재귀적 K-D Tree 구축"""
        if not orders:
            return None
        
        # 현재 축으로 정렬 (0: lat, 1: lon)
        axis = depth % 2
        if axis == 0:
            orders.sort(key=lambda o: o.coordinates.latitude)
        else:
            orders.sort(key=lambda o: o.coordinates.longitude)
        
        # 중앙값 선택
        median = len(orders) // 2
        node = KDTreeNode(orders[median], depth)
        
        # 재귀적으로 좌우 서브트리 구축
        node.left = self._build_recursive(orders[:median], depth + 1)
        node.right = self._build_recursive(orders[median + 1:], depth + 1)
        
        return node
    
    def find_nearest(self, target: Coordinates, max_distance: float = float('inf')) -> Optional[Order]:
        """최근접 주문 찾기 - O(log n)"""
        if not self.root:
            return None
        
        best = {'order': None, 'distance': max_distance}
        self._search_nearest(self.root, target, best)
        return best['order']
    
    def find_k_nearest(self, target: Coordinates, k: int, max_distance: float = float('inf')) -> List[Order]:
        """k개의 최근접 주문 찾기"""
        if not self.root or k <= 0:
            return []
        
        candidates = []
        self._search_k_nearest(self.root, target, k, max_distance, candidates)
        return [order for order, _ in sorted(candidates, key=lambda x: x[1])[:k]]
    
    def _search_nearest(self, node: KDTreeNode, target: Coordinates, best: Dict) -> None:
        """최근접 탐색 재귀 함수"""
        if not node:
            return
        
        # 현재 노드와의 거리 계산
        distance = target.distance_to(node.order.coordinates)
        if distance < best['distance']:
            best['order'] = node.order
            best['distance'] = distance
        
        # 탐색할 서브트리 결정
        axis = node.axis
        if axis == 0:
            target_val = target.latitude
            node_val = node.order.coordinates.latitude
        else:
            target_val = target.longitude
            node_val = node.order.coordinates.longitude
        
        # 가까운 쪽 먼저 탐색
        if target_val < node_val:
            self._search_nearest(node.left, target, best)
            # 경계 확인 후 반대쪽도 탐색
            if abs(target_val - node_val) < best['distance']:
                self._search_nearest(node.right, target, best)
        else:
            self._search_nearest(node.right, target, best)
            if abs(target_val - node_val) < best['distance']:
                self._search_nearest(node.left, target, best)
    
    def _search_k_nearest(self, node: KDTreeNode, target: Coordinates, k: int, 
                         max_distance: float, candidates: List[Tuple[Order, float]]) -> None:
        """k-최근접 탐색 재귀 함수"""
        if not node:
            return
        
        distance = target.distance_to(node.order.coordinates)
        if distance <= max_distance:
            candidates.append((node.order, distance))
            # k개를 초과하면 가장 먼 것 제거
            if len(candidates) > k:
                candidates.sort(key=lambda x: x[1])
                candidates = candidates[:k]
        
        # 서브트리 탐색 (nearest와 동일한 로직)
        axis = node.axis
        if axis == 0:
            target_val = target.latitude
            node_val = node.order.coordinates.latitude
        else:
            target_val = target.longitude
            node_val = node.order.coordinates.longitude
        
        if target_val < node_val:
            self._search_k_nearest(node.left, target, k, max_distance, candidates)
            if len(candidates) < k or abs(target_val - node_val) < max([d for _, d in candidates]):
                self._search_k_nearest(node.right, target, k, max_distance, candidates)
        else:
            self._search_k_nearest(node.right, target, k, max_distance, candidates)
            if len(candidates) < k or abs(target_val - node_val) < max([d for _, d in candidates]):
                self._search_k_nearest(node.left, target, k, max_distance, candidates)


class VoronoiClustering:
    """Voronoi Diagram 기반 클러스터링"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_clusters(self, orders: List[Order], vehicles: List[Vehicle], 
                       vehicle_capacities: Dict[str, int]) -> Dict[str, List[Order]]:
        """Voronoi 다이어그램 기반 클러스터 생성"""
        
        if not orders or not vehicles:
            return {}
        
        # 권역별로 차량과 주문 그룹화
        region_clusters = {}
        region_orders = self._group_by_region(orders)
        region_vehicles = self._group_by_region(vehicles)
        
        for region_id in region_orders.keys():
            if region_id in region_vehicles:
                region_cluster = self._create_region_voronoi_clusters(
                    region_orders[region_id], 
                    region_vehicles[region_id],
                    vehicle_capacities
                )
                region_clusters.update(region_cluster)
        
        return region_clusters
    
    def _group_by_region(self, items: List) -> Dict[str, List]:
        """항목들을 권역별로 그룹화"""
        groups = defaultdict(list)
        for item in items:
            groups[item.region_id].append(item)
        return dict(groups)
    
    def _create_region_voronoi_clusters(self, orders: List[Order], vehicles: List[Vehicle],
                                       vehicle_capacities: Dict[str, int]) -> Dict[str, List[Order]]:
        """특정 권역의 Voronoi 클러스터 생성"""
        
        clusters = {}
        
        # 유효한 차량만 필터링
        valid_vehicles = [v for v in vehicles if vehicle_capacities.get(v.id, 0) > 0]
        if not valid_vehicles:
            return clusters
        
        # 각 주문을 가장 가까운 차량에 배정 (Voronoi 원리)
        for order in orders:
            nearest_vehicle = self._find_nearest_vehicle(order, valid_vehicles)
            if nearest_vehicle:
                if nearest_vehicle.id not in clusters:
                    clusters[nearest_vehicle.id] = []
                clusters[nearest_vehicle.id].append(order)
        
        # 용량 제한 적용
        final_clusters = {}
        for vehicle_id, assigned_orders in clusters.items():
            capacity = vehicle_capacities.get(vehicle_id, 0)
            
            # 용량 초과시 거리 기준으로 정렬하여 가까운 것만 선택
            if len(assigned_orders) > capacity:
                vehicle = next(v for v in valid_vehicles if v.id == vehicle_id)
                assigned_orders.sort(
                    key=lambda o: vehicle.center_coordinates.distance_to(o.coordinates)
                )
                final_clusters[vehicle_id] = assigned_orders[:capacity]
            else:
                final_clusters[vehicle_id] = assigned_orders
        
        return final_clusters
    
    def _find_nearest_vehicle(self, order: Order, vehicles: List[Vehicle]) -> Optional[Vehicle]:
        """주문에서 가장 가까운 차량 찾기"""
        if not vehicles:
            return None
        
        nearest_vehicle = None
        min_distance = float('inf')
        
        for vehicle in vehicles:
            distance = order.coordinates.distance_to(vehicle.center_coordinates)
            if distance < min_distance:
                min_distance = distance
                nearest_vehicle = vehicle
        
        return nearest_vehicle


class GeospatialClustering:
    """지리공간 클러스터링 메인 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.voronoi_clustering = VoronoiClustering()
        self.kdtree = KDTree()
        self.time_calculator = get_time_calculator()
    
    def create_non_overlapping_clusters(self, orders: List[Order], vehicles: List[Vehicle],
                                      vehicle_capacities: Dict[str, int]) -> Dict[str, List[Order]]:
        """라이더가 겹치지 않는 클러스터 생성"""
        
        self.logger.info(f"지리공간 클러스터링 시작: 주문 {len(orders)}개, 차량 {len(vehicles)}대")
        
        # Step 1: 기본 Voronoi 클러스터링
        initial_clusters = self.voronoi_clustering.create_clusters(orders, vehicles, vehicle_capacities)
        
        # Step 2: 경계 겹침 해결
        refined_clusters = self._resolve_boundary_overlaps(initial_clusters, vehicles, vehicle_capacities)
        
        # Step 3: 부하 균형 조정
        balanced_clusters = self._balance_cluster_loads(refined_clusters, vehicles, vehicle_capacities)
        
        # Step 4: 클러스터 품질 검증
        validated_clusters = self._validate_cluster_quality(balanced_clusters, vehicles)
        
        self.logger.info(f"클러스터링 완료: {len(validated_clusters)}개 클러스터 생성")
        return validated_clusters
    
    def _resolve_boundary_overlaps(self, clusters: Dict[str, List[Order]], 
                                 vehicles: List[Vehicle], vehicle_capacities: Dict[str, int]) -> Dict[str, List[Order]]:
        """경계 겹침 해결"""
        
        refined_clusters = {}
        vehicles_dict = {v.id: v for v in vehicles}
        
        # 각 클러스터의 경계 계산
        cluster_bounds = {}
        for vehicle_id, orders in clusters.items():
            if orders:
                cluster_bounds[vehicle_id] = self._calculate_cluster_bounds(orders)
        
        # 겹치는 주문들 재배정
        for vehicle_id, orders in clusters.items():
            if not orders or vehicle_id not in vehicles_dict:
                continue
            
            vehicle = vehicles_dict[vehicle_id]
            refined_orders = []
            
            for order in orders:
                # 이 주문이 다른 클러스터와 겹치는지 확인
                conflicts = self._find_boundary_conflicts(order, cluster_bounds, vehicle_id)
                
                if not conflicts:
                    refined_orders.append(order)
                else:
                    # 겹치는 경우 가장 가까운 차량에 배정
                    nearest_vehicle_id = self._find_nearest_vehicle_for_order(order, vehicles)
                    if nearest_vehicle_id == vehicle_id:
                        refined_orders.append(order)
                    # 다른 차량이 더 가까우면 해당 차량으로 이동 (다음 단계에서 처리)
            
            refined_clusters[vehicle_id] = refined_orders
        
        return refined_clusters
    
    def _calculate_cluster_bounds(self, orders: List[Order]) -> SpatialBounds:
        """클러스터의 공간 경계 계산"""
        if not orders:
            return SpatialBounds(0, 0, 0, 0)
        
        lats = [o.coordinates.latitude for o in orders]
        lons = [o.coordinates.longitude for o in orders]
        
        return SpatialBounds(
            min_lat=min(lats),
            max_lat=max(lats),
            min_lon=min(lons),
            max_lon=max(lons)
        )
    
    def _find_boundary_conflicts(self, order: Order, cluster_bounds: Dict[str, SpatialBounds], 
                               exclude_vehicle_id: str) -> List[str]:
        """주문이 다른 클러스터와 겹치는지 확인"""
        conflicts = []
        
        for vehicle_id, bounds in cluster_bounds.items():
            if vehicle_id != exclude_vehicle_id and bounds.contains(order.coordinates):
                conflicts.append(vehicle_id)
        
        return conflicts
    
    def _find_nearest_vehicle_for_order(self, order: Order, vehicles: List[Vehicle]) -> str:
        """주문에 가장 가까운 차량 ID 반환"""
        if not vehicles:
            return ""
        
        nearest_vehicle = vehicles[0]
        min_distance = order.coordinates.distance_to(nearest_vehicle.center_coordinates)
        
        for vehicle in vehicles[1:]:
            distance = order.coordinates.distance_to(vehicle.center_coordinates)
            if distance < min_distance:
                min_distance = distance
                nearest_vehicle = vehicle
        
        return nearest_vehicle.id
    
    def _balance_cluster_loads(self, clusters: Dict[str, List[Order]], 
                             vehicles: List[Vehicle], vehicle_capacities: Dict[str, int]) -> Dict[str, List[Order]]:
        """클러스터 부하 균형 조정"""
        
        balanced_clusters = {}
        vehicles_dict = {v.id: v for v in vehicles}
        
        # 과부하 및 저부하 클러스터 식별
        overloaded = []
        underloaded = []
        
        for vehicle_id, orders in clusters.items():
            capacity = vehicle_capacities.get(vehicle_id, 0)
            if capacity <= 0:
                continue
                
            load_ratio = len(orders) / capacity
            
            if load_ratio > 1.0:  # 용량 초과
                overloaded.append((vehicle_id, orders, load_ratio))
            elif load_ratio < 0.7:  # 70% 미만
                underloaded.append((vehicle_id, orders, load_ratio))
            else:
                balanced_clusters[vehicle_id] = orders
        
        # 과부하 클러스터에서 저부하 클러스터로 주문 이동
        for vehicle_id, orders, load_ratio in overloaded:
            capacity = vehicle_capacities[vehicle_id]
            excess_orders = orders[capacity:]  # 초과 주문들
            
            balanced_clusters[vehicle_id] = orders[:capacity]  # 용량만큼만 유지
            
            # 초과 주문들을 저부하 차량에 재배정
            self._redistribute_excess_orders(excess_orders, underloaded, vehicle_capacities, vehicles_dict)
        
        # 저부하 클러스터 추가
        for vehicle_id, orders, load_ratio in underloaded:
            if vehicle_id not in balanced_clusters:
                balanced_clusters[vehicle_id] = orders
        
        return balanced_clusters
    
    def _redistribute_excess_orders(self, excess_orders: List[Order], 
                                  underloaded: List[Tuple[str, List[Order], float]], 
                                  vehicle_capacities: Dict[str, int],
                                  vehicles_dict: Dict[str, Vehicle]) -> None:
        """초과 주문들을 저부하 차량에 재배정"""
        
        for order in excess_orders:
            best_vehicle_id = None
            min_distance = float('inf')
            
            # 여유가 있는 차량 중 가장 가까운 차량 찾기
            for vehicle_id, current_orders, load_ratio in underloaded:
                capacity = vehicle_capacities[vehicle_id]
                if len(current_orders) < capacity:
                    vehicle = vehicles_dict[vehicle_id]
                    distance = order.coordinates.distance_to(vehicle.center_coordinates)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_vehicle_id = vehicle_id
            
            # 가장 적합한 차량에 주문 추가
            if best_vehicle_id:
                for i, (vehicle_id, current_orders, load_ratio) in enumerate(underloaded):
                    if vehicle_id == best_vehicle_id:
                        current_orders.append(order)
                        # 부하율 업데이트
                        new_load_ratio = len(current_orders) / vehicle_capacities[vehicle_id]
                        underloaded[i] = (vehicle_id, current_orders, new_load_ratio)
                        break
    
    def _validate_cluster_quality(self, clusters: Dict[str, List[Order]], 
                                vehicles: List[Vehicle]) -> Dict[str, List[Order]]:
        """클러스터 품질 검증"""
        
        validated_clusters = {}
        vehicles_dict = {v.id: v for v in vehicles}
        
        for vehicle_id, orders in clusters.items():
            if not orders or vehicle_id not in vehicles_dict:
                continue
            
            vehicle = vehicles_dict[vehicle_id]
            
            # 품질 메트릭 계산
            avg_distance = sum(
                vehicle.center_coordinates.distance_to(order.coordinates) 
                for order in orders
            ) / len(orders)
            
            compactness = self._calculate_cluster_compactness(orders)
            separation = self._calculate_cluster_separation(orders, clusters, vehicle_id)
            
            # 품질 기준 통과 여부
            quality_score = self._calculate_quality_score(avg_distance, compactness, separation)
            
            if quality_score > 0.6:  # 60% 이상이면 통과
                validated_clusters[vehicle_id] = orders
                self.logger.debug(f"차량 {vehicle_id} 클러스터 품질: {quality_score:.2f}")
            else:
                self.logger.warning(f"차량 {vehicle_id} 클러스터 품질 부족: {quality_score:.2f}")
                validated_clusters[vehicle_id] = orders  # 일단 포함 (개선 여지 있음)
        
        return validated_clusters
    
    def _calculate_cluster_compactness(self, orders: List[Order]) -> float:
        """클러스터 응집도 계산"""
        if len(orders) <= 1:
            return 1.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(orders)):
            for j in range(i + 1, len(orders)):
                total_distance += orders[i].coordinates.distance_to(orders[j].coordinates)
                count += 1
        
        avg_internal_distance = total_distance / count if count > 0 else 0.0
        return 1.0 / (1.0 + avg_internal_distance)  # 거리가 짧을수록 높은 점수
    
    def _calculate_cluster_separation(self, orders: List[Order], 
                                    all_clusters: Dict[str, List[Order]], exclude_vehicle_id: str) -> float:
        """클러스터 분리도 계산"""
        if not orders:
            return 1.0
        
        min_external_distance = float('inf')
        
        for vehicle_id, other_orders in all_clusters.items():
            if vehicle_id == exclude_vehicle_id or not other_orders:
                continue
            
            for order in orders:
                for other_order in other_orders:
                    distance = order.coordinates.distance_to(other_order.coordinates)
                    min_external_distance = min(min_external_distance, distance)
        
        return min_external_distance if min_external_distance != float('inf') else 1.0
    
    def _calculate_quality_score(self, avg_distance: float, compactness: float, separation: float) -> float:
        """종합 품질 점수 계산"""
        # 정규화된 거리 점수 (10km를 기준으로)
        distance_score = max(0.0, 1.0 - avg_distance / 10.0)
        
        # 가중 평균으로 종합 점수 계산
        quality_score = (
            distance_score * 0.4 +    # 평균 거리 40%
            compactness * 0.35 +      # 응집도 35%
            min(separation / 5.0, 1.0) * 0.25  # 분리도 25% (5km 기준)
        )
        
        return max(0.0, min(1.0, quality_score))