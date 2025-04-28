from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ..models.delivery_point import DeliveryPoint
from ..utils.geo_utils import calculate_distance, get_center_point

class AreaDivider:
    """배송 지역을 효율적으로 분할하는 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def divide(
        self,
        delivery_points: List[DeliveryPoint],
        num_areas: int,
        method: str = 'kmeans',
        **kwargs
    ) -> List[List[DeliveryPoint]]:
        """
        배송 지역을 분할하는 메인 메서드

        Args:
            delivery_points: 배송지점 목록
            num_areas: 분할할 구역 수
            method: 분할 방법 ('kmeans', 'grid', 'administrative')
            **kwargs: 추가 파라미터

        Returns:
            분할된 배송지점 리스트의 리스트
        """
        if method == 'kmeans':
            return self._divide_kmeans(delivery_points, num_areas, **kwargs)
        elif method == 'grid':
            return self._divide_grid(delivery_points, num_areas, **kwargs)
        elif method == 'administrative':
            return self._divide_administrative(delivery_points, num_areas, **kwargs)
        else:
            raise ValueError(f"Unsupported division method: {method}")

    def _divide_kmeans(
        self,
        delivery_points: List[DeliveryPoint],
        num_areas: int,
        **kwargs
    ) -> List[List[DeliveryPoint]]:
        """K-means 클러스터링을 사용한 지역 분할"""
        if not delivery_points:
            return []

        # 위경도 데이터 추출 및 정규화
        coordinates = np.array([
            [point.latitude, point.longitude] for point in delivery_points
        ])
        
        # 배송량과 우선순위를 고려한 가중치 추가
        weights = np.array([
            [point.volume, point.weight, point.priority] 
            for point in delivery_points
        ])
        
        # 데이터 정규화
        scaled_coords = self.scaler.fit_transform(coordinates)
        scaled_weights = self.scaler.fit_transform(weights)
        
        # 좌표와 가중치를 결합
        features = np.hstack([
            scaled_coords * kwargs.get('coord_weight', 0.7),
            scaled_weights * kwargs.get('attr_weight', 0.3)
        ])

        # K-means 클러스터링 수행
        kmeans = KMeans(
            n_clusters=num_areas,
            random_state=42,
            n_init=10
        )
        clusters = kmeans.fit_predict(features)

        # 결과 그룹화
        areas = [[] for _ in range(num_areas)]
        for point, cluster_id in zip(delivery_points, clusters):
            areas[cluster_id].append(point)

        # 클러스터 밸런싱
        return self._balance_areas(areas)

    def _divide_grid(
        self,
        delivery_points: List[DeliveryPoint],
        num_areas: int,
        **kwargs
    ) -> List[List[DeliveryPoint]]:
        """격자 기반 지역 분할"""
        if not delivery_points:
            return []

        # 배송 지역의 경계 계산
        lats = [p.latitude for p in delivery_points]
        lons = [p.longitude for p in delivery_points]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # 격자 크기 계산
        aspect_ratio = (max_lon - min_lon) / (max_lat - min_lat)
        grid_rows = int(np.sqrt(num_areas / aspect_ratio))
        grid_cols = int(np.sqrt(num_areas * aspect_ratio))

        # 격자 생성
        lat_step = (max_lat - min_lat) / grid_rows
        lon_step = (max_lon - min_lon) / grid_cols

        # 포인트를 격자에 할당
        grid = {}
        for point in delivery_points:
            row = int((point.latitude - min_lat) / lat_step)
            col = int((point.longitude - min_lon) / lon_step)
            row = min(row, grid_rows - 1)
            col = min(col, grid_cols - 1)
            
            grid_key = (row, col)
            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append(point)

        # 격자를 병합하여 요청된 구역 수 만큼 생성
        return self._merge_grid_cells(grid, num_areas)

    def _divide_administrative(
        self,
        delivery_points: List[DeliveryPoint],
        num_areas: int,
        **kwargs
    ) -> List[List[DeliveryPoint]]:
        """행정구역 기반 지역 분할"""
        # 실제 구현에서는 행정구역 데이터 필요
        # 임시로 K-means 사용
        return self._divide_kmeans(delivery_points, num_areas, **kwargs)

    def _balance_areas(
        self,
        areas: List[List[DeliveryPoint]]
    ) -> List[List[DeliveryPoint]]:
        """분할된 구역들의 작업량 밸런싱"""
        # 각 구역의 작업량 계산
        area_loads = []
        for area in areas:
            total_volume = sum(p.volume for p in area)
            total_weight = sum(p.weight for p in area)
            total_priority = sum(p.get_priority_weight() for p in area)
            
            area_loads.append({
                'points': area,
                'volume': total_volume,
                'weight': total_weight,
                'priority': total_priority,
                'size': len(area)
            })

        # 작업량이 가장 큰/작은 구역 간 재분배
        balanced = False
        while not balanced:
            max_load = max(area_loads, key=lambda x: x['size'])
            min_load = min(area_loads, key=lambda x: x['size'])
            
            if max_load['size'] - min_load['size'] <= 1:
                balanced = True
                continue

            # 가장 큰 구역에서 가장 작은 구역으로 포인트 이동
            point_to_move = self._find_best_point_to_move(
                max_load['points'],
                get_center_point(min_load['points'])
            )
            
            max_load['points'].remove(point_to_move)
            min_load['points'].append(point_to_move)
            
            # 작업량 업데이트
            max_load['size'] -= 1
            min_load['size'] += 1

        return [load['points'] for load in area_loads]

    def _merge_grid_cells(
        self,
        grid: Dict[Tuple[int, int], List[DeliveryPoint]],
        target_areas: int
    ) -> List[List[DeliveryPoint]]:
        """격자 셀들을 병합하여 목표 구역 수 만큼 생성"""
        # 현재 비어있지 않은 셀의 수
        current_cells = len([cells for cells in grid.values() if cells])
        
        if current_cells <= target_areas:
            return list(grid.values())

        # 셀 병합
        while len(grid) > target_areas:
            # 가장 가까운 두 셀 찾기
            min_dist = float('inf')
            merge_pair = None
            
            cells = list(grid.items())
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    key1, points1 = cells[i]
                    key2, points2 = cells[j]
                    
                    if not points1 or not points2:
                        continue
                    
                    center1 = get_center_point(points1)
                    center2 = get_center_point(points2)
                    dist = calculate_distance(center1, center2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (key1, key2)

            if merge_pair:
                key1, key2 = merge_pair
                grid[key1].extend(grid[key2])
                del grid[key2]

        return list(grid.values())

    def _find_best_point_to_move(
        self,
        source_points: List[DeliveryPoint],
        target_center: Tuple[float, float]
    ) -> DeliveryPoint:
        """이동하기 가장 적합한 포인트 찾기"""
        best_point = None
        min_cost = float('inf')
        
        for point in source_points:
            # 거리 기반 비용
            dist_cost = calculate_distance(
                (point.latitude, point.longitude),
                target_center
            )
            
            # 우선순위와 작업량을 고려한 비용
            priority_cost = point.get_priority_weight()
            volume_cost = point.volume / 10  # 정규화
            
            total_cost = (
                dist_cost * 0.5 +
                priority_cost * 0.3 +
                volume_cost * 0.2
            )
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_point = point
        
        return best_point