"""
클러스터 최적화 모듈
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from geopy.distance import geodesic

from ...models import Order, Vehicle, Coordinates
from .hdbscan_clusterer import OrderCluster


class ClusterOptimizer:
    """클러스터 최적화 및 차량 매칭"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 최적화 설정
        self.max_cluster_size = config.get('max_cluster_size', 35)
        self.min_cluster_size = config.get('min_cluster_size', 8)
        self.efficiency_threshold = config.get('efficiency_threshold', 0.8)
        self.distance_weight = config.get('distance_weight', 0.6)
        self.capacity_weight = config.get('capacity_weight', 0.4)
    
    def optimize_clusters_for_vehicles(self, clusters: List[OrderCluster], 
                                     vehicles: List[Vehicle]) -> Dict[str, OrderCluster]:
        """차량 용량에 맞게 클러스터 최적화"""
        
        self.logger.info(f"클러스터 최적화 시작: {len(clusters)}개 클러스터, {len(vehicles)}대 차량")
        
        # 1. 차량별 용량 정보 추출
        vehicle_capacities = self._extract_vehicle_capacities(vehicles)
        
        # 2. 클러스터-차량 매칭 점수 계산
        cluster_vehicle_scores = self._calculate_cluster_vehicle_scores(clusters, vehicles)
        
        # 3. 최적 매칭 실행 (헝가리안 알고리즘 대신 그리디 방식)
        optimal_assignments = self._assign_clusters_to_vehicles(
            clusters, vehicles, cluster_vehicle_scores
        )
        
        # 4. 클러스터 크기 조정
        optimized_assignments = self._adjust_cluster_sizes(optimal_assignments, vehicle_capacities)
        
        self.logger.info(f"클러스터 최적화 완료: {len(optimized_assignments)}개 차량 배정")
        
        return optimized_assignments
    
    def _extract_vehicle_capacities(self, vehicles: List[Vehicle]) -> Dict[str, int]:
        """차량별 용량 정보 추출"""
        
        capacities = {}
        
        for vehicle in vehicles:
            # 차량 타입별 기본 용량 (실제 로직에서는 설정에서 가져와야 함)
            base_capacity = {
                'MOTORCYCLE': 15,
                'CAR': 25, 
                'VAN': 40,
                'TRUCK': 60
            }.get(vehicle.vehicle_type.value, 25)
            
            # 기사 경험도 등을 고려한 용량 조정이 필요하다면 여기서 처리
            capacities[vehicle.id] = base_capacity
            
        return capacities
    
    def _calculate_cluster_vehicle_scores(self, clusters: List[OrderCluster], 
                                        vehicles: List[Vehicle]) -> Dict[Tuple[int, str], float]:
        """클러스터-차량 매칭 점수 계산"""
        
        scores = {}
        
        for i, cluster in enumerate(clusters):
            for vehicle in vehicles:
                score = self._calculate_assignment_score(cluster, vehicle)
                scores[(i, vehicle.id)] = score
        
        return scores
    
    def _calculate_assignment_score(self, cluster: OrderCluster, vehicle: Vehicle) -> float:
        """개별 클러스터-차량 배정 점수 계산"""
        
        # 1. 권역 매칭 점수
        region_score = 1.0 if self._is_same_region(cluster, vehicle) else 0.3
        
        # 2. 용량 적합성 점수
        capacity_score = self._calculate_capacity_fit_score(cluster, vehicle)
        
        # 3. 거리 효율성 점수
        distance_score = self._calculate_distance_efficiency_score(cluster, vehicle)
        
        # 4. 클러스터 신뢰도 점수
        confidence_score = cluster.confidence
        
        # 가중 평균으로 최종 점수 계산
        final_score = (
            region_score * 0.3 +
            capacity_score * 0.3 +
            distance_score * 0.25 +
            confidence_score * 0.15
        )
        
        return final_score
    
    def _is_same_region(self, cluster: OrderCluster, vehicle: Vehicle) -> bool:
        """클러스터와 차량이 같은 권역인지 확인"""
        
        if not cluster.orders:
            return False
            
        # 클러스터의 첫 번째 주문의 권역과 차량 권역 비교
        cluster_region = cluster.orders[0].region_id
        return cluster_region == vehicle.region_id
    
    def _calculate_capacity_fit_score(self, cluster: OrderCluster, vehicle: Vehicle) -> float:
        """용량 적합성 점수 계산"""
        
        cluster_size = cluster.size
        vehicle_capacity = self._get_vehicle_capacity(vehicle)
        
        if cluster_size == 0:
            return 0.0
        
        if cluster_size <= vehicle_capacity:
            # 클러스터가 차량 용량에 맞는 경우
            utilization = cluster_size / vehicle_capacity
            
            # 80-95% 활용률이 최적
            if 0.8 <= utilization <= 0.95:
                return 1.0
            elif 0.6 <= utilization < 0.8:
                return 0.9
            elif 0.95 < utilization <= 1.0:
                return 0.85
            else:
                return 0.7
        else:
            # 클러스터가 차량 용량을 초과하는 경우 (분할 필요)
            excess_ratio = cluster_size / vehicle_capacity
            if excess_ratio <= 1.5:
                return 0.6  # 적당한 분할로 처리 가능
            else:
                return 0.3  # 많은 분할 필요
    
    def _calculate_distance_efficiency_score(self, cluster: OrderCluster, vehicle: Vehicle) -> float:
        """거리 효율성 점수 계산"""
        
        if not cluster.orders:
            return 0.0
        
        # 차량 위치 (여기서는 권역 중심점으로 가정)
        vehicle_location = self._get_vehicle_location(vehicle)
        
        # 클러스터 중심점까지의 거리
        distance_to_cluster = geodesic(
            (vehicle_location.latitude, vehicle_location.longitude),
            (cluster.centroid.latitude, cluster.centroid.longitude)
        ).kilometers
        
        # 거리별 점수 (5km 이내가 최적)
        if distance_to_cluster <= 5.0:
            return 1.0
        elif distance_to_cluster <= 10.0:
            return 0.8
        elif distance_to_cluster <= 15.0:
            return 0.6
        else:
            return 0.4
    
    def _get_vehicle_capacity(self, vehicle: Vehicle) -> int:
        """차량 용량 조회"""
        
        # 차량 타입별 기본 용량
        base_capacity = {
            'MOTORCYCLE': 15,
            'CAR': 25,
            'VAN': 40,
            'TRUCK': 60
        }.get(vehicle.vehicle_type.value, 25)
        
        return base_capacity
    
    def _get_vehicle_location(self, vehicle: Vehicle) -> Coordinates:
        """차량 위치 조회 (현재는 권역 중심점으로 가정)"""
        
        # 실제로는 차량의 현재 위치나 권역 중심점을 사용해야 함
        # 여기서는 임시로 서울 중심점 사용
        region_centers = {
            'CENTER_GANGNAM': Coordinates(37.5172, 127.0473),
            'CENTER_GANGBUK': Coordinates(37.6396, 127.0258),
            'CENTER_MAPO': Coordinates(37.5663, 126.9013),
            'CENTER_SONGPA': Coordinates(37.5145, 127.1059),
            'CENTER_JONGRO': Coordinates(37.5735, 126.9788),
            'CENTER_YEONGDEUNGPO': Coordinates(37.5263, 126.8966),
        }
        
        return region_centers.get(vehicle.region_id, Coordinates(37.5665, 126.9780))
    
    def _assign_clusters_to_vehicles(self, clusters: List[OrderCluster], 
                                   vehicles: List[Vehicle], 
                                   scores: Dict[Tuple[int, str], float]) -> Dict[str, OrderCluster]:
        """그리디 방식으로 클러스터를 차량에 배정"""
        
        assignments = {}
        assigned_clusters = set()
        assigned_vehicles = set()
        
        # 점수 순으로 정렬
        sorted_assignments = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for (cluster_idx, vehicle_id), score in sorted_assignments:
            # 이미 배정된 클러스터나 차량은 건너뛰기
            if cluster_idx in assigned_clusters or vehicle_id in assigned_vehicles:
                continue
            
            # 최소 점수 임계값 확인
            if score < 0.5:  # 너무 낮은 점수는 배정하지 않음
                continue
            
            # 배정 실행
            cluster = clusters[cluster_idx]
            assignments[vehicle_id] = cluster
            
            assigned_clusters.add(cluster_idx)
            assigned_vehicles.add(vehicle_id)
            
            self.logger.debug(f"차량 {vehicle_id}에 클러스터 배정: {cluster.size}개 주문, 점수: {score:.3f}")
        
        # 배정되지 않은 클러스터들을 남는 차량에 배정
        unassigned_clusters = [clusters[i] for i in range(len(clusters)) if i not in assigned_clusters]
        available_vehicles = [v for v in vehicles if v.id not in assigned_vehicles]
        
        for i, cluster in enumerate(unassigned_clusters):
            if i < len(available_vehicles):
                vehicle = available_vehicles[i]
                assignments[vehicle.id] = cluster
                self.logger.debug(f"잉여 차량 {vehicle.id}에 클러스터 배정: {cluster.size}개 주문")
        
        return assignments
    
    def _adjust_cluster_sizes(self, assignments: Dict[str, OrderCluster], 
                            vehicle_capacities: Dict[str, int]) -> Dict[str, OrderCluster]:
        """차량 용량에 맞게 클러스터 크기 조정"""
        
        adjusted_assignments = {}
        
        for vehicle_id, cluster in assignments.items():
            capacity = vehicle_capacities.get(vehicle_id, 25)
            
            if cluster.size <= capacity:
                # 용량에 맞으면 그대로 사용
                adjusted_assignments[vehicle_id] = cluster
            else:
                # 용량을 초과하면 분할
                adjusted_cluster = self._split_cluster_for_capacity(cluster, capacity)
                adjusted_assignments[vehicle_id] = adjusted_cluster
                
                self.logger.info(f"차량 {vehicle_id}: 클러스터 크기 조정 {cluster.size} → {adjusted_cluster.size}")
        
        return adjusted_assignments
    
    def _split_cluster_for_capacity(self, cluster: OrderCluster, capacity: int) -> OrderCluster:
        """용량에 맞게 클러스터 분할"""
        
        if cluster.size <= capacity:
            return cluster
        
        # 중심점에서 가장 가까운 주문들을 용량만큼 선택
        orders_with_distance = []
        
        for order in cluster.orders:
            distance = geodesic(
                (cluster.centroid.latitude, cluster.centroid.longitude),
                (order.coordinates.latitude, order.coordinates.longitude)
            ).kilometers
            orders_with_distance.append((order, distance))
        
        # 거리순으로 정렬하여 용량만큼 선택
        orders_with_distance.sort(key=lambda x: x[1])
        selected_orders = [order for order, _ in orders_with_distance[:capacity]]
        
        # 새로운 클러스터 생성
        adjusted_cluster = OrderCluster(
            orders=selected_orders,
            centroid=self._calculate_centroid(selected_orders),
            confidence=cluster.confidence * 0.9,  # 분할로 인한 신뢰도 약간 감소
            is_noise_cluster=cluster.is_noise_cluster
        )
        
        return adjusted_cluster
    
    def _calculate_centroid(self, orders: List[Order]) -> Coordinates:
        """주문들의 중심점 계산"""
        
        if not orders:
            return Coordinates(latitude=0.0, longitude=0.0)
        
        total_lat = sum(order.coordinates.latitude for order in orders)
        total_lng = sum(order.coordinates.longitude for order in orders)
        
        return Coordinates(
            latitude=total_lat / len(orders),
            longitude=total_lng / len(orders)
        )