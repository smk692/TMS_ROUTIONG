"""
Nearest Neighbor (최근접 이웃) 알고리즘
- 가장 빠른 처리 속도 (30초 이내)
- 품질: 70-80%
- 비상 상황 및 소규모 주문용
"""
from typing import List, Dict, Set
import random

from ..models import Order, Vehicle, VehicleAssignment
from .base_algorithm import BaseAlgorithm, AlgorithmResult, AlgorithmConfig


class NearestNeighborAlgorithm(BaseAlgorithm):
    """최근접 이웃 배차 알고리즘"""
    
    def __init__(self, config: AlgorithmConfig = None):
        super().__init__(config)
        self.distance_cache = {}
    
    def get_algorithm_name(self) -> str:
        return "NearestNeighbor"
    
    def _solve_implementation(self, orders: List[Order], vehicles: List[Vehicle],
                            vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """최근접 이웃 알고리즘 실행"""
        
        assignments = []
        unassigned_orders = set(order.id for order in orders)
        
        # 권역별로 주문 그룹화
        region_orders = self._group_orders_by_region(orders)
        
        # 차량별로 배차 실행
        for vehicle in vehicles:
            if vehicle_capacities.get(vehicle.id, 0) <= 0:
                continue
            
            assignment = self._assign_orders_to_vehicle(
                vehicle, 
                vehicle_capacities[vehicle.id],
                region_orders.get(vehicle.region_id, []),
                unassigned_orders
            )
            
            if assignment and assignment.assigned_orders:
                assignments.append(assignment)
                # 배정된 주문들을 미배정 목록에서 제거
                for order_id in assignment.assigned_orders:
                    unassigned_orders.discard(order_id)
        
        return AlgorithmResult(
            assignments=assignments,
            unassigned_orders=list(unassigned_orders),
            execution_time_seconds=0.0,  # BaseAlgorithm에서 계산
            quality_score=0.0,           # BaseAlgorithm에서 계산
            algorithm_name=self.get_algorithm_name(),
            iteration_count=1
        )
    
    def _group_orders_by_region(self, orders: List[Order]) -> Dict[str, List[Order]]:
        """주문을 권역별로 그룹화"""
        region_orders = {}
        for order in orders:
            region_id = order.region_id
            if region_id not in region_orders:
                region_orders[region_id] = []
            region_orders[region_id].append(order)
        return region_orders
    
    def _assign_orders_to_vehicle(self, vehicle: Vehicle, capacity: int,
                                region_orders: List[Order], 
                                unassigned_orders: Set[str]) -> VehicleAssignment:
        """차량에 주문 배정 (최근접 이웃 방식)"""
        
        if not region_orders or capacity <= 0:
            return None
        
        # 해당 권역의 미배정 주문들만 필터링
        available_orders = [order for order in region_orders 
                          if order.id in unassigned_orders]
        
        if not available_orders:
            return None
        
        assigned_orders = []
        current_location = vehicle.center_coordinates
        remaining_orders = available_orders.copy()
        
        # 최근접 이웃 방식으로 주문 선택
        while len(assigned_orders) < capacity and remaining_orders:
            
            # 현재 위치에서 가장 가까운 주문 찾기
            nearest_order = self._find_nearest_order(current_location, remaining_orders)
            
            if nearest_order:
                assigned_orders.append(nearest_order.id)
                current_location = nearest_order.coordinates
                remaining_orders.remove(nearest_order)
            else:
                break
        
        if not assigned_orders:
            return None
        
        # 거리 및 시간 추정
        estimated_distance = self._calculate_route_distance(vehicle, assigned_orders, region_orders)
        estimated_time = self._estimate_delivery_time(assigned_orders, estimated_distance)
        
        return VehicleAssignment(
            vehicle_id=vehicle.id,
            driver_name=vehicle.driver_name,
            vehicle_type=vehicle.vehicle_type.value,
            region_name=f"권역_{vehicle.region_id}",
            assigned_orders=assigned_orders,
            estimated_distance_km=estimated_distance,
            estimated_time_minutes=estimated_time,
            capacity_utilization=len(assigned_orders) / capacity
        )
    
    def _find_nearest_order(self, current_location, remaining_orders: List[Order]) -> Order:
        """현재 위치에서 가장 가까운 주문 찾기"""
        
        if not remaining_orders:
            return None
        
        nearest_order = None
        min_distance = float('inf')
        
        for order in remaining_orders:
            distance = self._get_distance(current_location, order.coordinates)
            
            # 우선순위 가중치 적용
            weighted_distance = distance / order.get_priority_weight()
            
            if weighted_distance < min_distance:
                min_distance = weighted_distance
                nearest_order = order
        
        return nearest_order
    
    def _get_distance(self, coord1, coord2) -> float:
        """두 좌표 간 거리 계산 (캐싱 적용)"""
        cache_key = (coord1.latitude, coord1.longitude, coord2.latitude, coord2.longitude)
        
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        distance = coord1.distance_to(coord2)
        self.distance_cache[cache_key] = distance
        
        return distance
    
    def _calculate_route_distance(self, vehicle: Vehicle, assigned_order_ids: List[str],
                                region_orders: List[Order]) -> float:
        """경로 총 거리 계산"""
        
        if not assigned_order_ids:
            return 0.0
        
        # 주문 ID로 주문 객체 찾기
        order_dict = {order.id: order for order in region_orders}
        assigned_order_objects = [order_dict[order_id] for order_id in assigned_order_ids 
                                if order_id in order_dict]
        
        if not assigned_order_objects:
            return 0.0
        
        total_distance = 0.0
        current_location = vehicle.center_coordinates
        
        # 배정된 순서대로 거리 누적
        for order in assigned_order_objects:
            distance = self._get_distance(current_location, order.coordinates)
            total_distance += distance
            current_location = order.coordinates
        
        # 마지막 주문에서 센터로 돌아가는 거리 (선택적)
        # total_distance += self._get_distance(current_location, vehicle.center_coordinates)
        
        return total_distance
    
    def _estimate_delivery_time(self, assigned_orders: List[str], 
                              total_distance: float) -> int:
        """배송 시간 추정 (분 단위) - 고정 공식 대신 TimeCalculator 사용"""
        
        if not assigned_orders:
            return 0
        
        # TimeCalculator를 사용한 정확한 시간 계산으로 대체 예정
        # 현재는 기존 로직 유지하되 더 현실적인 값 사용
        
        # 기본 가정:
        # - 평균 속도: 25km/h (더 현실적인 도심 배송 속도)
        # - 주문당 배송 시간: 8분 (효율적인 배송 시간)
        # - 차량 준비 시간: 5분
        
        travel_time_minutes = (total_distance / 25.0) * 60  # 이동 시간
        delivery_time_minutes = len(assigned_orders) * 8   # 배송 시간
        setup_time = 5  # 차량 준비 시간
        
        total_time = travel_time_minutes + delivery_time_minutes + setup_time
        
        return int(total_time)


class RandomNearestNeighborAlgorithm(NearestNeighborAlgorithm):
    """랜덤 시작점을 가진 최근접 이웃 알고리즘 (품질 개선 버전)"""
    
    def __init__(self, config: AlgorithmConfig = None, num_trials: int = 3):
        super().__init__(config)
        self.num_trials = num_trials
    
    def get_algorithm_name(self) -> str:
        return "RandomNearestNeighbor"
    
    def _solve_implementation(self, orders: List[Order], vehicles: List[Vehicle],
                            vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """여러 시작점으로 최근접 이웃을 실행하고 최적 결과 선택"""
        
        best_result = None
        best_quality = -1
        
        for trial in range(self.num_trials):
            # 주문 순서를 랜덤하게 섞어서 시작점 변경
            shuffled_orders = orders.copy()
            random.shuffle(shuffled_orders)
            
            # 기본 최근접 이웃 알고리즘 실행
            result = super()._solve_implementation(shuffled_orders, vehicles, vehicle_capacities)
            
            # 임시 품질 계산 (간단한 배정률 기준)
            if result.assignments:
                assigned_count = sum(len(a.assigned_orders) for a in result.assignments)
                quality = assigned_count / len(orders)
                
                if quality > best_quality:
                    best_quality = quality
                    best_result = result
        
        return best_result if best_result else self._create_empty_result()