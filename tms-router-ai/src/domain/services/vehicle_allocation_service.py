"""
VehicleAllocationService - 차량 배정 도메인 서비스

차량과 주문 간의 최적 매칭을 담당하는 도메인 서비스입니다.
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.domain.entities.vehicle import Vehicle
from src.domain.entities.delivery_order import DeliveryOrder
from src.shared.constants import Priority


@dataclass
class AllocationScore:
    """차량-주문 배정 점수"""
    vehicle_id: str
    order_id: str
    total_score: float
    distance_score: float
    capacity_score: float
    capability_score: float
    priority_score: float
    time_score: float


class VehicleAllocationService:
    """차량 배정 도메인 서비스"""
    
    def find_optimal_vehicle_for_order(
        self, 
        order: DeliveryOrder, 
        available_vehicles: List[Vehicle]
    ) -> Optional[Vehicle]:
        """
        주문에 대한 최적 차량 찾기
        
        Args:
            order: 배송 주문
            available_vehicles: 사용 가능한 차량 리스트
            
        Returns:
            최적 차량, 없으면 None
        """
        if not available_vehicles:
            return None
        
        # 기본 제약 조건 필터링
        compatible_vehicles = self._filter_compatible_vehicles(order, available_vehicles)
        
        if not compatible_vehicles:
            return None
        
        # 각 차량에 대한 배정 점수 계산
        allocation_scores = []
        for vehicle in compatible_vehicles:
            score = self._calculate_allocation_score(order, vehicle)
            allocation_scores.append(score)
        
        # 가장 높은 점수의 차량 선택
        best_allocation = max(allocation_scores, key=lambda x: x.total_score)
        
        return next(v for v in compatible_vehicles if v.vehicle_id == best_allocation.vehicle_id)
    
    def allocate_orders_to_vehicles(
        self, 
        orders: List[DeliveryOrder], 
        vehicles: List[Vehicle]
    ) -> Dict[str, List[str]]:
        """
        여러 주문을 여러 차량에 최적 배정
        
        Args:
            orders: 배송 주문 리스트
            vehicles: 차량 리스트
            
        Returns:
            차량 ID별 할당된 주문 ID 리스트
        """
        allocation_map = {vehicle.vehicle_id: [] for vehicle in vehicles}
        unassigned_orders = orders.copy()
        available_vehicles = [v for v in vehicles if v.is_available_for_new_order]
        
        # 우선순위별로 주문 정렬 (긴급 → 높음 → 중간 → 낮음)
        unassigned_orders.sort(key=lambda x: (-x.priority_score, x.created_at))
        
        for order in unassigned_orders:
            # 현재 사용 가능한 차량 중에서 최적 차량 찾기
            best_vehicle = self.find_optimal_vehicle_for_order(order, available_vehicles)
            
            if best_vehicle:
                allocation_map[best_vehicle.vehicle_id].append(order.order_id)
                
                # 차량 용량 업데이트 (시뮬레이션)
                best_vehicle.current_load_tons += order.weight_tons
                
                # 용량이 가득 차면 사용 가능한 차량 목록에서 제거
                if best_vehicle.available_capacity_tons < 0.1:  # 100kg 미만 여유 용량
                    available_vehicles.remove(best_vehicle)
        
        return allocation_map
    
    def calculate_vehicle_efficiency(self, vehicle: Vehicle, assigned_orders: List[DeliveryOrder]) -> float:
        """
        차량 효율성 계산
        
        Args:
            vehicle: 평가할 차량
            assigned_orders: 배정된 주문 리스트
            
        Returns:
            효율성 점수 (0.0 ~ 1.0)
        """
        if not assigned_orders:
            return 0.0
        
        # 용량 활용률
        total_weight = sum(order.weight_tons for order in assigned_orders)
        capacity_utilization = min(1.0, total_weight / vehicle.capacity_tons)
        
        # 거리 효율성
        total_distance = self._calculate_total_route_distance(vehicle, assigned_orders)
        direct_distance = sum(order.delivery_distance_km for order in assigned_orders)
        
        distance_efficiency = 1.0
        if total_distance > 0:
            distance_efficiency = min(1.0, direct_distance / total_distance)
        
        # 우선순위 대응도
        high_priority_count = sum(1 for order in assigned_orders if order.is_high_priority)
        priority_response = high_priority_count / len(assigned_orders) if assigned_orders else 0.0
        
        # 종합 효율성
        return (capacity_utilization * 0.4) + (distance_efficiency * 0.4) + (priority_response * 0.2)
    
    def validate_vehicle_constraints(self, vehicle: Vehicle, orders: List[DeliveryOrder]) -> List[str]:
        """
        차량 제약 조건 검증
        
        Args:
            vehicle: 검증할 차량
            orders: 배정 예정 주문 리스트
            
        Returns:
            위반된 제약 조건 메시지 리스트
        """
        violations = []
        
        # 1. 용량 제약
        total_weight = sum(order.weight_tons for order in orders)
        if total_weight > vehicle.available_capacity_tons:
            violations.append(f"Weight exceeds capacity: {total_weight:.1f}t > {vehicle.available_capacity_tons:.1f}t")
        
        # 2. 특수 능력 제약
        for order in orders:
            for requirement in order.special_requirements:
                if not vehicle.has_special_capability(requirement):
                    violations.append(f"Vehicle lacks required capability: {requirement} for order {order.order_id}")
        
        # 3. 근무 시간 제약
        estimated_work_hours = len(orders) * 2.0  # 주문당 2시간 가정
        if vehicle.working_hours_today + estimated_work_hours > 8.0:
            violations.append(f"Would exceed working hours limit: {vehicle.working_hours_today + estimated_work_hours:.1f}h > 8h")
        
        # 4. 긴급 주문 제약
        urgent_orders = [order for order in orders if order.is_urgent]
        if len(urgent_orders) > 1:
            violations.append(f"Too many urgent orders for single vehicle: {len(urgent_orders)}")
        
        return violations
    
    def _filter_compatible_vehicles(self, order: DeliveryOrder, vehicles: List[Vehicle]) -> List[Vehicle]:
        """
        주문과 호환되는 차량 필터링
        
        Args:
            order: 배송 주문
            vehicles: 차량 리스트
            
        Returns:
            호환되는 차량 리스트
        """
        compatible = []
        
        for vehicle in vehicles:
            # 기본 가용성 확인
            if not vehicle.is_available_for_new_order:
                continue
            
            # 용량 확인
            if not vehicle.can_handle_load(order.weight_tons):
                continue
            
            # 특수 능력 확인
            if order.special_requirements:
                if not all(vehicle.has_special_capability(req) for req in order.special_requirements):
                    continue
            
            compatible.append(vehicle)
        
        return compatible
    
    def _calculate_allocation_score(self, order: DeliveryOrder, vehicle: Vehicle) -> AllocationScore:
        """
        차량-주문 배정 점수 계산
        
        Args:
            order: 배송 주문
            vehicle: 차량
            
        Returns:
            배정 점수 객체
        """
        # 1. 거리 점수 (가까울수록 높음)
        pickup_distance = vehicle.current_location.distance_to(order.pickup_location)
        max_distance = 100.0  # 최대 거리 기준 (km)
        distance_score = max(0, (max_distance - pickup_distance) / max_distance)
        
        # 2. 용량 점수 (여유 용량에 맞을수록 높음)
        capacity_ratio = order.weight_tons / vehicle.available_capacity_tons
        capacity_score = 1.0 - min(1.0, capacity_ratio)
        
        # 3. 특수 능력 점수
        capability_score = 1.0
        if order.special_requirements:
            matched_capabilities = sum(
                1 for req in order.special_requirements 
                if vehicle.has_special_capability(req)
            )
            capability_score = matched_capabilities / len(order.special_requirements)
        
        # 4. 우선순위 대응 점수
        priority_score = order.priority_score / 4.0  # 최대 4점을 1.0으로 정규화
        
        # 5. 시간 효율성 점수
        estimated_hours = pickup_distance / 50.0 + 2.0  # 이동 + 작업 시간
        time_score = max(0, (8.0 - vehicle.working_hours_today - estimated_hours) / 8.0)
        
        # 가중 평균으로 총 점수 계산
        total_score = (
            distance_score * 0.3 +
            capacity_score * 0.25 +
            capability_score * 0.2 +
            priority_score * 0.15 +
            time_score * 0.1
        )
        
        return AllocationScore(
            vehicle_id=vehicle.vehicle_id,
            order_id=order.order_id,
            total_score=total_score,
            distance_score=distance_score,
            capacity_score=capacity_score,
            capability_score=capability_score,
            priority_score=priority_score,
            time_score=time_score
        )
    
    def _calculate_total_route_distance(self, vehicle: Vehicle, orders: List[DeliveryOrder]) -> float:
        """
        차량의 총 경로 거리 계산
        
        Args:
            vehicle: 차량
            orders: 배정된 주문 리스트
            
        Returns:
            총 거리 (킬로미터)
        """
        if not orders:
            return 0.0
        
        total_distance = 0.0
        current_location = vehicle.current_location
        
        for order in orders:
            # 현재 위치 → 픽업 지점
            total_distance += current_location.distance_to(order.pickup_location)
            # 픽업 지점 → 배송 지점
            total_distance += order.delivery_distance_km
            
            current_location = order.delivery_location
        
        return total_distance 