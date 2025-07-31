"""
RouteOptimizationService - 경로 최적화 도메인 서비스

여러 엔티티에 걸친 경로 최적화 비즈니스 로직을 담당합니다.
"""
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

from src.domain.entities.vehicle import Vehicle
from src.domain.entities.delivery_order import DeliveryOrder
from src.domain.entities.route import Route
from src.domain.value_objects.coordinate import Coordinate
from src.domain.value_objects.route_segment import RouteSegment, SegmentType
from src.shared.constants import Priority, ScenarioType


class RouteOptimizationService:
    """경로 최적화 도메인 서비스"""
    
    def validate_optimization_constraints(
        self,
        vehicles: List[Vehicle], 
        orders: List[DeliveryOrder]
    ) -> List[str]:
        """
        최적화 제약 조건 검증
        
        Args:
            vehicles: 사용 가능한 차량 리스트
            orders: 배송 주문 리스트
            
        Returns:
            위반된 제약 조건 메시지 리스트
        """
        violations = []
        
        # 1. 기본 가용성 검증
        available_vehicles = [v for v in vehicles if v.is_available_for_new_order]
        if not available_vehicles:
            violations.append("No available vehicles for assignment")
        
        if not orders:
            violations.append("No delivery orders to optimize")
        
        # 2. 용량 제약 검증
        total_weight = sum(order.weight_tons for order in orders)
        total_capacity = sum(vehicle.available_capacity_tons for vehicle in available_vehicles)
        
        if total_weight > total_capacity:
            violations.append(f"Total order weight ({total_weight:.1f}t) exceeds available capacity ({total_capacity:.1f}t)")
        
        # 3. 특수 요구사항 검증
        for order in orders:
            if order.special_requirements:
                capable_vehicles = [
                    v for v in available_vehicles 
                    if all(v.has_special_capability(req) for req in order.special_requirements)
                ]
                if not capable_vehicles:
                    violations.append(f"No vehicles with required capabilities for order {order.order_id}: {order.special_requirements}")
        
        # 4. 시간 제약 검증
        urgent_orders = [order for order in orders if order.is_urgent]
        if len(urgent_orders) > len(available_vehicles):
            violations.append(f"Too many urgent orders ({len(urgent_orders)}) for available vehicles ({len(available_vehicles)})")
        
        return violations
    
    def calculate_route_priority_score(self, route: Route) -> float:
        """
        경로 우선순위 점수 계산
        
        Args:
            route: 평가할 경로
            
        Returns:
            우선순위 점수 (높을수록 우선)
        """
        if route.is_empty:
            return 0.0
        
        # 우선순위 점수 = 평균 주문 우선순위 + 긴급도 보너스 + 시간 제약 패널티
        avg_priority = sum(order.priority_score for order in route.delivery_orders) / len(route.delivery_orders)
        
        # 긴급 주문 보너스
        urgent_bonus = len(route.urgent_orders) * 2.0
        
        # 시간 제약 패널티
        time_constrained_orders = [order for order in route.delivery_orders if order.has_time_constraint]
        time_penalty = len(time_constrained_orders) * 0.5
        
        return avg_priority + urgent_bonus - time_penalty
    
    def calculate_route_efficiency_score(self, route: Route) -> float:
        """
        경로 효율성 점수 계산
        
        Args:
            route: 평가할 경로
            
        Returns:
            효율성 점수 (0.0 ~ 1.0)
        """
        if route.is_empty or route.total_distance_km <= 0:
            return 0.0
        
        # 거리 효율성 (직선 거리 대비)
        total_direct_distance = sum(
            order.delivery_distance_km for order in route.delivery_orders
        )
        
        distance_efficiency = 1.0
        if total_direct_distance > 0:
            distance_efficiency = min(1.0, total_direct_distance / route.total_distance_km)
        
        # 용량 활용률 (가정: 차량 용량 20톤)
        capacity_utilization = min(1.0, route.total_weight_tons / 20.0)
        
        # 시간 효율성 (예상 시간 대비)
        time_efficiency = 1.0
        if route.estimated_duration_hours > 0:
            optimal_time = (route.total_distance_km / 60.0) + (len(route.delivery_orders) * 0.5)
            time_efficiency = min(1.0, optimal_time / route.estimated_duration_hours)
        
        # 종합 효율성 점수
        return (distance_efficiency * 0.4) + (capacity_utilization * 0.3) + (time_efficiency * 0.3)
    
    def generate_route_segments(
        self, 
        vehicle: Vehicle, 
        orders: List[DeliveryOrder]
    ) -> List[RouteSegment]:
        """
        차량과 주문 리스트로부터 경로 구간 생성
        
        Args:
            vehicle: 배정된 차량
            orders: 배송 주문 리스트
            
        Returns:
            경로 구간 리스트
        """
        if not orders:
            return []
        
        segments = []
        current_location = vehicle.current_location
        
        for i, order in enumerate(orders):
            # 현재 위치에서 픽업 지점으로 이동
            if current_location != order.pickup_location:
                travel_segment = RouteSegment(
                    segment_id=f"travel_{i}_pickup",
                    start_location=current_location,
                    end_location=order.pickup_location,
                    segment_type=SegmentType.TRAVEL,
                    distance_km=current_location.distance_to(order.pickup_location),
                    estimated_duration_minutes=self._estimate_travel_time(
                        current_location.distance_to(order.pickup_location)
                    ),
                    sequence_number=len(segments) + 1
                )
                segments.append(travel_segment)
            
            # 픽업 작업
            pickup_segment = RouteSegment(
                segment_id=f"pickup_{order.order_id}",
                start_location=order.pickup_location,
                end_location=order.pickup_location,
                segment_type=SegmentType.PICKUP,
                distance_km=0.0,
                estimated_duration_minutes=30.0,  # 픽업 작업 시간
                order_id=order.order_id,
                sequence_number=len(segments) + 1
            )
            segments.append(pickup_segment)
            
            # 픽업 지점에서 배송 지점으로 이동
            delivery_travel_segment = RouteSegment(
                segment_id=f"travel_{i}_delivery",
                start_location=order.pickup_location,
                end_location=order.delivery_location,
                segment_type=SegmentType.TRAVEL,
                distance_km=order.delivery_distance_km,
                estimated_duration_minutes=self._estimate_travel_time(order.delivery_distance_km),
                sequence_number=len(segments) + 1
            )
            segments.append(delivery_travel_segment)
            
            # 배송 작업
            delivery_segment = RouteSegment(
                segment_id=f"delivery_{order.order_id}",
                start_location=order.delivery_location,
                end_location=order.delivery_location,
                segment_type=SegmentType.DELIVERY,
                distance_km=0.0,
                estimated_duration_minutes=20.0,  # 배송 작업 시간
                order_id=order.order_id,
                sequence_number=len(segments) + 1
            )
            segments.append(delivery_segment)
            
            current_location = order.delivery_location
        
        return segments
    
    def optimize_order_sequence_by_distance(self, orders: List[DeliveryOrder], start_location: Coordinate) -> List[DeliveryOrder]:
        """
        거리 기반 주문 순서 최적화 (Nearest Neighbor)
        
        Args:
            orders: 최적화할 주문 리스트
            start_location: 시작 위치
            
        Returns:
            최적화된 주문 순서
        """
        if len(orders) <= 1:
            return orders.copy()
        
        # 우선순위별로 먼저 정렬
        orders_by_priority = sorted(orders, key=lambda x: (-x.priority_score, x.created_at))
        
        # 긴급 주문은 앞에 배치
        urgent_orders = [order for order in orders_by_priority if order.is_urgent]
        normal_orders = [order for order in orders_by_priority if not order.is_urgent]
        
        # 일반 주문에 대해 거리 기반 최적화 적용
        optimized_normal = self._nearest_neighbor_optimization(normal_orders, start_location)
        
        return urgent_orders + optimized_normal
    
    def _estimate_travel_time(self, distance_km: float, average_speed_kmh: float = 50.0) -> float:
        """
        이동 시간 추정
        
        Args:
            distance_km: 거리 (킬로미터)
            average_speed_kmh: 평균 속도 (km/h)
            
        Returns:
            예상 시간 (분)
        """
        if distance_km <= 0:
            return 0.0
        
        hours = distance_km / average_speed_kmh
        return hours * 60.0  # 분으로 변환
    
    def _nearest_neighbor_optimization(self, orders: List[DeliveryOrder], start_location: Coordinate) -> List[DeliveryOrder]:
        """
        Nearest Neighbor 알고리즘으로 주문 순서 최적화
        
        Args:
            orders: 최적화할 주문 리스트
            start_location: 시작 위치
            
        Returns:
            최적화된 주문 순서
        """
        if not orders:
            return []
        
        remaining_orders = orders.copy()
        optimized_sequence = []
        current_location = start_location
        
        while remaining_orders:
            # 현재 위치에서 가장 가까운 픽업 지점 찾기
            nearest_order = min(
                remaining_orders,
                key=lambda order: current_location.distance_to(order.pickup_location)
            )
            
            optimized_sequence.append(nearest_order)
            remaining_orders.remove(nearest_order)
            current_location = nearest_order.delivery_location
        
        return optimized_sequence 