"""
배송 시간 계산 유틸리티
실제 경로 기반 정확한 배송 시간 계산
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from ..models import Order, Vehicle, Coordinates
from ..external.routing_client import RoutingClient


class TimeCalculator:
    """실제 배송 시간 계산기"""
    
    def __init__(self, use_routing_api: bool = True):
        self.logger = logging.getLogger(__name__)
        self.use_routing_api = use_routing_api
        self.routing_client = RoutingClient() if use_routing_api else None
        
        # 기본 설정값
        self.avg_speed_kmh = 30  # 도심 평균 속도
        self.delivery_time_per_order = 8  # 주문당 배송 시간 (분)
        self.setup_time_per_vehicle = 5  # 차량 준비 시간 (분)
        
        # 교통 상황별 속도 조정
        self.traffic_speed_factors = {
            'smooth': 1.2,    # 원활: 36km/h
            'normal': 1.0,    # 보통: 30km/h  
            'congested': 0.7, # 정체: 21km/h
            'heavy': 0.5      # 심각: 15km/h
        }
    
    def calculate_delivery_time(self, vehicle: Vehicle, orders: List[Order], 
                               traffic_factor: float = 1.0) -> int:
        """정확한 배송 시간 계산"""
        if not orders:
            return 0
        
        try:
            # 실제 경로 기반 계산
            if self.use_routing_api and self.routing_client:
                return self._calculate_with_routing_api(vehicle, orders, traffic_factor)
            else:
                return self._calculate_with_estimation(vehicle, orders, traffic_factor)
                
        except Exception as e:
            self.logger.warning(f"시간 계산 오류, 추정값 사용: {str(e)}")
            return self._calculate_with_estimation(vehicle, orders, traffic_factor)
    
    def _calculate_with_routing_api(self, vehicle: Vehicle, orders: List[Order], 
                                   traffic_factor: float) -> int:
        """실제 라우팅 API를 사용한 정확한 시간 계산"""
        total_time = 0
        current_location = vehicle.center_coordinates
        
        # 차량 준비 시간
        total_time += self.setup_time_per_vehicle
        
        for order in orders:
            try:
                # 실제 경로 정보 조회
                route_info = self.routing_client.calculate_route(
                    current_location, order.coordinates
                )
                
                if route_info and route_info.duration_seconds > 0:
                    # 실제 API 시간 + 교통 상황 반영
                    travel_time_minutes = (route_info.duration_seconds / 60) * traffic_factor
                else:
                    # API 실패시 추정
                    distance_km = current_location.distance_to(order.coordinates)
                    travel_time_minutes = (distance_km / self.avg_speed_kmh) * 60 * traffic_factor
                
                # 배송 시간 추가
                delivery_time = self.delivery_time_per_order
                
                total_time += travel_time_minutes + delivery_time
                current_location = order.coordinates
                
            except Exception as e:
                self.logger.warning(f"라우팅 API 오류 (주문 {order.id}): {str(e)}")
                # 추정값으로 대체
                distance_km = current_location.distance_to(order.coordinates)
                travel_time_minutes = (distance_km / self.avg_speed_kmh) * 60 * traffic_factor
                total_time += travel_time_minutes + self.delivery_time_per_order
                current_location = order.coordinates
        
        return int(total_time)
    
    def _calculate_with_estimation(self, vehicle: Vehicle, orders: List[Order], 
                                  traffic_factor: float) -> int:
        """추정 기반 시간 계산 (API 사용 불가시)"""
        total_time = self.setup_time_per_vehicle  # 준비 시간
        current_location = vehicle.center_coordinates
        
        for order in orders:
            # 직선거리 기반 이동시간 계산
            distance_km = current_location.distance_to(order.coordinates)
            
            # 도로거리는 직선거리의 1.4배로 추정 (기존 1.3배에서 개선)
            road_distance = distance_km * 1.4
            
            # 이동시간 (교통 상황 반영) - 도심 평균속도 25km/h 적용
            travel_time_minutes = (road_distance / 25) * 60 * traffic_factor
            
            # 배송시간
            delivery_time = self.delivery_time_per_order
            
            total_time += travel_time_minutes + delivery_time
            current_location = order.coordinates
        
        return int(total_time)
    
    def calculate_route_distance(self, vehicle: Vehicle, orders: List[Order]) -> float:
        """경로 총 거리 계산 (km)"""
        if not orders:
            return 0.0
        
        total_distance = 0.0
        current_location = vehicle.center_coordinates
        
        if self.use_routing_api and self.routing_client:
            # 실제 라우팅 API 사용
            for order in orders:
                try:
                    route_info = self.routing_client.calculate_route(
                        current_location, order.coordinates
                    )
                    if route_info and route_info.distance_meters > 0:
                        total_distance += route_info.distance_meters / 1000  # km 변환
                    else:
                        # API 실패시 추정 (1.4배 적용)
                        distance_km = current_location.distance_to(order.coordinates) * 1.4
                        total_distance += distance_km
                    
                    current_location = order.coordinates
                    
                except Exception as e:
                    self.logger.warning(f"거리 계산 오류: {str(e)}")
                    # 추정값 사용 (1.4배 적용)
                    distance_km = current_location.distance_to(order.coordinates) * 1.4
                    total_distance += distance_km
                    current_location = order.coordinates
        else:
            # 추정 기반 계산 (1.4배 적용)
            for order in orders:
                distance_km = current_location.distance_to(order.coordinates) * 1.4
                total_distance += distance_km
                current_location = order.coordinates
        
        return total_distance
    
    def calculate_time_efficiency(self, estimated_time: int, optimal_time: int) -> float:
        """시간 효율성 계산 (0.0-1.0)"""
        if optimal_time <= 0:
            return 0.0
        
        if estimated_time <= optimal_time:
            return 1.0
        
        # 효율성 = 최적시간 / 예상시간
        efficiency = optimal_time / estimated_time
        return max(0.0, min(1.0, efficiency))
    
    def get_traffic_adjusted_speed(self, base_speed: float, traffic_level: str) -> float:
        """교통 상황별 속도 조정"""
        factor = self.traffic_speed_factors.get(traffic_level, 1.0)
        return base_speed * factor
    
    def estimate_optimal_time_for_orders(self, orders: List[Order]) -> int:
        """주문들의 이론적 최적 시간 계산"""
        if not orders:
            return 0
        
        # 가장 효율적인 경로를 가정한 최소 시간
        min_distance = self._calculate_minimum_spanning_distance(orders)
        travel_time = (min_distance / (self.avg_speed_kmh * 1.2)) * 60  # 최적 속도
        delivery_time = len(orders) * self.delivery_time_per_order
        setup_time = self.setup_time_per_vehicle
        
        return int(travel_time + delivery_time + setup_time)
    
    def _calculate_minimum_spanning_distance(self, orders: List[Order]) -> float:
        """최소 신장 거리 계산 (TSP 근사)"""
        if len(orders) <= 1:
            return 0.0
        
        # 간단한 최근접 이웃 기반 최소 거리 추정
        visited = set()
        total_distance = 0.0
        current_order = orders[0]
        visited.add(current_order.id)
        
        while len(visited) < len(orders):
            nearest_order = None
            min_distance = float('inf')
            
            for order in orders:
                if order.id not in visited:
                    distance = current_order.coordinates.distance_to(order.coordinates)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_order = order
            
            if nearest_order:
                total_distance += min_distance * 1.4  # 도로거리 보정 (1.4배 적용)
                visited.add(nearest_order.id)
                current_order = nearest_order
        
        return total_distance
    
    def validate_time_estimation(self, estimated_time: int, orders: List[Order]) -> Dict[str, any]:
        """시간 추정 검증"""
        optimal_time = self.estimate_optimal_time_for_orders(orders)
        efficiency = self.calculate_time_efficiency(estimated_time, optimal_time)
        
        # 검증 결과
        status = "optimal"
        if estimated_time > optimal_time * 1.5:
            status = "inefficient"
        elif estimated_time > optimal_time * 1.2:
            status = "suboptimal"
        
        return {
            'estimated_time': estimated_time,
            'optimal_time': optimal_time,
            'efficiency': efficiency,
            'status': status,
            'time_difference': estimated_time - optimal_time,
            'order_count': len(orders)
        }


# 전역 인스턴스
_time_calculator = None

def get_time_calculator(use_routing_api: bool = True) -> TimeCalculator:
    """전역 TimeCalculator 인스턴스 반환"""
    global _time_calculator
    if _time_calculator is None:
        _time_calculator = TimeCalculator(use_routing_api)
    return _time_calculator