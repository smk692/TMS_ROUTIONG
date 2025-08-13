"""
SimpleDistanceBasedAlgorithm - 순수 거리 기반 간단한 배차 알고리즘
- 시간 균형화 로직 완전 제거
- O(n log n) 시간 복잡도
- 2명 라이더 최적화 특화
- 가장 가까운 것부터 처리하는 탐욕적 접근
"""
import logging
import time
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass

from ..models import Order, Vehicle, VehicleAssignment, Coordinates
from .base_algorithm import BaseAlgorithm, AlgorithmResult, AlgorithmConfig
from ..utils.time_calculator import get_time_calculator


@dataclass
class SimpleConfig(AlgorithmConfig):
    """Simple Distance Based Algorithm 설정"""
    
    def __init__(self, **kwargs):
        # 기본 AlgorithmConfig 파라미터들
        base_params = {
            'time_limit_seconds': kwargs.get('time_limit_seconds', 300),
            'quality_threshold': kwargs.get('quality_threshold', 0.8),
            'max_iterations': kwargs.get('max_iterations', 1),  # 단일 반복만
            'early_stopping_enabled': kwargs.get('early_stopping_enabled', True),
            'verbose': kwargs.get('verbose', False)
        }
        super().__init__(**base_params)
        
        # 단순 알고리즘 설정
        self.enable_priority_weighting = kwargs.get('enable_priority_weighting', True)
        self.max_distance_threshold = kwargs.get('max_distance_threshold', 50.0)  # 50km 제한
        self.enable_detailed_logging = kwargs.get('enable_detailed_logging', False)


class SimpleDistanceBasedAlgorithm(BaseAlgorithm):
    """순수 거리 기반 간단한 배차 알고리즘"""
    
    def __init__(self, config: SimpleConfig = None):
        if config is None:
            config = SimpleConfig()
        super().__init__(config)
        self.simple_config = config
        self.time_calculator = get_time_calculator()
        
        # 성능 통계
        self.stats = {
            'total_distance_calculations': 0,
            'orders_processed': 0,
            'vehicles_used': 0,
            'processing_time': 0.0
        }
    
    def get_algorithm_name(self) -> str:
        return "SimpleDistanceBased"
    
    def _solve_implementation(self, orders: List[Order], vehicles: List[Vehicle],
                            vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """간단한 거리 기반 알고리즘 실행"""
        
        start_time = time.perf_counter()
        
        try:
            # 초기화
            assignments = []
            unassigned_orders = set(order.id for order in orders)
            self.stats['orders_processed'] = len(orders)
            
            # 권역별로 주문 그룹화
            region_orders = self._group_orders_by_region(orders)
            
            # 차량별로 순수 거리 기반 배정
            for vehicle in vehicles:
                capacity = vehicle_capacities.get(vehicle.id, 0)
                if capacity <= 0:
                    continue
                
                assignment = self._create_distance_based_assignment(
                    vehicle, capacity, region_orders.get(vehicle.region_id, []), unassigned_orders
                )
                
                if assignment and assignment.assigned_orders:
                    assignments.append(assignment)
                    self.stats['vehicles_used'] += 1
                    
                    # 배정된 주문들을 미배정 목록에서 제거
                    for order_id in assignment.assigned_orders:
                        unassigned_orders.discard(order_id)
            
            processing_time = time.perf_counter() - start_time
            self.stats['processing_time'] = processing_time
            
            if self.simple_config.enable_detailed_logging:
                self._log_performance_stats()
            
            return AlgorithmResult(
                assignments=assignments,
                unassigned_orders=list(unassigned_orders),
                execution_time_seconds=processing_time,
                quality_score=0.0,  # BaseAlgorithm에서 계산
                algorithm_name=self.get_algorithm_name(),
                iteration_count=1
            )
            
        except Exception as e:
            self.logger.error(f"간단한 거리 기반 알고리즘 실행 오류: {str(e)}")
            return self._create_empty_result()
    
    def _group_orders_by_region(self, orders: List[Order]) -> Dict[str, List[Order]]:
        """주문을 권역별로 그룹화"""
        region_orders = {}
        for order in orders:
            region_id = order.region_id
            if region_id not in region_orders:
                region_orders[region_id] = []
            region_orders[region_id].append(order)
        return region_orders
    
    def _create_distance_based_assignment(self, vehicle: Vehicle, capacity: int,
                                        region_orders: List[Order], 
                                        unassigned_orders: Set[str]) -> Optional[VehicleAssignment]:
        """순수 거리 기반 주문 배정"""
        
        if not region_orders or capacity <= 0:
            return None
        
        # 해당 권역의 미배정 주문들만 필터링
        available_orders = [order for order in region_orders 
                          if order.id in unassigned_orders]
        
        if not available_orders:
            return None
        
        # 차량 위치에서 거리 순으로 정렬 (우선순위 가중치 적용)
        sorted_orders = self._sort_orders_by_distance(vehicle, available_orders)
        
        # 용량까지 가장 가까운 주문들 선택
        assigned_orders = []
        total_distance = 0.0
        current_location = vehicle.center_coordinates
        
        for order in sorted_orders:
            if len(assigned_orders) >= capacity:
                break
            
            # 거리 제한 확인
            distance_to_order = current_location.distance_to(order.coordinates)
            if distance_to_order > self.simple_config.max_distance_threshold:
                self.logger.warning(f"주문 {order.id}: 거리 {distance_to_order:.1f}km > 제한 {self.simple_config.max_distance_threshold}km")
                continue
            
            assigned_orders.append(order.id)
            total_distance += distance_to_order
            current_location = order.coordinates
            self.stats['total_distance_calculations'] += 1
        
        if not assigned_orders:
            return None
        
        # 시간 계산 (TimeCalculator 사용)
        assigned_order_objects = [order for order in region_orders if order.id in assigned_orders]
        estimated_time = self.time_calculator.calculate_delivery_time(vehicle, assigned_order_objects)
        
        return VehicleAssignment(
            vehicle_id=vehicle.id,
            driver_name=vehicle.driver_name,
            vehicle_type=vehicle.vehicle_type.value,
            region_name=f"권역_{vehicle.region_id}",
            assigned_orders=assigned_orders,
            estimated_distance_km=total_distance,
            estimated_time_minutes=estimated_time,
            capacity_utilization=len(assigned_orders) / capacity if capacity > 0 else 0
        )
    
    def _sort_orders_by_distance(self, vehicle: Vehicle, orders: List[Order]) -> List[Order]:
        """차량 위치에서 거리순으로 주문 정렬 (우선순위 가중치 적용)"""
        
        def get_weighted_distance(order: Order) -> float:
            """우선순위를 반영한 가중 거리 계산"""
            distance = vehicle.center_coordinates.distance_to(order.coordinates)
            
            if self.simple_config.enable_priority_weighting:
                # 우선순위 가중치 적용 (높은 우선순위 = 더 가까운 것으로 간주)
                priority_weight = order.get_priority_weight()
                weighted_distance = distance / priority_weight
            else:
                weighted_distance = distance
            
            return weighted_distance
        
        # 가중 거리 순으로 정렬
        return sorted(orders, key=get_weighted_distance)
    
    def _create_empty_result(self) -> AlgorithmResult:
        """빈 결과 생성"""
        return AlgorithmResult(
            assignments=[],
            unassigned_orders=[],
            execution_time_seconds=0.0,
            quality_score=0.0,
            algorithm_name=self.get_algorithm_name(),
            iteration_count=0
        )
    
    def _log_performance_stats(self) -> None:
        """성능 통계 로그"""
        self.logger.info("=== 간단한 거리 기반 알고리즘 성능 통계 ===")
        self.logger.info(f"처리된 주문: {self.stats['orders_processed']}개")
        self.logger.info(f"사용된 차량: {self.stats['vehicles_used']}대")
        self.logger.info(f"거리 계산: {self.stats['total_distance_calculations']}회")
        self.logger.info(f"처리 시간: {self.stats['processing_time']:.3f}초")
        
        if self.stats['orders_processed'] > 0:
            orders_per_second = self.stats['orders_processed'] / self.stats['processing_time']
            self.logger.info(f"처리 속도: {orders_per_second:.1f} 주문/초")
    
    def get_performance_stats(self) -> Dict[str, any]:
        """성능 통계 반환"""
        return {
            'algorithm_stats': self.stats.copy(),
            'algorithm_type': 'simple_distance_based',
            'optimization_features': {
                'priority_weighting': self.simple_config.enable_priority_weighting,
                'distance_threshold': self.simple_config.max_distance_threshold
            }
        }


class FastestDistanceAlgorithm(SimpleDistanceBasedAlgorithm):
    """초고속 거리 기반 알고리즘 - 2명 라이더 특화"""
    
    def __init__(self, config: SimpleConfig = None):
        if config is None:
            config = SimpleConfig()
        super().__init__(config)
        # 우선순위 가중치 비활성화로 최대 속도
        self.simple_config.enable_priority_weighting = False
        self.simple_config.max_distance_threshold = 30.0  # 더 엄격한 거리 제한
    
    def get_algorithm_name(self) -> str:
        return "FastestDistance"
    
    def _sort_orders_by_distance(self, vehicle: Vehicle, orders: List[Order]) -> List[Order]:
        """순수 거리순 정렬 (우선순위 무시)"""
        return sorted(orders, key=lambda order: vehicle.center_coordinates.distance_to(order.coordinates))