"""
데이터 어댑터 - TMS 모델을 VRP 모델로 변환
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ...models import Order, Vehicle, Coordinates, VehicleAssignment
from ..base_algorithm import AlgorithmResult
from ..vrp_solver import VRPSolution, VRPRoute
from ..optimization.vrp_model import VRPModel, VRPLocation, VRPVehicle, VehicleType


@dataclass
class AdapterConfig:
    """어댑터 설정"""
    default_service_time_minutes: int = 8
    default_max_distance_km: float = 120.0
    default_max_work_hours: int = 8
    depot_name_template: str = "depot_{vehicle_id}"
    order_name_template: str = "order_{order_id}"


class DataAdapter:
    """TMS 데이터와 VRP 모델 간 변환 어댑터"""
    
    def __init__(self, config: AdapterConfig = None):
        self.config = config or AdapterConfig()
        self.logger = logging.getLogger(__name__)
        
        # 변환 맵핑 정보
        self._order_id_to_vrp_id = {}
        self._vrp_id_to_order_id = {}
        self._vehicle_id_to_vrp_id = {}
        self._vrp_id_to_vehicle_id = {}
        self._depot_id_to_vehicle_id = {}
    
    def convert_to_vrp_model(self, orders: List[Order], vehicles: List[Vehicle]) -> VRPModel:
        """TMS 주문/차량을 VRP 모델로 변환"""
        
        self.logger.info(f"VRP 모델 변환 시작: {len(orders)}개 주문, {len(vehicles)}대 차량")
        
        # 맵핑 정보 초기화
        self._clear_mappings()
        
        # VRP 모델 생성
        model = VRPModel("TMS_VRP_Model")
        
        # 1. Depot 위치들 추가 (차량 출발지)
        depot_locations = self._create_depot_locations(vehicles, model)
        
        # 2. 주문 위치들 추가
        self._create_order_locations(orders, model)
        
        # 3. 차량들 추가
        self._create_vrp_vehicles(vehicles, depot_locations, model)
        
        self.logger.info(f"VRP 모델 변환 완료: {len(model.locations)}개 위치, {len(model.vehicles)}대 차량")
        
        return model
    
    def convert_from_vrp_solution(self, vrp_solution: VRPSolution, 
                                 original_orders: List[Order], 
                                 original_vehicles: List[Vehicle]) -> AlgorithmResult:
        """VRP 솔루션을 TMS 결과로 변환"""
        
        self.logger.info(f"VRP 솔루션 변환 시작: {len(vrp_solution.routes)}개 경로")
        
        # 차량 배정 결과 변환
        vehicle_assignments = []
        
        for vrp_route in vrp_solution.routes:
            assignment = self._convert_vrp_route_to_assignment(vrp_route, original_vehicles)
            if assignment:
                vehicle_assignments.append(assignment)
        
        # 통계 정보 계산
        total_orders = len(original_orders)
        assigned_orders = sum(len(assignment.assigned_orders) for assignment in vehicle_assignments)
        unassigned_orders = total_orders - assigned_orders
        
        # 품질 점수 계산
        assignment_rate = assigned_orders / total_orders if total_orders > 0 else 0.0
        distance_efficiency = 1.0 / (vrp_solution.total_distance + 1)  # 거리 역수 (짧을수록 좋음)
        time_efficiency = 1.0 / (vrp_solution.total_time + 1)  # 시간 역수
        
        quality_score = (assignment_rate * 0.6 + distance_efficiency * 0.2 + time_efficiency * 0.2)
        
        # AlgorithmResult 생성
        algorithm_result = AlgorithmResult(
            algorithm_name="OR-Tools VRP",
            vehicle_assignments=vehicle_assignments,
            execution_time=vrp_solution.solve_time_seconds,
            quality_score=quality_score,
            assigned_orders=assigned_orders,
            unassigned_orders=unassigned_orders,
            total_distance_km=vrp_solution.total_distance,
            total_time_minutes=vrp_solution.total_time,
            is_optimal=vrp_solution.is_optimal,
            metadata={
                'vrp_objective_value': vrp_solution.objective_value,
                'vrp_routes_count': len(vrp_solution.routes),
                'unassigned_order_ids': vrp_solution.unassigned_orders
            }
        )
        
        self.logger.info(f"VRP 솔루션 변환 완료: {assigned_orders}개 주문 배정, 품질점수: {quality_score:.3f}")
        
        return algorithm_result
    
    def _clear_mappings(self):
        """맵핑 정보 초기화"""
        self._order_id_to_vrp_id.clear()
        self._vrp_id_to_order_id.clear()
        self._vehicle_id_to_vrp_id.clear()
        self._vrp_id_to_vehicle_id.clear()
        self._depot_id_to_vehicle_id.clear()
    
    def _create_depot_locations(self, vehicles: List[Vehicle], model: VRPModel) -> Dict[str, VRPLocation]:
        """Depot 위치 생성"""
        
        depot_locations = {}
        
        for vehicle in vehicles:
            depot_id = self.config.depot_name_template.format(vehicle_id=vehicle.id)
            
            depot_location = VRPLocation(
                id=depot_id,
                coordinates=vehicle.center_coordinates,
                demand=0,  # Depot은 수요량 0
                service_time=0  # Depot에서 서비스 시간 없음
            )
            
            model.add_location(depot_location)
            depot_locations[depot_id] = depot_location
            
            # 맵핑 정보 저장
            self._depot_id_to_vehicle_id[depot_id] = vehicle.id
        
        self.logger.debug(f"Depot 위치 {len(depot_locations)}개 생성")
        
        return depot_locations
    
    def _create_order_locations(self, orders: List[Order], model: VRPModel):
        """주문 위치 생성"""
        
        for order in orders:
            vrp_order_id = self.config.order_name_template.format(order_id=order.id)
            
            # 우선순위에 따른 시간 창 설정
            # Priority enum 문자열을 숫자로 변환
            priority_mapping = {
                'low': 1,
                'normal': 2, 
                'high': 3,
                'urgent': 4
            }
            priority_value = priority_mapping.get(order.priority.value, 2)
            
            if priority_value >= 3:  # 높은 우선순위 (high, urgent)
                time_window_end = 4 * 60  # 4시간 이내 (오전 선호)
            else:
                time_window_end = self.config.default_max_work_hours * 60  # 8시간 이내
            
            order_location = VRPLocation(
                id=vrp_order_id,
                coordinates=order.coordinates,
                demand=1,  # 주문 하나당 수요량 1
                service_time=self.config.default_service_time_minutes,
                time_window_start=0,
                time_window_end=time_window_end,
                priority=priority_value
            )
            
            model.add_location(order_location)
            
            # 맵핑 정보 저장
            self._order_id_to_vrp_id[order.id] = vrp_order_id
            self._vrp_id_to_order_id[vrp_order_id] = order.id
        
        self.logger.debug(f"주문 위치 {len(orders)}개 생성")
    
    def _create_vrp_vehicles(self, vehicles: List[Vehicle], 
                           depot_locations: Dict[str, VRPLocation], 
                           model: VRPModel):
        """VRP 차량 생성"""
        
        for vehicle in vehicles:
            depot_id = self.config.depot_name_template.format(vehicle_id=vehicle.id)
            depot_location = depot_locations[depot_id]
            
            # 차량 타입 변환
            vehicle_type_map = {
                'MOTORCYCLE': VehicleType.MOTORCYCLE,
                'CAR': VehicleType.CAR,
                'VAN': VehicleType.VAN,
                'TRUCK': VehicleType.TRUCK
            }
            
            vrp_vehicle_id = f"vrp_{vehicle.id}"
            
            vrp_vehicle = VRPVehicle(
                id=vrp_vehicle_id,
                vehicle_type=vehicle_type_map.get(vehicle.vehicle_type.value, VehicleType.CAR),
                capacity=vehicle.safe_capacity,
                start_location=depot_location,
                end_location=depot_location,  # 같은 위치로 돌아옴
                max_distance=self.config.default_max_distance_km,
                max_time=self.config.default_max_work_hours * 60,  # 분 단위
                cost_per_km=1.0,
                fixed_cost=5000.0
            )
            
            model.add_vehicle(vrp_vehicle)
            
            # 맵핑 정보 저장
            self._vehicle_id_to_vrp_id[vehicle.id] = vrp_vehicle_id
            self._vrp_id_to_vehicle_id[vrp_vehicle_id] = vehicle.id
        
        self.logger.debug(f"VRP 차량 {len(vehicles)}대 생성")
    
    def _convert_vrp_route_to_assignment(self, vrp_route: VRPRoute, 
                                       original_vehicles: List[Vehicle]) -> Optional[VehicleAssignment]:
        """VRP 경로를 차량 배정으로 변환"""
        
        # VRP 차량 ID를 원본 차량 ID로 변환
        original_vehicle_id = self._vrp_id_to_vehicle_id.get(vrp_route.vehicle_id)
        
        if not original_vehicle_id:
            self.logger.warning(f"VRP 차량 ID {vrp_route.vehicle_id}에 대응하는 원본 차량을 찾을 수 없음")
            return None
        
        # 원본 차량 정보 조회
        original_vehicle = None
        for vehicle in original_vehicles:
            if vehicle.id == original_vehicle_id:
                original_vehicle = vehicle
                break
        
        if not original_vehicle:
            self.logger.warning(f"원본 차량 ID {original_vehicle_id}를 찾을 수 없음")
            return None
        
        # VRP 주문 ID들을 원본 주문 ID들로 변환
        assigned_order_ids = []
        for vrp_order_id in vrp_route.order_sequence:
            original_order_id = self._vrp_id_to_order_id.get(vrp_order_id)
            if original_order_id:
                assigned_order_ids.append(original_order_id)
        
        if not assigned_order_ids:
            self.logger.debug(f"차량 {original_vehicle_id}: 배정된 주문이 없음")
            return None
        
        # 차량 배정 결과 생성
        assignment = VehicleAssignment(
            vehicle_id=original_vehicle.id,
            driver_name=original_vehicle.driver_name,
            vehicle_type=original_vehicle.vehicle_type.value,
            region_name=f"권역_{original_vehicle.region_id}",
            assigned_orders=assigned_order_ids,
            estimated_distance_km=vrp_route.total_distance,
            estimated_time_minutes=vrp_route.total_time,
            capacity_utilization=vrp_route.capacity_usage
        )
        
        self.logger.debug(f"차량 {original_vehicle_id}: {len(assigned_order_ids)}개 주문 배정")
        
        return assignment
    
    def get_order_vrp_id(self, order_id: str) -> Optional[str]:
        """주문 ID에 대응하는 VRP ID 조회"""
        return self._order_id_to_vrp_id.get(order_id)
    
    def get_vehicle_vrp_id(self, vehicle_id: str) -> Optional[str]:
        """차량 ID에 대응하는 VRP ID 조회"""
        return self._vehicle_id_to_vrp_id.get(vehicle_id)
    
    def get_original_order_id(self, vrp_order_id: str) -> Optional[str]:
        """VRP 주문 ID에 대응하는 원본 주문 ID 조회"""
        return self._vrp_id_to_order_id.get(vrp_order_id)
    
    def get_original_vehicle_id(self, vrp_vehicle_id: str) -> Optional[str]:
        """VRP 차량 ID에 대응하는 원본 차량 ID 조회"""
        return self._vrp_id_to_vehicle_id.get(vrp_vehicle_id)
    
    def get_mapping_summary(self) -> Dict[str, Any]:
        """맵핑 정보 요약"""
        return {
            'orders_mapped': len(self._order_id_to_vrp_id),
            'vehicles_mapped': len(self._vehicle_id_to_vrp_id),
            'depots_created': len(self._depot_id_to_vehicle_id)
        }