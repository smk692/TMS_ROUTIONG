"""
Hybrid VRP+TSP 알고리즘
- VRP (1단계): 차량별 주문 배정
- TSP (2단계): 차량 내 경로 순서 최적화
- 품질: 90-95%
- 대규모 주문용 (300개 이상)
"""
from typing import List, Dict, Optional, Tuple
import random
import copy
import logging
from datetime import datetime

from ..models import Order, Vehicle, VehicleAssignment
from .base_algorithm import BaseAlgorithm, AlgorithmResult, AlgorithmConfig
from ..utils.time_calculator import get_time_calculator


class HybridVRPTSPConfig(AlgorithmConfig):
    """Hybrid VRP+TSP 전용 설정"""
    def __init__(self, **kwargs):
        # 기본 AlgorithmConfig 파라미터들 분리
        base_params = {
            'time_limit_seconds': kwargs.get('time_limit_seconds', 300),
            'quality_threshold': kwargs.get('quality_threshold', 0.8),
            'max_iterations': kwargs.get('max_iterations', 1000),
            'early_stopping_enabled': kwargs.get('early_stopping_enabled', True),
            'verbose': kwargs.get('verbose', False)
        }
        super().__init__(**base_params)
        
        # VRP 단계 설정
        self.vrp_method = kwargs.get('vrp_method', 'nearest_neighbor')  # 'nearest_neighbor', 'capacity_first'
        self.vrp_iterations = kwargs.get('vrp_iterations', 100)
        
        # TSP 단계 설정
        self.tsp_method = kwargs.get('tsp_method', '2opt')  # '2opt', '3opt', 'or_opt'
        self.tsp_max_iterations = kwargs.get('tsp_max_iterations', 50)
        self.tsp_improvement_threshold = kwargs.get('tsp_improvement_threshold', 0.01)  # 1% 개선 임계값
        
        # 하이브리드 설정
        self.balance_iterations = kwargs.get('balance_iterations', 20)  # 차량간 균형 조정 반복
        self.enable_inter_vehicle_optimization = kwargs.get('enable_inter_vehicle_optimization', False)  # 기본값을 False로 변경


class VRPSolution:
    """VRP 해 클래스"""
    def __init__(self):
        self.vehicle_assignments: Dict[str, List[str]] = {}  # {vehicle_id: [order_ids]}
        self.total_distance = 0.0
        self.total_time = 0
        self.unassigned_orders: List[str] = []
        self.quality_score = 0.0


class TSPSolver:
    """TSP 최적화 솔버"""
    
    def __init__(self, time_calculator):
        self.time_calculator = time_calculator
        self.logger = logging.getLogger(__name__)
    
    def optimize_route_order(self, vehicle: Vehicle, order_ids: List[str], 
                           orders_dict: Dict[str, Order], max_iterations: int = 50) -> List[str]:
        """2-opt 알고리즘으로 경로 순서 최적화"""
        if len(order_ids) <= 2:
            return order_ids.copy()
        
        current_route = order_ids.copy()
        orders = [orders_dict[oid] for oid in order_ids if oid in orders_dict]
        
        if not orders:
            return current_route
        
        best_distance = self._calculate_route_distance(vehicle, current_route, orders_dict)
        best_route = current_route.copy()
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 2-opt 최적화
            for i in range(1, len(current_route) - 1):
                for j in range(i + 1, len(current_route)):
                    # 경로 구간 뒤집기
                    new_route = current_route.copy()
                    new_route[i:j+1] = new_route[i:j+1][::-1]
                    
                    new_distance = self._calculate_route_distance(vehicle, new_route, orders_dict)
                    
                    # 개선된 경우 업데이트
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_route = new_route.copy()
                        current_route = new_route.copy()
                        improved = True
                        break
                
                if improved:
                    break
        
        self.logger.debug(f"TSP 최적화 완료: {iteration}회 반복, 거리 {best_distance:.2f}km")
        return best_route
    
    def _calculate_route_distance(self, vehicle: Vehicle, order_ids: List[str], 
                                orders_dict: Dict[str, Order]) -> float:
        """경로 총 거리 계산"""
        if not order_ids:
            return 0.0
        
        total_distance = 0.0
        current_location = vehicle.center_coordinates
        
        for order_id in order_ids:
            if order_id in orders_dict:
                order = orders_dict[order_id]
                distance = current_location.distance_to(order.coordinates)
                total_distance += distance
                current_location = order.coordinates
        
        return total_distance


class HybridVRPTSPAlgorithm(BaseAlgorithm):
    """하이브리드 VRP+TSP 배차 최적화"""
    
    def __init__(self, config: HybridVRPTSPConfig = None):
        if config is None:
            config = HybridVRPTSPConfig()
        super().__init__(config)
        self.hybrid_config = config
        self.orders_dict = {}
        self.vehicles_dict = {}
        self.vehicle_capacities = {}
        self.tsp_solver = TSPSolver(get_time_calculator())
    
    def get_algorithm_name(self) -> str:
        return "HybridVRPTSP"
    
    def _solve_implementation(self, orders: List[Order], vehicles: List[Vehicle],
                            vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """하이브리드 VRP+TSP 실행"""
        
        # 데이터 준비
        self.orders_dict = {order.id: order for order in orders}
        self.vehicles_dict = {vehicle.id: vehicle for vehicle in vehicles}
        self.vehicle_capacities = vehicle_capacities
        
        # Phase 1: VRP - 차량별 주문 배정
        self.logger.info("Phase 1: VRP 차량 배정 시작")
        vrp_solution = self._solve_vrp_phase(orders, vehicles)
        
        # Phase 2: TSP - 차량별 경로 순서 최적화
        self.logger.info("Phase 2: TSP 경로 최적화 시작")
        optimized_solution = self._solve_tsp_phase(vrp_solution, vehicles)
        
        # Phase 3: 차량간 균형 조정 (선택적)
        if self.hybrid_config.enable_inter_vehicle_optimization:
            self.logger.info("Phase 3: 차량간 균형 조정 시작")
            balanced_solution = self._balance_vehicles(optimized_solution, vehicles)
        else:
            balanced_solution = optimized_solution
        
        # 결과 생성
        assignments = self._convert_to_assignments(balanced_solution)
        unassigned_orders = balanced_solution.unassigned_orders
        
        return AlgorithmResult(
            assignments=assignments,
            unassigned_orders=unassigned_orders,
            execution_time_seconds=0.0,
            quality_score=balanced_solution.quality_score,
            algorithm_name=self.get_algorithm_name(),
            iteration_count=0
        )
    
    def _solve_vrp_phase(self, orders: List[Order], vehicles: List[Vehicle]) -> VRPSolution:
        """VRP 단계: 차량별 주문 배정"""
        solution = VRPSolution()
        
        # 차량별 배정 초기화
        for vehicle in vehicles:
            if self.vehicle_capacities.get(vehicle.id, 0) > 0:
                solution.vehicle_assignments[vehicle.id] = []
        
        # 권역별 주문 그룹화
        region_orders = self._group_orders_by_region(orders)
        
        # 각 권역별로 차량 배정
        for region_id, region_order_list in region_orders.items():
            region_vehicles = [v for v in vehicles if v.region_id == region_id 
                             and self.vehicle_capacities.get(v.id, 0) > 0]
            
            if not region_vehicles:
                solution.unassigned_orders.extend([o.id for o in region_order_list])
                continue
            
            # VRP 배정 방법 선택
            if self.hybrid_config.vrp_method == 'nearest_neighbor':
                self._assign_by_nearest_neighbor(solution, region_order_list, region_vehicles)
            elif self.hybrid_config.vrp_method == 'capacity_first':
                self._assign_by_capacity_first(solution, region_order_list, region_vehicles)
        
        return solution
    
    def _group_orders_by_region(self, orders: List[Order]) -> Dict[str, List[Order]]:
        """주문을 권역별로 그룹화"""
        region_orders = {}
        for order in orders:
            region_id = order.region_id
            if region_id not in region_orders:
                region_orders[region_id] = []
            region_orders[region_id].append(order)
        return region_orders
    
    def _assign_by_nearest_neighbor(self, solution: VRPSolution, orders: List[Order], 
                                  vehicles: List[Vehicle]):
        """최근접 이웃 기반 VRP 배정"""
        # 우선순위별로 주문 정렬
        sorted_orders = sorted(orders, key=lambda o: -o.get_priority_weight())
        
        for order in sorted_orders:
            best_vehicle = None
            min_additional_distance = float('inf')
            
            for vehicle in vehicles:
                current_assignments = solution.vehicle_assignments[vehicle.id]
                capacity = self.vehicle_capacities[vehicle.id]
                
                # 용량 확인
                if len(current_assignments) >= capacity:
                    continue
                
                # 추가 거리 계산
                additional_distance = self._calculate_additional_distance(
                    vehicle, order, current_assignments
                )
                
                if additional_distance < min_additional_distance:
                    min_additional_distance = additional_distance
                    best_vehicle = vehicle
            
            # 최적 차량에 배정
            if best_vehicle:
                solution.vehicle_assignments[best_vehicle.id].append(order.id)
            else:
                solution.unassigned_orders.append(order.id)
    
    def _assign_by_capacity_first(self, solution: VRPSolution, orders: List[Order], 
                                vehicles: List[Vehicle]):
        """용량 우선 VRP 배정"""
        # 차량을 용량 순으로 정렬
        sorted_vehicles = sorted(vehicles, 
                               key=lambda v: self.vehicle_capacities.get(v.id, 0), 
                               reverse=True)
        
        remaining_orders = orders.copy()
        
        for vehicle in sorted_vehicles:
            capacity = self.vehicle_capacities[vehicle.id]
            vehicle_orders = []
            
            # 현재 차량에 최대한 많은 주문 배정
            while len(vehicle_orders) < capacity and remaining_orders:
                # 가장 가까운 주문 찾기
                if not vehicle_orders:
                    # 첫 번째 주문은 차량 위치에서 가장 가까운 것
                    current_location = vehicle.center_coordinates
                else:
                    # 마지막 주문 위치에서 가장 가까운 것
                    last_order = self.orders_dict[vehicle_orders[-1]]
                    current_location = last_order.coordinates
                
                nearest_order = None
                min_distance = float('inf')
                
                for order in remaining_orders:
                    distance = current_location.distance_to(order.coordinates)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_order = order
                
                if nearest_order:
                    vehicle_orders.append(nearest_order.id)
                    remaining_orders.remove(nearest_order)
                else:
                    break
            
            solution.vehicle_assignments[vehicle.id] = vehicle_orders
        
        # 남은 주문들은 미배정으로 처리
        solution.unassigned_orders.extend([o.id for o in remaining_orders])
    
    def _calculate_additional_distance(self, vehicle: Vehicle, new_order: Order, 
                                     current_assignments: List[str]) -> float:
        """새 주문 추가 시 증가하는 거리 계산"""
        if not current_assignments:
            return vehicle.center_coordinates.distance_to(new_order.coordinates)
        
        # 마지막 주문에서 새 주문까지의 거리
        last_order_id = current_assignments[-1]
        if last_order_id in self.orders_dict:
            last_order = self.orders_dict[last_order_id]
            return last_order.coordinates.distance_to(new_order.coordinates)
        
        return 0.0
    
    def _solve_tsp_phase(self, vrp_solution: VRPSolution, vehicles: List[Vehicle]) -> VRPSolution:
        """TSP 단계: 차량별 경로 순서 최적화"""
        optimized_solution = copy.deepcopy(vrp_solution)
        total_improvement = 0.0
        
        for vehicle_id, order_ids in optimized_solution.vehicle_assignments.items():
            if len(order_ids) <= 2:
                continue  # 2개 이하는 최적화할 필요 없음
            
            vehicle = self.vehicles_dict[vehicle_id]
            original_distance = self.tsp_solver._calculate_route_distance(
                vehicle, order_ids, self.orders_dict
            )
            
            # TSP 최적화 실행
            optimized_order_ids = self.tsp_solver.optimize_route_order(
                vehicle, order_ids, self.orders_dict, 
                max_iterations=self.hybrid_config.tsp_max_iterations
            )
            
            optimized_distance = self.tsp_solver._calculate_route_distance(
                vehicle, optimized_order_ids, self.orders_dict
            )
            
            # 개선된 경우 적용
            if optimized_distance < original_distance:
                improvement = original_distance - optimized_distance
                total_improvement += improvement
                optimized_solution.vehicle_assignments[vehicle_id] = optimized_order_ids
                
                self.logger.debug(f"차량 {vehicle_id} TSP 최적화: "
                               f"{original_distance:.2f}km → {optimized_distance:.2f}km "
                               f"(개선: {improvement:.2f}km)")
        
        self.logger.info(f"TSP 총 개선량: {total_improvement:.2f}km")
        return optimized_solution
    
    def _balance_vehicles(self, solution: VRPSolution, vehicles: List[Vehicle]) -> VRPSolution:
        """차량간 균형 조정 - 거리 효율성 우선"""
        balanced_solution = copy.deepcopy(solution)
        
        for iteration in range(self.hybrid_config.balance_iterations):
            improved = False
            
            # 과부하/저부하 차량 찾기
            overloaded_vehicles = []
            underloaded_vehicles = []
            
            for vehicle_id, order_ids in balanced_solution.vehicle_assignments.items():
                capacity = self.vehicle_capacities[vehicle_id]
                utilization = len(order_ids) / capacity if capacity > 0 else 0
                
                if utilization > 0.95:  # 95% 이상만 과부하로 간주 (더 엄격하게)
                    overloaded_vehicles.append((vehicle_id, order_ids))
                elif utilization < 0.4:  # 40% 미만만 저부하로 간주 (더 엄격하게)
                    underloaded_vehicles.append((vehicle_id, order_ids))
            
            # 과부하 차량에서 저부하 차량으로 주문 이동 - 거리 효율성 고려
            for overloaded_id, overloaded_orders in overloaded_vehicles:
                for underloaded_id, underloaded_orders in underloaded_vehicles:
                    if self._can_transfer_order_efficiently(overloaded_id, underloaded_id, overloaded_orders):
                        # 거리 효율성을 고려한 주문 이동
                        transferred_order = self._find_best_transfer_candidate(
                            overloaded_id, underloaded_id, overloaded_orders
                        )
                        if transferred_order:
                            overloaded_orders.remove(transferred_order)
                            underloaded_orders.append(transferred_order)
                            improved = True
                            break
            
            if not improved:
                break
        
        # 품질 점수 계산
        balanced_solution.quality_score = self._calculate_solution_quality(balanced_solution)
        
        return balanced_solution
    
    def _can_transfer_order(self, from_vehicle_id: str, to_vehicle_id: str) -> bool:
        """주문 이전 가능 여부 확인"""
        from_vehicle = self.vehicles_dict[from_vehicle_id]
        to_vehicle = self.vehicles_dict[to_vehicle_id]
        
        # 같은 권역 내에서만 이전 가능
        return from_vehicle.region_id == to_vehicle.region_id
    
    def _can_transfer_order_efficiently(self, from_vehicle_id: str, to_vehicle_id: str, orders: List[str]) -> bool:
        """거리 효율성을 고려한 주문 이전 가능 여부 확인"""
        if not self._can_transfer_order(from_vehicle_id, to_vehicle_id):
            return False
        
        # 과부하 상황에서만 이전 허용 (거리 증가 최소화)
        from_vehicle = self.vehicles_dict[from_vehicle_id]
        to_vehicle = self.vehicles_dict[to_vehicle_id]
        
        # 차량들 사이의 거리가 너무 멀면 이전하지 않음
        vehicle_distance = from_vehicle.center_coordinates.distance_to(to_vehicle.center_coordinates)
        if vehicle_distance > 10.0:  # 10km 이상 떨어진 차량간 이전 금지
            return False
            
        return len(orders) > 0
    
    def _find_best_transfer_candidate(self, from_vehicle_id: str, to_vehicle_id: str, orders: List[str]) -> str:
        """거리 효율성을 고려한 최적 이전 후보 선택"""
        if not orders:
            return None
        
        to_vehicle = self.vehicles_dict[to_vehicle_id]
        best_order = None
        min_distance_increase = float('inf')
        
        for order_id in orders:
            if order_id in self.orders_dict:
                order = self.orders_dict[order_id]
                # 받는 차량 위치에서 해당 주문까지의 거리 계산
                distance_to_new_vehicle = to_vehicle.center_coordinates.distance_to(order.coordinates)
                
                # 거리 증가가 가장 적은 주문 선택
                if distance_to_new_vehicle < min_distance_increase:
                    min_distance_increase = distance_to_new_vehicle
                    best_order = order_id
        
        return best_order
    
    def _calculate_solution_quality(self, solution: VRPSolution) -> float:
        """해의 품질 점수 계산"""
        total_orders = len(self.orders_dict)
        assigned_orders = sum(len(orders) for orders in solution.vehicle_assignments.values())
        
        if total_orders == 0:
            return 0.0
        
        assignment_rate = assigned_orders / total_orders
        
        # 거리 효율성 계산
        total_distance = 0.0
        total_capacity_utilization = 0.0
        vehicle_count = 0
        
        time_calculator = get_time_calculator()
        time_efficiency_sum = 0.0
        
        for vehicle_id, order_ids in solution.vehicle_assignments.items():
            if not order_ids:
                continue
            
            vehicle = self.vehicles_dict[vehicle_id]
            vehicle_count += 1
            
            # 거리 계산
            route_distance = time_calculator.calculate_route_distance(
                vehicle, [self.orders_dict[oid] for oid in order_ids if oid in self.orders_dict]
            )
            total_distance += route_distance
            
            # 용량 활용도
            capacity = self.vehicle_capacities[vehicle_id]
            if capacity > 0:
                total_capacity_utilization += len(order_ids) / capacity
            
            # 시간 효율성
            assigned_order_objects = [self.orders_dict[oid] for oid in order_ids if oid in self.orders_dict]
            if assigned_order_objects:
                optimal_time = time_calculator.estimate_optimal_time_for_orders(assigned_order_objects)
                estimated_time = time_calculator.calculate_delivery_time(vehicle, assigned_order_objects)
                if optimal_time > 0:
                    time_efficiency = time_calculator.calculate_time_efficiency(estimated_time, optimal_time)
                    time_efficiency_sum += time_efficiency
        
        # 평균 계산
        if vehicle_count > 0:
            avg_capacity_utilization = total_capacity_utilization / vehicle_count
            avg_time_efficiency = time_efficiency_sum / vehicle_count
        else:
            avg_capacity_utilization = 0.0
            avg_time_efficiency = 0.0
        
        # 거리 효율성 (주문 수 / 총 거리)
        distance_efficiency = assigned_orders / total_distance if total_distance > 0 else 0.0
        
        # 품질 점수 계산 (BaseAlgorithm과 동일한 가중치)
        quality_score = (
            assignment_rate * 0.4 +           # 배정률 40%
            distance_efficiency * 0.25 +      # 거리 효율성 25%
            avg_capacity_utilization * 0.15 + # 용량 활용도 15%
            avg_time_efficiency * 0.2         # 시간 효율성 20%
        )
        
        return min(1.0, quality_score)
    
    def _convert_to_assignments(self, solution: VRPSolution) -> List[VehicleAssignment]:
        """VRPSolution을 VehicleAssignment 리스트로 변환"""
        assignments = []
        time_calculator = get_time_calculator()
        
        for vehicle_id, order_ids in solution.vehicle_assignments.items():
            if not order_ids:
                continue
            
            vehicle = self.vehicles_dict[vehicle_id]
            capacity = self.vehicle_capacities[vehicle_id]
            
            # 주문 객체들 가져오기
            assigned_order_objects = [self.orders_dict[oid] for oid in order_ids if oid in self.orders_dict]
            
            if not assigned_order_objects:
                continue
            
            # 거리 및 시간 계산
            estimated_distance = time_calculator.calculate_route_distance(vehicle, assigned_order_objects)
            estimated_time = time_calculator.calculate_delivery_time(vehicle, assigned_order_objects)
            
            assignment = VehicleAssignment(
                vehicle_id=vehicle_id,
                driver_name=vehicle.driver_name,
                vehicle_type=vehicle.vehicle_type.value,
                region_name=f"권역_{vehicle.region_id}",
                assigned_orders=order_ids,
                estimated_distance_km=estimated_distance,
                estimated_time_minutes=estimated_time,
                capacity_utilization=len(order_ids) / capacity if capacity > 0 else 0
            )
            
            assignments.append(assignment)
        
        return assignments