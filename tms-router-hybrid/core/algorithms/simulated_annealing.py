"""
Simulated Annealing (시뮬레이티드 어닐링) 알고리즘
- 품질 우선 처리 (5-10분)
- 품질: 88-93%
- 대규모 주문용 (101-300개)
"""
from typing import List, Dict, Optional
import random
import math
import copy

from ..models import Order, Vehicle, VehicleAssignment
from .base_algorithm import BaseAlgorithm, AlgorithmResult, AlgorithmConfig
from ..utils.time_calculator import get_time_calculator


class SimulatedAnnealingConfig(AlgorithmConfig):
    """시뮬레이티드 어닐링 전용 설정"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_temperature = kwargs.get('initial_temperature', 1000.0)
        self.final_temperature = kwargs.get('final_temperature', 1.0)
        self.cooling_rate = kwargs.get('cooling_rate', 0.95)
        self.max_iterations_per_temp = kwargs.get('max_iterations_per_temp', 100)
        self.neighbor_types = kwargs.get('neighbor_types', ['swap', 'relocate', 'reverse'])


class Solution:
    """해 클래스"""
    
    def __init__(self, vehicle_assignments: Dict[str, List[str]], cost: float = float('inf')):
        self.vehicle_assignments = vehicle_assignments  # {vehicle_id: [order_ids]}
        self.cost = cost  # 비용 (낮을수록 좋음)
        self.total_distance = 0.0
        self.assignment_count = 0
        
    def copy(self):
        """해 복사"""
        new_solution = Solution(
            vehicle_assignments=copy.deepcopy(self.vehicle_assignments),
            cost=self.cost
        )
        new_solution.total_distance = self.total_distance
        new_solution.assignment_count = self.assignment_count
        return new_solution


class SimulatedAnnealingAlgorithm(BaseAlgorithm):
    """시뮬레이티드 어닐링 배차 최적화"""
    
    def __init__(self, config: SimulatedAnnealingConfig = None):
        if config is None:
            config = SimulatedAnnealingConfig()
        super().__init__(config)
        self.sa_config = config
        self.orders_dict = {}
        self.vehicles_dict = {}
        self.vehicle_capacities = {}
        self.distance_cache = {}
    
    def get_algorithm_name(self) -> str:
        return "SimulatedAnnealing"
    
    def _solve_implementation(self, orders: List[Order], vehicles: List[Vehicle],
                            vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """시뮬레이티드 어닐링 실행"""
        
        # 데이터 준비
        self.orders_dict = {order.id: order for order in orders}
        self.vehicles_dict = {vehicle.id: vehicle for vehicle in vehicles}
        self.vehicle_capacities = vehicle_capacities
        
        # 초기 해 생성
        current_solution = self._create_initial_solution(orders, vehicles)
        current_solution.cost = self._calculate_cost(current_solution)
        
        # 최적 해 추적
        best_solution = current_solution.copy()
        
        # 온도 초기화
        temperature = self.sa_config.initial_temperature
        iteration = 0
        
        # 시뮬레이티드 어닐링 메인 루프
        while temperature > self.sa_config.final_temperature:
            
            # 현재 온도에서 여러 번 시도
            for _ in range(self.sa_config.max_iterations_per_temp):
                iteration += 1
                
                # 이웃해 생성
                neighbor_solution = self._generate_neighbor(current_solution)
                neighbor_solution.cost = self._calculate_cost(neighbor_solution)
                
                # 해 품질 비교
                delta_cost = neighbor_solution.cost - current_solution.cost
                
                # 수용 여부 결정
                if self._should_accept(delta_cost, temperature):
                    current_solution = neighbor_solution
                    
                    # 최적해 업데이트
                    if current_solution.cost < best_solution.cost:
                        best_solution = current_solution.copy()
                
                # 진행상황 로깅 (품질을 적합도로 변환)
                current_fitness = 1 / (1 + current_solution.cost)
                self._log_progress(iteration, current_fitness, 
                                 f"(온도: {temperature:.1f}, 비용: {current_solution.cost:.2f})")
                
                # 조기 종료 조건 확인
                if self._should_stop_early(current_fitness, iteration):
                    break
            
            # 온도 감소
            temperature *= self.sa_config.cooling_rate
        
        # 결과 생성
        assignments = self._convert_to_assignments(best_solution)
        unassigned_orders = self._find_unassigned_orders(best_solution, orders)
        
        return AlgorithmResult(
            assignments=assignments,
            unassigned_orders=unassigned_orders,
            execution_time_seconds=0.0,
            quality_score=0.0,
            algorithm_name=self.get_algorithm_name(),
            iteration_count=iteration
        )
    
    def _create_initial_solution(self, orders: List[Order], vehicles: List[Vehicle]) -> Solution:
        """초기 해 생성 (그리디 방식)"""
        vehicle_assignments = {vehicle.id: [] for vehicle in vehicles 
                             if self.vehicle_capacities.get(vehicle.id, 0) > 0}
        
        # 권역별로 주문 그룹화
        region_orders = {}
        for order in orders:
            region_id = order.region_id
            if region_id not in region_orders:
                region_orders[region_id] = []
            region_orders[region_id].append(order)
        
        # 각 권역별로 그리디 배정
        for region_id, region_order_list in region_orders.items():
            # 해당 권역의 차량들
            region_vehicles = [v for v in vehicles if v.region_id == region_id 
                             and self.vehicle_capacities.get(v.id, 0) > 0]
            
            if not region_vehicles:
                continue
            
            # 우선순위 기반으로 정렬
            sorted_orders = sorted(region_order_list, 
                                 key=lambda o: -o.get_priority_weight())
            
            # 차량별 용량을 고려하여 배정
            for order in sorted_orders:
                best_vehicle = None
                min_additional_distance = float('inf')
                
                for vehicle in region_vehicles:
                    current_orders = vehicle_assignments[vehicle.id]
                    capacity = self.vehicle_capacities[vehicle.id]
                    
                    if len(current_orders) < capacity:
                        # 추가 거리 계산
                        additional_distance = self._calculate_additional_distance(
                            vehicle, order, current_orders
                        )
                        
                        if additional_distance < min_additional_distance:
                            min_additional_distance = additional_distance
                            best_vehicle = vehicle
                
                if best_vehicle:
                    vehicle_assignments[best_vehicle.id].append(order.id)
        
        return Solution(vehicle_assignments)
    
    def _calculate_additional_distance(self, vehicle: Vehicle, new_order: Order, 
                                     current_orders: List[str]) -> float:
        """새 주문 추가 시 추가되는 거리 계산"""
        if not current_orders:
            # 첫 주문인 경우: 센터에서 주문지까지의 거리
            return self._get_distance(vehicle.center_coordinates, new_order.coordinates)
        
        # 마지막 주문에서 새 주문까지의 거리 (간단한 추정)
        last_order_id = current_orders[-1]
        if last_order_id in self.orders_dict:
            last_order = self.orders_dict[last_order_id]
            return self._get_distance(last_order.coordinates, new_order.coordinates)
        
        return 0.0
    
    def _generate_neighbor(self, solution: Solution) -> Solution:
        """이웃해 생성"""
        neighbor = solution.copy()
        
        # 랜덤하게 이웃해 생성 방법 선택
        neighbor_type = random.choice(self.sa_config.neighbor_types)
        
        if neighbor_type == 'swap':
            self._swap_neighbor(neighbor)
        elif neighbor_type == 'relocate':
            self._relocate_neighbor(neighbor)
        elif neighbor_type == 'reverse':
            self._reverse_neighbor(neighbor)
        
        return neighbor
    
    def _swap_neighbor(self, solution: Solution):
        """교환 이웃해: 두 차량 간 주문 교환"""
        vehicles_with_orders = [(vid, orders) for vid, orders in solution.vehicle_assignments.items() if orders]
        
        if len(vehicles_with_orders) < 2:
            return
        
        # 두 차량 선택
        (vehicle1_id, orders1), (vehicle2_id, orders2) = random.sample(vehicles_with_orders, 2)
        
        if orders1 and orders2:
            # 각 차량에서 주문 하나씩 선택하여 교환
            order1 = random.choice(orders1)
            order2 = random.choice(orders2)
            
            orders1.remove(order1)
            orders1.append(order2)
            orders2.remove(order2)
            orders2.append(order1)
    
    def _relocate_neighbor(self, solution: Solution):
        """재배치 이웃해: 한 주문을 다른 차량으로 이동"""
        non_empty_vehicles = [(vid, orders) for vid, orders in solution.vehicle_assignments.items() if orders]
        all_vehicles = list(solution.vehicle_assignments.keys())
        
        if not non_empty_vehicles:
            return
        
        # 소스 차량에서 주문 선택
        source_vehicle_id, source_orders = random.choice(non_empty_vehicles)
        relocated_order = random.choice(source_orders)
        
        # 타겟 차량 선택 (용량 확인)
        available_targets = []
        for vehicle_id in all_vehicles:
            if vehicle_id != source_vehicle_id:
                current_load = len(solution.vehicle_assignments[vehicle_id])
                capacity = self.vehicle_capacities.get(vehicle_id, 0)
                if current_load < capacity:
                    available_targets.append(vehicle_id)
        
        if available_targets:
            target_vehicle_id = random.choice(available_targets)
            source_orders.remove(relocated_order)
            solution.vehicle_assignments[target_vehicle_id].append(relocated_order)
    
    def _reverse_neighbor(self, solution: Solution):
        """역순 이웃해: 한 차량의 주문 순서를 부분적으로 뒤집기"""
        vehicles_with_orders = [(vid, orders) for vid, orders in solution.vehicle_assignments.items() 
                              if len(orders) > 2]
        
        if not vehicles_with_orders:
            return
        
        vehicle_id, orders = random.choice(vehicles_with_orders)
        
        # 뒤집을 구간 선택
        start = random.randint(0, len(orders) - 2)
        end = random.randint(start + 1, len(orders) - 1)
        
        # 해당 구간 뒤집기
        orders[start:end+1] = orders[start:end+1][::-1]
    
    def _calculate_cost(self, solution: Solution) -> float:
        """해의 비용 계산 (낮을수록 좋음)"""
        total_cost = 0.0
        total_distance = 0.0
        assignment_count = 0
        time_calculator = get_time_calculator()
        
        for vehicle_id, order_ids in solution.vehicle_assignments.items():
            if not order_ids:
                continue
            
            vehicle = self.vehicles_dict[vehicle_id]
            assignment_count += len(order_ids)
            
            # 용량 초과 패널티
            capacity = self.vehicle_capacities[vehicle_id]
            if len(order_ids) > capacity:
                total_cost += (len(order_ids) - capacity) * 1000  # 큰 패널티
                continue
            
            # 해당 주문들의 Order 객체 가져오기
            assigned_order_objects = [self.orders_dict[oid] for oid in order_ids if oid in self.orders_dict]
            
            if not assigned_order_objects:
                continue
            
            # 경로 거리 계산
            route_distance = self._calculate_route_distance(vehicle, order_ids)
            total_distance += route_distance
            
            # 거리 비용
            total_cost += route_distance * 8  # 가중치 조정 (10->8)
            
            # 시간 효율성 비용 추가
            optimal_time = time_calculator.estimate_optimal_time_for_orders(assigned_order_objects)
            estimated_time = self._calculate_estimated_time(vehicle, order_ids)
            
            if optimal_time > 0:
                time_inefficiency = max(0, (estimated_time - optimal_time) / optimal_time)
                total_cost += time_inefficiency * 30  # 시간 비효율성 패널티
            
            # 용량 활용도 보너스 (비용 감소)
            utilization = len(order_ids) / capacity
            total_cost -= utilization * 40  # 가중치 조정 (50->40)
        
        # 미배정 주문 패널티
        total_orders = len(self.orders_dict)
        unassigned_count = total_orders - assignment_count
        total_cost += unassigned_count * 100
        
        # 해 객체에 통계 저장
        solution.total_distance = total_distance
        solution.assignment_count = assignment_count
        
        return total_cost
    
    def _calculate_estimated_time(self, vehicle: Vehicle, order_ids: List[str]) -> int:
        """배정된 주문들의 예상 시간 계산"""
        if not order_ids:
            return 0
            
        route_distance = self._calculate_route_distance(vehicle, order_ids)
        travel_time = int(route_distance / 25 * 60)  # 25km/h 기준
        delivery_time = len(order_ids) * 8  # 주문당 8분
        setup_time = 5  # 차량 준비시간
        
        return travel_time + delivery_time + setup_time
    
    def _calculate_route_distance(self, vehicle: Vehicle, order_ids: List[str]) -> float:
        """경로 총 거리 계산"""
        if not order_ids:
            return 0.0
        
        total_distance = 0.0
        current_location = vehicle.center_coordinates
        
        for order_id in order_ids:
            if order_id in self.orders_dict:
                order = self.orders_dict[order_id]
                distance = self._get_distance(current_location, order.coordinates)
                total_distance += distance
                current_location = order.coordinates
        
        return total_distance
    
    def _get_distance(self, coord1, coord2) -> float:
        """거리 계산 (캐싱 적용)"""
        cache_key = (coord1.latitude, coord1.longitude, coord2.latitude, coord2.longitude)
        
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        distance = coord1.distance_to(coord2)
        self.distance_cache[cache_key] = distance
        
        return distance
    
    def _should_accept(self, delta_cost: float, temperature: float) -> bool:
        """수용 확률 계산"""
        if delta_cost <= 0:
            return True  # 개선되는 해는 항상 수용
        
        # 볼츠만 확률 계산
        probability = math.exp(-delta_cost / temperature)
        return random.random() < probability
    
    def _convert_to_assignments(self, solution: Solution) -> List[VehicleAssignment]:
        """Solution을 VehicleAssignment 리스트로 변환"""
        assignments = []
        
        for vehicle_id, order_ids in solution.vehicle_assignments.items():
            if not order_ids:
                continue
            
            vehicle = self.vehicles_dict[vehicle_id]
            capacity = self.vehicle_capacities[vehicle_id]
            
            # 거리 및 시간 계산
            estimated_distance = self._calculate_route_distance(vehicle, order_ids)
            # 개선된 시간 계산: 더 현실적인 공식 적용
            travel_time = int(estimated_distance / 25 * 60)  # 25km/h 기준 이동시간
            delivery_time = len(order_ids) * 8  # 주문당 8분 배송시간
            setup_time = 5  # 차량 준비시간
            estimated_time = travel_time + delivery_time + setup_time
            
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
    
    def _find_unassigned_orders(self, solution: Solution, orders: List[Order]) -> List[str]:
        """미배정 주문 찾기"""
        all_assigned = set()
        for order_ids in solution.vehicle_assignments.values():
            all_assigned.update(order_ids)
        
        all_order_ids = {order.id for order in orders}
        unassigned = all_order_ids - all_assigned
        
        return list(unassigned)