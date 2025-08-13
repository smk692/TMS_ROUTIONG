"""
Genetic Algorithm (유전자 알고리즘) 
- 균형잡힌 처리 시간 (2-5분)
- 품질: 85-90%
- 중간 규모 주문용 (31-100개)
"""
from typing import List, Dict, Tuple, Optional
import random
import copy

from ..models import Order, Vehicle, VehicleAssignment
from .base_algorithm import BaseAlgorithm, AlgorithmResult, AlgorithmConfig
from ..utils.time_calculator import get_time_calculator


class GeneticAlgorithmConfig(AlgorithmConfig):
    """유전자 알고리즘 전용 설정"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.population_size = kwargs.get('population_size', 100)
        self.generations = kwargs.get('generations', 200)
        self.crossover_rate = kwargs.get('crossover_rate', 0.7)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.elitism_count = kwargs.get('elitism_count', 10)
        self.tournament_size = kwargs.get('tournament_size', 5)


class Individual:
    """개체 (해) 클래스"""
    
    def __init__(self, vehicle_assignments: Dict[str, List[str]], fitness: float = 0.0):
        self.vehicle_assignments = vehicle_assignments  # {vehicle_id: [order_ids]}
        self.fitness = fitness
        self.age = 0
    
    def copy(self):
        """개체 복사"""
        return Individual(
            vehicle_assignments=copy.deepcopy(self.vehicle_assignments),
            fitness=self.fitness
        )


class GeneticAlgorithm(BaseAlgorithm):
    """유전자 알고리즘 배차 최적화"""
    
    def __init__(self, config: GeneticAlgorithmConfig = None):
        if config is None:
            config = GeneticAlgorithmConfig()
        super().__init__(config)
        self.ga_config = config
        self.orders_dict = {}
        self.vehicles_dict = {}
        self.vehicle_capacities = {}
    
    def get_algorithm_name(self) -> str:
        return "GeneticAlgorithm"
    
    def _solve_implementation(self, orders: List[Order], vehicles: List[Vehicle],
                            vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """유전자 알고리즘 실행"""
        
        # 데이터 준비
        self.orders_dict = {order.id: order for order in orders}
        self.vehicles_dict = {vehicle.id: vehicle for vehicle in vehicles}
        self.vehicle_capacities = vehicle_capacities
        
        # 초기 인구 생성
        population = self._create_initial_population(orders, vehicles)
        
        best_individual = None
        best_fitness = -1
        stagnation_count = 0
        
        # 진화 과정
        for generation in range(self.ga_config.generations):
            # 적합도 평가
            for individual in population:
                individual.fitness = self._evaluate_fitness(individual)
            
            # 인구를 적합도 순으로 정렬
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # 최적 개체 업데이트
            current_best = population[0]
            if current_best.fitness > best_fitness:
                best_individual = current_best.copy()
                best_fitness = current_best.fitness
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # 진행상황 로깅
            self._log_progress(generation, best_fitness, 
                             f"(평균: {sum(ind.fitness for ind in population)/len(population):.3f})")
            
            # 조기 종료 조건 확인
            if self._should_stop_early(best_fitness, generation):
                break
            
            # 정체 상태 확인 (50세대 동안 개선 없음)
            if stagnation_count >= 50:
                self.logger.info(f"정체 상태로 인한 조기 종료: {generation}세대")
                break
            
            # 새로운 세대 생성
            population = self._create_next_generation(population)
        
        # 결과 생성
        assignments = self._convert_to_assignments(best_individual) if best_individual else []
        unassigned_orders = self._find_unassigned_orders(best_individual, orders) if best_individual else [o.id for o in orders]
        
        return AlgorithmResult(
            assignments=assignments,
            unassigned_orders=unassigned_orders,
            execution_time_seconds=0.0,
            quality_score=0.0,
            algorithm_name=self.get_algorithm_name(),
            iteration_count=generation + 1
        )
    
    def _create_initial_population(self, orders: List[Order], vehicles: List[Vehicle]) -> List[Individual]:
        """초기 인구 생성"""
        population = []
        
        for _ in range(self.ga_config.population_size):
            # 랜덤 배정으로 개체 생성
            individual = self._create_random_individual(orders, vehicles)
            population.append(individual)
        
        return population
    
    def _create_random_individual(self, orders: List[Order], vehicles: List[Vehicle]) -> Individual:
        """랜덤 개체 생성"""
        vehicle_assignments = {vehicle.id: [] for vehicle in vehicles 
                             if self.vehicle_capacities.get(vehicle.id, 0) > 0}
        
        available_orders = orders.copy()
        random.shuffle(available_orders)
        
        for order in available_orders:
            # 해당 권역의 차량들 중에서 랜덤 선택
            eligible_vehicles = [v for v in vehicles 
                               if v.region_id == order.region_id and 
                               self.vehicle_capacities.get(v.id, 0) > 0]
            
            if not eligible_vehicles:
                continue
            
            # 용량이 남아있는 차량 중에서 랜덤 선택
            available_vehicles = []
            for vehicle in eligible_vehicles:
                current_load = len(vehicle_assignments[vehicle.id])
                capacity = self.vehicle_capacities[vehicle.id]
                if current_load < capacity:
                    available_vehicles.append(vehicle)
            
            if available_vehicles:
                selected_vehicle = random.choice(available_vehicles)
                vehicle_assignments[selected_vehicle.id].append(order.id)
        
        return Individual(vehicle_assignments)
    
    def _evaluate_fitness(self, individual: Individual) -> float:
        """개체의 적합도 평가"""
        
        total_fitness = 0.0
        total_orders = len(self.orders_dict)
        assigned_orders = 0
        time_calculator = get_time_calculator()
        
        for vehicle_id, order_ids in individual.vehicle_assignments.items():
            if not order_ids:
                continue
            
            vehicle = self.vehicles_dict[vehicle_id]
            assigned_orders += len(order_ids)
            
            # 용량 제약 위반 패널티
            capacity = self.vehicle_capacities[vehicle_id]
            if len(order_ids) > capacity:
                total_fitness -= (len(order_ids) - capacity) * 10  # 패널티
                continue
            
            # 해당 주문들의 Order 객체 가져오기
            assigned_order_objects = [self.orders_dict[oid] for oid in order_ids if oid in self.orders_dict]
            
            if not assigned_order_objects:
                continue
            
            # 거리 효율성 계산
            route_distance = self._calculate_route_distance_for_fitness(vehicle, order_ids)
            if route_distance > 0 and len(order_ids) > 0:
                distance_efficiency = len(order_ids) / route_distance
                total_fitness += distance_efficiency * 8  # 가중치 조정 (10->8)
            
            # 시간 효율성 계산 및 보너스
            optimal_time = time_calculator.estimate_optimal_time_for_orders(assigned_order_objects)
            estimated_time = self._calculate_estimated_time(vehicle, order_ids)
            
            if optimal_time > 0:
                time_efficiency = time_calculator.calculate_time_efficiency(estimated_time, optimal_time)
                total_fitness += time_efficiency * 6  # 시간 효율성 보너스
            
            # 용량 활용도 보너스
            utilization = len(order_ids) / capacity
            total_fitness += utilization * 4  # 가중치 조정 (5->4)
        
        # 배정률 보너스
        if total_orders > 0:
            assignment_rate = assigned_orders / total_orders
            total_fitness += assignment_rate * 40  # 가중치 조정 (50->40)
        
        return max(0, total_fitness)
    
    def _calculate_estimated_time(self, vehicle: Vehicle, order_ids: List[str]) -> int:
        """배정된 주문들의 예상 시간 계산"""
        if not order_ids:
            return 0
            
        route_distance = self._calculate_route_distance_for_fitness(vehicle, order_ids)
        travel_time = int(route_distance / 25 * 60)  # 25km/h 기준
        delivery_time = len(order_ids) * 8  # 주문당 8분
        setup_time = 5  # 차량 준비시간
        
        return travel_time + delivery_time + setup_time
    
    def _calculate_route_distance_for_fitness(self, vehicle: Vehicle, order_ids: List[str]) -> float:
        """적합도 계산용 간단한 거리 계산"""
        if not order_ids:
            return 0.0
        
        total_distance = 0.0
        current_location = vehicle.center_coordinates
        
        for order_id in order_ids:
            if order_id in self.orders_dict:
                order = self.orders_dict[order_id]
                distance = current_location.distance_to(order.coordinates)
                total_distance += distance
                current_location = order.coordinates
        
        return total_distance
    
    def _create_next_generation(self, population: List[Individual]) -> List[Individual]:
        """다음 세대 생성"""
        next_generation = []
        
        # 엘리티즘: 상위 개체들 보존
        elite_count = min(self.ga_config.elitism_count, len(population))
        for i in range(elite_count):
            next_generation.append(population[i].copy())
        
        # 나머지 개체들 생성
        while len(next_generation) < self.ga_config.population_size:
            # 부모 선택 (토너먼트 선택)
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # 교배
            if random.random() < self.ga_config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 돌연변이
            if random.random() < self.ga_config.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.ga_config.mutation_rate:
                child2 = self._mutate(child2)
            
            next_generation.append(child1)
            if len(next_generation) < self.ga_config.population_size:
                next_generation.append(child2)
        
        return next_generation[:self.ga_config.population_size]
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """토너먼트 선택"""
        tournament_size = min(self.ga_config.tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """교배 (단순 순서 기반 교배)"""
        
        child1_assignments = {}
        child2_assignments = {}
        
        # 각 차량에 대해 부모들의 배정을 조합
        all_vehicle_ids = set(parent1.vehicle_assignments.keys()) | set(parent2.vehicle_assignments.keys())
        
        for vehicle_id in all_vehicle_ids:
            orders1 = parent1.vehicle_assignments.get(vehicle_id, [])
            orders2 = parent2.vehicle_assignments.get(vehicle_id, [])
            
            # 교배 지점 결정
            if orders1 and orders2:
                crossover_point = random.randint(0, min(len(orders1), len(orders2)))
                child1_assignments[vehicle_id] = orders1[:crossover_point] + orders2[crossover_point:]
                child2_assignments[vehicle_id] = orders2[:crossover_point] + orders1[crossover_point:]
            else:
                child1_assignments[vehicle_id] = orders1.copy()
                child2_assignments[vehicle_id] = orders2.copy()
        
        # 중복 주문 제거 및 용량 제약 확인
        child1_assignments = self._repair_individual(child1_assignments)
        child2_assignments = self._repair_individual(child2_assignments)
        
        return Individual(child1_assignments), Individual(child2_assignments)
    
    def _mutate(self, individual: Individual) -> Individual:
        """돌연변이"""
        mutated = individual.copy()
        
        # 돌연변이 타입 랜덤 선택
        mutation_type = random.choice(['swap', 'relocate', 'shuffle'])
        
        if mutation_type == 'swap':
            self._swap_mutation(mutated)
        elif mutation_type == 'relocate':
            self._relocate_mutation(mutated)
        elif mutation_type == 'shuffle':
            self._shuffle_mutation(mutated)
        
        return mutated
    
    def _swap_mutation(self, individual: Individual):
        """교환 돌연변이"""
        vehicle_ids = list(individual.vehicle_assignments.keys())
        if len(vehicle_ids) < 2:
            return
        
        # 두 차량 선택
        vehicle1, vehicle2 = random.sample(vehicle_ids, 2)
        orders1 = individual.vehicle_assignments[vehicle1]
        orders2 = individual.vehicle_assignments[vehicle2]
        
        # 각 차량에서 주문 하나씩 교환
        if orders1 and orders2:
            order1 = random.choice(orders1)
            order2 = random.choice(orders2)
            
            orders1.remove(order1)
            orders1.append(order2)
            orders2.remove(order2)
            orders2.append(order1)
    
    def _relocate_mutation(self, individual: Individual):
        """재배치 돌연변이"""
        # 한 차량에서 주문을 다른 차량으로 이동
        non_empty_vehicles = [vid for vid, orders in individual.vehicle_assignments.items() if orders]
        
        if not non_empty_vehicles:
            return
        
        source_vehicle = random.choice(non_empty_vehicles)
        target_vehicle = random.choice(list(individual.vehicle_assignments.keys()))
        
        source_orders = individual.vehicle_assignments[source_vehicle]
        target_orders = individual.vehicle_assignments[target_vehicle]
        
        if source_orders and len(target_orders) < self.vehicle_capacities.get(target_vehicle, 0):
            relocated_order = random.choice(source_orders)
            source_orders.remove(relocated_order)
            target_orders.append(relocated_order)
    
    def _shuffle_mutation(self, individual: Individual):
        """섞기 돌연변이"""
        # 한 차량 내의 주문 순서를 랜덤하게 섞기
        vehicle_ids = [vid for vid, orders in individual.vehicle_assignments.items() if len(orders) > 1]
        
        if vehicle_ids:
            selected_vehicle = random.choice(vehicle_ids)
            orders = individual.vehicle_assignments[selected_vehicle]
            random.shuffle(orders)
    
    def _repair_individual(self, assignments: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """개체 수리 (중복 제거, 용량 제약 확인)"""
        
        # 중복 주문 제거
        all_assigned_orders = set()
        repaired_assignments = {}
        
        for vehicle_id, order_ids in assignments.items():
            unique_orders = []
            for order_id in order_ids:
                if order_id not in all_assigned_orders:
                    all_assigned_orders.add(order_id)
                    unique_orders.append(order_id)
            
            # 용량 제약 확인
            capacity = self.vehicle_capacities.get(vehicle_id, 0)
            repaired_assignments[vehicle_id] = unique_orders[:capacity]
        
        return repaired_assignments
    
    def _convert_to_assignments(self, individual: Individual) -> List[VehicleAssignment]:
        """Individual을 VehicleAssignment 리스트로 변환"""
        assignments = []
        
        for vehicle_id, order_ids in individual.vehicle_assignments.items():
            if not order_ids:
                continue
            
            vehicle = self.vehicles_dict[vehicle_id]
            capacity = self.vehicle_capacities[vehicle_id]
            
            # 거리 및 시간 계산
            estimated_distance = self._calculate_route_distance_for_fitness(vehicle, order_ids)
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
    
    def _find_unassigned_orders(self, individual: Individual, orders: List[Order]) -> List[str]:
        """미배정 주문 찾기"""
        all_assigned = set()
        for order_ids in individual.vehicle_assignments.values():
            all_assigned.update(order_ids)
        
        all_order_ids = {order.id for order in orders}
        unassigned = all_order_ids - all_assigned
        
        return list(unassigned)