"""
알고리즘 선택기 - 상황별 최적 알고리즘 자동 선택
"""
from typing import List, Dict, Optional, Type
from enum import Enum
import logging

from ..models import Order, Vehicle, Region
from .base_algorithm import BaseAlgorithm, AlgorithmConfig
from .nearest_neighbor import NearestNeighborAlgorithm, RandomNearestNeighborAlgorithm
from .genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmConfig
from .simulated_annealing import SimulatedAnnealingAlgorithm, SimulatedAnnealingConfig
from .simple_distance_based import SimpleDistanceBasedAlgorithm, FastestDistanceAlgorithm, SimpleConfig
from .ortools_vrp_algorithm import ORToolsVRPAlgorithm, ORToolsVRPConfig


class AlgorithmType(Enum):
    """알고리즘 타입"""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    RANDOM_NEAREST_NEIGHBOR = "random_nearest_neighbor"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    SIMPLE_DISTANCE_BASED = "simple_distance_based"
    FASTEST_DISTANCE = "fastest_distance"
    ORTOOLS_VRP = "ortools_vrp"


class SelectionStrategy(Enum):
    """선택 전략"""
    ORDER_COUNT_BASED = "order_count"      # 주문량 기반
    COMPLEXITY_BASED = "complexity"         # 복잡도 기반
    TIME_BASED = "time_limit"              # 시간 제한 기반
    EMERGENCY = "emergency"                 # 비상 상황


class AlgorithmSelector:
    """상황별 알고리즘 선택기"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 알고리즘 클래스 매핑
        self.algorithm_classes = {
            AlgorithmType.NEAREST_NEIGHBOR: NearestNeighborAlgorithm,
            AlgorithmType.RANDOM_NEAREST_NEIGHBOR: RandomNearestNeighborAlgorithm,
            AlgorithmType.GENETIC_ALGORITHM: GeneticAlgorithm,
            AlgorithmType.SIMULATED_ANNEALING: SimulatedAnnealingAlgorithm,
            AlgorithmType.SIMPLE_DISTANCE_BASED: SimpleDistanceBasedAlgorithm,
            AlgorithmType.FASTEST_DISTANCE: FastestDistanceAlgorithm,
            AlgorithmType.ORTOOLS_VRP: ORToolsVRPAlgorithm
        }
    
    def select_algorithm(self, orders: List[Order], vehicles: List[Vehicle],
                        regions: List[Region], conditions: Dict = None) -> BaseAlgorithm:
        """상황에 맞는 최적 알고리즘 선택"""
        
        conditions = conditions or {}
        
        # 선택 전략 결정
        selection_strategy = self._determine_selection_strategy(orders, vehicles, regions, conditions)
        
        # 전략에 따른 알고리즘 선택
        if selection_strategy == SelectionStrategy.EMERGENCY:
            algorithm_type = self._select_emergency_algorithm(conditions)
        elif selection_strategy == SelectionStrategy.TIME_BASED:
            algorithm_type = self._select_time_based_algorithm(orders, conditions)
        elif selection_strategy == SelectionStrategy.COMPLEXITY_BASED:
            algorithm_type = self._select_complexity_based_algorithm(orders, vehicles, regions)
        else:  # ORDER_COUNT_BASED
            algorithm_type = self._select_order_count_based_algorithm(orders)
        
        # 알고리즘 인스턴스 생성
        algorithm = self._create_algorithm_instance(algorithm_type, orders, conditions)
        
        self.logger.info(f"알고리즘 선택: {algorithm_type.value} (전략: {selection_strategy.value})")
        
        return algorithm
    
    def _determine_selection_strategy(self, orders: List[Order], vehicles: List[Vehicle],
                                    regions: List[Region], conditions: Dict) -> SelectionStrategy:
        """선택 전략 결정"""
        
        # 1. 비상 상황 확인
        if self._is_emergency_situation(conditions):
            return SelectionStrategy.EMERGENCY
        
        # 2. 시간 제한이 엄격한 경우
        time_limit = conditions.get('time_limit_seconds', 600)
        if time_limit < 120:  # 2분 미만
            return SelectionStrategy.TIME_BASED
        
        # 3. 복잡도 기반 선택이 유리한 경우
        if len(orders) > 50 and len(regions) > 2:
            return SelectionStrategy.COMPLEXITY_BASED
        
        # 4. 기본: 주문량 기반
        return SelectionStrategy.ORDER_COUNT_BASED
    
    def _is_emergency_situation(self, conditions: Dict) -> bool:
        """비상 상황 확인"""
        
        # 극한 날씨 조건
        weather_conditions = conditions.get('weather', {})
        for region_weather in weather_conditions.values():
            if isinstance(region_weather, dict):
                severity = region_weather.get('severity_score', 1.0)
                if severity >= 4.0:  # 폭풍 수준
                    return True
        
        # 심각한 교통 정체
        traffic_conditions = conditions.get('traffic', {})
        for region_traffic in traffic_conditions.values():
            if isinstance(region_traffic, dict):
                congestion = region_traffic.get('congestion_level', 0.5)
                if congestion >= 0.9:  # 90% 이상 정체
                    return True
        
        # 시스템 장애 상황
        if conditions.get('system_emergency', False):
            return True
        
        return False
    
    def _select_emergency_algorithm(self, conditions: Dict) -> AlgorithmType:
        """비상 상황용 알고리즘 선택"""
        # 비상 상황에서는 가장 빠른 알고리즘 사용
        return AlgorithmType.FASTEST_DISTANCE
    
    def _select_time_based_algorithm(self, orders: List[Order], conditions: Dict) -> AlgorithmType:
        """시간 기반 알고리즘 선택"""
        time_limit = conditions.get('time_limit_seconds', 600)
        order_count = len(orders)
        
        if time_limit < 60:  # 1분 미만
            return AlgorithmType.FASTEST_DISTANCE  # 초고속 알고리즘
        elif time_limit < 180:  # 3분 미만
            if order_count <= 50:
                return AlgorithmType.SIMPLE_DISTANCE_BASED
            else:
                return AlgorithmType.NEAREST_NEIGHBOR
        else:  # 3분 이상
            if order_count <= 100:
                return AlgorithmType.SIMPLE_DISTANCE_BASED  # 여전히 간단한 알고리즘 우선
            else:
                return AlgorithmType.GENETIC_ALGORITHM
    
    def _select_complexity_based_algorithm(self, orders: List[Order], vehicles: List[Vehicle],
                                         regions: List[Region]) -> AlgorithmType:
        """복잡도 기반 알고리즘 선택"""
        
        complexity_score = self._calculate_complexity_score(orders, vehicles, regions)
        
        # 2명 라이더 특수 케이스 확인
        if len(vehicles) <= 2:
            if complexity_score <= 2.0:
                return AlgorithmType.FASTEST_DISTANCE
            else:
                return AlgorithmType.SIMPLE_DISTANCE_BASED
        
        # 일반적인 경우
        if complexity_score <= 1.5:
            return AlgorithmType.SIMPLE_DISTANCE_BASED  # 간단한 알고리즘 우선
        elif complexity_score <= 2.5:
            return AlgorithmType.NEAREST_NEIGHBOR
        elif complexity_score <= 3.5:
            return AlgorithmType.GENETIC_ALGORITHM
        else:
            return AlgorithmType.SIMULATED_ANNEALING
    
    def _select_order_count_based_algorithm(self, orders: List[Order]) -> AlgorithmType:
        """주문량 기반 알고리즘 선택 - OR-Tools VRP 우선 사용"""
        order_count = len(orders)
        
        if order_count <= 15:
            return AlgorithmType.SIMPLE_DISTANCE_BASED  # 15개 이하는 간단한 알고리즘
        elif order_count <= 30:
            return AlgorithmType.NEAREST_NEIGHBOR  # 30개 이하는 최근접 이웃
        else:
            return AlgorithmType.ORTOOLS_VRP  # 30개 이상은 OR-Tools VRP 사용
    
    def _calculate_complexity_score(self, orders: List[Order], vehicles: List[Vehicle],
                                  regions: List[Region]) -> float:
        """복잡도 점수 계산 (1.0-5.0)"""
        
        # 주문량 복잡도 (40%)
        order_count = len(orders)
        if order_count <= 50:
            order_complexity = 1.0
        elif order_count <= 150:
            order_complexity = 2.0
        elif order_count <= 300:
            order_complexity = 3.0
        else:
            order_complexity = 4.0
        
        # 지리적 분산도 (25%)
        geo_complexity = self._calculate_geographical_complexity(orders, regions)
        
        # 시간 제약 복잡도 (20%)
        time_complexity = self._calculate_time_complexity(orders)
        
        # 용량 제약 복잡도 (15%)
        capacity_complexity = self._calculate_capacity_complexity(orders, vehicles)
        
        # 가중 평균
        total_complexity = (
            order_complexity * 0.4 +
            geo_complexity * 0.25 +
            time_complexity * 0.2 +
            capacity_complexity * 0.15
        )
        
        return min(5.0, max(1.0, total_complexity))
    
    def _calculate_geographical_complexity(self, orders: List[Order], regions: List[Region]) -> float:
        """지리적 분산도 계산"""
        if len(regions) <= 1:
            return 1.0
        elif len(regions) <= 3:
            return 2.0
        elif len(regions) <= 5:
            return 3.0
        else:
            return 4.0
    
    def _calculate_time_complexity(self, orders: List[Order]) -> float:
        """시간 제약 복잡도 계산"""
        # 고우선순위 주문의 비율
        high_priority_count = sum(1 for order in orders 
                                if order.priority.value in ['high', 'urgent'])
        
        if not orders:
            return 1.0
        
        high_priority_ratio = high_priority_count / len(orders)
        
        if high_priority_ratio <= 0.1:
            return 1.0
        elif high_priority_ratio <= 0.3:
            return 2.0
        elif high_priority_ratio <= 0.5:
            return 3.0
        else:
            return 4.0
    
    def _calculate_capacity_complexity(self, orders: List[Order], vehicles: List[Vehicle]) -> float:
        """용량 제약 복잡도 계산"""
        if not vehicles:
            return 4.0
        
        total_orders = len(orders)
        total_capacity = sum(v.safe_capacity for v in vehicles if v.is_auto_dispatch_eligible())
        
        if total_capacity == 0:
            return 4.0
        
        utilization_ratio = total_orders / total_capacity
        
        if utilization_ratio <= 0.5:
            return 1.0
        elif utilization_ratio <= 0.8:
            return 2.0
        elif utilization_ratio <= 1.0:
            return 3.0
        else:
            return 4.0
    
    def _create_algorithm_instance(self, algorithm_type: AlgorithmType, orders: List[Order],
                                 conditions: Dict) -> BaseAlgorithm:
        """알고리즘 인스턴스 생성"""
        
        algorithm_class = self.algorithm_classes[algorithm_type]
        
        # 알고리즘별 설정 생성
        if algorithm_type == AlgorithmType.GENETIC_ALGORITHM:
            config = self._create_ga_config(orders, conditions)
            return algorithm_class(config)
        elif algorithm_type == AlgorithmType.SIMULATED_ANNEALING:
            config = self._create_sa_config(orders, conditions)
            return algorithm_class(config)
        elif algorithm_type in [AlgorithmType.SIMPLE_DISTANCE_BASED, AlgorithmType.FASTEST_DISTANCE]:
            config = self._create_simple_config(orders, conditions)
            return algorithm_class(config)
        elif algorithm_type == AlgorithmType.ORTOOLS_VRP:
            config = self._create_ortools_config(orders, conditions)
            return algorithm_class(config)
        else:
            config = self._create_basic_config(conditions)
            return algorithm_class(config)
    
    def _create_basic_config(self, conditions: Dict) -> AlgorithmConfig:
        """기본 알고리즘 설정 생성"""
        time_limit = conditions.get('time_limit_seconds', 300)
        
        return AlgorithmConfig(
            time_limit_seconds=time_limit,
            quality_threshold=0.8,
            early_stopping_enabled=True,
            verbose=conditions.get('verbose', False)
        )
    
    def _create_ga_config(self, orders: List[Order], conditions: Dict) -> GeneticAlgorithmConfig:
        """유전자 알고리즘 설정 생성"""
        order_count = len(orders)
        time_limit = conditions.get('time_limit_seconds', 300)
        
        # 주문량에 따른 인구 크기 조정
        if order_count <= 50:
            population_size = 50
            generations = 100
        elif order_count <= 100:
            population_size = 100
            generations = 150
        else:
            population_size = 150
            generations = 200
        
        # 시간 제한에 따른 조정
        if time_limit < 180:
            generations = min(generations, 50)
            population_size = min(population_size, 50)
        
        return GeneticAlgorithmConfig(
            time_limit_seconds=time_limit,
            population_size=population_size,
            generations=generations,
            crossover_rate=0.7,
            mutation_rate=0.1,
            early_stopping_enabled=True,
            verbose=conditions.get('verbose', False)
        )
    
    def _create_sa_config(self, orders: List[Order], conditions: Dict) -> SimulatedAnnealingConfig:
        """시뮬레이티드 어닐링 설정 생성"""
        order_count = len(orders)
        time_limit = conditions.get('time_limit_seconds', 600)
        
        # 주문량에 따른 온도 조정
        if order_count <= 100:
            initial_temp = 500.0
            max_iterations = 50
        elif order_count <= 300:
            initial_temp = 1000.0
            max_iterations = 100
        else:
            initial_temp = 1500.0
            max_iterations = 150
        
        # 시간 제한에 따른 조정
        if time_limit < 300:
            initial_temp *= 0.5
            max_iterations = min(max_iterations, 30)
        
        return SimulatedAnnealingConfig(
            time_limit_seconds=time_limit,
            initial_temperature=initial_temp,
            final_temperature=1.0,
            cooling_rate=0.95,
            max_iterations_per_temp=max_iterations,
            early_stopping_enabled=True,
            verbose=conditions.get('verbose', False)
        )
    
    def _create_simple_config(self, orders: List[Order], conditions: Dict) -> SimpleConfig:
        """간단한 거리 기반 알고리즘 설정 생성"""
        order_count = len(orders)
        time_limit = conditions.get('time_limit_seconds', 300)
        
        # 주문량에 따른 설정 조정
        if order_count <= 20:
            # 초소규모: 최대 속도 설정
            enable_priority_weighting = False
            max_distance_threshold = 20.0
        elif order_count <= 50:
            # 소규모: 균형 설정
            enable_priority_weighting = True
            max_distance_threshold = 30.0
        else:
            # 중간 규모: 품질 우선
            enable_priority_weighting = True
            max_distance_threshold = 50.0
        
        return SimpleConfig(
            time_limit_seconds=time_limit,
            quality_threshold=0.8,
            early_stopping_enabled=True,
            verbose=conditions.get('verbose', False),
            enable_priority_weighting=enable_priority_weighting,
            max_distance_threshold=max_distance_threshold,
            enable_detailed_logging=conditions.get('detailed_logging', False)
        )
    
    def _create_ortools_config(self, orders: List[Order], conditions: Dict) -> ORToolsVRPConfig:
        """OR-Tools VRP 알고리즘 설정 생성"""
        order_count = len(orders)
        time_limit = conditions.get('time_limit_seconds', 120)
        
        # 주문량에 따른 설정 조정
        if order_count <= 50:
            # 소규모: 빠른 처리
            max_solve_time = min(60, time_limit)
            use_clustering = False
            min_cluster_size = 8
            max_cluster_size = 25
        elif order_count <= 200:
            # 중규모: 균형 처리
            max_solve_time = min(120, time_limit)
            use_clustering = True
            min_cluster_size = 10
            max_cluster_size = 30
        else:
            # 대규모: 품질 우선
            max_solve_time = min(180, time_limit)
            use_clustering = True
            min_cluster_size = 12
            max_cluster_size = 35
        
        # 비상 상황 조정
        if conditions.get('emergency', False):
            max_solve_time = min(max_solve_time, 30)
            use_clustering = False
        
        return ORToolsVRPConfig(
            max_solve_time_seconds=max_solve_time,
            use_clustering=use_clustering,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            epsilon=0.005,  # ~500m 클러스터 반경
            
            # 제약조건 설정
            max_work_hours=8,
            max_distance_km=120,
            break_interval_hours=4,
            break_duration_minutes=15,
            
            # 목적함수 가중치
            unassigned_penalty=100000,
            distance_weight=1.0,
            vehicle_fixed_cost=5000,
            time_balance_penalty=50,
            
            # 거리 API 설정
            distance_api={
                'api_priority': ['openroute', 'here', 'kakao', 'haversine'],
                'distance_cache_ttl': 24 * 3600,  # 24시간 캐시
                'max_locations_per_request': 50,
                'request_delay': 0.1,
                'openroute_api_key': conditions.get('openroute_api_key', 'demo_key'),
                'here_api_key': conditions.get('here_api_key', 'demo_key'),
                'kakao_api_key': conditions.get('kakao_api_key', 'demo_key'),
            }
        )