"""
알고리즘 팩토리 - 알고리즘 생성 및 관리
"""
from typing import Dict, List, Optional, Type, Union
import logging

from ..models import Order, Vehicle, Region
from .base_algorithm import BaseAlgorithm, AlgorithmConfig
from .algorithm_selector import AlgorithmSelector, AlgorithmType
from .nearest_neighbor import NearestNeighborAlgorithm, RandomNearestNeighborAlgorithm
from .genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmConfig
from .simulated_annealing import SimulatedAnnealingAlgorithm, SimulatedAnnealingConfig
from .ortools_vrp_algorithm import ORToolsVRPAlgorithm, ORToolsVRPConfig


class AlgorithmFactory:
    """알고리즘 팩토리"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.selector = AlgorithmSelector(config)
        
        # 알고리즘 등록
        self._algorithm_registry = {
            AlgorithmType.NEAREST_NEIGHBOR: NearestNeighborAlgorithm,
            AlgorithmType.RANDOM_NEAREST_NEIGHBOR: RandomNearestNeighborAlgorithm,
            AlgorithmType.GENETIC_ALGORITHM: GeneticAlgorithm,
            AlgorithmType.SIMULATED_ANNEALING: SimulatedAnnealingAlgorithm,
            AlgorithmType.ORTOOLS_VRP: ORToolsVRPAlgorithm,
        }
    
    def create_optimal_algorithm(self, orders: List[Order], vehicles: List[Vehicle],
                               regions: List[Region], conditions: Dict = None) -> BaseAlgorithm:
        """상황에 최적화된 알고리즘 자동 생성"""
        
        conditions = conditions or {}
        
        self.logger.info(f"최적 알고리즘 선택 시작 - 주문: {len(orders)}개, 차량: {len(vehicles)}대, 권역: {len(regions)}개")
        
        # 알고리즘 선택기를 통한 자동 선택
        algorithm = self.selector.select_algorithm(orders, vehicles, regions, conditions)
        
        return algorithm
    
    def create_algorithm(self, algorithm_type: Union[AlgorithmType, str], 
                        config: Optional[AlgorithmConfig] = None) -> BaseAlgorithm:
        """지정된 타입의 알고리즘 생성"""
        
        # 문자열인 경우 AlgorithmType으로 변환
        if isinstance(algorithm_type, str):
            try:
                algorithm_type = AlgorithmType(algorithm_type)
            except ValueError:
                raise ValueError(f"알 수 없는 알고리즘 타입: {algorithm_type}")
        
        # 알고리즘 클래스 조회
        if algorithm_type not in self._algorithm_registry:
            raise ValueError(f"등록되지 않은 알고리즘 타입: {algorithm_type}")
        
        algorithm_class = self._algorithm_registry[algorithm_type]
        
        # 설정이 없는 경우 기본 설정 생성
        if config is None:
            config = self._create_default_config(algorithm_type)
        
        self.logger.info(f"알고리즘 생성: {algorithm_type.value}")
        
        return algorithm_class(config)
    
    def create_algorithm_chain(self, algorithm_types: List[Union[AlgorithmType, str]], 
                             conditions: Dict = None) -> List[BaseAlgorithm]:
        """여러 알고리즘을 체인으로 생성"""
        
        algorithms = []
        conditions = conditions or {}
        
        for algorithm_type in algorithm_types:
            config = self._create_contextual_config(algorithm_type, conditions)
            algorithm = self.create_algorithm(algorithm_type, config)
            algorithms.append(algorithm)
        
        self.logger.info(f"알고리즘 체인 생성: {len(algorithms)}개 알고리즘")
        
        return algorithms
    
    def create_fallback_algorithm(self, primary_algorithm_type: Union[AlgorithmType, str]) -> BaseAlgorithm:
        """주 알고리즘에 대한 폴백 알고리즘 생성"""
        
        fallback_mapping = {
            AlgorithmType.SIMULATED_ANNEALING: AlgorithmType.GENETIC_ALGORITHM,
            AlgorithmType.GENETIC_ALGORITHM: AlgorithmType.RANDOM_NEAREST_NEIGHBOR,
            AlgorithmType.RANDOM_NEAREST_NEIGHBOR: AlgorithmType.NEAREST_NEIGHBOR,
            AlgorithmType.NEAREST_NEIGHBOR: AlgorithmType.NEAREST_NEIGHBOR  # 최종 폴백
        }
        
        # 문자열인 경우 AlgorithmType으로 변환
        if isinstance(primary_algorithm_type, str):
            primary_algorithm_type = AlgorithmType(primary_algorithm_type)
        
        fallback_type = fallback_mapping.get(primary_algorithm_type, AlgorithmType.NEAREST_NEIGHBOR)
        
        # 폴백 알고리즘은 더 빠른 설정 사용
        config = self._create_fallback_config(fallback_type)
        
        self.logger.info(f"폴백 알고리즘 생성: {primary_algorithm_type.value} -> {fallback_type.value}")
        
        return self.create_algorithm(fallback_type, config)
    
    def get_available_algorithms(self) -> List[AlgorithmType]:
        """사용 가능한 알고리즘 목록 반환"""
        return list(self._algorithm_registry.keys())
    
    def get_algorithm_info(self, algorithm_type: Union[AlgorithmType, str]) -> Dict:
        """알고리즘 정보 반환"""
        
        if isinstance(algorithm_type, str):
            algorithm_type = AlgorithmType(algorithm_type)
        
        info_mapping = {
            AlgorithmType.NEAREST_NEIGHBOR: {
                'name': 'Nearest Neighbor',
                'description': '최근접 이웃 알고리즘',
                'time_complexity': 'O(n²)',
                'expected_time': '30초 이내',
                'quality_range': '70-80%',
                'best_for': '소규모 주문 (≤30개), 비상상황'
            },
            AlgorithmType.RANDOM_NEAREST_NEIGHBOR: {
                'name': 'Random Nearest Neighbor',
                'description': '개선된 최근접 이웃 알고리즘',
                'time_complexity': 'O(k×n²)',
                'expected_time': '1-2분',
                'quality_range': '75-85%',
                'best_for': '소-중규모 주문 (≤50개)'
            },
            AlgorithmType.GENETIC_ALGORITHM: {
                'name': 'Genetic Algorithm',
                'description': '유전자 알고리즘',
                'time_complexity': 'O(g×p×n)',
                'expected_time': '2-5분',
                'quality_range': '85-90%',
                'best_for': '중규모 주문 (31-100개)'
            },
            AlgorithmType.SIMULATED_ANNEALING: {
                'name': 'Simulated Annealing',
                'description': '시뮬레이티드 어닐링',
                'time_complexity': 'O(i×n)',
                'expected_time': '5-10분',
                'quality_range': '88-93%',
                'best_for': '대규모 주문 (101-300개)'
            }
        }
        
        return info_mapping.get(algorithm_type, {})
    
    def _create_default_config(self, algorithm_type: AlgorithmType) -> AlgorithmConfig:
        """기본 설정 생성"""
        
        if algorithm_type == AlgorithmType.GENETIC_ALGORITHM:
            return GeneticAlgorithmConfig(
                time_limit_seconds=300,
                population_size=100,
                generations=200,
                early_stopping_enabled=True
            )
        elif algorithm_type == AlgorithmType.SIMULATED_ANNEALING:
            return SimulatedAnnealingConfig(
                time_limit_seconds=600,
                initial_temperature=1000.0,
                final_temperature=1.0,
                cooling_rate=0.95,
                max_iterations_per_temp=100,
                early_stopping_enabled=True
            )
        elif algorithm_type == AlgorithmType.ORTOOLS_VRP:
            return ORToolsVRPConfig(
                max_solve_time_seconds=120,
                use_clustering=True,
                min_cluster_size=8,
                max_cluster_size=35,
                max_work_hours=8,
                max_distance_km=120
            )
        else:
            return AlgorithmConfig(
                time_limit_seconds=120,
                quality_threshold=0.8,
                early_stopping_enabled=True
            )
    
    def _create_contextual_config(self, algorithm_type: Union[AlgorithmType, str],
                                conditions: Dict) -> AlgorithmConfig:
        """상황에 맞는 설정 생성"""
        
        if isinstance(algorithm_type, str):
            algorithm_type = AlgorithmType(algorithm_type)
        
        # 시간 제한 조정
        time_limit = conditions.get('time_limit_seconds', 300)
        is_emergency = conditions.get('emergency', False)
        
        if is_emergency:
            time_limit = min(time_limit, 60)  # 비상시 최대 1분
        
        if algorithm_type == AlgorithmType.GENETIC_ALGORITHM:
            return GeneticAlgorithmConfig(
                time_limit_seconds=time_limit,
                population_size=50 if is_emergency else 100,
                generations=50 if is_emergency else 200,
                early_stopping_enabled=True,
                verbose=conditions.get('verbose', False)
            )
        elif algorithm_type == AlgorithmType.SIMULATED_ANNEALING:
            return SimulatedAnnealingConfig(
                time_limit_seconds=time_limit,
                initial_temperature=500.0 if is_emergency else 1000.0,
                cooling_rate=0.90 if is_emergency else 0.95,
                max_iterations_per_temp=30 if is_emergency else 100,
                early_stopping_enabled=True,
                verbose=conditions.get('verbose', False)
            )
        else:
            return AlgorithmConfig(
                time_limit_seconds=time_limit,
                quality_threshold=0.7 if is_emergency else 0.8,
                early_stopping_enabled=True,
                verbose=conditions.get('verbose', False)
            )
    
    def _create_fallback_config(self, algorithm_type: AlgorithmType) -> AlgorithmConfig:
        """폴백 알고리즘용 빠른 설정 생성"""
        
        if algorithm_type == AlgorithmType.GENETIC_ALGORITHM:
            return GeneticAlgorithmConfig(
                time_limit_seconds=120,
                population_size=50,
                generations=50,
                early_stopping_enabled=True
            )
        elif algorithm_type == AlgorithmType.SIMULATED_ANNEALING:
            return SimulatedAnnealingConfig(
                time_limit_seconds=180,
                initial_temperature=500.0,
                cooling_rate=0.90,
                max_iterations_per_temp=30,
                early_stopping_enabled=True
            )
        else:
            return AlgorithmConfig(
                time_limit_seconds=60,
                quality_threshold=0.7,
                early_stopping_enabled=True
            )
    
    def register_algorithm(self, algorithm_type: AlgorithmType, 
                         algorithm_class: Type[BaseAlgorithm]):
        """새로운 알고리즘 등록"""
        self._algorithm_registry[algorithm_type] = algorithm_class
        self.logger.info(f"새 알고리즘 등록: {algorithm_type.value}")
    
    def unregister_algorithm(self, algorithm_type: AlgorithmType):
        """알고리즘 등록 해제"""
        if algorithm_type in self._algorithm_registry:
            del self._algorithm_registry[algorithm_type]
            self.logger.info(f"알고리즘 등록 해제: {algorithm_type.value}")


# 전역 팩토리 인스턴스
_global_factory = None

def get_algorithm_factory(config: Dict = None) -> AlgorithmFactory:
    """전역 알고리즘 팩토리 인스턴스 반환"""
    global _global_factory
    
    if _global_factory is None:
        _global_factory = AlgorithmFactory(config)
    
    return _global_factory