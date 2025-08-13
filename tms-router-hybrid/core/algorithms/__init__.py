"""
TMS Router Hybrid - 알고리즘 패키지
"""

# 기본 클래스
from .base_algorithm import BaseAlgorithm, AlgorithmResult, AlgorithmConfig, AlgorithmError

# 알고리즘 구현체들
from .nearest_neighbor import NearestNeighborAlgorithm, RandomNearestNeighborAlgorithm
from .genetic_algorithm import GeneticAlgorithm, GeneticAlgorithmConfig
from .simulated_annealing import SimulatedAnnealingAlgorithm, SimulatedAnnealingConfig
from .hybrid_vrp_tsp import HybridVRPTSPAlgorithm, HybridVRPTSPConfig
from .simple_distance_based import SimpleDistanceBasedAlgorithm, FastestDistanceAlgorithm, SimpleConfig
from .ortools_vrp_algorithm import ORToolsVRPAlgorithm, ORToolsVRPConfig

# 알고리즘 선택 및 팩토리
from .algorithm_selector import AlgorithmSelector, AlgorithmType, SelectionStrategy
from .algorithm_factory import AlgorithmFactory, get_algorithm_factory

__all__ = [
    # 기본 클래스
    'BaseAlgorithm', 'AlgorithmResult', 'AlgorithmConfig', 'AlgorithmError',
    
    # 알고리즘 구현체들
    'NearestNeighborAlgorithm', 'RandomNearestNeighborAlgorithm',
    'GeneticAlgorithm', 'GeneticAlgorithmConfig',
    'SimulatedAnnealingAlgorithm', 'SimulatedAnnealingConfig',
    'HybridVRPTSPAlgorithm', 'HybridVRPTSPConfig',
    'SimpleDistanceBasedAlgorithm', 'FastestDistanceAlgorithm', 'SimpleConfig',
    'ORToolsVRPAlgorithm', 'ORToolsVRPConfig',
    
    # 선택 및 팩토리
    'AlgorithmSelector', 'AlgorithmType', 'SelectionStrategy',
    'AlgorithmFactory', 'get_algorithm_factory'
]

# 편의 함수들
def create_algorithm(algorithm_type: str, config: dict = None) -> BaseAlgorithm:
    """편의 함수: 알고리즘 생성"""
    factory = get_algorithm_factory()
    return factory.create_algorithm(algorithm_type, config)

def get_optimal_algorithm(orders, vehicles, regions, conditions=None) -> BaseAlgorithm:
    """편의 함수: 최적 알고리즘 선택"""
    factory = get_algorithm_factory()
    return factory.create_optimal_algorithm(orders, vehicles, regions, conditions)