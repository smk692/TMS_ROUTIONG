"""
TMS Router Hybrid - 알고리즘 패키지 (OR-Tools VRP 전용)
"""

# 기본 클래스
from .base_algorithm import BaseAlgorithm, AlgorithmResult, AlgorithmConfig, AlgorithmError

# OR-Tools VRP 알고리즘 (유일한 알고리즘)
from .ortools_vrp_algorithm import ORToolsVRPAlgorithm, ORToolsVRPConfig

# 간소화된 팩토리
from .algorithm_factory_simplified import SimplifiedAlgorithmFactory

__all__ = [
    # 기본 클래스
    'BaseAlgorithm', 'AlgorithmResult', 'AlgorithmConfig', 'AlgorithmError',
    
    # OR-Tools VRP 알고리즘
    'ORToolsVRPAlgorithm', 'ORToolsVRPConfig',
    
    # 간소화된 팩토리
    'SimplifiedAlgorithmFactory'
]

# 편의 함수들 (OR-Tools VRP 전용)
def create_algorithm(orders, vehicles, regions=None, conditions=None) -> BaseAlgorithm:
    """편의 함수: OR-Tools VRP 알고리즘 생성"""
    factory = SimplifiedAlgorithmFactory()
    return factory.create_algorithm(orders, vehicles, regions or [], conditions)

def get_optimal_algorithm(orders, vehicles, regions=None, conditions=None) -> BaseAlgorithm:
    """편의 함수: OR-Tools VRP 알고리즘 선택 (항상 OR-Tools VRP 반환)"""
    return create_algorithm(orders, vehicles, regions, conditions)

# 하위 호환성을 위한 팩토리 함수
def get_algorithm_factory():
    """하위 호환성: 간소화된 팩토리 반환"""
    return SimplifiedAlgorithmFactory()