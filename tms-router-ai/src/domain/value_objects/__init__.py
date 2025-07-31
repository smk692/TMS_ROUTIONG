"""
Domain Value Objects - 불변 값 객체

비즈니스 로직에서 사용되는 불변 값 객체들을 정의합니다.
"""

from .coordinate import Coordinate
from .time_window import TimeWindow
from .route_segment import RouteSegment
from .optimization_result import OptimizationResult

__all__ = [
    'Coordinate',
    'TimeWindow',
    'RouteSegment',
    'OptimizationResult'
] 