"""
TMS Router AI 시스템 상수

Clean Code 원칙에 따라 모든 매직 넘버/문자열을 상수로 정의
"""
from enum import Enum
from typing import Final


# API 관련 상수
class HttpStatus:
    """HTTP 상태 코드"""
    OK: Final[int] = 200
    BAD_REQUEST: Final[int] = 400
    UNPROCESSABLE_ENTITY: Final[int] = 422
    INTERNAL_SERVER_ERROR: Final[int] = 500


# TMS 비즈니스 상수
class TmsLimits:
    """TMS 시스템 제한값"""
    MAX_VEHICLES_PER_REQUEST: Final[int] = 50
    MAX_ORDERS_PER_REQUEST: Final[int] = 100
    MAX_VEHICLE_CAPACITY_TONS: Final[float] = 30.0
    MAX_WORKING_HOURS_PER_DAY: Final[int] = 8
    MAX_DRIVING_DISTANCE_KM: Final[int] = 400


# AI 관련 상수
class AiConstants:
    """AI 서비스 관련 상수"""
    DEFAULT_MODEL: Final[str] = "gpt-4"
    MAX_TOKENS_PER_REQUEST: Final[int] = 4000
    MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.7
    DEFAULT_TEMPERATURE: Final[float] = 0.3
    MAX_RETRY_ATTEMPTS: Final[int] = 3
    REQUEST_TIMEOUT_SECONDS: Final[int] = 30
    MAX_RETRIES: Final[int] = 3


# 시나리오 타입
class ScenarioType(str, Enum):
    """TMS 배차 시나리오 타입"""
    VRP = "vrp"  # Vehicle Routing Problem
    TSP = "tsp"  # Traveling Salesman Problem
    LOAD_CONSOLIDATION = "load_consolidation"
    EMERGENCY_DISPATCH = "emergency_dispatch"
    REALTIME_ADJUSTMENT = "realtime_adjustment"


# 차량 상태
class VehicleStatus(str, Enum):
    """차량 상태"""
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    MAINTENANCE = "MAINTENANCE"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


# 주문 우선순위
class Priority(str, Enum):
    """배송 주문 우선순위"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


# 피드백 타입
class FeedbackType(str, Enum):
    """피드백 분류"""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    IMPROVEMENT_REQUEST = "IMPROVEMENT_REQUEST"
    PERFORMANCE_ISSUE = "PERFORMANCE_ISSUE"


# 메모리 관련 상수
class MemoryConstants:
    """대화 메모리 관련 상수"""
    DEFAULT_CONVERSATION_TTL_HOURS: Final[int] = 24
    MAX_CONVERSATION_HISTORY: Final[int] = 100
    FEEDBACK_RETENTION_DAYS: Final[int] = 30
    MESSAGE_TTL_DAYS: Final[int] = 30
    SUMMARY_TTL_DAYS: Final[int] = 90
    MAX_CONVERSATION_TURNS: Final[int] = 20
    DEFAULT_MESSAGE_LIMIT: Final[int] = 50 