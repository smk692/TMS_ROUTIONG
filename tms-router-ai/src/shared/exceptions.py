"""
TMS Router AI 공통 예외 클래스들

Clean Code 원칙에 따라 의미 있는 예외 타입 정의
"""
from typing import Any, Dict, Optional


class TmsError(Exception):
    """TMS 시스템 기본 예외 클래스"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(TmsError):
    """데이터 검증 실패 예외"""
    
    def __init__(self, field: str, message: str, value: Any = None):
        super().__init__(f"Validation failed for '{field}': {message}")
        self.field = field
        self.value = value


class AIServiceError(TmsError):
    """AI 서비스 관련 예외"""
    
    def __init__(self, message: str, service_type: str = "unknown", error_code: Optional[str] = None):
        super().__init__(message)
        self.service_type = service_type
        self.error_code = error_code


class MemoryRepositoryError(TmsError):
    """메모리 저장소 관련 예외"""
    
    def __init__(self, message: str, conversation_id: Optional[str] = None):
        super().__init__(message)
        self.conversation_id = conversation_id


class PromptSelectionError(TmsError):
    """프롬프트 선택 실패 예외"""
    
    def __init__(self, scenario_type: str, available_patterns: list):
        message = f"No suitable prompt found for scenario: {scenario_type}"
        super().__init__(message)
        self.scenario_type = scenario_type
        self.available_patterns = available_patterns


class RouteOptimizationError(TmsError):
    """경로 최적화 실패 예외"""
    
    def __init__(self, message: str, request_data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.request_data = request_data


class OptimizationError(TmsError):
    """최적화 실패 예외"""
    
    def __init__(self, message: str, request_data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.request_data = request_data


class ConfigurationError(TmsError):
    """설정 관련 예외"""
    
    def __init__(self, config_key: str, message: str):
        super().__init__(f"Configuration error for '{config_key}': {message}")
        self.config_key = config_key 