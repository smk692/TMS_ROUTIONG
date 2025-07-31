"""
TMS Router AI 로깅 설정

구조화된 로깅과 AWS CloudWatch 통합
"""
import logging
import os
import sys
from typing import Any, Dict

import structlog


def setup_logging() -> None:
    """로깅 시스템 초기화"""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    # 기본 로깅 설정
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )
    
    # structlog 설정
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """구조화된 로거 인스턴스 반환"""
    return structlog.get_logger(name)


class TmsLoggerMixin:
    """TMS 로깅 믹스인 클래스"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def log_request(self, request_id: str, request_type: str, **kwargs) -> None:
        """요청 로그 기록"""
        self.logger.info(
            "Request received",
            request_id=request_id,
            request_type=request_type,
            **kwargs
        )
    
    def log_response(self, request_id: str, status: str, duration_ms: float, **kwargs) -> None:
        """응답 로그 기록"""
        self.logger.info(
            "Request completed",
            request_id=request_id,
            status=status,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_error(self, request_id: str, error: Exception, **kwargs) -> None:
        """에러 로그 기록"""
        self.logger.error(
            "Request failed",
            request_id=request_id,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs
        )
    
    def log_ai_interaction(self, request_id: str, prompt_type: str, 
                          tokens_used: int, confidence_score: float, **kwargs) -> None:
        """AI 상호작용 로그 기록"""
        self.logger.info(
            "AI interaction",
            request_id=request_id,
            prompt_type=prompt_type,
            tokens_used=tokens_used,
            confidence_score=confidence_score,
            **kwargs
        ) 