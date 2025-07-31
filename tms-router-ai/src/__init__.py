"""
TMS Router AI - Clean Architecture Source Package

이 패키지는 Clean Architecture 원칙에 따라 구성되어 있습니다:

- domain/: 비즈니스 엔티티와 규칙 (최내부, 의존성 없음)
- use_cases/: 애플리케이션 로직 (domain에만 의존)
- interfaces/: 추상화 계층 (domain, use_cases에 의존)
- infrastructure/: 외부 시스템 구현체 (모든 레이어에 의존 가능)
- presentation/: API 및 요청/응답 처리 (use_cases에만 의존)
- shared/: 공통 유틸리티 (모든 레이어에서 사용 가능)
"""

__version__ = "1.0.0"
__author__ = "TMS Router AI Team" 