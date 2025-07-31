#!/usr/bin/env python3
"""
차량 ID 관리 서비스

현재: 순차적 ID 생성 방식
나중: 데이터베이스 조회 방식으로 쉽게 교체 가능하도록 설계
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import threading

class VehicleIdProvider(ABC):
    """차량 ID 제공자 인터페이스 - 나중에 데이터베이스 구현체로 교체 가능"""
    
    @abstractmethod
    def get_next_vehicle_id(self) -> int:
        """다음 차량 ID 반환"""
        pass
    
    @abstractmethod
    def reset_counter(self) -> None:
        """카운터 리셋 (테스트용)"""
        pass
    
    @abstractmethod
    def get_current_count(self) -> int:
        """현재 생성된 차량 수 반환"""
        pass

class SequentialVehicleIdProvider(VehicleIdProvider):
    """순차적 차량 ID 생성 구현체 (현재 방식)"""
    
    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()  # 스레드 안전성
    
    def get_next_vehicle_id(self) -> int:
        """전역적으로 고유한 차량 ID 반환"""
        with self._lock:
            self._counter += 1
            return self._counter
    
    def reset_counter(self) -> None:
        """카운터 리셋 (테스트용)"""
        with self._lock:
            self._counter = 0
    
    def get_current_count(self) -> int:
        """현재 생성된 차량 수 반환"""
        with self._lock:
            return self._counter

class DatabaseVehicleIdProvider(VehicleIdProvider):
    """데이터베이스 기반 차량 ID 제공자 (미래 구현용)"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        self.db_config = db_config
        # TODO: 데이터베이스 연결 초기화
        # self.db_connection = create_connection(db_config)
    
    def get_next_vehicle_id(self) -> int:
        """데이터베이스에서 다음 차량 ID 조회"""
        # TODO: 데이터베이스 구현
        # return self.db_connection.get_next_vehicle_id()
        raise NotImplementedError("데이터베이스 구현이 필요합니다")
    
    def reset_counter(self) -> None:
        """데이터베이스 카운터 리셋"""
        # TODO: 데이터베이스 구현
        # self.db_connection.reset_vehicle_counter()
        raise NotImplementedError("데이터베이스 구현이 필요합니다")
    
    def get_current_count(self) -> int:
        """데이터베이스에서 현재 차량 수 조회"""
        # TODO: 데이터베이스 구현
        # return self.db_connection.get_vehicle_count()
        raise NotImplementedError("데이터베이스 구현이 필요합니다")

class VehicleIdService:
    """차량 ID 관리 서비스 - 의존성 주입으로 구현체 교체 가능"""
    
    def __init__(self, provider: Optional[VehicleIdProvider] = None):
        """
        Args:
            provider: 차량 ID 제공자 (None이면 기본 순차적 제공자 사용)
        """
        if provider is None:
            self._provider = SequentialVehicleIdProvider()
        else:
            self._provider = provider
    
    def get_next_vehicle_id(self) -> int:
        """다음 차량 ID 반환"""
        return self._provider.get_next_vehicle_id()
    
    def reset_counter(self) -> None:
        """카운터 리셋 (테스트용)"""
        self._provider.reset_counter()
    
    def get_current_count(self) -> int:
        """현재 생성된 차량 수 반환"""
        return self._provider.get_current_count()
    
    def switch_to_database_provider(self, db_config: Dict) -> None:
        """데이터베이스 제공자로 교체 (나중에 사용)"""
        self._provider = DatabaseVehicleIdProvider(db_config)
    
    def switch_to_sequential_provider(self) -> None:
        """순차적 제공자로 교체"""
        current_count = self._provider.get_current_count()
        self._provider = SequentialVehicleIdProvider()
        # 기존 카운터 값 유지
        self._provider._counter = current_count

# 전역 싱글톤 인스턴스 (애플리케이션 전체에서 공유)
_global_vehicle_id_service = VehicleIdService()

def get_vehicle_id_service() -> VehicleIdService:
    """전역 차량 ID 서비스 인스턴스 반환"""
    return _global_vehicle_id_service

def configure_vehicle_id_service(provider: VehicleIdProvider) -> None:
    """차량 ID 서비스 설정 (애플리케이션 시작 시 호출)"""
    global _global_vehicle_id_service
    _global_vehicle_id_service = VehicleIdService(provider)

# 편의 함수들 (기존 코드와의 호환성)
def get_next_vehicle_id() -> int:
    """다음 차량 ID 반환 (전역 함수)"""
    return _global_vehicle_id_service.get_next_vehicle_id()

def reset_vehicle_counter() -> None:
    """차량 ID 카운터 리셋 (테스트용)"""
    _global_vehicle_id_service.reset_counter()

def get_vehicle_count() -> int:
    """현재 생성된 차량 수 반환"""
    return _global_vehicle_id_service.get_current_count() 