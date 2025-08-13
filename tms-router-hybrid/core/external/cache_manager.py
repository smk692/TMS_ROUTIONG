"""
캐싱 관리자
API 호출 최적화를 위한 다층 캐시 시스템
"""
import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Callable, TypeVar, Union
from dataclasses import dataclass, asdict
from functools import wraps

import diskcache
from ..models import Coordinates

T = TypeVar('T')


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    data: Any
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    source: str = "unknown"


class CacheManager:
    """다층 캐시 관리자"""
    
    def __init__(self, cache_dir: str = "./cache", memory_size_mb: int = 100):
        self.cache_dir = cache_dir
        self.memory_size_mb = memory_size_mb
        self.logger = logging.getLogger(__name__)
        
        # 메모리 캐시 (빠른 접근)
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._max_memory_items = 1000
        
        # 디스크 캐시 (영구 저장)
        os.makedirs(cache_dir, exist_ok=True)
        self._disk_cache = diskcache.Cache(
            directory=os.path.join(cache_dir, 'diskcache'),
            size_limit=memory_size_mb * 1024 * 1024  # MB to bytes
        )
        
        # 캐시 정책 설정
        self._cache_policies = {
            'weather': {'ttl_minutes': 30, 'max_items': 100},
            'traffic': {'ttl_minutes': 15, 'max_items': 200}, 
            'routing': {'ttl_minutes': 60, 'max_items': 500},
            'default': {'ttl_minutes': 60, 'max_items': 100}
        }
    
    def get_cache_key(self, prefix: str, **kwargs) -> str:
        """캐시 키 생성"""
        # 좌표는 소수점 4자리까지만 사용 (약 10m 정확도)
        normalized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, Coordinates):
                normalized_kwargs[key] = f"{value.latitude:.4f},{value.longitude:.4f}"
            elif isinstance(value, (list, tuple)):
                # 리스트는 문자열로 변환
                normalized_kwargs[key] = str(sorted(value))
            else:
                normalized_kwargs[key] = str(value)
        
        # 해시 생성
        content = f"{prefix}:" + ":".join(f"{k}={v}" for k, v in sorted(normalized_kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()[:16]  # 16자리만 사용
    
    def get(self, cache_type: str, cache_key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        
        # 1. 메모리 캐시 확인
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            if not self._is_expired(entry):
                entry.hit_count += 1
                self.logger.debug(f"메모리 캐시 히트: {cache_key}")
                return entry.data
            else:
                # 만료된 항목 제거
                del self._memory_cache[cache_key]
        
        # 2. 디스크 캐시 확인
        try:
            disk_entry = self._disk_cache.get(cache_key)
            if disk_entry:
                entry = CacheEntry(**disk_entry)
                if not self._is_expired(entry):
                    # 메모리 캐시에 복사 (자주 사용하는 경우)
                    if entry.hit_count > 2:
                        self._add_to_memory_cache(cache_key, entry)
                    
                    entry.hit_count += 1
                    # 디스크 캐시 업데이트
                    self._disk_cache.set(cache_key, asdict(entry))
                    
                    self.logger.debug(f"디스크 캐시 히트: {cache_key}")
                    return entry.data
                else:
                    # 만료된 항목 제거
                    self._disk_cache.delete(cache_key)
        
        except Exception as e:
            self.logger.warning(f"디스크 캐시 읽기 오류: {str(e)}")
        
        return None
    
    def set(self, cache_type: str, cache_key: str, data: Any, source: str = "api") -> bool:
        """캐시에 데이터 저장"""
        try:
            policy = self._cache_policies.get(cache_type, self._cache_policies['default'])
            ttl_minutes = policy['ttl_minutes']
            
            now = datetime.now()
            expires_at = now + timedelta(minutes=ttl_minutes)
            
            entry = CacheEntry(
                data=data,
                created_at=now,
                expires_at=expires_at,
                hit_count=0,
                source=source
            )
            
            # 메모리 캐시에 저장 (용량 제한 확인)
            self._add_to_memory_cache(cache_key, entry)
            
            # 디스크 캐시에 저장
            self._disk_cache.set(cache_key, asdict(entry))
            
            self.logger.debug(f"캐시 저장: {cache_key} (TTL: {ttl_minutes}분)")
            return True
            
        except Exception as e:
            self.logger.error(f"캐시 저장 오류: {str(e)}")
            return False
    
    def _add_to_memory_cache(self, cache_key: str, entry: CacheEntry):
        """메모리 캐시에 항목 추가 (LRU 정책)"""
        # 용량 초과 시 가장 오래된 항목 제거
        if len(self._memory_cache) >= self._max_memory_items:
            # 히트 카운트가 낮고 오래된 항목부터 제거
            sorted_items = sorted(
                self._memory_cache.items(),
                key=lambda x: (x[1].hit_count, x[1].created_at)
            )
            
            # 하위 10% 제거
            remove_count = max(1, len(sorted_items) // 10)
            for key, _ in sorted_items[:remove_count]:
                del self._memory_cache[key]
        
        self._memory_cache[cache_key] = entry
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """캐시 항목 만료 확인"""
        return datetime.now() > entry.expires_at
    
    def clear_expired(self, cache_type: Optional[str] = None):
        """만료된 캐시 항목 제거"""
        removed_count = 0
        
        # 메모리 캐시 정리
        expired_keys = []
        for key, entry in self._memory_cache.items():
            if self._is_expired(entry):
                if cache_type is None or key.startswith(cache_type):
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self._memory_cache[key]
            removed_count += 1
        
        # 디스크 캐시 정리 (더 복잡하므로 주기적으로만 실행)
        try:
            self._disk_cache.cull()  # 내장 정리 기능 사용
        except Exception as e:
            self.logger.warning(f"디스크 캐시 정리 오류: {str(e)}")
        
        if removed_count > 0:
            self.logger.info(f"만료된 캐시 {removed_count}개 항목 제거")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        memory_size = len(self._memory_cache)
        
        try:
            disk_size = len(self._disk_cache)
            disk_bytes = self._disk_cache.volume()
        except:
            disk_size = 0
            disk_bytes = 0
        
        return {
            'memory_cache': {
                'items': memory_size,
                'max_items': self._max_memory_items,
                'usage_percent': (memory_size / self._max_memory_items) * 100
            },
            'disk_cache': {
                'items': disk_size,
                'bytes_used': disk_bytes,
                'max_bytes': self.memory_size_mb * 1024 * 1024
            },
            'policies': self._cache_policies
        }
    
    def clear_all(self, cache_type: Optional[str] = None):
        """캐시 전체 또는 특정 타입 삭제"""
        if cache_type:
            # 특정 타입만 삭제
            keys_to_remove = [k for k in self._memory_cache.keys() if k.startswith(cache_type)]
            for key in keys_to_remove:
                del self._memory_cache[key]
                
            # 디스크 캐시에서도 제거 (비효율적이지만 정확)
            try:
                for key in list(self._disk_cache.iterkeys()):
                    if key.startswith(cache_type):
                        self._disk_cache.delete(key)
            except Exception as e:
                self.logger.warning(f"디스크 캐시 부분 삭제 오류: {str(e)}")
        else:
            # 전체 삭제
            self._memory_cache.clear()
            self._disk_cache.clear()
        
        self.logger.info(f"캐시 삭제 완료: {cache_type or '전체'}")


# 전역 캐시 관리자 인스턴스
_global_cache_manager = None

def get_cache_manager() -> CacheManager:
    """전역 캐시 관리자 반환"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


# 데코레이터
def cached(cache_type: str, ttl_minutes: Optional[int] = None):
    """함수 결과 캐싱 데코레이터"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_manager = get_cache_manager()
            
            # TTL 설정 업데이트
            if ttl_minutes:
                cache_manager._cache_policies[cache_type] = {
                    **cache_manager._cache_policies.get(cache_type, {}),
                    'ttl_minutes': ttl_minutes
                }
            
            # 캐시 키 생성
            func_name = f"{func.__module__}.{func.__name__}"
            cache_key = cache_manager.get_cache_key(
                f"{cache_type}:{func_name}",
                args=args,
                kwargs=kwargs
            )
            
            # 캐시 확인
            cached_result = cache_manager.get(cache_type, cache_key)
            if cached_result is not None:
                return cached_result
            
            # 함수 실행 및 결과 캐싱
            result = func(*args, **kwargs)
            cache_manager.set(cache_type, cache_key, result, source=func_name)
            
            return result
        
        return wrapper
    return decorator


# 사용 예시 데코레이터들
def cache_weather(ttl_minutes: int = 30):
    """날씨 정보 캐싱"""
    return cached('weather', ttl_minutes)

def cache_traffic(ttl_minutes: int = 15):
    """교통 정보 캐싱"""
    return cached('traffic', ttl_minutes)

def cache_routing(ttl_minutes: int = 60):
    """경로 정보 캐싱"""
    return cached('routing', ttl_minutes)