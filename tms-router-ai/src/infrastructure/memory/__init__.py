"""
Memory Infrastructure - 메모리 저장소 구현체

대화 기록, 피드백, 컨텍스트를 저장하는 인프라 구현체입니다.
"""

from .redis_memory_repository import RedisMemoryRepository

__all__ = [
    'RedisMemoryRepository',
    'get_memory_repository'
]


def get_memory_repository(**kwargs) -> RedisMemoryRepository:
    """
    Redis 메모리 저장소 팩토리
    
    Args:
        **kwargs: Redis 초기화 파라미터
        
    Returns:
        Redis 메모리 저장소 인스턴스
    """
    return RedisMemoryRepository(**kwargs) 