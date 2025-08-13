"""
API 캐싱 시스템
- 메모리 캐시: 세션 내 중복 요청 방지
- 파일 캐시: 재시작 시 캐시 유지
- TTL 기반 캐시 무효화
- API 호출 비용 최적화
"""
import json
import hashlib
import time
import os
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from threading import Lock
from pathlib import Path

from ..models import Coordinates
from ..external.routing_client import RouteInfo, MatrixResult


class RoutingCache:
    """라우팅 API 결과 캐싱 시스템"""
    
    def __init__(self, cache_dir: str = "cache", memory_ttl: int = 3600, file_ttl: int = 86400):
        """
        Args:
            cache_dir: 캐시 파일 저장 디렉토리
            memory_ttl: 메모리 캐시 TTL (초, 기본 1시간)
            file_ttl: 파일 캐시 TTL (초, 기본 24시간)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.memory_ttl = memory_ttl
        self.file_ttl = file_ttl
        
        # 메모리 캐시 (세션 내)
        self._memory_cache: Dict[str, Tuple[Any, float]] = {}
        self._memory_lock = Lock()
        
        # 파일 캐시 경로
        self.route_cache_file = self.cache_dir / "route_cache.json"
        self.matrix_cache_file = self.cache_dir / "matrix_cache.json"
        
        self.logger = logging.getLogger(__name__)
        
        # 캐시 통계
        self.stats = {
            'memory_hits': 0,
            'file_hits': 0,
            'misses': 0,
            'api_calls_saved': 0
        }
        
        # 파일 캐시 로드
        self._load_file_caches()
    
    def _generate_cache_key(self, origin: Coordinates, destination: Coordinates, 
                          additional_params: str = "") -> str:
        """좌표와 추가 파라미터로 캐시 키 생성"""
        # 좌표를 소수점 4자리로 반올림하여 근접한 좌표는 같은 캐시 사용
        lat1 = round(origin.latitude, 4)
        lon1 = round(origin.longitude, 4)
        lat2 = round(destination.latitude, 4)
        lon2 = round(destination.longitude, 4)
        
        key_string = f"{lat1},{lon1}|{lat2},{lon2}|{additional_params}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_route_cache(self, origin: Coordinates, destination: Coordinates) -> Optional[RouteInfo]:
        """경로 캐시 조회"""
        cache_key = self._generate_cache_key(origin, destination)
        
        # 메모리 캐시 확인
        memory_result = self._get_from_memory(cache_key)
        if memory_result:
            self.stats['memory_hits'] += 1
            self.stats['api_calls_saved'] += 1
            self.logger.debug(f"메모리 캐시 히트: {cache_key[:8]}")
            return self._deserialize_route_info(memory_result)
        
        # 파일 캐시 확인
        file_result = self._get_from_file_cache(self.route_cache_file, cache_key)
        if file_result:
            # 메모리 캐시에도 저장
            self._set_to_memory(cache_key, file_result)
            self.stats['file_hits'] += 1
            self.stats['api_calls_saved'] += 1
            self.logger.debug(f"파일 캐시 히트: {cache_key[:8]}")
            return self._deserialize_route_info(file_result)
        
        self.stats['misses'] += 1
        return None
    
    def set_route_cache(self, origin: Coordinates, destination: Coordinates, 
                       route_info: RouteInfo):
        """경로 캐시 저장"""
        cache_key = self._generate_cache_key(origin, destination)
        serialized_data = self._serialize_route_info(route_info)
        
        # 메모리 캐시에 저장
        self._set_to_memory(cache_key, serialized_data)
        
        # 파일 캐시에 저장
        self._set_to_file_cache(self.route_cache_file, cache_key, serialized_data)
        
        self.logger.debug(f"경로 캐시 저장: {cache_key[:8]}")
    
    def get_matrix_cache(self, origins: list, destinations: list) -> Optional[list]:
        """거리 매트릭스 캐시 조회"""
        # 매트릭스는 좌표 리스트를 키로 사용
        origins_str = "|".join([f"{round(c.latitude,4)},{round(c.longitude,4)}" for c in origins])
        destinations_str = "|".join([f"{round(c.latitude,4)},{round(c.longitude,4)}" for c in destinations])
        cache_key = self._generate_cache_key_from_string(f"matrix:{origins_str}:{destinations_str}")
        
        # 메모리 캐시 확인
        memory_result = self._get_from_memory(cache_key)
        if memory_result:
            self.stats['memory_hits'] += 1
            self.stats['api_calls_saved'] += len(origins) * len(destinations)
            return self._deserialize_matrix_results(memory_result)
        
        # 파일 캐시 확인
        file_result = self._get_from_file_cache(self.matrix_cache_file, cache_key)
        if file_result:
            self._set_to_memory(cache_key, file_result)
            self.stats['file_hits'] += 1
            self.stats['api_calls_saved'] += len(origins) * len(destinations)
            return self._deserialize_matrix_results(file_result)
        
        self.stats['misses'] += 1
        return None
    
    def set_matrix_cache(self, origins: list, destinations: list, matrix_results: list):
        """거리 매트릭스 캐시 저장"""
        origins_str = "|".join([f"{round(c.latitude,4)},{round(c.longitude,4)}" for c in origins])
        destinations_str = "|".join([f"{round(c.latitude,4)},{round(c.longitude,4)}" for c in destinations])
        cache_key = self._generate_cache_key_from_string(f"matrix:{origins_str}:{destinations_str}")
        
        serialized_data = self._serialize_matrix_results(matrix_results)
        
        self._set_to_memory(cache_key, serialized_data)
        self._set_to_file_cache(self.matrix_cache_file, cache_key, serialized_data)
    
    def _generate_cache_key_from_string(self, key_string: str) -> str:
        """문자열로부터 캐시 키 생성"""
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_memory(self, cache_key: str) -> Optional[Any]:
        """메모리 캐시에서 조회"""
        with self._memory_lock:
            if cache_key in self._memory_cache:
                data, timestamp = self._memory_cache[cache_key]
                
                # TTL 확인
                if time.time() - timestamp < self.memory_ttl:
                    return data
                else:
                    # 만료된 캐시 삭제
                    del self._memory_cache[cache_key]
        
        return None
    
    def _set_to_memory(self, cache_key: str, data: Any):
        """메모리 캐시에 저장"""
        with self._memory_lock:
            self._memory_cache[cache_key] = (data, time.time())
            
            # 메모리 캐시 크기 제한 (최대 1000개)
            if len(self._memory_cache) > 1000:
                # 가장 오래된 것부터 삭제
                oldest_key = min(self._memory_cache.keys(), 
                               key=lambda k: self._memory_cache[k][1])
                del self._memory_cache[oldest_key]
    
    def _get_from_file_cache(self, cache_file: Path, cache_key: str) -> Optional[Any]:
        """파일 캐시에서 조회"""
        try:
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if cache_key in cache_data:
                entry = cache_data[cache_key]
                
                # TTL 확인
                if time.time() - entry['timestamp'] < self.file_ttl:
                    return entry['data']
                else:
                    # 만료된 캐시는 다음 정리 시점에서 제거
                    pass
        
        except (json.JSONDecodeError, KeyError, OSError) as e:
            self.logger.warning(f"파일 캐시 읽기 오류: {str(e)}")
        
        return None
    
    def _set_to_file_cache(self, cache_file: Path, cache_key: str, data: Any):
        """파일 캐시에 저장"""
        try:
            # 기존 캐시 로드
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            else:
                cache_data = {}
            
            # 새 데이터 추가
            cache_data[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
            
            # 만료된 캐시 정리 (매 100개 저장마다)
            if len(cache_data) % 100 == 0:
                cache_data = self._cleanup_expired_cache(cache_data)
            
            # 캐시 크기 제한 (최대 5000개)
            if len(cache_data) > 5000:
                # 오래된 것부터 삭제
                sorted_items = sorted(cache_data.items(), 
                                    key=lambda x: x[1]['timestamp'])
                cache_data = dict(sorted_items[-4000:])  # 상위 4000개만 유지
            
            # 파일에 저장
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        except (OSError, json.JSONDecodeError) as e:
            self.logger.error(f"파일 캐시 저장 오류: {str(e)}")
    
    def _cleanup_expired_cache(self, cache_data: Dict) -> Dict:
        """만료된 캐시 항목 정리"""
        current_time = time.time()
        cleaned_data = {}
        
        for key, entry in cache_data.items():
            if current_time - entry['timestamp'] < self.file_ttl:
                cleaned_data[key] = entry
        
        removed_count = len(cache_data) - len(cleaned_data)
        if removed_count > 0:
            self.logger.info(f"만료된 캐시 {removed_count}개 정리")
        
        return cleaned_data
    
    def _load_file_caches(self):
        """시작시 파일 캐시 로드 및 정리"""
        for cache_file in [self.route_cache_file, self.matrix_cache_file]:
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    # 만료된 캐시 정리
                    cleaned_data = self._cleanup_expired_cache(cache_data)
                    
                    # 정리된 데이터로 다시 저장
                    if len(cleaned_data) != len(cache_data):
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    self.logger.warning(f"파일 캐시 로드 오류 ({cache_file.name}): {str(e)}")
    
    def _serialize_route_info(self, route_info: RouteInfo) -> Dict:
        """RouteInfo 객체를 직렬화"""
        return {
            'distance_meters': route_info.distance_meters,
            'duration_seconds': route_info.duration_seconds,
            'toll_fare': route_info.toll_fare,
            'taxi_fare': route_info.taxi_fare,
            'route_summary': route_info.route_summary,
            'waypoints': [
                {'latitude': wp.latitude, 'longitude': wp.longitude}
                for wp in route_info.waypoints
            ],
            'confidence': route_info.confidence
        }
    
    def _deserialize_route_info(self, data: Dict) -> RouteInfo:
        """직렬화된 데이터를 RouteInfo 객체로 변환"""
        from ..external.routing_client import RouteInfo
        
        waypoints = [
            Coordinates(latitude=wp['latitude'], longitude=wp['longitude'])
            for wp in data.get('waypoints', [])
        ]
        
        return RouteInfo(
            distance_meters=data['distance_meters'],
            duration_seconds=data['duration_seconds'],
            toll_fare=data['toll_fare'],
            taxi_fare=data['taxi_fare'],
            route_summary=data['route_summary'],
            waypoints=waypoints,
            confidence=data['confidence']
        )
    
    def _serialize_matrix_results(self, matrix_results: list) -> list:
        """MatrixResult 리스트를 직렬화"""
        return [
            {
                'origin': {'latitude': result.origin.latitude, 'longitude': result.origin.longitude},
                'destination': {'latitude': result.destination.latitude, 'longitude': result.destination.longitude},
                'distance_meters': result.distance_meters,
                'duration_seconds': result.duration_seconds,
                'success': result.success
            }
            for result in matrix_results
        ]
    
    def _deserialize_matrix_results(self, data: list) -> list:
        """직렬화된 데이터를 MatrixResult 리스트로 변환"""
        from ..external.routing_client import MatrixResult
        
        return [
            MatrixResult(
                origin=Coordinates(latitude=item['origin']['latitude'], 
                                 longitude=item['origin']['longitude']),
                destination=Coordinates(latitude=item['destination']['latitude'], 
                                      longitude=item['destination']['longitude']),
                distance_meters=item['distance_meters'],
                duration_seconds=item['duration_seconds'],
                success=item['success']
            )
            for item in data
        ]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_requests = self.stats['memory_hits'] + self.stats['file_hits'] + self.stats['misses']
        hit_rate = ((self.stats['memory_hits'] + self.stats['file_hits']) / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'memory_hits': self.stats['memory_hits'],
            'file_hits': self.stats['file_hits'],
            'misses': self.stats['misses'],
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'api_calls_saved': self.stats['api_calls_saved'],
            'memory_cache_size': len(self._memory_cache)
        }
    
    def clear_cache(self, cache_type: str = 'all'):
        """캐시 초기화"""
        if cache_type in ['memory', 'all']:
            with self._memory_lock:
                self._memory_cache.clear()
            self.logger.info("메모리 캐시 초기화 완료")
        
        if cache_type in ['file', 'all']:
            for cache_file in [self.route_cache_file, self.matrix_cache_file]:
                if cache_file.exists():
                    cache_file.unlink()
            self.logger.info("파일 캐시 초기화 완료")
        
        # 통계 초기화
        if cache_type == 'all':
            self.stats = {
                'memory_hits': 0,
                'file_hits': 0,
                'misses': 0,
                'api_calls_saved': 0
            }


# 전역 캐시 인스턴스
_routing_cache = None

def get_routing_cache() -> RoutingCache:
    """전역 RoutingCache 인스턴스 반환"""
    global _routing_cache
    if _routing_cache is None:
        _routing_cache = RoutingCache()
    return _routing_cache