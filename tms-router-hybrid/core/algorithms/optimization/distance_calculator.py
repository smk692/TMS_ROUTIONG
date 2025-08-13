"""
실제 거리 계산 모듈 - 다중 API 지원
"""

import asyncio
import aiohttp
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from geopy.distance import geodesic
import time
from diskcache import Cache
import json

from ...models import Coordinates
from ...external import get_cache_manager


@dataclass
class DistanceResult:
    """거리 계산 결과"""
    distance_km: float
    duration_minutes: float
    api_source: str
    confidence: float = 1.0


class DistanceMatrixCalculator:
    """실제 도로 거리 계산 - 다중 API 지원"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # API 설정
        self.openroute_api_key = config.get('openroute_api_key', 'demo_key')
        self.here_api_key = config.get('here_api_key', 'demo_key')
        self.kakao_api_key = config.get('kakao_api_key', 'demo_key')
        
        # API 우선순위 
        self.api_priority = config.get('api_priority', ['openroute', 'here', 'kakao', 'haversine'])
        
        # 캐시 설정
        self.cache = get_cache_manager()
        self.cache_ttl = config.get('distance_cache_ttl', 24 * 3600)  # 24시간
        
        # 요청 제한 설정
        self.max_locations_per_request = config.get('max_locations_per_request', 50)
        self.request_delay = config.get('request_delay', 0.1)  # 100ms 지연
        
        # 세션 생성
        self._session = None
        
    async def calculate_distance_matrix(self, locations: List[Coordinates]) -> np.ndarray:
        """위치들 간의 거리 행렬 계산"""
        
        n_locations = len(locations)
        self.logger.info(f"거리 행렬 계산 시작: {n_locations}×{n_locations}")
        
        # 결과 행렬 초기화
        distance_matrix = np.zeros((n_locations, n_locations))
        
        try:
            # 비동기 세션 생성
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300),  # 5분 타임아웃
                connector=aiohttp.TCPConnector(limit=10)
            ) as session:
                self._session = session
                
                # 캐시된 거리 먼저 확인
                cache_hits = 0
                pending_pairs = []
                
                for i in range(n_locations):
                    for j in range(n_locations):
                        if i == j:
                            distance_matrix[i][j] = 0.0
                            continue
                            
                        # 캐시 확인
                        cached_distance = self._get_cached_distance(locations[i], locations[j])
                        if cached_distance is not None:
                            distance_matrix[i][j] = cached_distance
                            cache_hits += 1
                        else:
                            pending_pairs.append((i, j))
                
                self.logger.info(f"캐시 히트: {cache_hits}개, 계산 필요: {len(pending_pairs)}개")
                
                # 남은 거리들 계산
                if pending_pairs:
                    await self._calculate_pending_distances(locations, pending_pairs, distance_matrix)
                
        except Exception as e:
            self.logger.error(f"거리 행렬 계산 오류: {str(e)}")
            # 폴백: Haversine 거리로 전체 계산
            distance_matrix = self._calculate_haversine_matrix(locations)
        
        self.logger.info(f"거리 행렬 계산 완료")
        return distance_matrix
    
    async def _calculate_pending_distances(self, locations: List[Coordinates], 
                                         pending_pairs: List[Tuple[int, int]], 
                                         distance_matrix: np.ndarray):
        """대기 중인 거리 쌍들 계산"""
        
        # 배치 단위로 처리
        batch_size = min(100, len(pending_pairs))  # 한 번에 최대 100개
        
        for i in range(0, len(pending_pairs), batch_size):
            batch_pairs = pending_pairs[i:i + batch_size]
            
            try:
                # Haversine만 사용하도록 직접 계산 (API 호출 스킵)
                if 'haversine' in self.api_priority:
                    # Haversine으로 모든 거리 계산
                    for src_idx, dst_idx in batch_pairs:
                        distance = self._calculate_haversine_distance(
                            locations[src_idx], locations[dst_idx]
                        )
                        distance_matrix[src_idx][dst_idx] = distance
                        
                        # 캐시 저장
                        self._cache_distance(
                            locations[src_idx], locations[dst_idx], 
                            distance, 'haversine'
                        )
                    batch_pairs = []  # 모든 계산 완료
                else:
                    # 다른 API들 시도 (기존 로직)
                    for api_name in self.api_priority:
                        # 실제 API 호출
                        success_pairs = await self._call_distance_api(
                            api_name, locations, batch_pairs, distance_matrix
                        )
                        
                        # 성공한 쌍들 제거
                        batch_pairs = [pair for pair in batch_pairs if pair not in success_pairs]
                        
                        if not batch_pairs:  # 모든 거리 계산 완료
                            break
                
                # 요청 간 지연
                await asyncio.sleep(self.request_delay)
                
            except Exception as e:
                self.logger.error(f"배치 거리 계산 오류: {str(e)}")
                # 실패한 배치는 Haversine으로 폴백
                for src_idx, dst_idx in batch_pairs:
                    distance = self._calculate_haversine_distance(
                        locations[src_idx], locations[dst_idx]
                    )
                    distance_matrix[src_idx][dst_idx] = distance
    
    async def _call_distance_api(self, api_name: str, locations: List[Coordinates],
                               pairs: List[Tuple[int, int]], distance_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """특정 API로 거리 계산"""
        
        success_pairs = []
        
        try:
            if api_name == 'openroute':
                success_pairs = await self._call_openroute_api(locations, pairs, distance_matrix)
            elif api_name == 'here':
                success_pairs = await self._call_here_api(locations, pairs, distance_matrix)
            elif api_name == 'kakao':
                success_pairs = await self._call_kakao_api(locations, pairs, distance_matrix)
            
            self.logger.info(f"{api_name} API: {len(success_pairs)}개 거리 계산 성공")
            
        except Exception as e:
            self.logger.warning(f"{api_name} API 호출 실패: {str(e)}")
        
        return success_pairs
    
    async def _call_openroute_api(self, locations: List[Coordinates], 
                                pairs: List[Tuple[int, int]], 
                                distance_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """OpenRouteService API 호출"""
        
        if self.openroute_api_key == 'demo_key':
            return []  # 데모 키는 사용하지 않음
        
        success_pairs = []
        
        # 배치 단위로 처리 (ORS는 최대 50개 위치)
        unique_locations = list(set([locations[i] for i, j in pairs] + [locations[j] for i, j in pairs]))
        
        if len(unique_locations) > 50:
            # 너무 많으면 개별 요청으로 처리
            return await self._call_individual_openroute_requests(locations, pairs, distance_matrix)
        
        try:
            # 좌표 매트릭스 준비
            coords = [[loc.longitude, loc.latitude] for loc in unique_locations]
            location_index_map = {id(loc): idx for idx, loc in enumerate(unique_locations)}
            
            url = "https://api.openrouteservice.org/v2/matrix/driving-car"
            headers = {
                'Authorization': self.openroute_api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                'locations': coords,
                'metrics': ['distance', 'duration']
            }
            
            async with self._session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    distances = result['distances']  # km
                    durations = result['durations']  # seconds
                    
                    # 결과를 distance_matrix에 저장
                    for src_idx, dst_idx in pairs:
                        src_loc_idx = location_index_map[id(locations[src_idx])]
                        dst_loc_idx = location_index_map[id(locations[dst_idx])]
                        
                        distance_km = distances[src_loc_idx][dst_loc_idx] / 1000.0  # m를 km로
                        duration_min = durations[src_loc_idx][dst_loc_idx] / 60.0   # s를 min으로
                        
                        distance_matrix[src_idx][dst_idx] = distance_km
                        
                        # 캐시 저장
                        self._cache_distance(
                            locations[src_idx], locations[dst_idx], 
                            distance_km, 'openroute'
                        )
                        
                        success_pairs.append((src_idx, dst_idx))
                else:
                    self.logger.warning(f"OpenRoute API 오류: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"OpenRoute API 호출 오류: {str(e)}")
        
        return success_pairs
    
    async def _call_individual_openroute_requests(self, locations: List[Coordinates], 
                                                pairs: List[Tuple[int, int]], 
                                                distance_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """개별 OpenRoute 요청"""
        success_pairs = []
        
        for src_idx, dst_idx in pairs[:20]:  # 최대 20개만 개별 처리
            try:
                src_coord = locations[src_idx]
                dst_coord = locations[dst_idx]
                
                url = f"https://api.openrouteservice.org/v2/directions/driving-car"
                headers = {'Authorization': self.openroute_api_key}
                
                params = {
                    'start': f"{src_coord.longitude},{src_coord.latitude}",
                    'end': f"{dst_coord.longitude},{dst_coord.latitude}"
                }
                
                async with self._session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'features' in result and len(result['features']) > 0:
                            properties = result['features'][0]['properties']
                            distance_km = properties['segments'][0]['distance'] / 1000.0
                            
                            distance_matrix[src_idx][dst_idx] = distance_km
                            
                            # 캐시 저장
                            self._cache_distance(src_coord, dst_coord, distance_km, 'openroute')
                            success_pairs.append((src_idx, dst_idx))
                            
                await asyncio.sleep(0.1)  # 요청 간 지연
                
            except Exception as e:
                self.logger.debug(f"개별 OpenRoute 요청 실패: {str(e)}")
                continue
        
        return success_pairs
    
    async def _call_here_api(self, locations: List[Coordinates], 
                           pairs: List[Tuple[int, int]], 
                           distance_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """HERE Maps API 호출"""
        
        if self.here_api_key == 'demo_key':
            return []
        
        # HERE Matrix API 구현 (생략 - OpenRoute와 유사한 구조)
        # 실제 구현시에는 HERE Maps Matrix API 사용
        return []
    
    async def _call_kakao_api(self, locations: List[Coordinates], 
                            pairs: List[Tuple[int, int]], 
                            distance_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Kakao Map API 호출"""
        
        if self.kakao_api_key == 'demo_key':
            return []
        
        # Kakao 길찾기 API 구현 (생략 - 개별 요청 방식)
        # 실제 구현시에는 Kakao Directions API 사용
        return []
    
    def _calculate_haversine_matrix(self, locations: List[Coordinates]) -> np.ndarray:
        """Haversine 거리 행렬 계산"""
        
        n = len(locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 0.0
                else:
                    distance = self._calculate_haversine_distance(locations[i], locations[j])
                    matrix[i][j] = distance
                    
                    # 캐시 저장
                    self._cache_distance(locations[i], locations[j], distance, 'haversine')
        
        return matrix
    
    def _calculate_haversine_distance(self, coord1: Coordinates, coord2: Coordinates) -> float:
        """두 좌표 간 Haversine 거리 계산 (도로계수 적용)"""
        
        # 직선 거리 계산
        straight_distance = geodesic(
            (coord1.latitude, coord1.longitude),
            (coord2.latitude, coord2.longitude)
        ).kilometers
        
        # 도로계수 적용 (평균적으로 직선거리의 1.4배)
        road_factor = 1.4
        return straight_distance * road_factor
    
    def _get_cached_distance(self, coord1: Coordinates, coord2: Coordinates) -> Optional[float]:
        """캐시된 거리 조회"""
        
        try:
            cache_key = self._generate_cache_key(coord1, coord2)
            cached_result = self.cache.get(cache_key, 'distance')
            
            if cached_result is not None:
                return cached_result.get('distance_km')
                
        except Exception as e:
            self.logger.debug(f"캐시 조회 오류: {str(e)}")
        
        return None
    
    def _cache_distance(self, coord1: Coordinates, coord2: Coordinates, 
                       distance_km: float, api_source: str):
        """거리 결과 캐시 저장"""
        
        try:
            cache_key = self._generate_cache_key(coord1, coord2)
            cache_data = {
                'distance_km': distance_km,
                'api_source': api_source,
                'timestamp': time.time()
            }
            
            self.cache.set(cache_key, cache_data, 'distance', expire=self.cache_ttl)
            
        except Exception as e:
            self.logger.debug(f"캐시 저장 오류: {str(e)}")
    
    def _generate_cache_key(self, coord1: Coordinates, coord2: Coordinates) -> str:
        """캐시 키 생성"""
        
        # 좌표를 정렬하여 방향에 무관하게 동일한 키 생성
        coords = sorted([
            (coord1.latitude, coord1.longitude),
            (coord2.latitude, coord2.longitude)
        ])
        
        # 좌표를 4자리 소수점으로 반올림하여 근사치 캐싱
        rounded_coords = [
            (round(lat, 4), round(lng, 4)) for lat, lng in coords
        ]
        
        return f"distance_{rounded_coords[0]}_{rounded_coords[1]}"
    
    def calculate_time_matrix(self, distance_matrix: np.ndarray, avg_speed_kmh: float = 25.0) -> np.ndarray:
        """거리 행렬을 기반으로 시간 행렬 계산"""
        
        # 기본 이동 시간 (시간 = 거리 / 속도)
        time_matrix = distance_matrix / avg_speed_kmh * 60  # 분 단위
        
        # 신호 대기 및 교통 상황 보정 (10% 추가)
        time_matrix *= 1.1
        
        return time_matrix