"""
카카오맵 API 클라이언트
경로 계산 및 거리/시간 정보 수집
"""
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..models import Coordinates


@dataclass 
class RouteInfo:
    """경로 정보"""
    distance_meters: int      # 총 거리 (m)
    duration_seconds: int     # 예상 시간 (초)
    toll_fare: int           # 통행료 (원)
    taxi_fare: int           # 택시 요금 (원)
    route_summary: str       # 경로 요약
    waypoints: List[Coordinates]  # 경유지 좌표
    confidence: float        # 신뢰도 0.0-1.0


@dataclass
class MatrixResult:
    """거리/시간 매트릭스 결과"""
    origin: Coordinates
    destination: Coordinates
    distance_meters: int
    duration_seconds: int
    success: bool


class KakaoRoutingClient:
    """카카오맵 Mobility API 클라이언트"""
    
    BASE_URL = "https://apis-navi.kakaomobility.com/v1"
    
    def __init__(self, rest_api_key: Optional[str] = None, timeout: int = 10):
        self.rest_api_key = rest_api_key or self._get_default_api_key()
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self._session = None
    
    def _get_default_api_key(self) -> str:
        """기본 REST API 키 반환"""
        import os
        return os.getenv('KAKAO_REST_API_KEY', 'demo_key')
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        headers = {
            'Authorization': f'KakaoAK {self.rest_api_key}',
            'Content-Type': 'application/json'
        }
        self._session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self._session:
            await self._session.close()
    
    async def calculate_route(self, origin: Coordinates, destination: Coordinates,
                            waypoints: Optional[List[Coordinates]] = None) -> Optional[RouteInfo]:
        """단일 경로 계산"""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.BASE_URL}/directions"
        
        # API 요청 파라미터
        params = {
            'origin': f"{origin.longitude},{origin.latitude}",
            'destination': f"{destination.longitude},{destination.latitude}",
            'priority': 'RECOMMEND',  # 추천 경로
            'car_fuel': 'GASOLINE',
            'car_hipass': 'false',
            'alternatives': 'false',
            'road_details': 'false'
        }
        
        # 경유지가 있는 경우
        if waypoints:
            waypoint_str = '|'.join([f"{wp.longitude},{wp.latitude}" for wp in waypoints])
            params['waypoints'] = waypoint_str
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_route_response(data)
                elif response.status == 401:
                    self.logger.error("카카오맵 API 인증 실패 - REST API 키를 확인하세요")
                    return self._get_fallback_route_info(origin, destination)
                else:
                    self.logger.warning(f"카카오맵 API 오류: {response.status}")
                    return self._get_fallback_route_info(origin, destination)
                    
        except asyncio.TimeoutError:
            self.logger.warning("카카오맵 API 타임아웃")
            return self._get_fallback_route_info(origin, destination)
        except Exception as e:
            self.logger.error(f"카카오맵 API 호출 오류: {str(e)}")
            return self._get_fallback_route_info(origin, destination)
    
    async def calculate_distance_matrix(self, origins: List[Coordinates], 
                                      destinations: List[Coordinates]) -> List[MatrixResult]:
        """거리/시간 매트릭스 계산"""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        results = []
        
        # 카카오맵 API는 매트릭스를 직접 지원하지 않으므로 개별 경로로 계산
        for origin in origins:
            for destination in destinations:
                route_info = await self.calculate_route(origin, destination)
                
                if route_info:
                    result = MatrixResult(
                        origin=origin,
                        destination=destination,
                        distance_meters=route_info.distance_meters,
                        duration_seconds=route_info.duration_seconds,
                        success=True
                    )
                else:
                    # 실패 시 직선거리 기반 추정
                    result = MatrixResult(
                        origin=origin,
                        destination=destination,
                        distance_meters=self._calculate_haversine_distance(origin, destination),
                        duration_seconds=self._estimate_duration(origin, destination),
                        success=False
                    )
                
                results.append(result)
                
                # API 호출 제한 고려 (초당 10회)
                await asyncio.sleep(0.1)
        
        return results
    
    def _parse_route_response(self, api_response: Dict) -> RouteInfo:
        """카카오맵 API 응답 파싱"""
        routes = api_response.get('routes', [])
        if not routes:
            raise ValueError("경로 정보가 없습니다")
        
        route = routes[0]  # 첫 번째 경로 사용
        summary = route.get('summary', {})
        sections = route.get('sections', [])
        
        # 경유지 좌표 추출
        waypoints = []
        for section in sections:
            for road in section.get('roads', []):
                if 'vertexes' in road:
                    vertices = road['vertexes']
                    # 좌표는 [lng, lat, lng, lat, ...] 형식
                    for i in range(0, len(vertices), 2):
                        if i + 1 < len(vertices):
                            waypoints.append(Coordinates(
                                latitude=vertices[i + 1],
                                longitude=vertices[i]
                            ))
        
        return RouteInfo(
            distance_meters=summary.get('distance', 0),
            duration_seconds=summary.get('duration', 0),
            toll_fare=summary.get('fare', {}).get('toll', 0),
            taxi_fare=summary.get('fare', {}).get('taxi', 0),
            route_summary=self._generate_route_summary(sections),
            waypoints=waypoints[:10],  # 최대 10개 포인트만
            confidence=0.9  # 카카오맵은 높은 신뢰도
        )
    
    def _generate_route_summary(self, sections: List[Dict]) -> str:
        """경로 요약 생성"""
        road_names = []
        for section in sections:
            for road in section.get('roads', []):
                name = road.get('name', '')
                if name and name not in road_names:
                    road_names.append(name)
        
        if road_names:
            return ' → '.join(road_names[:5])  # 최대 5개 도로명
        else:
            return "경로 정보 없음"
    
    def _calculate_haversine_distance(self, coord1: Coordinates, coord2: Coordinates) -> int:
        """Haversine 공식으로 직선 거리 계산 (미터)"""
        from haversine import haversine, Unit
        
        point1 = (coord1.latitude, coord1.longitude)
        point2 = (coord2.latitude, coord2.longitude)
        
        distance_km = haversine(point1, point2, unit=Unit.KILOMETERS)
        return int(distance_km * 1000)  # 미터로 변환
    
    def _estimate_duration(self, coord1: Coordinates, coord2: Coordinates) -> int:
        """거리 기반 예상 시간 계산 (초)"""
        distance_m = self._calculate_haversine_distance(coord1, coord2)
        
        # 평균 속도 30km/h로 가정
        avg_speed_mps = 30 * 1000 / 3600  # m/s
        return int(distance_m / avg_speed_mps)
    
    def _get_fallback_route_info(self, origin: Coordinates, destination: Coordinates) -> RouteInfo:
        """API 실패 시 기본 경로 정보"""
        self.logger.info("기본 경로 정보 사용 (직선거리 기반)")
        
        distance_m = self._calculate_haversine_distance(origin, destination)
        duration_s = self._estimate_duration(origin, destination)
        
        return RouteInfo(
            distance_meters=distance_m,
            duration_seconds=duration_s,
            toll_fare=0,
            taxi_fare=int(distance_m * 0.001 * 3000),  # km당 3000원 추정
            route_summary="직선거리 추정",
            waypoints=[origin, destination],
            confidence=0.3  # 낮은 신뢰도
        )


# 대안 API 클라이언트들
class OpenRouteServiceClient:
    """OpenRouteService API 클라이언트"""
    
    BASE_URL = "https://api.openrouteservice.org/v2"
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or self._get_default_api_key()
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self._session = None
    
    def _get_default_api_key(self) -> str:
        """기본 OpenRouteService API 키 반환"""
        import os
        return os.getenv('OPENROUTE_API_KEY', 'demo_key')
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self._session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self._session:
            await self._session.close()
    
    async def calculate_route(self, origin: Coordinates, destination: Coordinates) -> Optional[RouteInfo]:
        """OpenRouteService로 실제 경로 계산"""
        if self.api_key == 'demo_key':
            # 데모키인 경우 추정 방식 사용
            return self._get_estimated_route_info(origin, destination)
        
        try:
            if not self._session:
                # 세션이 없으면 컨텍스트 매니저 없이 임시 세션 생성
                async with self as client:
                    return await client._calculate_route_with_api(origin, destination)
            else:
                return await self._calculate_route_with_api(origin, destination)
        except Exception as e:
            self.logger.warning(f"OpenRouteService API 호출 실패: {str(e)}")
            return self._get_estimated_route_info(origin, destination)
    
    async def _calculate_route_with_api(self, origin: Coordinates, destination: Coordinates) -> Optional[RouteInfo]:
        """실제 API 호출로 경로 계산"""
        url = f"{self.BASE_URL}/directions/driving-car"
        
        # API 요청 데이터
        coordinates = [
            [origin.longitude, origin.latitude],
            [destination.longitude, destination.latitude]
        ]
        
        data = {
            "coordinates": coordinates,
            "format": "json",
            "instructions": False,
            "geometry": False
        }
        
        try:
            async with self._session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._parse_ors_response(result, origin, destination)
                elif response.status == 401:
                    self.logger.error("OpenRouteService API 인증 실패 - API 키를 확인하세요")
                    return self._get_estimated_route_info(origin, destination)
                else:
                    self.logger.warning(f"OpenRouteService API 오류: {response.status}")
                    return self._get_estimated_route_info(origin, destination)
        
        except asyncio.TimeoutError:
            self.logger.warning("OpenRouteService API 타임아웃")
            return self._get_estimated_route_info(origin, destination)
        except Exception as e:
            self.logger.error(f"OpenRouteService API 호출 오류: {str(e)}")
            return self._get_estimated_route_info(origin, destination)
    
    def _parse_ors_response(self, api_response: Dict, origin: Coordinates, destination: Coordinates) -> RouteInfo:
        """OpenRouteService API 응답 파싱"""
        routes = api_response.get('routes', [])
        if not routes:
            return self._get_estimated_route_info(origin, destination)
        
        route = routes[0]
        summary = route.get('summary', {})
        
        return RouteInfo(
            distance_meters=int(summary.get('distance', 0)),
            duration_seconds=int(summary.get('duration', 0)),
            toll_fare=0,  # OpenRouteService는 통행료 정보 없음
            taxi_fare=0,  # 택시 요금 추정 안함
            route_summary="OpenRouteService 실제 경로",
            waypoints=[origin, destination],
            confidence=0.85  # 실제 API 결과는 높은 신뢰도
        )
    
    def _get_estimated_route_info(self, origin: Coordinates, destination: Coordinates) -> RouteInfo:
        """API 실패 또는 데모키 사용시 추정 방식"""
        from haversine import haversine, Unit
        
        point1 = (origin.latitude, origin.longitude)
        point2 = (destination.latitude, destination.longitude)
        distance_km = haversine(point1, point2, unit=Unit.KILOMETERS)
        
        # 실제 도로 거리는 직선거리보다 길기 때문에 1.4배 적용 (기존 1.3배에서 개선)
        road_distance_km = distance_km * 1.4
        road_distance_m = int(road_distance_km * 1000)
        
        # 도심 배송 평균 속도 25km/h 적용
        duration_seconds = int(road_distance_km / 25 * 3600)
        
        return RouteInfo(
            distance_meters=road_distance_m,
            duration_seconds=duration_seconds,
            toll_fare=0,
            taxi_fare=int(road_distance_km * 3000),  # km당 3000원 추정
            route_summary="OpenRouteService 추정 경로",
            waypoints=[origin, destination],
            confidence=0.65  # 추정이므로 중간 신뢰도
        )


# 동기 래퍼 클래스
class RoutingClient:
    """동기 경로 계산 클라이언트"""
    
    def __init__(self, kakao_api_key: Optional[str] = None, ors_api_key: Optional[str] = None, use_fallback: bool = True):
        self.ors_client = OpenRouteServiceClient(ors_api_key) if use_fallback else None
        self.kakao_client = KakaoRoutingClient(kakao_api_key) if use_fallback else None
        self.logger = logging.getLogger(__name__)
    
    def calculate_route(self, origin: Coordinates, destination: Coordinates) -> Optional[RouteInfo]:
        """동기 방식으로 경로 계산"""
        return asyncio.run(self._calculate_route_async(origin, destination))
    
    def calculate_distance_matrix(self, origins: List[Coordinates], 
                                destinations: List[Coordinates]) -> List[MatrixResult]:
        """동기 방식으로 거리 매트릭스 계산"""
        return asyncio.run(self._calculate_matrix_async(origins, destinations))
    
    async def _calculate_route_async(self, origin: Coordinates, destination: Coordinates) -> Optional[RouteInfo]:
        """비동기 경로 계산 - OpenRouteService 우선"""
        try:
            # OpenRouteService API 우선 시도
            if self.ors_client:
                route_info = await self.ors_client.calculate_route(origin, destination)
                if route_info and route_info.confidence > 0.5:
                    self.logger.info("OpenRouteService API로 경로 계산 성공")
                    return route_info
        except Exception as e:
            self.logger.warning(f"OpenRouteService API 실패, 카카오맵 API 사용: {str(e)}")
        
        # Fallback: 카카오맵 API 사용
        if self.kakao_client:
            try:
                async with self.kakao_client as client:
                    route_info = await client.calculate_route(origin, destination)
                    if route_info and route_info.confidence > 0.5:
                        self.logger.info("Fallback: 카카오맵 API로 경로 계산 성공")
                        return route_info
            except Exception as e:
                self.logger.warning(f"카카오맵 API도 실패: {str(e)}")
        
        return None
    
    async def _calculate_matrix_async(self, origins: List[Coordinates], 
                                    destinations: List[Coordinates]) -> List[MatrixResult]:
        """비동기 거리 매트릭스 계산 - OpenRouteService 우선"""
        try:
            # OpenRouteService 매트릭스 계산 (향후 구현)
            if self.ors_client:
                # 현재는 개별 경로로 계산
                results = []
                for origin in origins:
                    for destination in destinations:
                        route_info = await self.ors_client.calculate_route(origin, destination)
                        if route_info:
                            result = MatrixResult(
                                origin=origin,
                                destination=destination,
                                distance_meters=route_info.distance_meters,
                                duration_seconds=route_info.duration_seconds,
                                success=True
                            )
                        else:
                            # 실패 시 직선거리 추정
                            result = MatrixResult(
                                origin=origin,
                                destination=destination,
                                distance_meters=self._calculate_haversine_distance(origin, destination),
                                duration_seconds=self._estimate_duration(origin, destination),
                                success=False
                            )
                        results.append(result)
                        await asyncio.sleep(0.05)  # API 제한 고려
                return results
        except Exception as e:
            self.logger.warning(f"OpenRouteService 매트릭스 실패: {str(e)}")
        
        # Fallback: 카카오맵 API 사용
        if self.kakao_client:
            async with self.kakao_client as client:
                return await client.calculate_distance_matrix(origins, destinations)
        
        return []
    
    def _calculate_haversine_distance(self, coord1: Coordinates, coord2: Coordinates) -> int:
        """Haversine 공식으로 직선 거리 계산 (미터)"""
        from haversine import haversine, Unit
        
        point1 = (coord1.latitude, coord1.longitude)
        point2 = (coord2.latitude, coord2.longitude)
        
        distance_km = haversine(point1, point2, unit=Unit.KILOMETERS)
        return int(distance_km * 1000)  # 미터로 변환
    
    def _estimate_duration(self, coord1: Coordinates, coord2: Coordinates) -> int:
        """거리 기반 예상 시간 계산 (초)"""
        distance_m = self._calculate_haversine_distance(coord1, coord2)
        
        # 평균 속도 25km/h로 가정 (도심 배송)
        avg_speed_mps = 25 * 1000 / 3600  # m/s
        return int(distance_m / avg_speed_mps)