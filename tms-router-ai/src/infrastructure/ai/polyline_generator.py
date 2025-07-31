"""
PolylineGenerator - 경로 폴리라인 생성기

Google Maps 호환 폴리라인을 생성하는 유틸리티입니다.
OSRM (Open Source Routing Machine) API를 활용하여 실제 도로 경로를 계산합니다.

OSRM 데모 서버: http://router.project-osrm.org
- 무료 사용 가능 (API 키 불필요)
- Route API: 경로 계산
- Trip API: TSP (여행하는 판매원 문제) 해결
"""
import math
from typing import List, Tuple, Dict, Any, Optional
import requests
import json
import time
from dataclasses import dataclass

from src.domain.value_objects.coordinate import Coordinate
from src.shared.exceptions import TmsError


@dataclass
class RoutePoint:
    """경로 지점"""
    coordinate: Coordinate
    type: str  # start, pickup, delivery, end
    order_id: Optional[str] = None


class PolylineEncoder:
    """폴리라인 인코딩 유틸리티 (Google Algorithm)"""
    
    @staticmethod
    def encode_coordinates(coordinates: List[Tuple[float, float]]) -> str:
        """
        좌표 리스트를 Google Maps 호환 폴리라인으로 인코딩
        
        Args:
            coordinates: (위도, 경도) 튜플 리스트
            
        Returns:
            인코딩된 폴리라인 문자열
        """
        if not coordinates:
            return ""
        
        encoded = []
        prev_lat = prev_lng = 0
        
        for lat, lng in coordinates:
            # 좌표를 1e5배 하고 정수로 변환
            lat_e5 = int(lat * 1e5)
            lng_e5 = int(lng * 1e5)
            
            # 이전 좌표와의 차이 계산
            dlat = lat_e5 - prev_lat
            dlng = lng_e5 - prev_lng
            
            # 각 차이값을 인코딩
            encoded.extend(PolylineEncoder._encode_value(dlat))
            encoded.extend(PolylineEncoder._encode_value(dlng))
            
            prev_lat = lat_e5
            prev_lng = lng_e5
        
        return ''.join(encoded)
    
    @staticmethod
    def _encode_value(value: int) -> List[str]:
        """단일 값을 폴리라인 형식으로 인코딩"""
        # 음수 처리: 좌측으로 1비트 시프트하고 XOR
        value = ~(value << 1) if value < 0 else value << 1
        
        encoded = []
        while value >= 0x20:
            encoded.append(chr((0x20 | (value & 0x1f)) + 63))
            value >>= 5
        
        encoded.append(chr(value + 63))
        return encoded
    
    @staticmethod
    def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
        """
        폴리라인 문자열을 좌표 리스트로 디코딩
        
        Args:
            polyline_str: 인코딩된 폴리라인 문자열
            
        Returns:
            (위도, 경도) 튜플 리스트
        """
        coordinates = []
        index = lat = lng = 0
        
        while index < len(polyline_str):
            # 위도 디코딩
            result = 1
            shift = 0
            while True:
                b = ord(polyline_str[index]) - 63 - 1
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1f:
                    break
            
            lat += (~result >> 1) if (result & 1) != 0 else (result >> 1)
            
            # 경도 디코딩
            result = 1
            shift = 0
            while True:
                b = ord(polyline_str[index]) - 63 - 1
                index += 1
                result += b << shift
                shift += 5
                if b < 0x1f:
                    break
            
            lng += (~result >> 1) if (result & 1) != 0 else (result >> 1)
            
            coordinates.append((lat / 1e5, lng / 1e5))
        
        return coordinates


class OSRMClient:
    """
    OSRM (Open Source Routing Machine) API 클라이언트
    https://project-osrm.org/
    
    무료 데모 서버 사용: router.project-osrm.org
    """
    
    def __init__(self, base_url: Optional[str] = None):
        # OSRM 데모 서버 사용 (API 키 불필요)
        self.base_url = base_url or "http://router.project-osrm.org"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'TMS-Router-AI/1.0'
        })
    
    def get_route_polyline(self, coordinates: List[Coordinate], profile: str = "driving") -> str:
        """
        OSRM API를 사용하여 경로 폴리라인 생성
        
        Args:
            coordinates: 경유지 좌표 리스트
            profile: 라우팅 프로필 (driving, walking, cycling)
            
        Returns:
            Google Maps 호환 폴리라인
        """
        if len(coordinates) < 2:
            return ""
        
        try:
            # OSRM 좌표 형식: longitude,latitude;longitude,latitude;...
            coords_str = ";".join([
                f"{coord.longitude},{coord.latitude}" 
                for coord in coordinates
            ])
            
            # OSRM Route API 호출
            url = f"{self.base_url}/route/v1/{profile}/{coords_str}"
            
            params = {
                'overview': 'full',           # 전체 경로 geometry 반환
                'geometries': 'polyline',     # 폴리라인 형식으로 반환
                'steps': 'false',             # 단계별 정보 불필요
                'annotations': 'false'        # 추가 주석 불필요
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # OSRM 응답에서 폴리라인 추출
                if data.get('code') == 'Ok' and data.get('routes'):
                    route = data['routes'][0]
                    polyline = route.get('geometry', '')
                    
                    if polyline:
                        return polyline
                
                # API 성공했지만 경로 없는 경우 빈 폴리라인 반환
                return ""
            
            else:
                # HTTP 오류시 빈 폴리라인 반환 (의미없는 직선보다 낫다)
                return ""
                
        except Exception as e:
            # 예외 발생시 빈 폴리라인 반환 (프론트엔드에서 처리)
            return ""
    
    def get_route_info(self, coordinates: List[Coordinate], profile: str = "driving") -> Dict[str, Any]:
        """
        경로 정보 (거리, 시간) 조회
        
        Args:
            coordinates: 경유지 좌표 리스트  
            profile: 라우팅 프로필
            
        Returns:
            경로 정보 딕셔너리 (distance_m, duration_s, polyline)
        """
        if len(coordinates) < 2:
            return {'distance_m': 0, 'duration_s': 0, 'polyline': ''}
        
        try:
            coords_str = ";".join([
                f"{coord.longitude},{coord.latitude}" 
                for coord in coordinates
            ])
            
            url = f"{self.base_url}/route/v1/{profile}/{coords_str}"
            params = {
                'overview': 'full',
                'geometries': 'polyline'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == 'Ok' and data.get('routes'):
                    route = data['routes'][0]
                    return {
                        'distance_m': route.get('distance', 0),      # 미터
                        'duration_s': route.get('duration', 0),      # 초
                        'polyline': route.get('geometry', '')
                    }
            
            # 실패시 기본값 반환 (거리/시간 정보는 추정치로)
            return self._calculate_straight_distance(coordinates)
            
        except Exception:
            return self._calculate_straight_distance(coordinates)
    
    def get_trip_polyline(self, coordinates: List[Coordinate], profile: str = "driving") -> str:
        """
        TSP (여행하는 판매원 문제) 해결 후 폴리라인 생성
        OSRM Trip API 사용
        
        Args:
            coordinates: 방문할 좌표 리스트
            profile: 라우팅 프로필
            
        Returns:
            최적화된 순서의 폴리라인
        """
        if len(coordinates) < 2:
            return ""
        
        try:
            coords_str = ";".join([
                f"{coord.longitude},{coord.latitude}" 
                for coord in coordinates
            ])
            
            # OSRM Trip API 호출 (TSP 해결)
            url = f"{self.base_url}/trip/v1/{profile}/{coords_str}"
            params = {
                'overview': 'full',
                'geometries': 'polyline',
                'source': 'first',      # 첫 번째 점에서 시작
                'destination': 'last',  # 마지막 점에서 종료
                'roundtrip': 'false'    # 왕복 여행 아님
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == 'Ok' and data.get('trips'):
                    trip = data['trips'][0]
                    return trip.get('geometry', '')
            
            # Trip API 실패시 일반 Route API로 대체
            return self.get_route_polyline(coordinates, profile)
            
        except Exception:
            return self.get_route_polyline(coordinates, profile)
    
    def _generate_straight_line_polyline(self, coordinates: List[Coordinate]) -> str:
        """직선 경로 폴리라인 생성 (대체 방안)"""
        coords = [(coord.latitude, coord.longitude) for coord in coordinates]
        return PolylineEncoder.encode_coordinates(coords)
    
    def _calculate_straight_distance(self, coordinates: List[Coordinate]) -> Dict[str, Any]:
        """직선 거리 기반 경로 정보 계산"""
        if len(coordinates) < 2:
            return {'distance_m': 0, 'duration_s': 0, 'polyline': ''}
        
        total_distance = 0
        for i in range(len(coordinates) - 1):
            distance = coordinates[i].distance_to(coordinates[i + 1])
            total_distance += distance
        
        # 평균 속도 50km/h 가정
        duration_s = (total_distance / 50) * 3600  # 초 단위
        
        return {
            'distance_m': total_distance * 1000,  # 미터 변환
            'duration_s': duration_s,
            'polyline': self._generate_straight_line_polyline(coordinates)
        }


class PolylineGenerator:
    """폴리라인 생성기 메인 클래스"""
    
    def __init__(self, use_external_api: bool = True, osrm_base_url: Optional[str] = None):
        self.use_external_api = use_external_api
        self.osrm_client = OSRMClient(osrm_base_url) if use_external_api else None
        self.encoder = PolylineEncoder()
    
    def generate_route_polyline(self, waypoints: List[Dict[str, Any]]) -> str:
        """
        경유지 정보로부터 경로 폴리라인 생성
        
        Args:
            waypoints: 경유지 정보 리스트
            
        Returns:
            Google Maps 호환 폴리라인
        """
        if not waypoints:
            return ""
        
        # 경유지에서 좌표 추출
        coordinates = []
        for waypoint in waypoints:
            location = waypoint.get('location', {})
            lat = location.get('lat')
            lng = location.get('lng')
            
            if lat is not None and lng is not None:
                coordinates.append(Coordinate(latitude=lat, longitude=lng))
        
        if len(coordinates) < 2:
            return ""
        
        # OSRM API 사용 가능한 경우
        if self.use_external_api and self.osrm_client:
            try:
                return self.osrm_client.get_route_polyline(coordinates)
            except Exception:
                # API 실패 시 직선 경로로 대체
                pass
        
        # 직선 경로 생성 (기본 동작)
        return self._generate_simple_polyline(coordinates)
    
    def generate_optimized_polyline(self, route_points: List[RoutePoint]) -> str:
        """
        최적화된 경로 순서를 고려한 폴리라인 생성
        
        Args:
            route_points: 경로 지점 리스트
            
        Returns:
            최적화된 폴리라인
        """
        if not route_points:
            return ""
        
        # 경로 지점을 타입별로 정렬
        sorted_points = self._sort_route_points(route_points)
        coordinates = [point.coordinate for point in sorted_points]
        
        if self.use_external_api and self.osrm_client:
            try:
                # TSP 해결이 필요한 경우 OSRM Trip API 사용
                if len(coordinates) > 2:
                    return self.osrm_client.get_trip_polyline(coordinates)
                else:
                    return self.osrm_client.get_route_polyline(coordinates)
            except Exception:
                pass
        
        return self._generate_simple_polyline(coordinates)
    
    def _generate_simple_polyline(self, coordinates: List[Coordinate]) -> str:
        """간단한 직선 경로 폴리라인 생성"""
        coords = [(coord.latitude, coord.longitude) for coord in coordinates]
        return self.encoder.encode_coordinates(coords)
    
    def _sort_route_points(self, route_points: List[RoutePoint]) -> List[RoutePoint]:
        """경로 지점을 논리적 순서로 정렬"""
        # 타입별 우선순위
        type_priority = {
            'start': 0,
            'pickup': 1,
            'delivery': 2,
            'end': 3
        }
        
        return sorted(route_points, key=lambda p: type_priority.get(p.type, 1))
    
    def validate_polyline(self, polyline: str) -> bool:
        """폴리라인 유효성 검증"""
        try:
            decoded = self.encoder.decode_polyline(polyline)
            return len(decoded) >= 2
        except Exception:
            return False
    
    def get_polyline_bounds(self, polyline: str) -> Optional[Dict[str, float]]:
        """
        폴리라인의 경계 좌표 계산
        
        Args:
            polyline: 폴리라인 문자열
            
        Returns:
            경계 좌표 딕셔너리 (north, south, east, west)
        """
        try:
            coordinates = self.encoder.decode_polyline(polyline)
            
            if not coordinates:
                return None
            
            lats = [coord[0] for coord in coordinates]
            lngs = [coord[1] for coord in coordinates]
            
            return {
                'north': max(lats),
                'south': min(lats),
                'east': max(lngs),
                'west': min(lngs)
            }
            
        except Exception:
            return None

    def get_route_with_info(self, waypoints: List[Dict[str, Any]], profile: str = "driving") -> Dict[str, Any]:
        """
        경로 정보와 함께 폴리라인 생성
        
        Args:
            waypoints: 경유지 정보 리스트
            profile: 라우팅 프로필 (driving, walking, cycling)
            
        Returns:
            경로 정보와 폴리라인을 포함한 딕셔너리
        """
        if not waypoints:
            return {'polyline': '', 'distance_km': 0, 'duration_hours': 0}
        
        # 경유지에서 좌표 추출
        coordinates = []
        for waypoint in waypoints:
            location = waypoint.get('location', {})
            lat = location.get('lat')
            lng = location.get('lng')
            
            if lat is not None and lng is not None:
                coordinates.append(Coordinate(latitude=lat, longitude=lng))
        
        if len(coordinates) < 2:
            return {'polyline': '', 'distance_km': 0, 'duration_hours': 0}
        
        # OSRM API로 상세 정보 조회
        if self.use_external_api and self.osrm_client:
            try:
                route_info = self.osrm_client.get_route_info(coordinates, profile)
                return {
                    'polyline': route_info['polyline'],
                    'distance_km': route_info['distance_m'] / 1000,  # km 변환
                    'duration_hours': route_info['duration_s'] / 3600  # 시간 변환
                }
            except Exception:
                pass
        
        # 대체 방안: 직선 거리 계산
        polyline = self._generate_simple_polyline(coordinates)
        total_distance = sum(
            coordinates[i].distance_to(coordinates[i + 1])
            for i in range(len(coordinates) - 1)
        )
        duration_hours = total_distance / 50  # 평균 50km/h 가정
        
        return {
            'polyline': polyline,
            'distance_km': total_distance,
            'duration_hours': duration_hours
        }
    
    def generate_tsp_optimized_polyline(self, waypoints: List[Dict[str, Any]], 
                                      start_point: Optional[Dict[str, Any]] = None,
                                      end_point: Optional[Dict[str, Any]] = None) -> str:
        """
        TSP 최적화된 경로 폴리라인 생성
        
        Args:
            waypoints: 방문할 경유지 리스트
            start_point: 시작점 (선택사항)
            end_point: 종료점 (선택사항)
            
        Returns:
            최적화된 순서의 폴리라인
        """
        coordinates = []
        
        # 시작점 추가
        if start_point:
            location = start_point.get('location', {})
            if location.get('lat') and location.get('lng'):
                coordinates.append(Coordinate(
                    latitude=location['lat'], 
                    longitude=location['lng']
                ))
        
        # 경유지 추가
        for waypoint in waypoints:
            location = waypoint.get('location', {})
            if location.get('lat') and location.get('lng'):
                coordinates.append(Coordinate(
                    latitude=location['lat'], 
                    longitude=location['lng']
                ))
        
        # 종료점 추가
        if end_point and end_point != start_point:
            location = end_point.get('location', {})
            if location.get('lat') and location.get('lng'):
                coordinates.append(Coordinate(
                    latitude=location['lat'], 
                    longitude=location['lng']
                ))
        
        if len(coordinates) < 2:
            return ""
        
        # OSRM Trip API 사용 (TSP 해결)
        if self.use_external_api and self.osrm_client:
            try:
                return self.osrm_client.get_trip_polyline(coordinates)
            except Exception:
                pass
        
        # 대체 방안: 단순 순서로 폴리라인 생성
        return self._generate_simple_polyline(coordinates)
    
    def get_multiple_routes_polylines(self, vehicle_routes: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        여러 차량의 경로에 대해 일괄 폴리라인 생성
        
        Args:
            vehicle_routes: 차량별 경로 정보 리스트
            
        Returns:
            차량 ID별 폴리라인 딕셔너리
        """
        polylines = {}
        
        for route in vehicle_routes:
            vehicle_id = route.get('vehicle_id', '')
            waypoints = route.get('waypoints', [])
            
            if vehicle_id and waypoints:
                polyline = self.generate_route_polyline(waypoints)
                polylines[vehicle_id] = polyline
        
        return polylines

    def get_route_with_status(self, waypoints: List[Dict[str, Any]], profile: str = "driving") -> Dict[str, Any]:
        """
        경로 정보와 상태를 포함한 폴리라인 생성
        
        Args:
            waypoints: 경유지 정보 리스트
            profile: 라우팅 프로필 (driving, walking, cycling)
            
        Returns:
            경로 정보, 폴리라인, 상태를 포함한 딕셔너리
        """
        if not waypoints:
            return {
                'polyline': '', 
                'distance_km': 0, 
                'duration_hours': 0,
                'status': 'no_waypoints',
                'source': 'none'
            }
        
        # 경유지에서 좌표 추출
        coordinates = []
        for waypoint in waypoints:
            location = waypoint.get('location', {})
            lat = location.get('lat')
            lng = location.get('lng')
            
            if lat is not None and lng is not None:
                coordinates.append(Coordinate(latitude=lat, longitude=lng))
        
        if len(coordinates) < 2:
            return {
                'polyline': '', 
                'distance_km': 0, 
                'duration_hours': 0,
                'status': 'insufficient_coordinates',
                'source': 'none'
            }
        
        # OSRM API로 실제 도로 경로 시도
        if self.use_external_api and self.osrm_client:
            try:
                route_info = self.osrm_client.get_route_info(coordinates, profile)
                
                if route_info['polyline']:
                    return {
                        'polyline': route_info['polyline'],
                        'distance_km': route_info['distance_m'] / 1000,
                        'duration_hours': route_info['duration_s'] / 3600,
                        'status': 'success',
                        'source': 'osrm_api',
                        'profile': profile
                    }
                else:
                    # OSRM에서 경로를 찾을 수 없음
                    return {
                        'polyline': '',
                        'distance_km': 0,
                        'duration_hours': 0,
                        'status': 'no_route_found',
                        'source': 'osrm_api',
                        'error': 'OSRM could not find a route between these points'
                    }
                    
            except Exception as e:
                # OSRM API 오류
                return {
                    'polyline': '',
                    'distance_km': 0,
                    'duration_hours': 0,
                    'status': 'api_error',
                    'source': 'osrm_api',
                    'error': str(e)
                }
        
        # OSRM API 사용 불가능한 경우
        return {
            'polyline': '',
            'distance_km': 0,
            'duration_hours': 0,
            'status': 'api_disabled',
            'source': 'none',
            'note': 'OSRM API is disabled. Only real road routes are meaningful.'
        }


# 전역 폴리라인 생성기 인스턴스
_polyline_generator = None

def get_polyline_generator() -> PolylineGenerator:
    """폴리라인 생성기 싱글톤 인스턴스 반환"""
    global _polyline_generator
    if _polyline_generator is None:
        _polyline_generator = PolylineGenerator()
    return _polyline_generator 