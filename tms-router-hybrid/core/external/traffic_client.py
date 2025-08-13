"""
HERE Maps Traffic API 클라이언트
실시간 교통 정보 수집 및 처리
"""
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from ..models import Coordinates, Region


@dataclass
class TrafficIncident:
    """교통 사고/이벤트 정보"""
    incident_id: str
    incident_type: str      # ACCIDENT, CONSTRUCTION, ROAD_CLOSURE 등
    description: str
    severity: int          # 1(낮음) - 4(높음)
    start_time: datetime
    end_time: Optional[datetime]
    coordinates: Coordinates
    affected_road: str


@dataclass
class TrafficData:
    """교통 정보 데이터"""
    region_id: str
    congestion_level: float    # 0.0(원활) - 1.0(정체)
    average_speed: float       # km/h
    incidents: List[TrafficIncident]
    jam_factor: float         # 1.0(정상) - 10.0(심각한 정체) 
    confidence: float         # 데이터 신뢰도 0.0-1.0
    timestamp: datetime


class HereTrafficClient:
    """HERE Maps Traffic API 클라이언트"""
    
    BASE_URL = "https://traffic.ls.hereapi.com/traffic/6.0"
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key or self._get_default_api_key()
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self._session = None
    
    def _get_default_api_key(self) -> str:
        """기본 API 키 반환"""
        import os
        return os.getenv('HERE_API_KEY', 'demo_key')
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self._session:
            await self._session.close()
    
    async def get_traffic_data(self, region: Region) -> Optional[TrafficData]:
        """권역의 교통 정보 조회"""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # HERE API는 bbox(bounding box) 형식 필요
        bbox = self._calculate_region_bbox(region)
        
        try:
            # 교통 사고/이벤트 정보
            incidents = await self._get_traffic_incidents(bbox)
            
            # 교통 흐름 정보 
            flow_data = await self._get_traffic_flow(bbox)
            
            # 통합 교통 데이터 생성
            return self._create_traffic_data(region.id, incidents, flow_data)
            
        except Exception as e:
            self.logger.error(f"HERE Traffic API 호출 오류: {str(e)}")
            return self._get_fallback_traffic_data(region.id)
    
    def _calculate_region_bbox(self, region: Region) -> str:
        """권역의 경계 상자 계산"""
        # 권역 중심점 기준 ±0.05도 범위 (약 5km)
        center = region.center_coordinates
        margin = 0.05
        
        # bbox format: "북서경도,북서위도,남동경도,남동위도"
        return f"{center.longitude - margin},{center.latitude + margin},{center.longitude + margin},{center.latitude - margin}"
    
    async def _get_traffic_incidents(self, bbox: str) -> List[TrafficIncident]:
        """교통 사고/이벤트 정보 조회"""
        url = f"{self.BASE_URL}/incidents.json"
        params = {
            'bbox': bbox,
            'apikey': self.api_key
        }
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_incidents(data)
                elif response.status == 401:
                    self.logger.error("HERE Traffic API 인증 실패")
                    return []
                else:
                    self.logger.warning(f"HERE Incidents API 오류: {response.status}")
                    return []
                    
        except asyncio.TimeoutError:
            self.logger.warning("HERE Incidents API 타임아웃")
            return []
        except Exception as e:
            self.logger.error(f"교통 사고 정보 조회 오류: {str(e)}")
            return []
    
    async def _get_traffic_flow(self, bbox: str) -> Dict:
        """교통 흐름 정보 조회"""
        url = f"{self.BASE_URL}/flow.json"
        params = {
            'bbox': bbox,
            'apikey': self.api_key
        }
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.warning(f"HERE Flow API 오류: {response.status}")
                    return {}
                    
        except asyncio.TimeoutError:
            self.logger.warning("HERE Flow API 타임아웃")
            return {}
        except Exception as e:
            self.logger.error(f"교통 흐름 정보 조회 오류: {str(e)}")
            return {}
    
    def _parse_incidents(self, api_response: Dict) -> List[TrafficIncident]:
        """교통 사고 API 응답 파싱"""
        incidents = []
        
        for item in api_response.get('TRAFFIC_ITEMS', {}).get('TRAFFIC_ITEM', []):
            try:
                # 위치 정보 추출
                location = item.get('LOCATION', {})
                geoloc = location.get('GEOLOC', {})
                
                if 'ORIGIN' in geoloc:
                    coords = geoloc['ORIGIN']
                    coordinates = Coordinates(
                        latitude=float(coords['LATITUDE']),
                        longitude=float(coords['LONGITUDE'])
                    )
                else:
                    continue
                
                # 시간 정보 파싱
                start_time = self._parse_here_datetime(item.get('START_TIME'))
                end_time = self._parse_here_datetime(item.get('END_TIME'))
                
                incident = TrafficIncident(
                    incident_id=item.get('TRAFFIC_ITEM_ID', ''),
                    incident_type=item.get('TRAFFIC_ITEM_TYPE_DESC', 'UNKNOWN'),
                    description=item.get('TRAFFIC_ITEM_DESCRIPTION', [{}])[0].get('content', ''),
                    severity=int(item.get('CRITICALITY', {}).get('ID', 1)),
                    start_time=start_time,
                    end_time=end_time,
                    coordinates=coordinates,
                    affected_road=location.get('DEFINED', {}).get('ORIGIN', {}).get('ROADWAY', {}).get('content', '')
                )
                
                incidents.append(incident)
                
            except (KeyError, ValueError, TypeError) as e:
                self.logger.debug(f"교통 사고 정보 파싱 오류: {str(e)}")
                continue
        
        return incidents
    
    def _parse_here_datetime(self, datetime_str: Optional[str]) -> Optional[datetime]:
        """HERE API 날짜/시간 문자열 파싱"""
        if not datetime_str:
            return None
        
        try:
            # HERE API 시간 형식: "MM/dd/yyyy HH:mm:ss UTC"
            return datetime.strptime(datetime_str.replace(' UTC', ''), '%m/%d/%Y %H:%M:%S')
        except (ValueError, TypeError):
            return None
    
    def _create_traffic_data(self, region_id: str, incidents: List[TrafficIncident], 
                           flow_data: Dict) -> TrafficData:
        """교통 데이터 통합 생성"""
        
        # 교통 흐름에서 평균 속도 및 정체도 계산
        flow_items = flow_data.get('RWS', [])
        if not flow_items:
            # 기본값 사용
            congestion_level = 0.3  # 보통 정체
            average_speed = 40.0    # 40km/h
            jam_factor = 3.0        # 보통
            confidence = 0.5        # 낮은 신뢰도
        else:
            speeds = []
            jam_factors = []
            
            for rws in flow_items:
                for rw in rws.get('RW', []):
                    for fis in rw.get('FIS', []):
                        for fi in fis.get('FI', []):
                            # 현재 흐름 정보
                            cf = fi.get('CF', [{}])[0]
                            speeds.append(float(cf.get('SP', 40.0)))
                            jam_factors.append(float(cf.get('JF', 3.0)))
            
            if speeds:
                average_speed = sum(speeds) / len(speeds)
                avg_jam_factor = sum(jam_factors) / len(jam_factors)
                
                # 정체 수준 계산 (jam_factor를 0-1 범위로 변환)
                congestion_level = min(1.0, (avg_jam_factor - 1.0) / 9.0)
                jam_factor = avg_jam_factor
                confidence = 0.9
            else:
                congestion_level = 0.3
                average_speed = 40.0
                jam_factor = 3.0
                confidence = 0.5
        
        # 교통사고 영향 추가
        if incidents:
            # 심각한 사고가 있으면 정체 수준 증가
            max_severity = max(inc.severity for inc in incidents)
            congestion_level = min(1.0, congestion_level + (max_severity * 0.1))
        
        return TrafficData(
            region_id=region_id,
            congestion_level=congestion_level,
            average_speed=average_speed,
            incidents=incidents,
            jam_factor=jam_factor,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def _get_fallback_traffic_data(self, region_id: str) -> TrafficData:
        """API 실패 시 기본 교통 데이터"""
        self.logger.info("기본 교통 데이터 사용")
        return TrafficData(
            region_id=region_id,
            congestion_level=0.4,  # 보통 정체
            average_speed=35.0,    # 35km/h
            incidents=[],
            jam_factor=4.0,
            confidence=0.3,        # 낮은 신뢰도
            timestamp=datetime.now()
        )
    
    def get_traffic_impact_multiplier(self, traffic_data: TrafficData) -> float:
        """교통 영향 계수 계산 (0.6-1.1)"""
        # 정체 수준에 따른 배송 용량 조정
        # 0.0 (원활) -> 1.1 (10% 증가)
        # 1.0 (완전 정체) -> 0.6 (40% 감소)
        
        congestion = traffic_data.congestion_level
        
        if congestion <= 0.2:
            return 1.1  # 원활 - 10% 증가
        elif congestion <= 0.6:
            return 1.0  # 보통 - 변화 없음
        elif congestion <= 0.8:
            return 0.8  # 정체 - 20% 감소
        else:
            return 0.6  # 심각한 정체 - 40% 감소


# 동기 래퍼 클래스
class TrafficClient:
    """동기 교통 정보 클라이언트"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.async_client = HereTrafficClient(api_key)
    
    def get_traffic_data(self, region: Region) -> Optional[TrafficData]:
        """동기 방식으로 교통 데이터 조회"""
        return asyncio.run(self._get_traffic_async(region))
    
    async def _get_traffic_async(self, region: Region) -> Optional[TrafficData]:
        """비동기 교통 데이터 조회"""
        async with self.async_client as client:
            return await client.get_traffic_data(region)