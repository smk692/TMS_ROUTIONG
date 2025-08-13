"""
OpenWeatherMap API 클라이언트
날씨 데이터 수집 및 처리
"""
import asyncio
import aiohttp
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..models import Coordinates


@dataclass
class WeatherData:
    """날씨 데이터"""
    temperature: float  # 온도 (섭씨)
    humidity: int      # 습도 (%)
    visibility: int    # 가시거리 (m)
    wind_speed: float  # 풍속 (m/s)
    precipitation: float  # 강수량 (mm)
    weather_main: str  # 주요 날씨 상태
    weather_description: str  # 날씨 설명
    severity_score: float  # 심각도 점수 (1.0-5.0)
    timestamp: datetime


class OpenWeatherMapClient:
    """OpenWeatherMap API 클라이언트"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or self._get_default_api_key()
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self._session = None
    
    def _get_default_api_key(self) -> str:
        """기본 API 키 반환 (환경변수 또는 기본값)"""
        import os
        return os.getenv('OPENWEATHER_API_KEY', 'demo_key')
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self._session:
            await self._session.close()
    
    async def get_current_weather(self, coordinates: Coordinates) -> Optional[WeatherData]:
        """현재 날씨 정보 조회"""
        if not self._session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.BASE_URL}/weather"
        params = {
            'lat': coordinates.latitude,
            'lon': coordinates.longitude,
            'appid': self.api_key,
            'units': 'metric',  # 섭씨 온도
            'lang': 'kr'       # 한국어
        }
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_weather_data(data)
                elif response.status == 401:
                    self.logger.error("OpenWeatherMap API 인증 실패 - API 키를 확인하세요")
                    return self._get_fallback_weather_data()
                else:
                    self.logger.warning(f"OpenWeatherMap API 오류: {response.status}")
                    return self._get_fallback_weather_data()
                    
        except asyncio.TimeoutError:
            self.logger.warning("OpenWeatherMap API 타임아웃")
            return self._get_fallback_weather_data()
        except Exception as e:
            self.logger.error(f"OpenWeatherMap API 호출 오류: {str(e)}")
            return self._get_fallback_weather_data()
    
    def _parse_weather_data(self, api_response: Dict) -> WeatherData:
        """API 응답을 WeatherData 객체로 변환"""
        weather = api_response['weather'][0]
        main = api_response['main']
        wind = api_response.get('wind', {})
        rain = api_response.get('rain', {})
        
        # 강수량 계산 (1시간)
        precipitation = rain.get('1h', 0.0)
        
        # 심각도 점수 계산
        severity_score = self._calculate_weather_severity(
            temperature=main['temp'],
            humidity=main['humidity'],
            wind_speed=wind.get('speed', 0),
            precipitation=precipitation,
            weather_main=weather['main']
        )
        
        return WeatherData(
            temperature=main['temp'],
            humidity=main['humidity'],
            visibility=api_response.get('visibility', 10000),
            wind_speed=wind.get('speed', 0),
            precipitation=precipitation,
            weather_main=weather['main'],
            weather_description=weather['description'],
            severity_score=severity_score,
            timestamp=datetime.now()
        )
    
    def _calculate_weather_severity(self, temperature: float, humidity: int, 
                                  wind_speed: float, precipitation: float, 
                                  weather_main: str) -> float:
        """날씨 심각도 점수 계산 (1.0-5.0)"""
        severity = 1.0  # 기본값 (양호)
        
        # 온도 영향 (-10°C 이하 또는 35°C 이상)
        if temperature <= -10 or temperature >= 35:
            severity += 1.5
        elif temperature <= 0 or temperature >= 30:
            severity += 1.0
        elif temperature <= 5 or temperature >= 28:
            severity += 0.5
        
        # 강수량 영향
        if precipitation >= 20:  # 폭우
            severity += 2.0
        elif precipitation >= 10:  # 많은 비
            severity += 1.5
        elif precipitation >= 5:   # 비
            severity += 1.0
        elif precipitation >= 1:   # 약간의 비
            severity += 0.5
        
        # 풍속 영향
        if wind_speed >= 20:  # 강풍
            severity += 1.5
        elif wind_speed >= 15:  # 센 바람
            severity += 1.0
        elif wind_speed >= 10:  # 바람
            severity += 0.5
        
        # 날씨 상태별 가중치
        weather_weights = {
            'Thunderstorm': 2.0,  # 뇌우
            'Snow': 1.5,          # 눈
            'Fog': 1.0,           # 안개
            'Mist': 0.5,          # 박무
            'Clear': -0.5,        # 맑음 (보너스)
        }
        
        if weather_main in weather_weights:
            severity += weather_weights[weather_main]
        
        # 1.0-5.0 범위로 제한
        return max(1.0, min(5.0, severity))
    
    def _get_fallback_weather_data(self) -> WeatherData:
        """API 실패 시 기본값 반환"""
        self.logger.info("기본 날씨 데이터 사용")
        return WeatherData(
            temperature=20.0,
            humidity=60,
            visibility=10000,
            wind_speed=3.0,
            precipitation=0.0,
            weather_main="Clear",
            weather_description="맑음 (기본값)",
            severity_score=1.0,  # 양호 상태
            timestamp=datetime.now()
        )
    
    def get_weather_impact_multiplier(self, weather_data: WeatherData) -> float:
        """날씨 영향 계수 계산 (0.3-1.1)"""
        # 심각도 점수를 배송 용량 계수로 변환
        # 1.0 (양호) -> 1.1 (10% 증가)
        # 5.0 (매우 나쁨) -> 0.3 (70% 감소)
        
        if weather_data.severity_score <= 1.5:
            return 1.1  # 좋은 날씨 - 10% 증가
        elif weather_data.severity_score <= 2.5:
            return 1.0  # 보통 날씨 - 변화 없음
        elif weather_data.severity_score <= 3.5:
            return 0.8  # 나쁜 날씨 - 20% 감소
        elif weather_data.severity_score <= 4.5:
            return 0.6  # 매우 나쁜 날씨 - 40% 감소
        else:
            return 0.3  # 극한 날씨 - 70% 감소


# 동기 래퍼 클래스
class WeatherClient:
    """동기 날씨 클라이언트 (기존 코드 호환성)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.async_client = OpenWeatherMapClient(api_key)
    
    def get_weather_data(self, coordinates: Coordinates) -> Optional[WeatherData]:
        """동기 방식으로 날씨 데이터 조회"""
        return asyncio.run(self._get_weather_async(coordinates))
    
    async def _get_weather_async(self, coordinates: Coordinates) -> Optional[WeatherData]:
        """비동기 날씨 데이터 조회"""
        async with self.async_client as client:
            return await client.get_current_weather(coordinates)