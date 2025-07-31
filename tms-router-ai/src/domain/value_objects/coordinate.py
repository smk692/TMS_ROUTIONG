"""
Coordinate Value Object - 지리적 좌표

GPS 좌표를 나타내는 불변 값 객체입니다.
"""
from dataclasses import dataclass
from typing import Tuple
import math


@dataclass(frozen=True)
class Coordinate:
    """지리적 좌표 (위도, 경도)"""
    
    latitude: float
    longitude: float
    
    def __post_init__(self) -> None:
        """좌표 유효성 검증"""
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Invalid latitude: {self.latitude}. Must be between -90 and 90")
        
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Invalid longitude: {self.longitude}. Must be between -180 and 180")
    
    def distance_to(self, other: 'Coordinate') -> float:
        """
        두 좌표 간의 거리를 계산 (하버사인 공식)
        
        Args:
            other: 목표 좌표
            
        Returns:
            거리 (킬로미터)
        """
        # 지구의 반지름 (km)
        EARTH_RADIUS_KM = 6371.0
        
        # 라디안으로 변환
        lat1_rad = math.radians(self.latitude)
        lon1_rad = math.radians(self.longitude)
        lat2_rad = math.radians(other.latitude)
        lon2_rad = math.radians(other.longitude)
        
        # 위도와 경도 차이
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # 하버사인 공식
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return EARTH_RADIUS_KM * c
    
    def to_tuple(self) -> Tuple[float, float]:
        """좌표를 튜플로 반환"""
        return (self.latitude, self.longitude)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Coordinate':
        """딕셔너리에서 Coordinate 객체 생성"""
        return cls(
            latitude=data['lat'],
            longitude=data['lng']
        )
    
    def to_dict(self) -> dict:
        """좌표를 딕셔너리로 반환"""
        return {
            'lat': self.latitude,
            'lng': self.longitude
        }
    
    def __str__(self) -> str:
        return f"Coordinate(lat={self.latitude:.6f}, lng={self.longitude:.6f})" 