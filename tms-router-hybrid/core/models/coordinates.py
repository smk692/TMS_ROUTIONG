"""
좌표 값 객체
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Coordinates:
    """좌표 값 객체 - 불변"""
    latitude: float
    longitude: float
    
    def __post_init__(self):
        """좌표 유효성 검증"""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"잘못된 위도: {self.latitude} (범위: -90~90)")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"잘못된 경도: {self.longitude} (범위: -180~180)")
    
    def to_tuple(self) -> Tuple[float, float]:
        """튜플 형태로 반환"""
        return (self.latitude, self.longitude)
    
    def distance_to(self, other: 'Coordinates') -> float:
        """다른 좌표까지의 거리 계산 (km)"""
        from geopy.distance import geodesic
        return geodesic(self.to_tuple(), other.to_tuple()).kilometers
    
    def __str__(self) -> str:
        return f"({self.latitude:.6f}, {self.longitude:.6f})"