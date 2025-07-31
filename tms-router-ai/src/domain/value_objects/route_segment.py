"""
RouteSegment Value Object - 경로 구간

경로의 개별 구간을 나타내는 불변 값 객체입니다.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

from src.domain.value_objects.coordinate import Coordinate


class SegmentType(str, Enum):
    """구간 타입"""
    TRAVEL = "TRAVEL"           # 이동
    PICKUP = "PICKUP"           # 픽업
    DELIVERY = "DELIVERY"       # 배송
    BREAK = "BREAK"             # 휴식
    REFUEL = "REFUEL"           # 주유


@dataclass(frozen=True)
class RouteSegment:
    """경로 구간"""
    
    segment_id: str
    start_location: Coordinate
    end_location: Coordinate
    segment_type: SegmentType
    distance_km: float
    estimated_duration_minutes: float
    order_id: Optional[str] = None
    sequence_number: int = 0
    notes: str = ""
    
    def __post_init__(self) -> None:
        """구간 데이터 유효성 검증"""
        if self.distance_km < 0:
            raise ValueError(f"Distance cannot be negative: {self.distance_km}")
        
        if self.estimated_duration_minutes < 0:
            raise ValueError(f"Duration cannot be negative: {self.estimated_duration_minutes}")
        
        if self.sequence_number < 0:
            raise ValueError(f"Sequence number cannot be negative: {self.sequence_number}")
    
    @property
    def estimated_duration_hours(self) -> float:
        """예상 소요 시간 (시간 단위)"""
        return self.estimated_duration_minutes / 60.0
    
    @property
    def estimated_duration_timedelta(self) -> timedelta:
        """예상 소요 시간 (timedelta)"""
        return timedelta(minutes=self.estimated_duration_minutes)
    
    @property
    def is_service_stop(self) -> bool:
        """서비스 정차 지점인지 확인 (픽업/배송)"""
        return self.segment_type in [SegmentType.PICKUP, SegmentType.DELIVERY]
    
    @property
    def is_travel_segment(self) -> bool:
        """이동 구간인지 확인"""
        return self.segment_type == SegmentType.TRAVEL
    
    @property
    def requires_order(self) -> bool:
        """주문 ID가 필요한 구간인지 확인"""
        return self.segment_type in [SegmentType.PICKUP, SegmentType.DELIVERY]
    
    def calculate_average_speed_kmh(self) -> float:
        """
        평균 속도 계산
        
        Returns:
            평균 속도 (km/h)
        """
        if self.estimated_duration_hours <= 0:
            return 0.0
        
        return self.distance_km / self.estimated_duration_hours
    
    def estimate_fuel_consumption(self, fuel_efficiency_kmpl: float = 10.0) -> float:
        """
        연료 소모량 추정
        
        Args:
            fuel_efficiency_kmpl: 연비 (km/L)
            
        Returns:
            예상 연료 소모량 (리터)
        """
        if fuel_efficiency_kmpl <= 0:
            return 0.0
        
        return self.distance_km / fuel_efficiency_kmpl
    
    def estimate_cost(self, cost_per_km: float = 1000.0) -> float:
        """
        구간 비용 추정
        
        Args:
            cost_per_km: 킬로미터당 비용 (원)
            
        Returns:
            예상 비용 (원)
        """
        return self.distance_km * cost_per_km
    
    def is_compatible_with_order(self, order_id: str) -> bool:
        """
        주문과 호환되는지 확인
        
        Args:
            order_id: 확인할 주문 ID
            
        Returns:
            호환되면 True
        """
        if not self.requires_order:
            return True
        
        return self.order_id == order_id
    
    def to_dict(self) -> dict:
        """구간 정보를 딕셔너리로 변환"""
        return {
            'segment_id': self.segment_id,
            'start_location': self.start_location.to_dict(),
            'end_location': self.end_location.to_dict(),
            'segment_type': self.segment_type.value,
            'distance_km': self.distance_km,
            'estimated_duration_minutes': self.estimated_duration_minutes,
            'estimated_duration_hours': self.estimated_duration_hours,
            'average_speed_kmh': self.calculate_average_speed_kmh(),
            'order_id': self.order_id,
            'sequence_number': self.sequence_number,
            'notes': self.notes,
            'is_service_stop': self.is_service_stop,
            'is_travel_segment': self.is_travel_segment
        }
    
    def __str__(self) -> str:
        return (f"RouteSegment({self.segment_id}, {self.segment_type.value}, "
                f"{self.distance_km:.1f}km, {self.estimated_duration_minutes:.0f}min)") 