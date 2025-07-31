"""
Vehicle Entity - 차량 엔티티

TMS 시스템의 핵심 차량 엔티티입니다.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Set
from uuid import uuid4

from src.shared.constants import VehicleStatus, TmsLimits
from src.domain.value_objects.coordinate import Coordinate


@dataclass
class Vehicle:
    """차량 엔티티"""
    
    vehicle_id: str
    capacity_tons: float
    current_location: Coordinate
    status: VehicleStatus = VehicleStatus.AVAILABLE
    driver_id: Optional[str] = None
    fuel_efficiency_km_per_liter: float = 10.0
    current_load_tons: float = 0.0
    working_hours_today: float = 0.0
    last_maintenance_date: Optional[datetime] = None
    special_capabilities: Set[str] = field(default_factory=set)
    
    def __post_init__(self) -> None:
        """차량 데이터 유효성 검증"""
        if self.capacity_tons <= 0:
            raise ValueError(f"Vehicle capacity must be positive: {self.capacity_tons}")
        
        if self.capacity_tons > TmsLimits.MAX_VEHICLE_CAPACITY_TONS:
            raise ValueError(f"Vehicle capacity exceeds maximum: {self.capacity_tons} > "
                           f"{TmsLimits.MAX_VEHICLE_CAPACITY_TONS}")
        
        if self.current_load_tons < 0:
            raise ValueError(f"Current load cannot be negative: {self.current_load_tons}")
        
        if self.current_load_tons > self.capacity_tons:
            raise ValueError(f"Current load ({self.current_load_tons}) exceeds capacity "
                           f"({self.capacity_tons})")
    
    @property
    def available_capacity_tons(self) -> float:
        """사용 가능한 적재 용량"""
        return self.capacity_tons - self.current_load_tons
    
    @property
    def capacity_utilization_percent(self) -> float:
        """용량 사용률 (백분율)"""
        return (self.current_load_tons / self.capacity_tons) * 100.0
    
    @property
    def is_available_for_new_order(self) -> bool:
        """새로운 주문을 받을 수 있는지 확인"""
        return (self.status == VehicleStatus.AVAILABLE and
                self.driver_id is not None and
                self.working_hours_today < TmsLimits.MAX_WORKING_HOURS_PER_DAY)
    
    @property
    def remaining_working_hours(self) -> float:
        """남은 근무 시간"""
        return max(0, TmsLimits.MAX_WORKING_HOURS_PER_DAY - self.working_hours_today)
    
    def can_handle_load(self, weight_tons: float) -> bool:
        """
        지정된 중량을 처리할 수 있는지 확인
        
        Args:
            weight_tons: 확인할 중량 (톤)
            
        Returns:
            처리 가능하면 True
        """
        return (self.is_available_for_new_order and 
                self.available_capacity_tons >= weight_tons)
    
    def has_special_capability(self, required_capability: str) -> bool:
        """
        특수 능력을 보유하고 있는지 확인
        
        Args:
            required_capability: 필요한 특수 능력 (예: "냉장", "위험물")
            
        Returns:
            보유하고 있으면 True
        """
        return required_capability in self.special_capabilities
    
    def estimate_fuel_consumption(self, distance_km: float) -> float:
        """
        주행 거리에 따른 연료 소모량 추정
        
        Args:
            distance_km: 주행 거리 (킬로미터)
            
        Returns:
            예상 연료 소모량 (리터)
        """
        return distance_km / self.fuel_efficiency_km_per_liter
    
    def estimate_driving_cost(self, distance_km: float, fuel_price_per_liter: float = 1500.0) -> float:
        """
        주행 비용 추정
        
        Args:
            distance_km: 주행 거리 (킬로미터)
            fuel_price_per_liter: 연료 가격 (원/리터)
            
        Returns:
            예상 주행 비용 (원)
        """
        fuel_consumption = self.estimate_fuel_consumption(distance_km)
        return fuel_consumption * fuel_price_per_liter
    
    def add_working_hours(self, hours: float) -> None:
        """
        근무 시간 추가
        
        Args:
            hours: 추가할 근무 시간
        """
        self.working_hours_today += hours
        
        # 최대 근무 시간 초과 시 상태 변경
        if self.working_hours_today >= TmsLimits.MAX_WORKING_HOURS_PER_DAY:
            self.status = VehicleStatus.OUT_OF_SERVICE
    
    def load_cargo(self, weight_tons: float) -> None:
        """
        화물 적재
        
        Args:
            weight_tons: 적재할 중량 (톤)
        """
        if not self.can_handle_load(weight_tons):
            raise ValueError(f"Cannot load {weight_tons} tons. Available capacity: "
                           f"{self.available_capacity_tons} tons")
        
        self.current_load_tons += weight_tons
        
        # 적재 후 상태 업데이트
        if self.current_load_tons > 0:
            self.status = VehicleStatus.BUSY
    
    def unload_cargo(self, weight_tons: float) -> None:
        """
        화물 하역
        
        Args:
            weight_tons: 하역할 중량 (톤)
        """
        if weight_tons > self.current_load_tons:
            raise ValueError(f"Cannot unload {weight_tons} tons. Current load: "
                           f"{self.current_load_tons} tons")
        
        self.current_load_tons -= weight_tons
        
        # 하역 후 상태 업데이트
        if self.current_load_tons == 0 and self.status == VehicleStatus.BUSY:
            self.status = VehicleStatus.AVAILABLE
    
    def move_to(self, new_location: Coordinate) -> float:
        """
        새로운 위치로 이동
        
        Args:
            new_location: 목표 위치
            
        Returns:
            이동 거리 (킬로미터)
        """
        distance = self.current_location.distance_to(new_location)
        self.current_location = new_location
        return distance
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Vehicle':
        """딕셔너리에서 Vehicle 객체 생성"""
        # status 필드 처리 - 문자열이면 VehicleStatus로 변환
        status = data.get('status', 'AVAILABLE')
        if isinstance(status, str):
            status = VehicleStatus(status)
        
        return cls(
            vehicle_id=data['vehicle_id'],
            capacity_tons=data['capacity_tons'],
            current_location=Coordinate.from_dict(data['current_location']),
            status=status,
            driver_id=data.get('driver_id'),
            fuel_efficiency_km_per_liter=data.get('fuel_efficiency_km_per_liter', 10.0),
            current_load_tons=data.get('current_load_tons', 0.0),
            working_hours_today=data.get('working_hours_today', 0.0),
            special_capabilities=set(data.get('special_capabilities', []))
        )
    
    def to_dict(self) -> dict:
        """차량 정보를 딕셔너리로 변환"""
        return {
            'vehicle_id': self.vehicle_id,
            'capacity_tons': self.capacity_tons,
            'current_location': self.current_location.to_dict(),
            'status': self.status.value,
            'driver_id': self.driver_id,
            'fuel_efficiency': self.fuel_efficiency_km_per_liter,
            'current_load_tons': self.current_load_tons,
            'available_capacity_tons': self.available_capacity_tons,
            'capacity_utilization_percent': self.capacity_utilization_percent,
            'working_hours_today': self.working_hours_today,
            'remaining_working_hours': self.remaining_working_hours,
            'special_capabilities': list(self.special_capabilities),
            'is_available': self.is_available_for_new_order
        }
    
    def __str__(self) -> str:
        return (f"Vehicle({self.vehicle_id}, capacity={self.capacity_tons}t, "
                f"status={self.status.value}, load={self.current_load_tons}t)") 