"""
차량 도메인 모델
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .coordinates import Coordinates


class VehicleType(Enum):
    """차량 유형"""
    TOP_CAR = "TOP_CAR"
    CARGO = "CARGO"
    OTHER = "OTHER"


class VehicleStatus(Enum):
    """차량 상태"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MAINTENANCE = "MAINTENANCE"
    IN_DELIVERY = "IN_DELIVERY"


class ExperienceLevel(Enum):
    """기사 경험 수준"""
    BEGINNER = 1      # 신입 - 70% 용량
    JUNIOR = 2        # 초급 - 85% 용량  
    INTERMEDIATE = 3  # 중급 - 100% 용량
    SENIOR = 4        # 고급 - 115% 용량
    EXPERT = 5        # 전문가 - 130% 용량


@dataclass
class Vehicle:
    """차량 엔티티"""
    id: str
    driver_name: str
    vehicle_type: VehicleType
    region_id: str
    center_coordinates: Coordinates
    experience_months: int
    max_capacity: int = 40
    safe_capacity: int = 35
    status: VehicleStatus = VehicleStatus.ACTIVE
    auto_dispatch: bool = True
    assigned_orders: List[str] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.assigned_orders is None:
            self.assigned_orders = []
    
    def get_experience_level(self) -> ExperienceLevel:
        """경험 수준 계산"""
        if self.experience_months < 6:
            return ExperienceLevel.BEGINNER
        elif self.experience_months < 12:
            return ExperienceLevel.JUNIOR
        elif self.experience_months < 36:
            return ExperienceLevel.INTERMEDIATE
        elif self.experience_months < 60:
            return ExperienceLevel.SENIOR
        else:
            return ExperienceLevel.EXPERT
    
    def get_experience_multiplier(self) -> float:
        """경험도 계수 반환"""
        multipliers = {
            ExperienceLevel.BEGINNER: 0.70,
            ExperienceLevel.JUNIOR: 0.85,
            ExperienceLevel.INTERMEDIATE: 1.00,
            ExperienceLevel.SENIOR: 1.15,
            ExperienceLevel.EXPERT: 1.30
        }
        return multipliers[self.get_experience_level()]
    
    def calculate_adjusted_capacity(self, weather_factor: float = 1.0, 
                                   traffic_factor: float = 1.0) -> int:
        """조정된 용량 계산"""
        if not self.is_auto_dispatch_eligible():
            return 0
        
        base_capacity = self.safe_capacity
        experience_multiplier = self.get_experience_multiplier()
        
        adjusted = base_capacity * experience_multiplier * weather_factor * traffic_factor
        return min(int(adjusted), self.max_capacity)
    
    def is_auto_dispatch_eligible(self) -> bool:
        """자동 배차 대상 여부"""
        return (self.auto_dispatch and 
                self.status == VehicleStatus.ACTIVE and
                self.vehicle_type in [VehicleType.TOP_CAR, VehicleType.CARGO])
    
    def is_available(self) -> bool:
        """배차 가능 상태 확인"""
        return (self.status == VehicleStatus.ACTIVE and 
                len(self.assigned_orders) == 0)
    
    def assign_orders(self, order_ids: List[str]):
        """주문 배정"""
        self.assigned_orders.extend(order_ids)
        if self.assigned_orders:
            self.status = VehicleStatus.IN_DELIVERY
    
    def clear_assignments(self):
        """배정 초기화"""
        self.assigned_orders.clear()
        self.status = VehicleStatus.ACTIVE
    
    def __str__(self) -> str:
        exp_level = self.get_experience_level().name
        return f"Vehicle({self.id}, {self.driver_name}, {self.vehicle_type.value}, {exp_level})"