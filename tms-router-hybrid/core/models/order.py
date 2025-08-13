"""
주문 도메인 모델
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from .coordinates import Coordinates


class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"
    ASSIGNED = "assigned"  
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """우선순위"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Order:
    """주문 엔티티"""
    id: str
    center_id: str
    region_id: str
    coordinates: Coordinates
    address: str
    priority: Priority = Priority.NORMAL
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    assigned_vehicle_id: Optional[str] = None
    estimated_delivery_time: Optional[int] = None  # 분 단위
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def assign_to_vehicle(self, vehicle_id: str, estimated_time: int = None):
        """차량에 할당"""
        self.assigned_vehicle_id = vehicle_id
        self.status = OrderStatus.ASSIGNED
        if estimated_time:
            self.estimated_delivery_time = estimated_time
    
    def is_assignable(self) -> bool:
        """배정 가능한 상태인지 확인"""
        return self.status == OrderStatus.PENDING
    
    def get_priority_weight(self) -> float:
        """우선순위 가중치 반환"""
        weights = {
            Priority.LOW: 0.8,
            Priority.NORMAL: 1.0,
            Priority.HIGH: 1.3,
            Priority.URGENT: 2.0
        }
        return weights[self.priority]
    
    def __str__(self) -> str:
        return f"Order({self.id}, {self.address}, {self.priority.value})"