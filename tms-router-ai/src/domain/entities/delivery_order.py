"""
DeliveryOrder Entity - 배송 주문 엔티티

TMS 시스템의 핵심 배송 주문 엔티티입니다.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set
from enum import Enum

from src.shared.constants import Priority
from src.domain.value_objects.coordinate import Coordinate
from src.domain.value_objects.time_window import TimeWindow


class OrderStatus(str, Enum):
    """주문 상태"""
    PENDING = "PENDING"                 # 대기 중
    ASSIGNED = "ASSIGNED"               # 차량 배정됨
    IN_TRANSIT = "IN_TRANSIT"           # 운송 중
    DELIVERED = "DELIVERED"             # 배송 완료
    CANCELLED = "CANCELLED"             # 취소됨
    FAILED = "FAILED"                   # 배송 실패


@dataclass
class DeliveryOrder:
    """배송 주문 엔티티"""
    
    order_id: str
    pickup_location: Coordinate
    delivery_location: Coordinate
    weight_tons: float
    priority: Priority = Priority.MEDIUM
    time_window: Optional[TimeWindow] = None
    special_requirements: Set[str] = field(default_factory=set)
    customer_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    assigned_vehicle_id: Optional[str] = None
    estimated_pickup_time: Optional[datetime] = None
    estimated_delivery_time: Optional[datetime] = None
    actual_pickup_time: Optional[datetime] = None
    actual_delivery_time: Optional[datetime] = None
    delivery_notes: str = ""
    
    def __post_init__(self) -> None:
        """주문 데이터 유효성 검증"""
        if self.weight_tons <= 0:
            raise ValueError(f"Order weight must be positive: {self.weight_tons}")
        
        if self.pickup_location == self.delivery_location:
            raise ValueError("Pickup and delivery locations cannot be the same")
    
    @property
    def delivery_distance_km(self) -> float:
        """픽업 지점에서 배송 지점까지의 거리"""
        return self.pickup_location.distance_to(self.delivery_location)
    
    @property
    def is_urgent(self) -> bool:
        """긴급 주문인지 확인"""
        return self.priority == Priority.URGENT
    
    @property
    def is_high_priority(self) -> bool:
        """높은 우선순위 주문인지 확인"""
        return self.priority in [Priority.HIGH, Priority.URGENT]
    
    @property
    def is_assigned(self) -> bool:
        """차량이 배정되었는지 확인"""
        return self.assigned_vehicle_id is not None
    
    @property
    def is_completed(self) -> bool:
        """배송이 완료되었는지 확인"""
        return self.status == OrderStatus.DELIVERED
    
    @property
    def is_in_progress(self) -> bool:
        """배송이 진행 중인지 확인"""
        return self.status in [OrderStatus.ASSIGNED, OrderStatus.IN_TRANSIT]
    
    @property
    def has_time_constraint(self) -> bool:
        """시간 제약이 있는지 확인"""
        return self.time_window is not None
    
    @property
    def priority_score(self) -> int:
        """우선순위 점수 (높을수록 우선)"""
        priority_scores = {
            Priority.LOW: 1,
            Priority.MEDIUM: 2, 
            Priority.HIGH: 3,
            Priority.URGENT: 4
        }
        return priority_scores[self.priority]
    
    def requires_special_capability(self, capability: str) -> bool:
        """
        특정 특수 능력이 필요한지 확인
        
        Args:
            capability: 확인할 특수 능력
            
        Returns:
            필요하면 True
        """
        return capability in self.special_requirements
    
    def can_be_delivered_at(self, target_time: datetime) -> bool:
        """
        지정된 시간에 배송 가능한지 확인
        
        Args:
            target_time: 확인할 배송 시간
            
        Returns:
            배송 가능하면 True
        """
        if not self.has_time_constraint:
            return True
        
        return self.time_window.contains(target_time)
    
    def assign_to_vehicle(self, vehicle_id: str) -> None:
        """
        차량에 주문 배정
        
        Args:
            vehicle_id: 배정할 차량 ID
        """
        if self.is_assigned:
            raise ValueError(f"Order {self.order_id} is already assigned to vehicle "
                           f"{self.assigned_vehicle_id}")
        
        self.assigned_vehicle_id = vehicle_id
        self.status = OrderStatus.ASSIGNED
    
    def start_delivery(self, pickup_time: datetime) -> None:
        """
        배송 시작
        
        Args:
            pickup_time: 실제 픽업 시간
        """
        if not self.is_assigned:
            raise ValueError(f"Order {self.order_id} is not assigned to any vehicle")
        
        self.actual_pickup_time = pickup_time
        self.status = OrderStatus.IN_TRANSIT
    
    def complete_delivery(self, delivery_time: datetime, notes: str = "") -> None:
        """
        배송 완료
        
        Args:
            delivery_time: 실제 배송 완료 시간
            notes: 배송 관련 메모
        """
        if self.status != OrderStatus.IN_TRANSIT:
            raise ValueError(f"Order {self.order_id} is not in transit")
        
        self.actual_delivery_time = delivery_time
        self.delivery_notes = notes
        self.status = OrderStatus.DELIVERED
    
    def cancel_order(self, reason: str = "") -> None:
        """
        주문 취소
        
        Args:
            reason: 취소 사유
        """
        if self.is_completed:
            raise ValueError(f"Cannot cancel completed order {self.order_id}")
        
        self.delivery_notes = f"Cancelled: {reason}"
        self.status = OrderStatus.CANCELLED
        self.assigned_vehicle_id = None
    
    def calculate_delivery_efficiency(self) -> Optional[float]:
        """
        배송 효율성 계산 (실제 시간 / 예상 시간)
        
        Returns:
            효율성 점수, 계산 불가능하면 None
        """
        if not (self.estimated_delivery_time and self.actual_delivery_time and 
                self.estimated_pickup_time and self.actual_pickup_time):
            return None
        
        estimated_duration = (self.estimated_delivery_time - self.estimated_pickup_time).total_seconds()
        actual_duration = (self.actual_delivery_time - self.actual_pickup_time).total_seconds()
        
        if estimated_duration <= 0:
            return None
        
        return estimated_duration / actual_duration
    
    def is_time_window_violated(self) -> bool:
        """
        시간 창 위반 여부 확인
        
        Returns:
            시간 창을 위반했으면 True
        """
        if not self.has_time_constraint or not self.actual_delivery_time:
            return False
        
        return not self.time_window.contains(self.actual_delivery_time)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DeliveryOrder':
        """딕셔너리에서 DeliveryOrder 객체 생성"""
        return cls(
            order_id=data['order_id'],
            pickup_location=Coordinate.from_dict(data['pickup_location']),
            delivery_location=Coordinate.from_dict(data['delivery_location']),
            weight_tons=data['weight_tons'],
            priority=Priority(data['priority']),
            time_window=TimeWindow.from_dict(data['time_window']) if data.get('time_window') else None,
            special_requirements=set(data.get('special_requirements', [])),
            customer_id=data.get('customer_id'),
            status=OrderStatus(data.get('status', 'PENDING')),
            assigned_vehicle_id=data.get('assigned_vehicle_id'),
            delivery_notes=data.get('delivery_notes', '')
        )
    
    def to_dict(self) -> dict:
        """주문 정보를 딕셔너리로 변환"""
        return {
            'order_id': self.order_id,
            'pickup_location': self.pickup_location.to_dict(),
            'delivery_location': self.delivery_location.to_dict(),
            'weight_tons': self.weight_tons,
            'priority': self.priority.value,
            'priority_score': self.priority_score,
            'status': self.status.value,
            'delivery_distance_km': self.delivery_distance_km,
            'time_window': self.time_window.to_dict() if self.time_window else None,
            'special_requirements': list(self.special_requirements),
            'customer_id': self.customer_id,
            'assigned_vehicle_id': self.assigned_vehicle_id,
            'created_at': self.created_at.isoformat(),
            'estimated_pickup_time': self.estimated_pickup_time.isoformat() if self.estimated_pickup_time else None,
            'estimated_delivery_time': self.estimated_delivery_time.isoformat() if self.estimated_delivery_time else None,
            'actual_pickup_time': self.actual_pickup_time.isoformat() if self.actual_pickup_time else None,
            'actual_delivery_time': self.actual_delivery_time.isoformat() if self.actual_delivery_time else None,
            'delivery_notes': self.delivery_notes,
            'is_urgent': self.is_urgent,
            'is_assigned': self.is_assigned,
            'is_completed': self.is_completed,
            'has_time_constraint': self.has_time_constraint
        }
    
    def __str__(self) -> str:
        return (f"DeliveryOrder({self.order_id}, weight={self.weight_tons}t, "
                f"priority={self.priority.value}, status={self.status.value})") 