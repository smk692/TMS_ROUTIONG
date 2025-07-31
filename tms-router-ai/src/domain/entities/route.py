"""
Route Entity - 경로 엔티티

TMS 시스템의 배차 경로를 나타내는 엔티티입니다.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from src.domain.entities.vehicle import Vehicle

from src.domain.entities.delivery_order import DeliveryOrder
from src.domain.value_objects.coordinate import Coordinate
from src.domain.value_objects.route_segment import RouteSegment


class RouteStatus(str, Enum):
    """경로 상태"""
    PLANNED = "PLANNED"                 # 계획됨
    IN_PROGRESS = "IN_PROGRESS"         # 진행 중
    COMPLETED = "COMPLETED"             # 완료됨
    CANCELLED = "CANCELLED"             # 취소됨
    OPTIMIZING = "OPTIMIZING"           # 최적화 중


@dataclass
class Route:
    """배차 경로 엔티티"""
    
    route_id: str
    vehicle_id: str
    delivery_orders: List[DeliveryOrder] = field(default_factory=list)
    segments: List[RouteSegment] = field(default_factory=list)
    status: RouteStatus = RouteStatus.PLANNED
    created_at: datetime = field(default_factory=datetime.utcnow)
    planned_start_time: Optional[datetime] = None
    planned_end_time: Optional[datetime] = None
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    total_distance_km: float = 0.0
    estimated_duration_hours: float = 0.0
    estimated_cost: float = 0.0
    optimization_score: float = 0.0
    notes: str = ""
    
    def __post_init__(self) -> None:
        """경로 데이터 유효성 검증"""
        if not self.route_id:
            raise ValueError("Route ID cannot be empty")
        
        if not self.vehicle_id:
            raise ValueError("Vehicle ID cannot be empty")
    
    @property
    def order_count(self) -> int:
        """배송 주문 개수"""
        return len(self.delivery_orders)
    
    @property
    def total_weight_tons(self) -> float:
        """총 적재 중량"""
        return sum(order.weight_tons for order in self.delivery_orders)
    
    @property
    def is_empty(self) -> bool:
        """빈 경로인지 확인"""
        return len(self.delivery_orders) == 0
    
    @property
    def is_in_progress(self) -> bool:
        """진행 중인지 확인"""
        return self.status == RouteStatus.IN_PROGRESS
    
    @property
    def is_completed(self) -> bool:
        """완료되었는지 확인"""
        return self.status == RouteStatus.COMPLETED
    
    @property
    def high_priority_orders(self) -> List[DeliveryOrder]:
        """높은 우선순위 주문들"""
        return [order for order in self.delivery_orders if order.is_high_priority]
    
    @property
    def urgent_orders(self) -> List[DeliveryOrder]:
        """긴급 주문들"""
        return [order for order in self.delivery_orders if order.is_urgent]
    
    @property
    def estimated_fuel_consumption_liters(self) -> float:
        """예상 연료 소모량 (리터)"""
        # 평균 연비를 10km/L로 가정
        return self.total_distance_km / 10.0
    
    def add_delivery_order(self, order: DeliveryOrder) -> None:
        """
        배송 주문 추가
        
        Args:
            order: 추가할 배송 주문
        """
        if order.order_id in [o.order_id for o in self.delivery_orders]:
            raise ValueError(f"Order {order.order_id} already exists in route")
        
        if order.is_assigned and order.assigned_vehicle_id != self.vehicle_id:
            raise ValueError(f"Order {order.order_id} is assigned to different vehicle")
        
        self.delivery_orders.append(order)
        order.assign_to_vehicle(self.vehicle_id)
        self._recalculate_metrics()
    
    def remove_delivery_order(self, order_id: str) -> bool:
        """
        배송 주문 제거
        
        Args:
            order_id: 제거할 주문 ID
            
        Returns:
            제거 성공 시 True
        """
        for i, order in enumerate(self.delivery_orders):
            if order.order_id == order_id:
                removed_order = self.delivery_orders.pop(i)
                removed_order.assigned_vehicle_id = None
                self._recalculate_metrics()
                return True
        return False
    
    def get_delivery_order(self, order_id: str) -> Optional[DeliveryOrder]:
        """
        주문 ID로 배송 주문 조회
        
        Args:
            order_id: 조회할 주문 ID
            
        Returns:
            배송 주문, 없으면 None
        """
        for order in self.delivery_orders:
            if order.order_id == order_id:
                return order
        return None
    
    def optimize_order_sequence(self) -> None:
        """
        배송 주문 순서 최적화 (간단한 nearest neighbor 알고리즘)
        """
        if len(self.delivery_orders) <= 1:
            return
        
        # 우선순위별로 정렬 (긴급 → 높음 → 중간 → 낮음)
        self.delivery_orders.sort(key=lambda x: (-x.priority_score, x.created_at))
        
        self._recalculate_metrics()
    
    def calculate_total_distance(self) -> float:
        """
        총 주행 거리 계산
        
        Returns:
            총 거리 (킬로미터)
        """
        if not self.segments:
            return 0.0
        
        return sum(segment.distance_km for segment in self.segments)
    
    def calculate_estimated_duration(self, average_speed_kmh: float = 50.0) -> float:
        """
        예상 소요 시간 계산
        
        Args:
            average_speed_kmh: 평균 속도 (km/h)
            
        Returns:
            예상 시간 (시간)
        """
        if self.total_distance_km <= 0:
            return 0.0
        
        # 순수 주행 시간 + 배송 작업 시간 (주문당 30분)
        driving_time = self.total_distance_km / average_speed_kmh
        service_time = len(self.delivery_orders) * 0.5  # 30분 = 0.5시간
        
        return driving_time + service_time
    
    def start_route(self, start_time: datetime) -> None:
        """
        경로 시작
        
        Args:
            start_time: 시작 시간
        """
        if self.status != RouteStatus.PLANNED:
            raise ValueError(f"Cannot start route in status: {self.status}")
        
        if self.is_empty:
            raise ValueError("Cannot start empty route")
        
        self.actual_start_time = start_time
        self.status = RouteStatus.IN_PROGRESS
    
    def complete_route(self, end_time: datetime, notes: str = "") -> None:
        """
        경로 완료
        
        Args:
            end_time: 완료 시간
            notes: 완료 메모
        """
        if self.status != RouteStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete route in status: {self.status}")
        
        self.actual_end_time = end_time
        self.status = RouteStatus.COMPLETED
        self.notes = notes
    
    def cancel_route(self, reason: str = "") -> None:
        """
        경로 취소
        
        Args:
            reason: 취소 사유
        """
        if self.status == RouteStatus.COMPLETED:
            raise ValueError("Cannot cancel completed route")
        
        self.status = RouteStatus.CANCELLED
        self.notes = f"Cancelled: {reason}"
        
        # 배정된 주문들의 상태 초기화
        for order in self.delivery_orders:
            order.assigned_vehicle_id = None
    
    def calculate_efficiency_score(self) -> float:
        """
        경로 효율성 점수 계산 (0.0 ~ 1.0)
        
        Returns:
            효율성 점수
        """
        if not self.is_completed or not self.actual_start_time or not self.actual_end_time:
            return 0.0
        
        # 실제 소요 시간 vs 예상 소요 시간
        actual_duration = (self.actual_end_time - self.actual_start_time).total_seconds() / 3600.0
        
        if self.estimated_duration_hours <= 0 or actual_duration <= 0:
            return 0.0
        
        time_efficiency = min(1.0, self.estimated_duration_hours / actual_duration)
        
        # 용량 활용률
        vehicle_capacity = 20.0  # 가정값, 실제로는 Vehicle 엔티티에서 가져와야 함
        capacity_efficiency = min(1.0, self.total_weight_tons / vehicle_capacity)
        
        # 전체 효율성 = (시간 효율성 * 0.6) + (용량 효율성 * 0.4)
        return (time_efficiency * 0.6) + (capacity_efficiency * 0.4)
    
    def _recalculate_metrics(self) -> None:
        """내부 메트릭 재계산"""
        self.total_distance_km = self.calculate_total_distance()
        self.estimated_duration_hours = self.calculate_estimated_duration()
        # 단순 비용 계산 (거리 * 1000원/km)
        self.estimated_cost = self.total_distance_km * 1000.0
    
    def to_dict(self) -> dict:
        """경로 정보를 딕셔너리로 변환"""
        return {
            'route_id': self.route_id,
            'vehicle_id': self.vehicle_id,
            'status': self.status.value,
            'order_count': self.order_count,
            'total_weight_tons': self.total_weight_tons,
            'total_distance_km': self.total_distance_km,
            'estimated_duration_hours': self.estimated_duration_hours,
            'estimated_cost': self.estimated_cost,
            'optimization_score': self.optimization_score,
            'delivery_orders': [order.to_dict() for order in self.delivery_orders],
            'segments': [segment.to_dict() for segment in self.segments],
            'created_at': self.created_at.isoformat(),
            'planned_start_time': self.planned_start_time.isoformat() if self.planned_start_time else None,
            'planned_end_time': self.planned_end_time.isoformat() if self.planned_end_time else None,
            'actual_start_time': self.actual_start_time.isoformat() if self.actual_start_time else None,
            'actual_end_time': self.actual_end_time.isoformat() if self.actual_end_time else None,
            'high_priority_orders': len(self.high_priority_orders),
            'urgent_orders': len(self.urgent_orders),
            'estimated_fuel_consumption_liters': self.estimated_fuel_consumption_liters,
            'notes': self.notes
        }
    
    def __str__(self) -> str:
        return (f"Route({self.route_id}, vehicle={self.vehicle_id}, "
                f"orders={self.order_count}, status={self.status.value})") 