"""
지도 표시용 배차 결과 모델
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from .coordinates import Coordinates
from .order import Order


@dataclass
class VehicleAssignmentResult:
    """차량별 배정 결과 (지도 표시용)"""
    vehicle_id: str
    driver_name: str
    vehicle_type: str
    region_name: str
    assigned_orders: List[Order] = field(default_factory=list)
    route_coordinates: List[List[float]] = field(default_factory=list)  # [[lat, lon], ...]
    estimated_distance_km: float = 0.0
    estimated_time_minutes: int = 0
    capacity_utilization: float = 0.0
    color: str = "blue"  # 지도에서 표시할 색상
    
    def get_order_count(self) -> int:
        """배정된 주문 수"""
        return len(self.assigned_orders)


@dataclass
class MapDisplayResult:
    """지도 표시용 배차 결과"""
    center: Optional[Coordinates] = None
    vehicle_assignments: List[VehicleAssignmentResult] = field(default_factory=list)
    unassigned_orders: List[Order] = field(default_factory=list)
    total_orders: int = 0
    assigned_orders: int = 0
    total_vehicles: int = 0
    used_vehicles: int = 0
    total_distance: float = 0.0
    total_time: int = 0
    algorithm_used: str = ""
    execution_time: float = 0.0
    batch_id: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """초기화 후 계산"""
        if not self.total_orders:
            self.total_orders = self.assigned_orders + len(self.unassigned_orders)
        if not self.assigned_orders:
            self.assigned_orders = sum(len(va.assigned_orders) for va in self.vehicle_assignments)
        if not self.used_vehicles:
            self.used_vehicles = len(self.vehicle_assignments)
        if not self.total_distance:
            self.total_distance = sum(va.estimated_distance_km for va in self.vehicle_assignments)
        if not self.total_time:
            self.total_time = sum(va.estimated_time_minutes for va in self.vehicle_assignments)
