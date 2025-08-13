"""
웹 인터페이스용 데이터 모델
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime


@dataclass
class WebOrder:
    """웹 표시용 주문 정보"""
    order_id: str
    center_id: str
    region_id: str
    address: str
    latitude: float
    longitude: float
    priority: str
    status: str
    created_at: datetime
    
    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'center_id': self.center_id,
            'region_id': self.region_id,
            'address': self.address,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'priority': self.priority,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class WebVehicleAssignment:
    """웹 표시용 차량 배정 정보"""
    vehicle_id: str
    driver_name: str
    vehicle_type: str
    region_name: str
    assigned_orders: List[WebOrder]
    route_coordinates: List[Tuple[float, float]] = field(default_factory=list)
    estimated_distance_km: float = 0.0
    estimated_time_minutes: int = 0
    capacity_utilization: float = 0.0
    color: str = 'blue'  # 지도 표시 색상
    
    def to_dict(self) -> dict:
        return {
            'vehicle_id': self.vehicle_id,
            'driver_name': self.driver_name,
            'vehicle_type': self.vehicle_type,
            'region_name': self.region_name,
            'order_count': len(self.assigned_orders),
            'assigned_orders': [order.to_dict() for order in self.assigned_orders],
            'route_coordinates': self.route_coordinates,
            'estimated_distance_km': self.estimated_distance_km,
            'estimated_time_minutes': self.estimated_time_minutes,
            'capacity_utilization': self.capacity_utilization,
            'color': self.color
        }


@dataclass
class WebCenter:
    """웹 표시용 물류센터 정보"""
    center_id: str
    name: str
    address: str
    latitude: float
    longitude: float
    is_active: bool
    
    def to_dict(self) -> dict:
        return {
            'center_id': self.center_id,
            'name': self.name,
            'address': self.address,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'is_active': self.is_active
        }


@dataclass
class WebDispatchResult:
    """웹 표시용 배차 결과"""
    batch_id: str
    timestamp: datetime
    status: str
    center: WebCenter
    vehicle_assignments: List[WebVehicleAssignment] = field(default_factory=list)
    unassigned_orders: List[WebOrder] = field(default_factory=list)
    total_orders: int = 0
    assigned_orders: int = 0
    total_vehicles: int = 0
    used_vehicles: int = 0
    total_distance: float = 0.0
    total_time: int = 0
    execution_time: float = 0.0
    algorithm_used: str = ""
    quality_score: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'batch_id': self.batch_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'status': self.status,
            'center': self.center.to_dict() if self.center else None,
            'vehicle_assignments': [va.to_dict() for va in self.vehicle_assignments],
            'unassigned_orders': [order.to_dict() for order in self.unassigned_orders],
            'statistics': {
                'total_orders': self.total_orders,
                'assigned_orders': self.assigned_orders,
                'unassigned_orders': len(self.unassigned_orders),
                'total_vehicles': self.total_vehicles,
                'used_vehicles': self.used_vehicles,
                'unused_vehicles': self.total_vehicles - self.used_vehicles,
                'total_distance': self.total_distance,
                'total_time': self.total_time,
                'execution_time': self.execution_time,
                'algorithm_used': self.algorithm_used,
                'quality_score': self.quality_score,
                'assignment_rate': (self.assigned_orders / self.total_orders * 100) if self.total_orders > 0 else 0,
                'vehicle_utilization_rate': (self.used_vehicles / self.total_vehicles * 100) if self.total_vehicles > 0 else 0
            },
            'error_message': self.error_message,
            'warnings': self.warnings
        }
    
    def is_successful(self) -> bool:
        return self.status in ['success', 'partial_success']