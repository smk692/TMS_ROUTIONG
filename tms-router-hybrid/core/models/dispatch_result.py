"""
배차 결과 도메인 모델
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional


class DispatchStatus(Enum):
    """배차 상태"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VehicleAssignment:
    """차량별 배정 정보"""
    vehicle_id: str
    driver_name: str
    vehicle_type: str
    region_name: str
    assigned_orders: List[str]
    estimated_distance_km: float = 0.0
    estimated_time_minutes: int = 0
    capacity_utilization: float = 0.0  # 용량 활용률
    
    def get_order_count(self) -> int:
        """배정된 주문 수"""
        return len(self.assigned_orders)
    
    def __str__(self) -> str:
        return f"{self.driver_name}({self.vehicle_type}): {len(self.assigned_orders)}개 주문"


@dataclass 
class DispatchMetrics:
    """배차 품질 지표"""
    total_orders: int
    assigned_orders: int
    unassigned_orders: int
    total_vehicles: int
    used_vehicles: int
    unused_vehicles: int
    average_capacity_utilization: float = 0.0
    total_estimated_distance: float = 0.0
    total_estimated_time: int = 0
    algorithm_used: str = ""
    execution_time_seconds: float = 0.0
    quality_score: float = 0.0  # 0.0-1.0
    
    def get_assignment_rate(self) -> float:
        """배정률 계산"""
        if self.total_orders == 0:
            return 0.0
        return self.assigned_orders / self.total_orders
    
    def get_vehicle_utilization_rate(self) -> float:
        """차량 활용률 계산"""
        if self.total_vehicles == 0:
            return 0.0
        return self.used_vehicles / self.total_vehicles


@dataclass
class DispatchResult:
    """배차 결과"""
    batch_id: str
    timestamp: datetime
    status: DispatchStatus
    vehicle_assignments: List[VehicleAssignment] = field(default_factory=list)
    unassigned_orders: List[str] = field(default_factory=list)
    excluded_vehicles: List[str] = field(default_factory=list)  # 수동 배차 대상
    metrics: Optional[DispatchMetrics] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    external_conditions: Dict[str, any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0  # 실행 시간 (초)
    
    def __post_init__(self):
        """초기화 후 메트릭스 계산"""
        if self.metrics is None:
            self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> DispatchMetrics:
        """메트릭스 자동 계산"""
        total_assigned = sum(len(assignment.assigned_orders) 
                           for assignment in self.vehicle_assignments)
        total_distance = sum(assignment.estimated_distance_km 
                           for assignment in self.vehicle_assignments)
        total_time = sum(assignment.estimated_time_minutes 
                        for assignment in self.vehicle_assignments)
        
        avg_utilization = 0.0
        if self.vehicle_assignments:
            avg_utilization = sum(assignment.capacity_utilization 
                                for assignment in self.vehicle_assignments) / len(self.vehicle_assignments)
        
        return DispatchMetrics(
            total_orders=total_assigned + len(self.unassigned_orders),
            assigned_orders=total_assigned,
            unassigned_orders=len(self.unassigned_orders),
            total_vehicles=len(self.vehicle_assignments) + len(self.excluded_vehicles),
            used_vehicles=len(self.vehicle_assignments),
            unused_vehicles=len(self.excluded_vehicles),
            average_capacity_utilization=avg_utilization,
            total_estimated_distance=total_distance,
            total_estimated_time=total_time
        )
    
    def add_vehicle_assignment(self, assignment: VehicleAssignment):
        """차량 배정 추가"""
        self.vehicle_assignments.append(assignment)
        self.metrics = self._calculate_metrics()
    
    def add_warning(self, message: str):
        """경고 메시지 추가"""
        self.warnings.append(message)
    
    def is_successful(self) -> bool:
        """성공 여부"""
        return self.status in [DispatchStatus.SUCCESS, DispatchStatus.PARTIAL_SUCCESS]
    
    def get_summary_text(self) -> str:
        """요약 텍스트 생성"""
        if not self.is_successful():
            return f"배차 실패: {self.error_message}"
        
        metrics = self.metrics
        return (f"배차 완료: {metrics.assigned_orders}/{metrics.total_orders}개 주문, "
                f"{metrics.used_vehicles}/{metrics.total_vehicles}대 차량 사용, "
                f"총 거리 {metrics.total_estimated_distance:.1f}km")
    
    def __str__(self) -> str:
        return f"DispatchResult({self.batch_id}, {self.status.value}, {len(self.vehicle_assignments)} assignments)"