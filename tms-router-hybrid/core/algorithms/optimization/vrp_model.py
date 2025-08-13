"""
VRP 모델 정의 및 데이터 구조
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ...models import Order, Vehicle, Coordinates


class VehicleType(Enum):
    """차량 타입"""
    MOTORCYCLE = "MOTORCYCLE"
    CAR = "CAR"
    VAN = "VAN"
    TRUCK = "TRUCK"


class VRPConstraintType(Enum):
    """VRP 제약조건 타입"""
    CAPACITY = "capacity"
    DISTANCE = "distance"
    TIME_WINDOW = "time_window"
    SKILL = "skill"
    BREAK = "break"


@dataclass
class VRPLocation:
    """VRP 위치 정보"""
    id: str
    coordinates: Coordinates
    demand: int = 0
    service_time: int = 0  # 분
    time_window_start: int = 0  # 분
    time_window_end: int = 1440  # 분 (24시간)
    priority: int = 1
    
    def __post_init__(self):
        """유효성 검증"""
        if self.demand < 0:
            raise ValueError("수요량은 0 이상이어야 합니다")
        if self.time_window_start >= self.time_window_end:
            raise ValueError("시간 창 시작이 끝보다 늦을 수 없습니다")


@dataclass
class VRPVehicle:
    """VRP 차량 정보"""
    id: str
    vehicle_type: VehicleType
    capacity: int
    start_location: VRPLocation
    end_location: Optional[VRPLocation] = None
    max_distance: float = 120.0  # km
    max_time: int = 480  # 분 (8시간)
    cost_per_km: float = 1.0
    fixed_cost: float = 5000.0
    skills: List[str] = None
    
    def __post_init__(self):
        """기본값 설정 및 유효성 검증"""
        if self.skills is None:
            self.skills = []
        if self.end_location is None:
            self.end_location = self.start_location
        if self.capacity <= 0:
            raise ValueError("차량 용량은 0보다 커야 합니다")
        if self.max_distance <= 0:
            raise ValueError("최대 거리는 0보다 커야 합니다")


@dataclass
class VRPConstraint:
    """VRP 제약조건"""
    constraint_type: VRPConstraintType
    parameters: Dict[str, Any]
    penalty_weight: float = 1.0
    is_hard_constraint: bool = True


class VRPModel:
    """Vehicle Routing Problem 모델"""
    
    def __init__(self, name: str = "VRP_Model"):
        self.name = name
        self.locations: List[VRPLocation] = []
        self.vehicles: List[VRPVehicle] = []
        self.constraints: List[VRPConstraint] = []
        self.distance_matrix: Optional[np.ndarray] = None
        self.time_matrix: Optional[np.ndarray] = None
        
        # 목적함수 가중치
        self.objective_weights = {
            'distance': 1.0,
            'time': 0.5,
            'vehicle_count': 5000.0,
            'unassigned_penalty': 100000.0
        }
        
        # 위치 인덱스 매핑
        self._location_index_map: Dict[str, int] = {}
        self._vehicle_index_map: Dict[str, int] = {}
    
    def add_location(self, location: VRPLocation) -> None:
        """위치 추가"""
        if location.id in self._location_index_map:
            raise ValueError(f"위치 ID '{location.id}'가 이미 존재합니다")
        
        index = len(self.locations)
        self.locations.append(location)
        self._location_index_map[location.id] = index
    
    def add_vehicle(self, vehicle: VRPVehicle) -> None:
        """차량 추가"""
        if vehicle.id in self._vehicle_index_map:
            raise ValueError(f"차량 ID '{vehicle.id}'가 이미 존재합니다")
        
        index = len(self.vehicles)
        self.vehicles.append(vehicle)
        self._vehicle_index_map[vehicle.id] = index
    
    def add_constraint(self, constraint: VRPConstraint) -> None:
        """제약조건 추가"""
        self.constraints.append(constraint)
    
    def set_distance_matrix(self, matrix: np.ndarray) -> None:
        """거리 행렬 설정"""
        expected_size = len(self.locations)
        if matrix.shape != (expected_size, expected_size):
            raise ValueError(f"거리 행렬 크기가 맞지 않습니다. 예상: ({expected_size}, {expected_size}), 실제: {matrix.shape}")
        
        self.distance_matrix = matrix
    
    def set_time_matrix(self, matrix: np.ndarray) -> None:
        """시간 행렬 설정"""
        expected_size = len(self.locations)
        if matrix.shape != (expected_size, expected_size):
            raise ValueError(f"시간 행렬 크기가 맞지 않습니다. 예상: ({expected_size}, {expected_size}), 실제: {matrix.shape}")
        
        self.time_matrix = matrix
    
    def get_location_index(self, location_id: str) -> int:
        """위치 ID로 인덱스 조회"""
        if location_id not in self._location_index_map:
            raise KeyError(f"위치 ID '{location_id}'를 찾을 수 없습니다")
        return self._location_index_map[location_id]
    
    def get_vehicle_index(self, vehicle_id: str) -> int:
        """차량 ID로 인덱스 조회"""
        if vehicle_id not in self._vehicle_index_map:
            raise KeyError(f"차량 ID '{vehicle_id}'를 찾을 수 없습니다")
        return self._vehicle_index_map[vehicle_id]
    
    def get_location_by_index(self, index: int) -> VRPLocation:
        """인덱스로 위치 조회"""
        if index < 0 or index >= len(self.locations):
            raise IndexError(f"위치 인덱스 {index}가 범위를 벗어났습니다")
        return self.locations[index]
    
    def get_vehicle_by_index(self, index: int) -> VRPVehicle:
        """인덱스로 차량 조회"""
        if index < 0 or index >= len(self.vehicles):
            raise IndexError(f"차량 인덱스 {index}가 범위를 벗어났습니다")
        return self.vehicles[index]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """모델 유효성 검증"""
        errors = []
        
        # 기본 데이터 검증
        if not self.locations:
            errors.append("위치가 하나도 없습니다")
        
        if not self.vehicles:
            errors.append("차량이 하나도 없습니다")
        
        # 거리 행렬 검증
        if self.distance_matrix is None:
            errors.append("거리 행렬이 설정되지 않았습니다")
        elif self.distance_matrix.shape[0] != len(self.locations):
            errors.append("거리 행렬 크기가 위치 수와 일치하지 않습니다")
        
        # 시간 행렬 검증
        if self.time_matrix is None:
            errors.append("시간 행렬이 설정되지 않았습니다")
        elif self.time_matrix.shape[0] != len(self.locations):
            errors.append("시간 행렬 크기가 위치 수와 일치하지 않습니다")
        
        # 차량 출발/도착지 검증
        for vehicle in self.vehicles:
            if vehicle.start_location.id not in self._location_index_map:
                errors.append(f"차량 {vehicle.id}의 출발지가 위치 목록에 없습니다")
            
            if vehicle.end_location and vehicle.end_location.id not in self._location_index_map:
                errors.append(f"차량 {vehicle.id}의 도착지가 위치 목록에 없습니다")
        
        return len(errors) == 0, errors
    
    def get_depot_indices(self) -> List[int]:
        """Depot (차량 출발지) 인덱스 목록 반환"""
        depot_indices = []
        
        for vehicle in self.vehicles:
            start_index = self.get_location_index(vehicle.start_location.id)
            depot_indices.append(start_index)
        
        return depot_indices
    
    def get_customer_indices(self) -> List[int]:
        """고객 (주문) 위치 인덱스 목록 반환"""
        depot_indices = set(self.get_depot_indices())
        customer_indices = []
        
        for i in range(len(self.locations)):
            if i not in depot_indices:
                customer_indices.append(i)
        
        return customer_indices
    
    def calculate_total_demand(self) -> int:
        """총 수요량 계산"""
        return sum(location.demand for location in self.locations)
    
    def calculate_total_capacity(self) -> int:
        """총 차량 용량 계산"""
        return sum(vehicle.capacity for vehicle in self.vehicles)
    
    def get_capacity_utilization(self) -> float:
        """용량 활용률 계산"""
        total_demand = self.calculate_total_demand()
        total_capacity = self.calculate_total_capacity()
        
        if total_capacity == 0:
            return 0.0
        
        return total_demand / total_capacity
    
    def get_model_summary(self) -> Dict[str, Any]:
        """모델 요약 정보"""
        return {
            'name': self.name,
            'locations_count': len(self.locations),
            'vehicles_count': len(self.vehicles),
            'constraints_count': len(self.constraints),
            'total_demand': self.calculate_total_demand(),
            'total_capacity': self.calculate_total_capacity(),
            'capacity_utilization': self.get_capacity_utilization(),
            'has_distance_matrix': self.distance_matrix is not None,
            'has_time_matrix': self.time_matrix is not None
        }
    
    def __str__(self) -> str:
        """문자열 표현"""
        summary = self.get_model_summary()
        return f"VRPModel(name='{summary['name']}', locations={summary['locations_count']}, vehicles={summary['vehicles_count']})"
    
    def __repr__(self) -> str:
        return self.__str__()


def create_vrp_model_from_orders_vehicles(orders: List[Order], 
                                        vehicles: List[Vehicle],
                                        distance_matrix: np.ndarray = None,
                                        time_matrix: np.ndarray = None) -> VRPModel:
    """주문과 차량 리스트로부터 VRP 모델 생성"""
    
    model = VRPModel("TMS_VRP_Model")
    
    # 1. Depot 위치들 추가 (차량 출발지)
    depot_locations = {}
    for vehicle in vehicles:
        depot_id = f"depot_{vehicle.id}"
        if depot_id not in depot_locations:
            depot_location = VRPLocation(
                id=depot_id,
                coordinates=vehicle.center_coordinates,
                demand=0,
                service_time=0
            )
            model.add_location(depot_location)
            depot_locations[depot_id] = depot_location
    
    # 2. 주문 위치들 추가
    for order in orders:
        order_location = VRPLocation(
            id=f"order_{order.id}",
            coordinates=order.coordinates,
            demand=1,  # 주문 하나당 수요량 1
            service_time=8,  # 8분 서비스 시간
            priority=order.priority.value
        )
        model.add_location(order_location)
    
    # 3. 차량들 추가
    for vehicle in vehicles:
        depot_id = f"depot_{vehicle.id}"
        depot_location = depot_locations[depot_id]
        
        # 차량 타입 변환
        vehicle_type_map = {
            'MOTORCYCLE': VehicleType.MOTORCYCLE,
            'CAR': VehicleType.CAR,
            'VAN': VehicleType.VAN,
            'TRUCK': VehicleType.TRUCK
        }
        
        vrp_vehicle = VRPVehicle(
            id=vehicle.id,
            vehicle_type=vehicle_type_map.get(vehicle.vehicle_type.value, VehicleType.CAR),
            capacity=vehicle.safe_capacity,
            start_location=depot_location,
            end_location=depot_location,
            max_distance=120.0,
            max_time=480  # 8시간
        )
        model.add_vehicle(vrp_vehicle)
    
    # 4. 기본 제약조건 추가
    capacity_constraint = VRPConstraint(
        constraint_type=VRPConstraintType.CAPACITY,
        parameters={'enforce_capacity': True},
        is_hard_constraint=True
    )
    model.add_constraint(capacity_constraint)
    
    distance_constraint = VRPConstraint(
        constraint_type=VRPConstraintType.DISTANCE,
        parameters={'max_distance_km': 120.0},
        is_hard_constraint=True
    )
    model.add_constraint(distance_constraint)
    
    # 5. 거리/시간 행렬 설정
    if distance_matrix is not None:
        model.set_distance_matrix(distance_matrix)
    
    if time_matrix is not None:
        model.set_time_matrix(time_matrix)
    
    return model