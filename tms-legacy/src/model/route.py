from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
from .delivery_point import DeliveryPoint
from .vehicle import Vehicle

@dataclass
class RoutePoint:
    """경로의 각 지점 정보"""
    point: DeliveryPoint
    arrival_time: datetime
    departure_time: datetime
    cumulative_distance: float  # 시작점부터의 누적 거리
    cumulative_load: dict  # {'volume': float, 'weight': float}

@dataclass
class Route:
    """배송 경로 정보를 담는 클래스"""
    id: str
    vehicle: Vehicle
    points: List[RoutePoint]
    total_distance: float
    total_time: int  # 분 단위
    total_load: dict  # {'volume': float, 'weight': float}
    start_time: datetime
    end_time: datetime
    status: str  # 'PLANNED', 'IN_PROGRESS', 'COMPLETED', 'FAILED'
    depot_id: Optional[str] = None  # 물류센터 ID 추가
    depot_name: Optional[str] = None  # 물류센터 이름 추가

    @property
    def is_valid(self) -> bool:
        """경로의 유효성 검사"""
        if not self.points:
            return False

        # 시간 제약 검사
        if self.start_time < self.vehicle.start_time or self.end_time > self.vehicle.end_time:
            return False

        # 적재량 제약 검사 - 품목 정보가 있는 경우에만 확인
        volume_exceeded = False
        weight_exceeded = False
        
        if any(p.point.volume > 0 for p in self.points):
            volume_exceeded = self.total_load['volume'] > self.vehicle.capacity.volume
            
        if any(p.point.weight > 0 for p in self.points):
            weight_exceeded = self.total_load['weight'] > self.vehicle.capacity.weight
            
        if volume_exceeded or weight_exceeded:
            return False

        # 연속성 검사
        for i in range(len(self.points) - 1):
            if self.points[i].departure_time > self.points[i + 1].arrival_time:
                return False

        return True

    def add_point(self, point: DeliveryPoint, position: int = -1) -> bool:
        """경로에 새로운 배송지점 추가"""
        if not self.can_add_point(point):
            return False

        new_point = self._create_route_point(point, position)
        if position == -1:
            self.points.append(new_point)
        else:
            self.points.insert(position, new_point)

        self._update_route_metrics()
        return True

    def can_add_point(self, point: DeliveryPoint) -> bool:
        """새로운 배송지점을 추가할 수 있는지 확인"""
        # 차량 용량 검사 - 품목 정보가 있는 경우에만 확인
        volume_check = True
        weight_check = True
        
        if point.volume > 0:
            new_volume = self.total_load['volume'] + point.volume
            volume_check = new_volume <= self.vehicle.capacity.volume
            
        if point.weight > 0:
            new_weight = self.total_load['weight'] + point.weight
            weight_check = new_weight <= self.vehicle.capacity.weight
        
        if not (volume_check and weight_check):
            return False

        # 특수 요구사항 검사
        if not self.vehicle.has_required_features(point.special_requirements):
            return False

        return True

    def _create_route_point(self, point: DeliveryPoint, position: int) -> RoutePoint:
        """새로운 RoutePoint 객체 생성"""
        # 이전/다음 지점과의 시간 계산 로직 필요
        # 임시로 더미 데이터 반환
        return RoutePoint(
            point=point,
            arrival_time=datetime.now(),  # 실제 계산 필요
            departure_time=datetime.now(),  # 실제 계산 필요
            cumulative_distance=0.0,  # 실제 계산 필요
            cumulative_load={
                'volume': self.total_load['volume'] + point.volume,
                'weight': self.total_load['weight'] + point.weight
            }
        )

    def _update_route_metrics(self):
        """경로 메트릭 업데이트"""
        # 거리, 시간, 적재량 등 재계산
        self.total_distance = sum(p.cumulative_distance for p in self.points)
        self.total_load = self.points[-1].cumulative_load if self.points else {'volume': 0, 'weight': 0}
        self.start_time = self.points[0].arrival_time if self.points else datetime.now()
        self.end_time = self.points[-1].departure_time if self.points else datetime.now()

    def get_route_summary(self) -> dict:
        """경로 요약 정보 반환"""
        return {
            'route_id': self.id,
            'vehicle_id': self.vehicle.id,
            'depot_id': self.depot_id,
            'depot_name': self.depot_name,
            'num_points': len(self.points),
            'total_distance': self.total_distance,
            'total_time': self.total_time,
            'total_load': self.total_load,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'status': self.status
        }

    def to_dict(self) -> dict:
        """Route 객체를 딕셔너리로 변환"""
        return {
            'id': self.id,
            'vehicle': self.vehicle.to_dict(),
            'depot_id': self.depot_id,
            'depot_name': self.depot_name,
            'points': [
                {
                    'point': p.point.to_dict(),
                    'arrival_time': p.arrival_time.isoformat(),
                    'departure_time': p.departure_time.isoformat(),
                    'cumulative_distance': p.cumulative_distance,
                    'cumulative_load': p.cumulative_load
                }
                for p in self.points
            ],
            'total_distance': self.total_distance,
            'total_time': self.total_time,
            'total_load': self.total_load,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'status': self.status
        }

    def is_capacity_exceeded(self) -> bool:
        """용량 초과 여부 확인"""
        # 품목 부피 정보가 있는 경우에만 부피 제약 확인
        volume_exceeded = False
        if any(p.point.volume > 0 for p in self.points):
            volume_exceeded = self.total_load['volume'] > self.vehicle.capacity.volume
        
        # 품목 무게 정보가 있는 경우에만 무게 제약 확인
        weight_exceeded = False
        if any(p.point.weight > 0 for p in self.points):
            weight_exceeded = self.total_load['weight'] > self.vehicle.capacity.weight
        
        return volume_exceeded or weight_exceeded