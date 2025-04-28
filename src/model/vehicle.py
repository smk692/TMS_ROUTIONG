from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class VehicleCapacity:
    """차량 적재 용량 정보"""
    volume: float  # 적재 부피(m³)
    weight: float  # 적재 무게(kg)

@dataclass
class Vehicle:
    """차량 정보를 담는 클래스"""
    id: str
    type: str  # 차량 유형 (예: 'TRUCK_1TON', 'VAN_1TON' 등)
    capacity: VehicleCapacity
    features: List[str]  # 특수 기능 (예: ['REFRIGERATED', 'LIFT'])
    cost_per_km: float
    start_time: datetime
    end_time: datetime
    current_location: Optional[tuple] = None  # (latitude, longitude)
    current_load: Optional[VehicleCapacity] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'Vehicle':
        """딕셔너리에서 Vehicle 객체 생성"""
        return cls(
            id=data['id'],
            type=data['type'],
            capacity=VehicleCapacity(
                volume=data['capacity']['volume'],
                weight=data['capacity']['weight']
            ),
            features=data.get('features', []),
            cost_per_km=data['cost_per_km'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            current_location=tuple(data['current_location']) if 'current_location' in data else None,
            current_load=VehicleCapacity(
                volume=data['current_load']['volume'],
                weight=data['current_load']['weight']
            ) if 'current_load' in data else None
        )

    def to_dict(self) -> dict:
        """Vehicle 객체를 딕셔너리로 변환"""
        return {
            'id': self.id,
            'type': self.type,
            'capacity': {
                'volume': self.capacity.volume,
                'weight': self.capacity.weight
            },
            'features': self.features,
            'cost_per_km': self.cost_per_km,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'current_location': self.current_location,
            'current_load': {
                'volume': self.current_load.volume,
                'weight': self.current_load.weight
            } if self.current_load else None
        }

    def can_handle_delivery(self, delivery_point: 'DeliveryPoint') -> bool:
        """특정 배송지점을 처리할 수 있는지 확인"""
        if self.current_load is None:
            return True
        
        return (
            self.current_load.volume + delivery_point.volume <= self.capacity.volume and
            self.current_load.weight + delivery_point.weight <= self.capacity.weight
        )

    def has_required_features(self, requirements: List[str]) -> bool:
        """필요한 특수 기능을 가지고 있는지 확인"""
        return all(req in self.features for req in requirements)

    def is_available_at(self, time: datetime) -> bool:
        """주어진 시간에 운행 가능한지 확인"""
        return self.start_time <= time <= self.end_time