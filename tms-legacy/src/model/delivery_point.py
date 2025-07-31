from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Optional

@dataclass
class DeliveryPoint:
    """배송지점 정보를 담는 클래스"""
    id: int
    latitude: float
    longitude: float
    address1: str
    address2: str
    time_window: Tuple[datetime, datetime]  # 배송 가능 시간대
    service_time: int  # 예상 서비스 시간(분)
    special_requirements: List[str]  # 특수 요구사항
    volume: float = 0.0  # 화물 부피(m³) - 기본값 0.0 (부피 정보 없어도 작동)
    weight: float = 0.0  # 화물 무게(kg) - 기본값 0.0
    priority: int = 3  # 우선순위 (1: 최우선 ~ 5: 최저)

    @classmethod
    def from_dict(cls, data: dict) -> 'DeliveryPoint':
        """딕셔너리에서 DeliveryPoint 객체 생성"""
        return cls(
            id=data['id'],
            latitude=data['latitude'],
            longitude=data['longitude'],
            address1=data['address1'],
            address2=data.get('address2', ''),
            time_window=(
                datetime.fromisoformat(data['time_window'][0]),
                datetime.fromisoformat(data['time_window'][1])
            ) if 'time_window' in data else (None, None),
            service_time=data.get('service_time', 5),
            special_requirements=data.get('special_requirements', []),
            volume=data.get('volume', 0.0),  # 부피 정보 없으면 0.0 사용
            weight=data.get('weight', 0.0),  # 무게 정보 없으면 0.0 사용
            priority=data.get('priority', 3)
        )

    def to_dict(self) -> dict:
        """DeliveryPoint 객체를 딕셔너리로 변환"""
        return {
            'id': self.id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'address1': self.address1,
            'address2': self.address2,
            'time_window': (
                self.time_window[0].isoformat(),
                self.time_window[1].isoformat()
            ) if all(self.time_window) else None,
            'service_time': self.service_time,
            'special_requirements': self.special_requirements,
            'volume': self.volume,
            'weight': self.weight,
            'priority': self.priority
        }

    def is_valid_time(self, current_time: datetime) -> bool:
        """주어진 시간이 배송 가능 시간대인지 확인"""
        if not all(self.time_window):
            return True
        return self.time_window[0] <= current_time <= self.time_window[1]

    def get_priority_weight(self) -> float:
        """우선순위에 따른 가중치 반환"""
        return 1.0 + (5 - self.priority) * 0.2  # 우선순위가 높을수록 가중치 증가