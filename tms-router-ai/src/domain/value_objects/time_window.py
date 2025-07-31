"""
TimeWindow Value Object - 시간 창

배송 시간 제약을 나타내는 불변 값 객체입니다.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass(frozen=True)
class TimeWindow:
    """배송 시간 창"""
    
    start_time: datetime
    end_time: datetime
    
    def __post_init__(self) -> None:
        """시간 창 유효성 검증"""
        if self.start_time >= self.end_time:
            raise ValueError(f"Invalid time window: start_time ({self.start_time}) "
                           f"must be before end_time ({self.end_time})")
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TimeWindow':
        """딕셔너리에서 TimeWindow 객체 생성"""
        # ISO 형식 문자열을 datetime으로 변환
        start_time = datetime.fromisoformat(data['start']) if isinstance(data['start'], str) else data['start']
        end_time = datetime.fromisoformat(data['end']) if isinstance(data['end'], str) else data['end']
        
        return cls(
            start_time=start_time,
            end_time=end_time
        )
    
    @property
    def duration(self) -> timedelta:
        """시간 창의 지속 시간"""
        return self.end_time - self.start_time
    
    @property
    def duration_hours(self) -> float:
        """시간 창의 지속 시간 (시간 단위)"""
        return self.duration.total_seconds() / 3600.0
    
    def contains(self, target_time: datetime) -> bool:
        """
        주어진 시간이 시간 창 내에 있는지 확인
        
        Args:
            target_time: 확인할 시간
            
        Returns:
            시간 창 내에 있으면 True
        """
        return self.start_time <= target_time <= self.end_time
    
    def overlaps_with(self, other: 'TimeWindow') -> bool:
        """
        다른 시간 창과 겹치는지 확인
        
        Args:
            other: 비교할 시간 창
            
        Returns:
            겹치면 True
        """
        return (self.start_time < other.end_time and 
                self.end_time > other.start_time)
    
    def intersection_with(self, other: 'TimeWindow') -> Optional['TimeWindow']:
        """
        다른 시간 창과의 교집합을 계산
        
        Args:
            other: 비교할 시간 창
            
        Returns:
            교집합 시간 창, 겹치지 않으면 None
        """
        if not self.overlaps_with(other):
            return None
        
        intersection_start = max(self.start_time, other.start_time)
        intersection_end = min(self.end_time, other.end_time)
        
        return TimeWindow(intersection_start, intersection_end)
    
    def is_flexible(self, flexibility_hours: float = 1.0) -> bool:
        """
        시간 창이 유연한지 확인 (충분한 여유 시간이 있는지)
        
        Args:
            flexibility_hours: 최소 유연성 시간 (시간 단위)
            
        Returns:
            유연하면 True
        """
        return self.duration_hours >= flexibility_hours
    
    def to_dict(self) -> dict:
        """시간 창을 딕셔너리로 변환"""
        return {
            'start': self.start_time.isoformat(),
            'end': self.end_time.isoformat(),
            'duration_hours': self.duration_hours
        }
    
    def __str__(self) -> str:
        return f"TimeWindow({self.start_time} ~ {self.end_time})" 