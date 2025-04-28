from dataclasses import dataclass
from datetime import datetime

@dataclass
class TimeWindow:
    """배송 가능 시간대를 나타내는 클래스"""
    start: datetime
    end: datetime

    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end = end

    def is_valid(self, current_time: datetime) -> bool:
        """주어진 시간이 시간 윈도우 내에 있는지 확인"""
        return self.start <= current_time <= self.end

    def get_duration(self) -> float:
        """시간 윈도우의 길이를 시간 단위로 반환"""
        return (self.end - self.start).total_seconds() / 3600
