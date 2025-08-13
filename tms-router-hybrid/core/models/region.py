"""
권역 도메인 모델
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .coordinates import Coordinates


class RegionDifficulty(Enum):
    """권역 난이도"""
    EASY = 1      # 쉬움 - 도심, 접근성 좋음
    NORMAL = 2    # 보통 - 일반 주거지역  
    HARD = 3      # 어려움 - 언덕, 골목길
    VERY_HARD = 4 # 매우 어려움 - 산간, 접근 제한


@dataclass
class Region:
    """권역 엔티티"""
    id: str
    name: str
    center_id: str
    center_coordinates: Coordinates
    difficulty_score: float = 2.5  # 1.0-5.0 스케일
    road_access_score: int = 2     # 1-4 (1:매우어려움, 4:매우쉬움)
    parking_score: int = 2         # 1-4 (1:매우어려움, 4:매우쉬움)
    average_distance_km: Optional[float] = None
    max_delivery_distance_km: Optional[float] = 20.0  # 최대 배송 거리
    weather_severity: Optional[float] = None  # 실시간 날씨 심각도
    traffic_congestion: Optional[float] = None  # 실시간 교통 정체도
    
    def get_difficulty_level(self) -> RegionDifficulty:
        """난이도 레벨 계산"""
        if self.difficulty_score <= 1.5:
            return RegionDifficulty.EASY
        elif self.difficulty_score <= 2.5:
            return RegionDifficulty.NORMAL
        elif self.difficulty_score <= 3.5:
            return RegionDifficulty.HARD
        else:
            return RegionDifficulty.VERY_HARD
    
    def get_difficulty_multiplier(self) -> float:
        """난이도 계수 반환 (배송 시간에 영향)"""
        multipliers = {
            RegionDifficulty.EASY: 0.9,
            RegionDifficulty.NORMAL: 1.0,
            RegionDifficulty.HARD: 1.2,
            RegionDifficulty.VERY_HARD: 1.5
        }
        return multipliers[self.get_difficulty_level()]
    
    def update_real_time_conditions(self, weather_severity: float = None, 
                                   traffic_congestion: float = None):
        """실시간 조건 업데이트"""
        if weather_severity is not None:
            self.weather_severity = weather_severity
        if traffic_congestion is not None:
            self.traffic_congestion = traffic_congestion
    
    def get_total_adjustment_factor(self) -> float:
        """전체 조정 계수 계산"""
        base_factor = 1.0
        
        # 날씨 영향
        if self.weather_severity:
            if self.weather_severity >= 4.0:  # 폭풍
                base_factor *= 0.3
            elif self.weather_severity >= 3.0:  # 폭우/폭설
                base_factor *= 0.6
            elif self.weather_severity >= 2.0:  # 비/눈
                base_factor *= 0.8
            else:  # 맑음
                base_factor *= 1.1
        
        # 교통 영향
        if self.traffic_congestion:
            if self.traffic_congestion >= 0.8:  # 심각한 정체
                base_factor *= 0.6
            elif self.traffic_congestion >= 0.6:  # 정체
                base_factor *= 0.8
            elif self.traffic_congestion <= 0.2:  # 원활
                base_factor *= 1.1
        
        # 권역 난이도 영향
        base_factor *= self.get_difficulty_multiplier()
        
        return base_factor
    
    def is_delivery_feasible(self) -> bool:
        """배송 실행 가능 여부"""
        # 극악의 날씨 조건에서는 배송 중단
        if self.weather_severity and self.weather_severity >= 4.5:
            return False
        return True
    
    def __str__(self) -> str:
        difficulty = self.get_difficulty_level().name
        return f"Region({self.id}, {self.name}, {difficulty})"