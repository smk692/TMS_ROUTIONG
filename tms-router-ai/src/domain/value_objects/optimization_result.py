"""
OptimizationResult Value Object - 최적화 결과

AI 기반 배차 최적화 결과를 나타내는 불변 값 객체입니다.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.domain.entities.route import Route
from src.shared.constants import ScenarioType


@dataclass(frozen=True)
class OptimizationResult:
    """배차 최적화 결과"""
    
    request_id: str
    scenario_type: ScenarioType
    routes: List[Route]
    confidence_score: float
    ai_reasoning: str
    optimization_metrics: Dict[str, float] = field(default_factory=dict)
    total_distance_km: float = 0.0
    total_estimated_cost: float = 0.0
    total_estimated_duration_hours: float = 0.0
    processing_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    ai_model_version: str = "gpt-4"
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """최적화 결과 유효성 검증"""
        if not self.request_id:
            raise ValueError("Request ID cannot be empty")
        
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"Confidence score must be between 0 and 1: {self.confidence_score}")
        
        if self.total_distance_km < 0:
            raise ValueError(f"Total distance cannot be negative: {self.total_distance_km}")
        
        if self.total_estimated_cost < 0:
            raise ValueError(f"Total cost cannot be negative: {self.total_estimated_cost}")
    
    @property
    def route_count(self) -> int:
        """경로 개수"""
        return len(self.routes)
    
    @property
    def total_orders(self) -> int:
        """총 주문 개수"""
        return sum(route.order_count for route in self.routes)
    
    @property
    def total_weight_tons(self) -> float:
        """총 적재 중량"""
        return sum(route.total_weight_tons for route in self.routes)
    
    @property
    def average_confidence_score(self) -> float:
        """평균 신뢰도 점수"""
        if not self.routes:
            return self.confidence_score
        
        route_scores = [route.optimization_score for route in self.routes if route.optimization_score > 0]
        if not route_scores:
            return self.confidence_score
        
        return (self.confidence_score + sum(route_scores) / len(route_scores)) / 2.0
    
    @property
    def is_high_confidence(self) -> bool:
        """높은 신뢰도인지 확인"""
        return self.confidence_score >= 0.8
    
    @property
    def is_low_confidence(self) -> bool:
        """낮은 신뢰도인지 확인"""
        return self.confidence_score < 0.5
    
    @property
    def has_warnings(self) -> bool:
        """경고가 있는지 확인"""
        return len(self.warnings) > 0
    
    @property
    def total_tokens_used(self) -> int:
        """총 사용된 토큰 수"""
        return self.prompt_tokens_used + self.completion_tokens_used
    
    @property
    def estimated_fuel_consumption_liters(self) -> float:
        """예상 총 연료 소모량"""
        return sum(route.estimated_fuel_consumption_liters for route in self.routes)
    
    def get_route_by_vehicle(self, vehicle_id: str) -> Optional[Route]:
        """
        차량 ID로 경로 조회
        
        Args:
            vehicle_id: 차량 ID
            
        Returns:
            해당 차량의 경로, 없으면 None
        """
        for route in self.routes:
            if route.vehicle_id == vehicle_id:
                return route
        return None
    
    def get_routes_by_priority(self, urgent_only: bool = False) -> List[Route]:
        """
        우선순위별 경로 조회
        
        Args:
            urgent_only: 긴급 주문만 포함하는 경로만 반환할지 여부
            
        Returns:
            우선순위별 정렬된 경로 리스트
        """
        if urgent_only:
            return [route for route in self.routes if route.urgent_orders]
        
        # 높은 우선순위 주문 수가 많은 순서로 정렬
        return sorted(self.routes, key=lambda r: len(r.high_priority_orders), reverse=True)
    
    def calculate_efficiency_metrics(self) -> Dict[str, float]:
        """
        효율성 지표 계산
        
        Returns:
            효율성 지표 딕셔너리
        """
        if not self.routes:
            return {}
        
        # 평균 경로 거리
        avg_distance = self.total_distance_km / len(self.routes) if self.routes else 0.0
        
        # 평균 차량 활용률 (가정: 차량 용량 20톤)
        avg_utilization = (self.total_weight_tons / (len(self.routes) * 20.0)) * 100.0
        
        # 비용 효율성 (톤당 비용)
        cost_per_ton = self.total_estimated_cost / self.total_weight_tons if self.total_weight_tons > 0 else 0.0
        
        return {
            'average_route_distance_km': avg_distance,
            'average_vehicle_utilization_percent': min(100.0, avg_utilization),
            'cost_per_ton': cost_per_ton,
            'cost_per_km': self.total_estimated_cost / self.total_distance_km if self.total_distance_km > 0 else 0.0,
            'orders_per_route': self.total_orders / len(self.routes) if self.routes else 0.0,
            'estimated_fuel_efficiency': self.total_distance_km / self.estimated_fuel_consumption_liters if self.estimated_fuel_consumption_liters > 0 else 0.0
        }
    
    def add_warning(self, warning_message: str) -> None:
        """
        경고 메시지 추가 (불변 객체이므로 새 인스턴스 생성이 필요하지만, 
        실제로는 생성 시점에서만 경고를 추가한다고 가정)
        
        Args:
            warning_message: 경고 메시지
        """
        # 불변 객체이므로 직접 수정 불가
        # 실제 구현에서는 builder 패턴이나 factory 메서드 사용
        pass
    
    def to_dict(self) -> dict:
        """최적화 결과를 딕셔너리로 변환"""
        efficiency_metrics = self.calculate_efficiency_metrics()
        
        return {
            'request_id': self.request_id,
            'scenario_type': self.scenario_type.value,
            'confidence_score': self.confidence_score,
            'ai_reasoning': self.ai_reasoning,
            'route_count': self.route_count,
            'total_orders': self.total_orders,
            'total_weight_tons': self.total_weight_tons,
            'total_distance_km': self.total_distance_km,
            'total_estimated_cost': self.total_estimated_cost,
            'total_estimated_duration_hours': self.total_estimated_duration_hours,
            'estimated_fuel_consumption_liters': self.estimated_fuel_consumption_liters,
            'processing_time_ms': self.processing_time_ms,
            'created_at': self.created_at.isoformat(),
            'ai_model_version': self.ai_model_version,
            'prompt_tokens_used': self.prompt_tokens_used,
            'completion_tokens_used': self.completion_tokens_used,
            'total_tokens_used': self.total_tokens_used,
            'optimization_metrics': {**self.optimization_metrics, **efficiency_metrics},
            'routes': [route.to_dict() for route in self.routes],
            'warnings': self.warnings,
            'is_high_confidence': self.is_high_confidence,
            'is_low_confidence': self.is_low_confidence,
            'has_warnings': self.has_warnings
        }
    
    def __str__(self) -> str:
        return (f"OptimizationResult({self.request_id}, {self.scenario_type.value}, "
                f"routes={self.route_count}, confidence={self.confidence_score:.2f})") 