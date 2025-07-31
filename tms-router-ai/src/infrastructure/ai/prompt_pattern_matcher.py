"""
프롬프트 패턴 매칭 엔진

규칙 기반 매칭과 유사도 기반 매칭을 조합한 지능형 프롬프트 선택 시스템입니다.
파라미터 분석을 통해 최적의 프롬프트를 자동으로 선택합니다.
"""
import json
import math
import uuid
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from enum import Enum

from src.shared.constants import ScenarioType, Priority
from src.shared.logging_config import TmsLoggerMixin
from src.shared.exceptions import PromptSelectionError
from src.infrastructure.memory.redis_memory_repository import RedisMemoryRepository


class MatchingStrategy(str, Enum):
    """매칭 전략"""
    RULE_BASED = "rule_based"
    SIMILARITY_BASED = "similarity_based"
    HYBRID = "hybrid"
    LEARNING_BASED = "learning_based"


@dataclass
class ParameterFeatures:
    """파라미터 특성 벡터"""
    vehicle_count: int
    order_count: int
    total_weight: float
    total_volume: float
    geographic_span: float
    time_urgency: float
    complexity_score: float
    priority_distribution: Dict[str, float]
    vehicle_diversity: float
    time_window_flexibility: float
    
    def to_vector(self) -> List[float]:
        """특성을 벡터로 변환"""
        return [
            self.vehicle_count,
            self.order_count,
            self.total_weight,
            self.total_volume,
            self.geographic_span,
            self.time_urgency,
            self.complexity_score,
            self.priority_distribution.get('HIGH', 0),
            self.priority_distribution.get('URGENT', 0),
            self.vehicle_diversity,
            self.time_window_flexibility
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)


@dataclass
class PromptPattern:
    """프롬프트 패턴"""
    pattern_id: str
    scenario_type: ScenarioType
    parameter_features: ParameterFeatures
    success_rate: float
    usage_count: int
    confidence_score: float
    effectiveness_score: float
    last_used: datetime
    feedback_scores: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['scenario_type'] = self.scenario_type.value
        data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptPattern':
        """딕셔너리에서 생성"""
        data['scenario_type'] = ScenarioType(data['scenario_type'])
        data['last_used'] = datetime.fromisoformat(data['last_used'])
        data['parameter_features'] = ParameterFeatures(**data['parameter_features'])
        return cls(**data)


@dataclass
class MatchingResult:
    """매칭 결과"""
    scenario_type: ScenarioType
    confidence_score: float
    rule_based_score: float
    similarity_score: float
    learning_score: float
    final_score: float
    matching_strategy: MatchingStrategy
    similar_patterns: List[str]
    reasoning: str
    alternatives: List[Tuple[ScenarioType, float]]


class PromptPatternMatcher(TmsLoggerMixin):
    """프롬프트 패턴 매칭 엔진"""
    
    def __init__(self, memory_repository: RedisMemoryRepository):
        """
        Args:
            memory_repository: Redis 메모리 저장소
        """
        super().__init__()
        self.memory_repository = memory_repository
        
        # 매칭 전략 가중치
        self.strategy_weights = {
            'rule_based': 0.4,
            'similarity_based': 0.3,
            'learning_based': 0.3
        }
        
        # 유사도 임계값
        self.similarity_threshold = 0.7
        self.min_confidence_threshold = 0.5
        
        # 패턴 캐시
        self._pattern_cache: Dict[str, PromptPattern] = {}
        self._cache_expiry = timedelta(hours=1)
        self._last_cache_update = datetime.min
        
        self.logger.info("PromptPatternMatcher initialized")
    
    def match_optimal_prompt(self, parameters: Dict[str, Any], 
                           conversation_id: Optional[str] = None) -> MatchingResult:
        """
        파라미터 분석을 통한 최적 프롬프트 매칭
        
        Args:
            parameters: TMS 요청 파라미터
            conversation_id: 대화 ID (피드백 히스토리 참조용)
            
        Returns:
            매칭 결과
        """
        try:
            # 1. 파라미터 특성 추출
            features = self._extract_parameter_features(parameters)
            
            # 2. 규칙 기반 매칭
            rule_scores = self._rule_based_matching(features, parameters)
            
            # 3. 유사도 기반 매칭
            similarity_scores = self._similarity_based_matching(features)
            
            # 4. 학습 기반 매칭
            learning_scores = self._learning_based_matching(features, conversation_id)
            
            # 5. 최종 점수 계산 및 시나리오 선택
            final_result = self._combine_scores_and_select(
                rule_scores, similarity_scores, learning_scores, features, parameters
            )
            
            # 6. 패턴 저장 (성공 케이스 학습용)
            self._save_pattern_for_learning(features, final_result.scenario_type, final_result.confidence_score)
            
            self.logger.info("Prompt pattern matching completed", extra={
                'scenario_type': final_result.scenario_type.value,
                'confidence_score': final_result.confidence_score,
                'strategy': final_result.matching_strategy.value,
                'rule_score': final_result.rule_based_score,
                'similarity_score': final_result.similarity_score,
                'learning_score': final_result.learning_score
            })
            
            return final_result
            
        except Exception as e:
            self.logger.error("Failed to match prompt pattern", extra={
                'error': str(e),
                'conversation_id': conversation_id
            })
            # 폴백: 기본 VRP 시나리오
            return MatchingResult(
                scenario_type=ScenarioType.VRP,
                confidence_score=0.5,
                rule_based_score=0.5,
                similarity_score=0.0,
                learning_score=0.0,
                final_score=0.5,
                matching_strategy=MatchingStrategy.RULE_BASED,
                similar_patterns=[],
                reasoning="오류 발생으로 기본 VRP 시나리오 선택",
                alternatives=[]
            )
    
    def update_pattern_effectiveness(self, pattern_id: str, feedback_score: float, 
                                   optimization_success: bool) -> None:
        """패턴 효과성 업데이트"""
        try:
            pattern = self._get_pattern_by_id(pattern_id)
            if not pattern:
                return
            
            # 피드백 점수 추가
            pattern.feedback_scores.append(feedback_score)
            if len(pattern.feedback_scores) > 20:  # 최근 20개만 유지
                pattern.feedback_scores = pattern.feedback_scores[-20:]
            
            # 성공률 업데이트
            if optimization_success:
                pattern.success_rate = (pattern.success_rate * pattern.usage_count + 1.0) / (pattern.usage_count + 1)
            else:
                pattern.success_rate = (pattern.success_rate * pattern.usage_count) / (pattern.usage_count + 1)
            
            # 효과성 점수 재계산
            pattern.effectiveness_score = self._calculate_effectiveness_score(pattern)
            pattern.usage_count += 1
            pattern.last_used = datetime.now()
            
            # Redis에 저장
            self._save_pattern(pattern)
            
            self.logger.info("Pattern effectiveness updated", extra={
                'pattern_id': pattern_id,
                'feedback_score': feedback_score,
                'success': optimization_success,
                'new_effectiveness': pattern.effectiveness_score
            })
            
        except Exception as e:
            self.logger.error("Failed to update pattern effectiveness", extra={
                'pattern_id': pattern_id,
                'error': str(e)
            })
    
    def get_pattern_analytics(self, scenario_type: Optional[ScenarioType] = None) -> Dict[str, Any]:
        """패턴 분석 결과 조회"""
        try:
            patterns = self._load_all_patterns()
            
            if scenario_type:
                patterns = [p for p in patterns if p.scenario_type == scenario_type]
            
            if not patterns:
                return {'total_patterns': 0, 'analysis': {}}
            
            # 통계 계산
            total_patterns = len(patterns)
            avg_success_rate = sum(p.success_rate for p in patterns) / total_patterns
            avg_effectiveness = sum(p.effectiveness_score for p in patterns) / total_patterns
            
            # 시나리오별 분포
            scenario_distribution = Counter(p.scenario_type.value for p in patterns)
            
            # 효과성별 분류
            high_performance = [p for p in patterns if p.effectiveness_score >= 0.8]
            medium_performance = [p for p in patterns if 0.5 <= p.effectiveness_score < 0.8]
            low_performance = [p for p in patterns if p.effectiveness_score < 0.5]
            
            # 최근 트렌드 (최근 7일)
            recent_patterns = [
                p for p in patterns 
                if (datetime.now() - p.last_used).days <= 7
            ]
            
            analytics = {
                'total_patterns': total_patterns,
                'avg_success_rate': round(avg_success_rate, 3),
                'avg_effectiveness': round(avg_effectiveness, 3),
                'scenario_distribution': dict(scenario_distribution),
                'performance_distribution': {
                    'high': len(high_performance),
                    'medium': len(medium_performance),
                    'low': len(low_performance)
                },
                'recent_activity': {
                    'patterns_used_last_7_days': len(recent_patterns),
                    'most_used_scenario': scenario_distribution.most_common(1)[0] if scenario_distribution else None
                },
                'top_patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'scenario_type': p.scenario_type.value,
                        'effectiveness_score': p.effectiveness_score,
                        'usage_count': p.usage_count
                    }
                    for p in sorted(patterns, key=lambda x: x.effectiveness_score, reverse=True)[:5]
                ]
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error("Failed to get pattern analytics", extra={'error': str(e)})
            return {'error': str(e)}
    
    # Private methods
    
    def _extract_parameter_features(self, parameters: Dict[str, Any]) -> ParameterFeatures:
        """파라미터에서 특성 추출"""
        vehicles = parameters.get('vehicles', [])
        orders = parameters.get('orders', [])
        
        # 기본 통계
        vehicle_count = len(vehicles)
        order_count = len(orders)
        
        # 중량/부피 계산
        total_weight = sum(order.get('weight_tons', 0) for order in orders)
        total_volume = sum(order.get('volume_m3', 0) for order in orders)
        
        # 지리적 분산도 계산
        geographic_span = self._calculate_geographic_span(orders)
        
        # 시간 긴급도 계산
        time_urgency = self._calculate_time_urgency(orders)
        
        # 복잡도 점수 계산
        complexity_score = self._calculate_complexity_score(vehicles, orders)
        
        # 우선순위 분포
        priority_distribution = self._calculate_priority_distribution(orders)
        
        # 차량 다양성
        vehicle_diversity = self._calculate_vehicle_diversity(vehicles)
        
        # 시간 창 유연성
        time_window_flexibility = self._calculate_time_window_flexibility(orders)
        
        return ParameterFeatures(
            vehicle_count=vehicle_count,
            order_count=order_count,
            total_weight=total_weight,
            total_volume=total_volume,
            geographic_span=geographic_span,
            time_urgency=time_urgency,
            complexity_score=complexity_score,
            priority_distribution=priority_distribution,
            vehicle_diversity=vehicle_diversity,
            time_window_flexibility=time_window_flexibility
        )
    
    def _rule_based_matching(self, features: ParameterFeatures, 
                           parameters: Dict[str, Any]) -> Dict[ScenarioType, float]:
        """규칙 기반 매칭 (기존 로직 활용)"""
        scores = {}
        
        # VRP 점수
        vrp_score = 0.0
        if features.vehicle_count >= 2:
            vrp_score += 0.4
        if features.order_count > features.vehicle_count:
            vrp_score += 0.3
        if features.geographic_span > 0.5:
            vrp_score += 0.2
        if features.vehicle_diversity > 0.3:
            vrp_score += 0.1
        scores[ScenarioType.VRP] = min(1.0, vrp_score)
        
        # TSP 점수
        tsp_score = 0.0
        if features.vehicle_count == 1:
            tsp_score += 0.5
        if 2 <= features.order_count <= 20:
            tsp_score += 0.3
        if features.complexity_score < 0.5:
            tsp_score += 0.2
        scores[ScenarioType.TSP] = min(1.0, tsp_score)
        
        # 적재 통합 점수
        consolidation_score = 0.0
        if features.total_weight / max(features.order_count, 1) < 1.0:  # 소량 주문
            consolidation_score += 0.4
        if features.geographic_span < 0.3:  # 지역 집중
            consolidation_score += 0.3
        if features.time_window_flexibility > 0.6:
            consolidation_score += 0.2
        scores[ScenarioType.LOAD_CONSOLIDATION] = min(1.0, consolidation_score)
        
        # 긴급 배송 점수
        emergency_score = 0.0
        if features.time_urgency > 0.7:
            emergency_score += 0.5
        if features.priority_distribution.get('URGENT', 0) > 0:
            emergency_score += 0.3
        if parameters.get('existing_routes'):
            emergency_score += 0.2
        scores[ScenarioType.EMERGENCY_DISPATCH] = min(1.0, emergency_score)
        
        # 실시간 조정 점수
        realtime_score = 0.0
        if parameters.get('active_routes'):
            realtime_score += 0.4
        if parameters.get('change_reason'):
            realtime_score += 0.3
        if parameters.get('current_situation'):
            realtime_score += 0.3
        scores[ScenarioType.REALTIME_ADJUSTMENT] = min(1.0, realtime_score)
        
        return scores
    
    def _similarity_based_matching(self, features: ParameterFeatures) -> Dict[ScenarioType, float]:
        """유사도 기반 매칭"""
        similarity_scores = defaultdict(float)
        
        # 과거 패턴들과 유사도 계산
        patterns = self._load_recent_effective_patterns()
        target_vector = features.to_vector()
        
        for pattern in patterns:
            pattern_vector = pattern.parameter_features.to_vector()
            similarity = self._calculate_cosine_similarity(target_vector, pattern_vector)
            
            if similarity >= self.similarity_threshold:
                # 유사도와 패턴의 효과성을 조합
                weighted_score = similarity * pattern.effectiveness_score
                similarity_scores[pattern.scenario_type] = max(
                    similarity_scores[pattern.scenario_type], 
                    weighted_score
                )
        
        return dict(similarity_scores)
    
    def _learning_based_matching(self, features: ParameterFeatures, 
                               conversation_id: Optional[str]) -> Dict[ScenarioType, float]:
        """학습 기반 매칭 (과거 성공 패턴 학습)"""
        learning_scores = defaultdict(float)
        
        # 사용자별 선호 패턴 (대화 ID 기반)
        if conversation_id:
            user_patterns = self._get_user_preference_patterns(conversation_id)
            for scenario_type, preference_score in user_patterns.items():
                learning_scores[scenario_type] += preference_score * 0.5
        
        # 글로벌 성공 패턴
        global_patterns = self._get_global_success_patterns()
        target_vector = features.to_vector()
        
        for pattern in global_patterns:
            pattern_vector = pattern.parameter_features.to_vector()
            similarity = self._calculate_cosine_similarity(target_vector, pattern_vector)
            
            if similarity >= 0.6:  # 학습 기반은 임계값 낮게
                # 유사도 * 성공률 * 효과성
                score = similarity * pattern.success_rate * pattern.effectiveness_score
                learning_scores[pattern.scenario_type] = max(
                    learning_scores[pattern.scenario_type],
                    score
                )
        
        return dict(learning_scores)
    
    def _combine_scores_and_select(self, rule_scores: Dict[ScenarioType, float],
                                 similarity_scores: Dict[ScenarioType, float],
                                 learning_scores: Dict[ScenarioType, float],
                                 features: ParameterFeatures,
                                 parameters: Dict[str, Any]) -> MatchingResult:
        """점수 조합 및 최종 시나리오 선택"""
        
        # 모든 시나리오 수집
        all_scenarios = set(rule_scores.keys()) | set(similarity_scores.keys()) | set(learning_scores.keys())
        
        final_scores = {}
        detailed_scores = {}
        
        for scenario in all_scenarios:
            rule_score = rule_scores.get(scenario, 0.0)
            sim_score = similarity_scores.get(scenario, 0.0)
            learn_score = learning_scores.get(scenario, 0.0)
            
            # 가중 평균 계산
            final_score = (
                rule_score * self.strategy_weights['rule_based'] +
                sim_score * self.strategy_weights['similarity_based'] +
                learn_score * self.strategy_weights['learning_based']
            )
            
            final_scores[scenario] = final_score
            detailed_scores[scenario] = {
                'rule': rule_score,
                'similarity': sim_score,
                'learning': learn_score,
                'final': final_score
            }
        
        # 최고 점수 시나리오 선택
        if not final_scores:
            # 폴백: VRP
            return MatchingResult(
                scenario_type=ScenarioType.VRP,
                confidence_score=0.5,
                rule_based_score=0.5,
                similarity_score=0.0,
                learning_score=0.0,
                final_score=0.5,
                matching_strategy=MatchingStrategy.RULE_BASED,
                similar_patterns=[],
                reasoning="점수 없음으로 기본 VRP 선택",
                alternatives=[]
            )
        
        # 명시적 시나리오 타입이 있으면 보너스
        explicit_scenario = parameters.get('scenario_type')
        if explicit_scenario:
            try:
                explicit_type = ScenarioType(explicit_scenario)
                if explicit_type in final_scores:
                    final_scores[explicit_type] += 0.2
            except ValueError:
                pass
        
        best_scenario = max(final_scores.keys(), key=lambda x: final_scores[x])
        best_score = final_scores[best_scenario]
        
        # 매칭 전략 결정
        best_details = detailed_scores[best_scenario]
        if best_details['rule'] >= best_details['similarity'] and best_details['rule'] >= best_details['learning']:
            strategy = MatchingStrategy.RULE_BASED
        elif best_details['similarity'] >= best_details['learning']:
            strategy = MatchingStrategy.SIMILARITY_BASED
        elif best_details['learning'] > 0:
            strategy = MatchingStrategy.LEARNING_BASED
        else:
            strategy = MatchingStrategy.HYBRID
        
        # 대안 시나리오 정렬
        alternatives = sorted(
            [(s, score) for s, score in final_scores.items() if s != best_scenario],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # 유사 패턴 ID 수집
        similar_patterns = self._find_similar_pattern_ids(features)
        
        # 선택 근거 생성
        reasoning = self._generate_selection_reasoning(
            best_scenario, best_details, features, parameters
        )
        
        return MatchingResult(
            scenario_type=best_scenario,
            confidence_score=min(1.0, best_score),
            rule_based_score=best_details['rule'],
            similarity_score=best_details['similarity'],
            learning_score=best_details['learning'],
            final_score=best_score,
            matching_strategy=strategy,
            similar_patterns=similar_patterns,
            reasoning=reasoning,
            alternatives=alternatives
        )
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        if len(vec1) != len(vec2):
            return 0.0
        
        # 벡터 정규화
        norm1 = math.sqrt(sum(x**2 for x in vec1))
        norm2 = math.sqrt(sum(x**2 for x in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 내적 계산
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_geographic_span(self, orders: List[Dict[str, Any]]) -> float:
        """지리적 분산도 계산"""
        if len(orders) < 2:
            return 0.0
        
        locations = []
        for order in orders:
            pickup = order.get('pickup_location', {})
            delivery = order.get('delivery_location', {})
            
            if pickup.get('lat') and pickup.get('lng'):
                locations.append((pickup['lat'], pickup['lng']))
            if delivery.get('lat') and delivery.get('lng'):
                locations.append((delivery['lat'], delivery['lng']))
        
        if len(locations) < 2:
            return 0.0
        
        # 최대 거리 계산
        max_distance = 0.0
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                lat1, lng1 = locations[i]
                lat2, lng2 = locations[j]
                distance = math.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)
                max_distance = max(max_distance, distance)
        
        # 정규화 (0.1도 = 1.0으로 설정)
        return min(1.0, max_distance / 0.1)
    
    def _calculate_time_urgency(self, orders: List[Dict[str, Any]]) -> float:
        """시간 긴급도 계산"""
        if not orders:
            return 0.0
        
        urgent_count = 0
        high_count = 0
        
        for order in orders:
            priority = order.get('priority', 'MEDIUM')
            if priority == 'URGENT':
                urgent_count += 1
            elif priority == 'HIGH':
                high_count += 1
        
        urgency_ratio = (urgent_count * 1.0 + high_count * 0.5) / len(orders)
        return min(1.0, urgency_ratio)
    
    def _calculate_complexity_score(self, vehicles: List[Dict[str, Any]], 
                                  orders: List[Dict[str, Any]]) -> float:
        """복잡도 점수 계산"""
        score = 0.0
        
        # 차량-주문 비율
        if vehicles and orders:
            ratio = len(orders) / len(vehicles)
            score += min(0.3, ratio / 10.0)  # 10:1 비율에서 최대값
        
        # 제약 조건 수
        constraint_count = 0
        for order in orders:
            if order.get('time_window'):
                constraint_count += 1
            if order.get('special_requirements'):
                constraint_count += 1
        
        score += min(0.4, constraint_count / (len(orders) * 2))
        
        # 차량 특수 능력
        special_vehicles = sum(1 for v in vehicles if v.get('special_capabilities'))
        if vehicles:
            score += min(0.3, special_vehicles / len(vehicles))
        
        return min(1.0, score)
    
    def _calculate_priority_distribution(self, orders: List[Dict[str, Any]]) -> Dict[str, float]:
        """우선순위 분포 계산"""
        if not orders:
            return {}
        
        priority_counts = Counter(order.get('priority', 'MEDIUM') for order in orders)
        total_orders = len(orders)
        
        return {
            priority: count / total_orders 
            for priority, count in priority_counts.items()
        }
    
    def _calculate_vehicle_diversity(self, vehicles: List[Dict[str, Any]]) -> float:
        """차량 다양성 계산"""
        if len(vehicles) <= 1:
            return 0.0
        
        # 용량 다양성
        capacities = set(v.get('capacity_tons', 0) for v in vehicles)
        capacity_diversity = len(capacities) / len(vehicles)
        
        # 특수 능력 다양성
        all_capabilities = set()
        for v in vehicles:
            capabilities = v.get('special_capabilities', [])
            all_capabilities.update(capabilities)
        
        capability_diversity = len(all_capabilities) / max(len(vehicles), 1)
        
        return min(1.0, (capacity_diversity + capability_diversity) / 2)
    
    def _calculate_time_window_flexibility(self, orders: List[Dict[str, Any]]) -> float:
        """시간 창 유연성 계산"""
        if not orders:
            return 1.0
        
        flexible_count = 0
        for order in orders:
            time_window = order.get('time_window')
            if not time_window:
                flexible_count += 1  # 제약 없음
            # 실제로는 시간 창 길이를 계산해야 함
        
        return flexible_count / len(orders)
    
    def _calculate_effectiveness_score(self, pattern: PromptPattern) -> float:
        """패턴 효과성 점수 계산"""
        if not pattern.feedback_scores:
            return pattern.success_rate
        
        # 최근 피드백 평균
        recent_feedback = pattern.feedback_scores[-10:]  # 최근 10개
        avg_feedback = sum(recent_feedback) / len(recent_feedback)
        normalized_feedback = avg_feedback / 5.0  # 5점 만점 정규화
        
        # 성공률과 피드백 점수 조합
        effectiveness = (pattern.success_rate * 0.6 + normalized_feedback * 0.4)
        
        # 사용 빈도 보너스 (많이 사용될수록 신뢰도 높음)
        usage_bonus = min(0.1, pattern.usage_count / 100.0)
        
        return min(1.0, effectiveness + usage_bonus)
    
    def _load_recent_effective_patterns(self, days: int = 30) -> List[PromptPattern]:
        """최근 효과적인 패턴들 로드"""
        try:
            all_patterns = self._load_all_patterns()
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 최근 사용되고 효과성이 높은 패턴들 필터링
            effective_patterns = [
                p for p in all_patterns
                if p.last_used >= cutoff_date and p.effectiveness_score >= 0.6
            ]
            
            # 효과성 순으로 정렬
            effective_patterns.sort(key=lambda x: x.effectiveness_score, reverse=True)
            
            return effective_patterns[:50]  # 상위 50개
            
        except Exception as e:
            self.logger.error("Failed to load recent effective patterns", extra={'error': str(e)})
            return []
    
    def _get_user_preference_patterns(self, conversation_id: str) -> Dict[ScenarioType, float]:
        """사용자 선호 패턴 조회"""
        try:
            # 대화 메모리에서 사용자 선호도 조회
            memory_data = self.memory_repository.get_conversation_memory(conversation_id)
            if not memory_data or 'context' not in memory_data:
                return {}
            
            context = memory_data['context']
            preference_weights = context.get('learned_patterns', {}).get('preference_weights', {})
            
            # 시나리오별 선호도로 변환 (간단한 매핑)
            preferences = {}
            if preference_weights:
                # VRP: 균형잡힌 최적화
                if preference_weights.get('distance', 0) > 0.4:
                    preferences[ScenarioType.VRP] = 0.7
                
                # TSP: 시간 우선
                if preference_weights.get('time', 0) > 0.5:
                    preferences[ScenarioType.TSP] = 0.6
                
                # 비용 우선시 적재 통합
                if preference_weights.get('cost', 0) > 0.5:
                    preferences[ScenarioType.LOAD_CONSOLIDATION] = 0.6
            
            return preferences
            
        except Exception as e:
            self.logger.error("Failed to get user preference patterns", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            return {}
    
    def _get_global_success_patterns(self) -> List[PromptPattern]:
        """글로벌 성공 패턴 조회"""
        try:
            all_patterns = self._load_all_patterns()
            
            # 높은 성공률과 효과성을 가진 패턴들
            success_patterns = [
                p for p in all_patterns
                if p.success_rate >= 0.7 and p.effectiveness_score >= 0.7 and p.usage_count >= 3
            ]
            
            # 효과성 순으로 정렬
            success_patterns.sort(key=lambda x: x.effectiveness_score, reverse=True)
            
            return success_patterns[:30]  # 상위 30개
            
        except Exception as e:
            self.logger.error("Failed to get global success patterns", extra={'error': str(e)})
            return []
    
    def _find_similar_pattern_ids(self, features: ParameterFeatures) -> List[str]:
        """유사한 패턴 ID들 찾기"""
        try:
            patterns = self._load_recent_effective_patterns()
            target_vector = features.to_vector()
            
            similar_patterns = []
            for pattern in patterns:
                pattern_vector = pattern.parameter_features.to_vector()
                similarity = self._calculate_cosine_similarity(target_vector, pattern_vector)
                
                if similarity >= self.similarity_threshold:
                    similar_patterns.append(pattern.pattern_id)
            
            return similar_patterns[:5]  # 상위 5개
            
        except Exception as e:
            self.logger.error("Failed to find similar patterns", extra={'error': str(e)})
            return []
    
    def _generate_selection_reasoning(self, scenario: ScenarioType, 
                                    scores: Dict[str, float],
                                    features: ParameterFeatures,
                                    parameters: Dict[str, Any]) -> str:
        """선택 근거 생성"""
        reasoning_parts = []
        
        # 주요 점수 정보
        reasoning_parts.append(f"최종 점수: {scores['final']:.3f}")
        
        # 주요 기여 요소
        main_contributor = max(scores.keys(), key=lambda x: scores[x] if x != 'final' else 0)
        if main_contributor != 'final':
            reasoning_parts.append(f"주요 근거: {main_contributor} ({scores[main_contributor]:.3f})")
        
        # 시나리오별 특성
        if scenario == ScenarioType.VRP:
            reasoning_parts.append(f"다중 차량({features.vehicle_count}대), 복잡도 {features.complexity_score:.2f}")
        elif scenario == ScenarioType.TSP:
            reasoning_parts.append(f"단일 차량 최적화, 주문 수 {features.order_count}개")
        elif scenario == ScenarioType.LOAD_CONSOLIDATION:
            reasoning_parts.append(f"적재 통합 최적화, 평균 중량 {features.total_weight/max(features.order_count,1):.1f}톤")
        elif scenario == ScenarioType.EMERGENCY_DISPATCH:
            reasoning_parts.append(f"긴급 배송, 시간 긴급도 {features.time_urgency:.2f}")
        elif scenario == ScenarioType.REALTIME_ADJUSTMENT:
            reasoning_parts.append("실시간 상황 대응")
        
        # 지리적 특성
        if features.geographic_span > 0.5:
            reasoning_parts.append(f"광역 배송 (분산도 {features.geographic_span:.2f})")
        elif features.geographic_span < 0.2:
            reasoning_parts.append("지역 집중 배송")
        
        return " | ".join(reasoning_parts)
    
    def _save_pattern_for_learning(self, features: ParameterFeatures, 
                                 scenario_type: ScenarioType, confidence: float) -> None:
        """학습용 패턴 저장"""
        try:
            pattern_id = str(uuid.uuid4())
            pattern = PromptPattern(
                pattern_id=pattern_id,
                scenario_type=scenario_type,
                parameter_features=features,
                success_rate=0.8,  # 초기값
                usage_count=1,
                confidence_score=confidence,
                effectiveness_score=0.8,  # 초기값
                last_used=datetime.now(),
                feedback_scores=[]
            )
            
            self._save_pattern(pattern)
            
        except Exception as e:
            self.logger.error("Failed to save pattern for learning", extra={'error': str(e)})
    
    def _save_pattern(self, pattern: PromptPattern) -> None:
        """패턴을 Redis에 저장"""
        try:
            key = f"prompt_pattern:{pattern.pattern_id}"
            data = json.dumps(pattern.to_dict(), ensure_ascii=False)
            
            # 90일 TTL
            self.memory_repository.redis_client.setex(key, timedelta(days=90), data)
            
            # 캐시 업데이트
            self._pattern_cache[pattern.pattern_id] = pattern
            
        except Exception as e:
            self.logger.error("Failed to save pattern", extra={
                'pattern_id': pattern.pattern_id,
                'error': str(e)
            })
    
    def _load_all_patterns(self) -> List[PromptPattern]:
        """모든 패턴 로드"""
        try:
            # 캐시 확인
            if (datetime.now() - self._last_cache_update) < self._cache_expiry and self._pattern_cache:
                return list(self._pattern_cache.values())
            
            # Redis에서 로드
            pattern_keys = self.memory_repository.redis_client.keys("prompt_pattern:*")
            patterns = []
            
            for key in pattern_keys:
                try:
                    data = self.memory_repository.redis_client.get(key)
                    if data:
                        pattern_dict = json.loads(data)
                        pattern = PromptPattern.from_dict(pattern_dict)
                        patterns.append(pattern)
                        self._pattern_cache[pattern.pattern_id] = pattern
                except Exception as e:
                    self.logger.warning(f"Failed to load pattern {key}: {e}")
                    continue
            
            self._last_cache_update = datetime.now()
            return patterns
            
        except Exception as e:
            self.logger.error("Failed to load all patterns", extra={'error': str(e)})
            return []
    
    def _get_pattern_by_id(self, pattern_id: str) -> Optional[PromptPattern]:
        """ID로 패턴 조회"""
        try:
            # 캐시 확인
            if pattern_id in self._pattern_cache:
                return self._pattern_cache[pattern_id]
            
            # Redis에서 조회
            key = f"prompt_pattern:{pattern_id}"
            data = self.memory_repository.redis_client.get(key)
            
            if data:
                pattern_dict = json.loads(data)
                pattern = PromptPattern.from_dict(pattern_dict)
                self._pattern_cache[pattern_id] = pattern
                return pattern
            
            return None
            
        except Exception as e:
            self.logger.error("Failed to get pattern by ID", extra={
                'pattern_id': pattern_id,
                'error': str(e)
            })
            return None 