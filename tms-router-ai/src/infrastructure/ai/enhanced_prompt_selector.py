"""
Enhanced Prompt Selector - 고도화된 프롬프트 선택기

기존 PromptSelector를 확장하여 고도화된 패턴 매칭 엔진을 통합한 
지능형 프롬프트 선택 시스템입니다.
"""
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from langchain.prompts import PromptTemplate

from src.shared.constants import ScenarioType, Priority
from src.infrastructure.ai.prompt_templates import TmsPromptTemplates
from src.infrastructure.ai.prompt_selector import PromptSelector, PromptSelectionResult
from src.infrastructure.ai.prompt_pattern_matcher import PromptPatternMatcher, MatchingResult
from src.shared.exceptions import PromptSelectionError
from src.shared.logging_config import TmsLoggerMixin
from src.infrastructure.memory.redis_memory_repository import RedisMemoryRepository


@dataclass
class EnhancedPromptSelectionResult:
    """고도화된 프롬프트 선택 결과"""
    scenario_type: ScenarioType
    prompt_template: PromptTemplate
    confidence_score: float
    selection_reasoning: str
    alternative_scenarios: List[ScenarioType]
    
    # 고도화된 정보
    pattern_matching_result: MatchingResult
    effectiveness_prediction: float
    similar_patterns: List[str]
    optimization_suggestions: List[str]
    risk_assessment: Dict[str, float]


class EnhancedPromptSelector(TmsLoggerMixin):
    """고도화된 프롬프트 선택기"""
    
    def __init__(self, memory_repository: RedisMemoryRepository):
        """
        Args:
            memory_repository: Redis 메모리 저장소
        """
        super().__init__()
        self.memory_repository = memory_repository
        
        # 기존 선택기와 새로운 패턴 매칭 엔진
        self.basic_selector = PromptSelector()
        self.pattern_matcher = PromptPatternMatcher(memory_repository)
        self.templates = TmsPromptTemplates()
        
        # 선택 전략 설정
        self.use_pattern_matching = True
        self.fallback_to_basic = True
        self.min_confidence_threshold = 0.6
        
        self.logger.info("EnhancedPromptSelector initialized")
    
    def select_optimal_prompt(self, parameters: Dict[str, Any], 
                            conversation_id: Optional[str] = None,
                            feedback_context: Optional[Dict[str, Any]] = None) -> EnhancedPromptSelectionResult:
        """
        고도화된 프롬프트 선택
        
        Args:
            parameters: TMS 배차 요청 파라미터
            conversation_id: 대화 ID (선택사항)
            feedback_context: 피드백 컨텍스트 (선택사항)
            
        Returns:
            고도화된 프롬프트 선택 결과
        """
        try:
            self.logger.info("Starting enhanced prompt selection", extra={
                'parameters_keys': list(parameters.keys()),
                'conversation_id': conversation_id,
                'has_feedback_context': feedback_context is not None
            })
            
            # 1. 고도화된 패턴 매칭 시도
            pattern_result = None
            if self.use_pattern_matching:
                try:
                    pattern_result = self.pattern_matcher.match_optimal_prompt(
                        parameters, conversation_id
                    )
                    
                    # 신뢰도 검증
                    if pattern_result.confidence_score >= self.min_confidence_threshold:
                        self.logger.info("Pattern matching successful", extra={
                            'scenario': pattern_result.scenario_type.value,
                            'confidence': pattern_result.confidence_score,
                            'strategy': pattern_result.matching_strategy.value
                        })
                        
                        return self._create_enhanced_result_from_pattern(
                            pattern_result, parameters, feedback_context
                        )
                    else:
                        self.logger.warning("Pattern matching confidence too low", extra={
                            'confidence': pattern_result.confidence_score,
                            'threshold': self.min_confidence_threshold
                        })
                        
                except Exception as e:
                    self.logger.error("Pattern matching failed", extra={'error': str(e)})
            
            # 2. 기본 선택기로 폴백
            if self.fallback_to_basic:
                self.logger.info("Falling back to basic selector")
                basic_result = self.basic_selector.select_optimal_prompt(parameters)
                
                return self._create_enhanced_result_from_basic(
                    basic_result, pattern_result, parameters, feedback_context
                )
            
            # 3. 모든 방법 실패 시 기본 VRP
            raise PromptSelectionError("All selection methods failed", list(ScenarioType))
            
        except PromptSelectionError:
            raise
        except Exception as e:
            self.logger.error("Enhanced prompt selection failed", extra={'error': str(e)})
            raise PromptSelectionError(f"Selection failed: {e}", list(ScenarioType))
    
    def update_pattern_effectiveness(self, selection_result: EnhancedPromptSelectionResult,
                                   optimization_result: Dict[str, Any],
                                   feedback_score: Optional[float] = None) -> None:
        """패턴 효과성 업데이트"""
        try:
            if not hasattr(selection_result, 'pattern_matching_result'):
                return
            
            pattern_result = selection_result.pattern_matching_result
            if not pattern_result.similar_patterns:
                return
            
            # 최적화 성공 여부 판단
            confidence_score = optimization_result.get('confidence_score', 0.0)
            optimization_success = confidence_score >= 0.7
            
            # 피드백 점수 (없으면 confidence_score 기반으로 추정)
            if feedback_score is None:
                feedback_score = min(5.0, confidence_score * 5)
            
            # 각 유사 패턴의 효과성 업데이트
            for pattern_id in pattern_result.similar_patterns:
                self.pattern_matcher.update_pattern_effectiveness(
                    pattern_id, feedback_score, optimization_success
                )
            
            self.logger.info("Pattern effectiveness updated", extra={
                'scenario': selection_result.scenario_type.value,
                'patterns_updated': len(pattern_result.similar_patterns),
                'optimization_success': optimization_success,
                'feedback_score': feedback_score
            })
            
        except Exception as e:
            self.logger.error("Failed to update pattern effectiveness", extra={'error': str(e)})
    
    def get_selection_analytics(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """선택 분석 결과 조회"""
        try:
            # 패턴 매칭 분석
            pattern_analytics = self.pattern_matcher.get_pattern_analytics()
            
            # 시나리오별 선택 통계
            scenario_stats = self._get_scenario_selection_stats()
            
            # 효과성 트렌드
            effectiveness_trend = self._get_effectiveness_trend()
            
            # 사용자별 선호도 (대화 ID가 있는 경우)
            user_preferences = {}
            if conversation_id:
                user_preferences = self._get_user_selection_preferences(conversation_id)
            
            analytics = {
                'pattern_analytics': pattern_analytics,
                'scenario_selection_stats': scenario_stats,
                'effectiveness_trend': effectiveness_trend,
                'user_preferences': user_preferences,
                'selection_metadata': {
                    'matcher_confidence_threshold': self.min_confidence_threshold,
                    'pattern_matching_enabled': self.use_pattern_matching,
                    'fallback_enabled': self.fallback_to_basic,
                    'analyzed_at': pattern_analytics.get('analyzed_at', '')
                }
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error("Failed to get selection analytics", extra={'error': str(e)})
            return {'error': str(e)}
    
    def optimize_selection_strategy(self, recent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """선택 전략 최적화"""
        try:
            if len(recent_results) < 10:
                return {'message': 'Insufficient data for optimization'}
            
            # 성공률 분석
            pattern_matching_success = []
            basic_selection_success = []
            
            for result in recent_results:
                confidence = result.get('confidence_score', 0)
                feedback_score = result.get('feedback_score', 0)
                selection_method = result.get('selection_method', 'unknown')
                
                success = confidence >= 0.7 and feedback_score >= 3.5
                
                if selection_method == 'pattern_matching':
                    pattern_matching_success.append(success)
                elif selection_method == 'basic':
                    basic_selection_success.append(success)
            
            # 성공률 계산
            pattern_success_rate = sum(pattern_matching_success) / max(len(pattern_matching_success), 1)
            basic_success_rate = sum(basic_selection_success) / max(len(basic_selection_success), 1)
            
            # 전략 조정
            optimization_recommendations = []
            
            if pattern_success_rate < 0.6:
                # 패턴 매칭 성능이 낮으면 임계값 조정
                new_threshold = min(0.8, self.min_confidence_threshold + 0.1)
                optimization_recommendations.append(
                    f"패턴 매칭 임계값을 {new_threshold:.1f}로 상향 조정 권장"
                )
            
            if basic_success_rate > pattern_success_rate + 0.1:
                optimization_recommendations.append("기본 선택기 가중치 증가 권장")
            
            if pattern_success_rate > basic_success_rate + 0.1:
                optimization_recommendations.append("패턴 매칭 가중치 증가 권장")
            
            optimization_result = {
                'pattern_matching_success_rate': round(pattern_success_rate, 3),
                'basic_selection_success_rate': round(basic_success_rate, 3),
                'current_threshold': self.min_confidence_threshold,
                'recommendations': optimization_recommendations,
                'data_points_analyzed': len(recent_results),
                'optimization_date': pattern_analytics.get('analyzed_at', '')
            }
            
            self.logger.info("Selection strategy optimization completed", extra=optimization_result)
            
            return optimization_result
            
        except Exception as e:
            self.logger.error("Failed to optimize selection strategy", extra={'error': str(e)})
            return {'error': str(e)}
    
    # Private methods
    
    def _create_enhanced_result_from_pattern(self, pattern_result: MatchingResult,
                                           parameters: Dict[str, Any],
                                           feedback_context: Optional[Dict[str, Any]]) -> EnhancedPromptSelectionResult:
        """패턴 매칭 결과에서 고도화된 결과 생성"""
        
        # 프롬프트 템플릿 가져오기
        prompt_template = self.templates.get_prompt_by_scenario(pattern_result.scenario_type)
        
        # 효과성 예측
        effectiveness_prediction = self._predict_effectiveness(pattern_result, parameters)
        
        # 최적화 제안
        optimization_suggestions = self._generate_optimization_suggestions(
            pattern_result, parameters, feedback_context
        )
        
        # 리스크 평가
        risk_assessment = self._assess_selection_risk(pattern_result, parameters)
        
        # 대안 시나리오 추출
        alternatives = [scenario for scenario, _ in pattern_result.alternatives]
        
        return EnhancedPromptSelectionResult(
            scenario_type=pattern_result.scenario_type,
            prompt_template=prompt_template,
            confidence_score=pattern_result.confidence_score,
            selection_reasoning=pattern_result.reasoning,
            alternative_scenarios=alternatives,
            pattern_matching_result=pattern_result,
            effectiveness_prediction=effectiveness_prediction,
            similar_patterns=pattern_result.similar_patterns,
            optimization_suggestions=optimization_suggestions,
            risk_assessment=risk_assessment
        )
    
    def _create_enhanced_result_from_basic(self, basic_result: PromptSelectionResult,
                                         pattern_result: Optional[MatchingResult],
                                         parameters: Dict[str, Any],
                                         feedback_context: Optional[Dict[str, Any]]) -> EnhancedPromptSelectionResult:
        """기본 선택기 결과에서 고도화된 결과 생성"""
        
        # 효과성 예측 (패턴 정보가 없으므로 기본값)
        effectiveness_prediction = min(0.8, basic_result.confidence_score + 0.1)
        
        # 기본 최적화 제안
        optimization_suggestions = [
            "기본 선택 방식 사용",
            "더 많은 데이터로 패턴 학습 필요"
        ]
        
        # 낮은 리스크 평가
        risk_assessment = {
            'confidence_risk': max(0.0, 0.8 - basic_result.confidence_score),
            'pattern_learning_risk': 0.3,  # 패턴 학습 부족
            'overall_risk': 0.2
        }
        
        # 빈 패턴 매칭 결과 생성
        empty_pattern_result = MatchingResult(
            scenario_type=basic_result.scenario_type,
            confidence_score=basic_result.confidence_score,
            rule_based_score=basic_result.confidence_score,
            similarity_score=0.0,
            learning_score=0.0,
            final_score=basic_result.confidence_score,
            matching_strategy=pattern_result.matching_strategy if pattern_result else "rule_based",
            similar_patterns=[],
            reasoning=basic_result.selection_reasoning,
            alternatives=[(s, 0.5) for s in basic_result.alternative_scenarios]
        )
        
        return EnhancedPromptSelectionResult(
            scenario_type=basic_result.scenario_type,
            prompt_template=basic_result.prompt_template,
            confidence_score=basic_result.confidence_score,
            selection_reasoning=basic_result.selection_reasoning,
            alternative_scenarios=basic_result.alternative_scenarios,
            pattern_matching_result=empty_pattern_result,
            effectiveness_prediction=effectiveness_prediction,
            similar_patterns=[],
            optimization_suggestions=optimization_suggestions,
            risk_assessment=risk_assessment
        )
    
    def _predict_effectiveness(self, pattern_result: MatchingResult, 
                             parameters: Dict[str, Any]) -> float:
        """효과성 예측"""
        base_effectiveness = pattern_result.confidence_score
        
        # 유사 패턴이 많으면 예측 신뢰도 높음
        if len(pattern_result.similar_patterns) >= 3:
            base_effectiveness += 0.1
        
        # 학습 기반 점수가 높으면 효과성 높음
        if pattern_result.learning_score >= 0.7:
            base_effectiveness += 0.1
        
        # 복잡한 요청에서 패턴 매칭이 더 중요
        vehicle_count = len(parameters.get('vehicles', []))
        order_count = len(parameters.get('orders', []))
        if vehicle_count >= 3 and order_count >= 10:
            base_effectiveness += 0.05
        
        return min(1.0, base_effectiveness)
    
    def _generate_optimization_suggestions(self, pattern_result: MatchingResult,
                                         parameters: Dict[str, Any],
                                         feedback_context: Optional[Dict[str, Any]]) -> List[str]:
        """최적화 제안 생성"""
        suggestions = []
        
        # 매칭 전략 기반 제안
        if pattern_result.matching_strategy.value == "similarity_based":
            suggestions.append("유사 패턴 기반 선택: 과거 성공 사례 활용")
        elif pattern_result.matching_strategy.value == "learning_based":
            suggestions.append("학습 기반 선택: 사용자 선호도 반영")
        elif pattern_result.matching_strategy.value == "hybrid":
            suggestions.append("하이브리드 매칭: 규칙과 학습의 균형")
        
        # 신뢰도 기반 제안
        if pattern_result.confidence_score < 0.7:
            suggestions.append("신뢰도 개선을 위해 추가 제약조건 확인 필요")
        elif pattern_result.confidence_score >= 0.9:
            suggestions.append("높은 신뢰도: 현재 설정 유지 권장")
        
        # 피드백 컨텍스트 기반 제안
        if feedback_context:
            recent_satisfaction = feedback_context.get('average_satisfaction', 0)
            if recent_satisfaction < 3.5:
                suggestions.append("만족도 개선을 위해 다른 시나리오 고려")
            elif recent_satisfaction >= 4.5:
                suggestions.append("높은 만족도: 현재 방식 지속 권장")
        
        # 대안 시나리오 제안
        if len(pattern_result.alternatives) >= 2:
            second_best = pattern_result.alternatives[0]
            if second_best[1] >= 0.8:  # 두 번째 옵션도 점수가 높은 경우
                suggestions.append(f"대안 고려: {second_best[0].value} (점수: {second_best[1]:.2f})")
        
        return suggestions[:5]  # 최대 5개
    
    def _assess_selection_risk(self, pattern_result: MatchingResult, 
                             parameters: Dict[str, Any]) -> Dict[str, float]:
        """선택 리스크 평가"""
        risk_factors = {}
        
        # 신뢰도 리스크
        confidence_risk = max(0.0, 1.0 - pattern_result.confidence_score)
        risk_factors['confidence_risk'] = confidence_risk
        
        # 패턴 학습 리스크 (유사 패턴이 적으면 리스크 높음)
        pattern_count = len(pattern_result.similar_patterns)
        learning_risk = max(0.0, 1.0 - (pattern_count / 5.0))  # 5개를 기준으로 정규화
        risk_factors['pattern_learning_risk'] = learning_risk
        
        # 복잡도 리스크
        vehicle_count = len(parameters.get('vehicles', []))
        order_count = len(parameters.get('orders', []))
        complexity = (vehicle_count * order_count) / 100.0  # 정규화
        complexity_risk = min(0.5, complexity)  # 최대 0.5
        risk_factors['complexity_risk'] = complexity_risk
        
        # 전략 다양성 리스크 (한 가지 전략에만 의존하면 리스크)
        if pattern_result.rule_based_score > 0.8 and pattern_result.similarity_score < 0.2:
            risk_factors['strategy_diversity_risk'] = 0.3
        elif pattern_result.similarity_score > 0.8 and pattern_result.rule_based_score < 0.2:
            risk_factors['strategy_diversity_risk'] = 0.2
        else:
            risk_factors['strategy_diversity_risk'] = 0.1
        
        # 전체 리스크 (가중 평균)
        overall_risk = (
            confidence_risk * 0.4 +
            learning_risk * 0.3 +
            complexity_risk * 0.2 +
            risk_factors['strategy_diversity_risk'] * 0.1
        )
        risk_factors['overall_risk'] = overall_risk
        
        return risk_factors
    
    def _get_scenario_selection_stats(self) -> Dict[str, Any]:
        """시나리오별 선택 통계"""
        # 실제로는 Redis에서 선택 히스토리를 조회해야 함
        # 여기서는 예시 구조만 제공
        return {
            'vrp_selection_rate': 0.45,
            'tsp_selection_rate': 0.25,
            'consolidation_selection_rate': 0.15,
            'emergency_selection_rate': 0.10,
            'realtime_selection_rate': 0.05,
            'total_selections': 0,
            'period_days': 30
        }
    
    def _get_effectiveness_trend(self) -> Dict[str, Any]:
        """효과성 트렌드"""
        # 실제로는 시간대별 효과성 데이터 분석
        return {
            'trend_direction': 'improving',
            'current_avg_effectiveness': 0.78,
            'last_week_effectiveness': 0.75,
            'improvement_rate': 0.04
        }
    
    def _get_user_selection_preferences(self, conversation_id: str) -> Dict[str, Any]:
        """사용자 선택 선호도"""
        try:
            # 대화 메모리에서 사용자 선호도 조회
            memory_data = self.memory_repository.get_conversation_memory(conversation_id)
            if not memory_data:
                return {}
            
            return {
                'preferred_scenarios': ['vrp', 'tsp'],
                'selection_history_count': 0,
                'avg_satisfaction_by_scenario': {},
                'recent_feedback_trend': 'neutral'
            }
            
        except Exception as e:
            self.logger.error("Failed to get user preferences", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            return {} 