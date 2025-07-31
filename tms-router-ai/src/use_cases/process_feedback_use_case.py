"""
피드백 처리 Use Case

사용자 피드백을 처리하고 학습하여 TMS 시스템의 지속적인 개선을 수행합니다.
"""
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.shared.logging_config import TmsLoggerMixin
from src.shared.exceptions import ValidationError, MemoryRepositoryError
from src.infrastructure.memory.conversation_manager import TmsConversationManager
from src.infrastructure.memory.feedback_processor import TmsFeedbackProcessor, LearningInsight


@dataclass
class FeedbackRequest:
    """피드백 요청"""
    feedback_id: Optional[str]
    conversation_id: str
    feedback_type: str  # positive, negative, suggestion, bug_report, feature_request
    feedback_content: str
    rating: int  # 1-5
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class FeedbackResponse:
    """피드백 처리 응답"""
    feedback_id: str
    conversation_id: str
    processing_status: str
    analysis_summary: Dict[str, Any]
    learning_insights: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    next_recommendations: List[str]
    processing_metadata: Dict[str, Any]


@dataclass
class FeedbackResult:
    """피드백 결과"""
    feedback_id: str
    status: str
    message: str
    conversation_summary: Optional[Dict[str, Any]] = None


class ProcessFeedbackUseCase(TmsLoggerMixin):
    """피드백 처리 Use Case"""
    
    def __init__(self,
                 conversation_manager: TmsConversationManager,
                 feedback_processor: TmsFeedbackProcessor):
        """
        Args:
            conversation_manager: 대화 메모리 관리자
            feedback_processor: 피드백 처리기
        """
        self.conversation_manager = conversation_manager
        self.feedback_processor = feedback_processor
        
        self.logger.info("ProcessFeedbackUseCase initialized")
    
    def execute(self, request: FeedbackRequest) -> FeedbackResponse:
        """피드백 처리 실행"""
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting feedback processing", extra={
                'conversation_id': request.conversation_id,
                'feedback_type': request.feedback_type,
                'rating': request.rating
            })
            
            # 1. 피드백 요청 검증
            self._validate_feedback_request(request)
            
            # 2. 피드백 ID 생성 (없는 경우)
            if not request.feedback_id:
                request.feedback_id = str(uuid.uuid4())
            
            # 3. 대화 컨텍스트 조회
            conversation_context = self._get_conversation_context(request.conversation_id)
            
            # 4. 피드백 처리 및 분석
            processed_feedback = self._process_and_analyze_feedback(request, conversation_context)
            
            # 5. 학습 인사이트 생성
            learning_insights = self._generate_learning_insights(request, processed_feedback)
            
            # 6. 개선 제안 생성
            improvement_suggestions = self._generate_improvement_suggestions(
                request, processed_feedback, learning_insights
            )
            
            # 7. 다음 최적화를 위한 추천사항 생성
            next_recommendations = self._generate_next_recommendations(
                request.conversation_id, processed_feedback
            )
            
            # 8. 대화 메모리 업데이트
            self._update_conversation_with_feedback(request, processed_feedback)
            
            # 9. 응답 생성
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = FeedbackResponse(
                feedback_id=request.feedback_id,
                conversation_id=request.conversation_id,
                processing_status="success",
                analysis_summary=self._create_analysis_summary(processed_feedback),
                learning_insights=[insight.to_dict() for insight in learning_insights],
                improvement_suggestions=improvement_suggestions,
                next_recommendations=next_recommendations,
                processing_metadata={
                    'processing_time_seconds': processing_time,
                    'analysis_version': '1.0',
                    'insights_generated': len(learning_insights),
                    'patterns_updated': len(processed_feedback.get('pattern_matches', [])),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            self.logger.info("Feedback processing completed successfully", extra={
                'feedback_id': request.feedback_id,
                'processing_time': processing_time,
                'insights_count': len(learning_insights),
                'sentiment_score': processed_feedback.get('content_analysis', {}).get('sentiment_score', 0)
            })
            
            return response
            
        except ValidationError as e:
            self.logger.error("Feedback validation failed", extra={
                'conversation_id': request.conversation_id,
                'error': str(e)
            })
            return self._create_error_response(request, f"Validation error: {e}")
            
        except MemoryRepositoryError as e:
            self.logger.error("Memory operation failed", extra={
                'conversation_id': request.conversation_id,
                'error': str(e)
            })
            return self._create_error_response(request, f"Memory error: {e}")
            
        except Exception as e:
            self.logger.error("Unexpected error during feedback processing", extra={
                'conversation_id': request.conversation_id,
                'error': str(e)
            })
            return self._create_error_response(request, f"Internal error: {e}")
    
    def get_feedback_analytics(self, conversation_id: Optional[str] = None, 
                              days: int = 30) -> Dict[str, Any]:
        """피드백 분석 조회"""
        try:
            # 기본 분석 데이터
            analytics = self.feedback_processor.memory_repository.get_feedback_analytics(
                conversation_id, days
            )
            
            # 학습 인사이트 추가
            insights = self.feedback_processor.get_learning_insights(conversation_id, days)
            
            # 최적화 추천사항 추가
            if conversation_id:
                recommendations = self.feedback_processor.get_optimization_recommendations(conversation_id)
            else:
                recommendations = {'recommendations': [], 'confidence': 0.0}
            
            # 종합 분석 결과
            comprehensive_analytics = {
                'basic_analytics': analytics,
                'learning_insights': [insight.to_dict() for insight in insights],
                'optimization_recommendations': recommendations,
                'analysis_metadata': {
                    'analyzed_at': datetime.now().isoformat(),
                    'conversation_id': conversation_id,
                    'period_days': days,
                    'insights_count': len(insights)
                }
            }
            
            self.logger.info("Generated feedback analytics", extra={
                'conversation_id': conversation_id,
                'days': days,
                'insights_count': len(insights)
            })
            
            return comprehensive_analytics
            
        except Exception as e:
            self.logger.error("Failed to get feedback analytics", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            return {
                'error': str(e),
                'basic_analytics': {},
                'learning_insights': [],
                'optimization_recommendations': {'recommendations': [], 'confidence': 0.0}
            }
    
    def get_conversation_insights(self, conversation_id: str) -> Dict[str, Any]:
        """특정 대화의 인사이트 조회"""
        try:
            # 대화 요약
            conversation_summary = self.conversation_manager.get_conversation_summary(conversation_id)
            
            # 학습 인사이트
            learning_insights = self.feedback_processor.get_learning_insights(conversation_id, 30)
            
            # 피드백 패턴
            feedback_patterns = self.feedback_processor.get_feedback_patterns(conversation_id)
            
            # 최적화 추천
            optimization_recommendations = self.feedback_processor.get_optimization_recommendations(conversation_id)
            
            insights = {
                'conversation_summary': conversation_summary,
                'learning_insights': [insight.to_dict() for insight in learning_insights],
                'feedback_patterns': [pattern.to_dict() for pattern in feedback_patterns],
                'optimization_recommendations': optimization_recommendations,
                'insights_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'conversation_id': conversation_id,
                    'total_insights': len(learning_insights),
                    'total_patterns': len(feedback_patterns)
                }
            }
            
            self.logger.info("Generated conversation insights", extra={
                'conversation_id': conversation_id,
                'insights_count': len(learning_insights),
                'patterns_count': len(feedback_patterns)
            })
            
            return insights
            
        except Exception as e:
            self.logger.error("Failed to get conversation insights", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            return {
                'error': str(e),
                'conversation_summary': {},
                'learning_insights': [],
                'feedback_patterns': [],
                'optimization_recommendations': {'recommendations': [], 'confidence': 0.0}
            }
    
    # Private methods
    
    def _validate_feedback_request(self, request: FeedbackRequest) -> None:
        """피드백 요청 검증"""
        if not request.conversation_id:
            raise ValidationError("conversation_id is required")
        
        if not request.feedback_type:
            raise ValidationError("feedback_type is required")
        
        valid_feedback_types = ['positive', 'negative', 'suggestion', 'bug_report', 'feature_request']
        if request.feedback_type not in valid_feedback_types:
            raise ValidationError(f"feedback_type must be one of: {valid_feedback_types}")
        
        if not isinstance(request.rating, int) or not (1 <= request.rating <= 5):
            raise ValidationError("rating must be an integer between 1 and 5")
        
        if not request.feedback_content or len(request.feedback_content.strip()) < 3:
            raise ValidationError("feedback_content must be at least 3 characters long")
    
    def _get_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """대화 컨텍스트 조회"""
        try:
            context = self.conversation_manager.get_or_create_conversation_context(conversation_id)
            return {
                'context': context,
                'recent_optimization_history': context.optimization_history[-3:],  # 최근 3개
                'feedback_history': context.feedback_history,
                'user_preferences': context.user_preferences,
                'learned_patterns': context.learned_patterns
            }
        except Exception as e:
            self.logger.warning("Failed to get conversation context", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            return {}
    
    def _process_and_analyze_feedback(self, 
                                    request: FeedbackRequest, 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """피드백 처리 및 분석"""
        feedback_data = {
            'feedback_id': request.feedback_id,
            'feedback_type': request.feedback_type,
            'feedback_content': request.feedback_content,
            'rating': request.rating
        }
        
        if request.metadata:
            feedback_data.update(request.metadata)
        
        # 피드백 처리기를 통한 분석
        processed_feedback = self.feedback_processor.process_feedback(
            request.conversation_id, 
            feedback_data
        )
        
        return processed_feedback
    
    def _generate_learning_insights(self, 
                                  request: FeedbackRequest,
                                  processed_feedback: Dict[str, Any]) -> List[LearningInsight]:
        """학습 인사이트 생성"""
        # 피드백 처리기에서 생성된 인사이트 조회
        insights = self.feedback_processor.get_learning_insights(request.conversation_id, 7)  # 최근 7일
        
        # 현재 피드백 기반 추가 인사이트 생성
        current_analysis = processed_feedback.get('content_analysis', {})
        
        # 고신뢰도 피드백에 대한 즉시 인사이트
        if (request.rating >= 4 and current_analysis.get('sentiment_score', 0) > 0.5) or \
           (request.rating <= 2 and current_analysis.get('sentiment_score', 0) < -0.3):
            
            immediate_insight = LearningInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='immediate',
                title=f"즉시 피드백 인사이트 (평점: {request.rating})",
                description=f"강한 감정의 피드백이 수신되었습니다. "
                           f"감정 점수: {current_analysis.get('sentiment_score', 0):.2f}",
                confidence_score=abs(current_analysis.get('sentiment_score', 0)),
                supporting_feedback_count=1,
                actionable_recommendations=self._generate_immediate_recommendations(request, current_analysis),
                created_at=datetime.now()
            )
            insights.append(immediate_insight)
        
        return insights
    
    def _generate_improvement_suggestions(self, 
                                        request: FeedbackRequest,
                                        processed_feedback: Dict[str, Any],
                                        insights: List[LearningInsight]) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        
        # 처리된 피드백에서 제안 추출
        optimization_suggestions = processed_feedback.get('optimization_suggestions', [])
        suggestions.extend(optimization_suggestions)
        
        # 인사이트 기반 제안 추가
        for insight in insights:
            if insight.insight_type in ['preference', 'pattern'] and insight.confidence_score > 0.7:
                suggestions.extend(insight.actionable_recommendations)
        
        # 평점 기반 제안
        if request.rating <= 2:
            suggestions.append("사용자 요구사항 재분석 및 최적화 알고리즘 개선 필요")
        elif request.rating >= 4:
            suggestions.append("현재 최적화 방식의 성공 패턴을 다른 시나리오에도 적용")
        
        # 중복 제거 및 상위 5개만 반환
        unique_suggestions = list(dict.fromkeys(suggestions))  # 순서 유지하면서 중복 제거
        return unique_suggestions[:5]
    
    def _generate_next_recommendations(self, 
                                     conversation_id: str,
                                     processed_feedback: Dict[str, Any]) -> List[str]:
        """다음 최적화를 위한 추천사항"""
        try:
            recommendations_data = self.feedback_processor.get_optimization_recommendations(conversation_id)
            
            base_recommendations = recommendations_data.get('recommendations', [])
            confidence = recommendations_data.get('confidence', 0.0)
            
            # 신뢰도 기반 추가 추천
            if confidence >= 0.8:
                base_recommendations.append("높은 신뢰도의 학습 패턴을 적극 활용하세요")
            elif confidence <= 0.3:
                base_recommendations.append("더 많은 피드백을 통해 학습 정확도를 높이세요")
            
            # 현재 피드백 분석 기반 추천
            content_analysis = processed_feedback.get('content_analysis', {})
            detected_topics = content_analysis.get('detected_topics', [])
            
            if 'cost' in detected_topics:
                base_recommendations.append("비용 최적화에 더 집중하여 다음 경로를 계획하세요")
            if 'time' in detected_topics:
                base_recommendations.append("시간 효율성을 우선 고려하여 다음 최적화를 수행하세요")
            
            return base_recommendations[:5]  # 상위 5개만 반환
            
        except Exception as e:
            self.logger.warning("Failed to generate next recommendations", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            return ["피드백을 반영하여 다음 최적화를 개선하겠습니다"]
    
    def _update_conversation_with_feedback(self, 
                                         request: FeedbackRequest,
                                         processed_feedback: Dict[str, Any]) -> None:
        """피드백으로 대화 컨텍스트 업데이트"""
        try:
            # 대화 메모리 관리자를 통한 피드백 처리
            feedback_data = {
                'feedback_type': request.feedback_type,
                'feedback_content': request.feedback_content,
                'rating': request.rating
            }
            
            if request.metadata:
                feedback_data.update(request.metadata)
            
            analysis = self.conversation_manager.process_feedback(
                request.conversation_id,
                feedback_data
            )
            
            self.logger.debug("Updated conversation context with feedback", extra={
                'conversation_id': request.conversation_id,
                'feedback_id': request.feedback_id,
                'sentiment_score': analysis.sentiment_score
            })
            
        except Exception as e:
            self.logger.warning("Failed to update conversation with feedback", extra={
                'conversation_id': request.conversation_id,
                'error': str(e)
            })
    
    def _create_analysis_summary(self, processed_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """분석 요약 생성"""
        content_analysis = processed_feedback.get('content_analysis', {})
        context_analysis = processed_feedback.get('context_analysis', {})
        
        return {
            'sentiment': {
                'level': content_analysis.get('sentiment_level', 'neutral'),
                'score': content_analysis.get('sentiment_score', 0.0)
            },
            'topics': {
                'detected': content_analysis.get('detected_topics', []),
                'count': len(content_analysis.get('detected_topics', []))
            },
            'urgency': content_analysis.get('urgency', 'low'),
            'actionability': content_analysis.get('actionability', 'low'),
            'context': {
                'timing': context_analysis.get('feedback_timing', 'unknown'),
                'conversation_length': context_analysis.get('conversation_length', 0),
                'has_recent_optimization': context_analysis.get('recent_optimization_count', 0) > 0
            },
            'patterns_matched': len(processed_feedback.get('pattern_matches', []))
        }
    
    def _generate_immediate_recommendations(self, 
                                          request: FeedbackRequest,
                                          analysis: Dict[str, Any]) -> List[str]:
        """즉시 피드백에 대한 추천사항"""
        recommendations = []
        
        if request.rating >= 4:
            recommendations.extend([
                "성공적인 최적화 패턴을 다른 유사 시나리오에 적용",
                "현재 설정을 기본 선호도로 저장"
            ])
        elif request.rating <= 2:
            recommendations.extend([
                "최적화 알고리즘 파라미터 재조정 필요",
                "사용자 요구사항 재확인 및 제약조건 재검토"
            ])
        
        # 토픽 기반 추천
        topics = analysis.get('detected_topics', [])
        if 'cost' in topics:
            recommendations.append("비용 최적화 가중치 조정")
        if 'time' in topics:
            recommendations.append("시간 효율성 알고리즘 개선")
        
        return recommendations
    
    def _create_error_response(self, request: FeedbackRequest, error_message: str) -> FeedbackResponse:
        """에러 응답 생성"""
        return FeedbackResponse(
            feedback_id=request.feedback_id or str(uuid.uuid4()),
            conversation_id=request.conversation_id,
            processing_status="error",
            analysis_summary={},
            learning_insights=[],
            improvement_suggestions=[],
            next_recommendations=[],
            processing_metadata={
                'error': True,
                'error_message': error_message,
                'timestamp': datetime.now().isoformat()
            }
        ) 