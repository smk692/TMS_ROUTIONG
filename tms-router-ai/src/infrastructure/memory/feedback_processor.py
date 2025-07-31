"""
피드백 처리 및 학습 시스템

사용자 피드백을 분석하고 학습하여 TMS 최적화 성능을 지속적으로 개선합니다.
"""
import json
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
from enum import Enum

from src.shared.logging_config import TmsLoggerMixin
from src.shared.exceptions import MemoryRepositoryError
from src.infrastructure.memory.redis_memory_repository import RedisMemoryRepository


class FeedbackType(Enum):
    """피드백 유형"""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    SUGGESTION = "suggestion"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"


class SentimentLevel(Enum):
    """감정 레벨"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class FeedbackPattern:
    """피드백 패턴"""
    pattern_id: str
    pattern_type: str  # topic, sentiment, scenario
    keywords: List[str]
    context_conditions: Dict[str, Any]
    success_rate: float
    occurrence_count: int
    last_seen: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'keywords': self.keywords,
            'context_conditions': self.context_conditions,
            'success_rate': self.success_rate,
            'occurrence_count': self.occurrence_count,
            'last_seen': self.last_seen.isoformat()
        }


@dataclass
class LearningInsight:
    """학습 인사이트"""
    insight_id: str
    insight_type: str  # preference, pattern, optimization
    title: str
    description: str
    confidence_score: float
    supporting_feedback_count: int
    actionable_recommendations: List[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'insight_id': self.insight_id,
            'insight_type': self.insight_type,
            'title': self.title,
            'description': self.description,
            'confidence_score': self.confidence_score,
            'supporting_feedback_count': self.supporting_feedback_count,
            'actionable_recommendations': self.actionable_recommendations,
            'created_at': self.created_at.isoformat()
        }


class TmsFeedbackProcessor(TmsLoggerMixin):
    """TMS 피드백 처리 및 학습 엔진"""
    
    def __init__(self, memory_repository: RedisMemoryRepository):
        """
        Args:
            memory_repository: Redis 메모리 저장소
        """
        super().__init__()  # TmsLoggerMixin 초기화
        self.memory_repository = memory_repository
        
        # TMS 도메인 특화 키워드 사전
        self.tms_keywords = {
            'cost': ['비용', '가격', '요금', '돈', '경제적', '저렴', '비싸', '절약'],
            'time': ['시간', '속도', '빠른', '느린', '지연', '신속', '즉시', '소요시간'],
            'distance': ['거리', '킬로미터', 'km', '경로', '루트', '길', '도로'],
            'efficiency': ['효율', '최적', '개선', '향상', '성능', '품질'],
            'vehicle': ['차량', '트럭', '운송수단', '배송차', '운전자', '기사'],
            'delivery': ['배송', '배달', '운송', '납품', '픽업', '집하'],
            'satisfaction': ['만족', '좋다', '훌륭', '완벽', '우수', '최고'],
            'dissatisfaction': ['불만', '실망', '나쁘다', '문제', '오류', '개선필요']
        }
        
        # 감정 분석 키워드
        self.sentiment_keywords = {
            'very_positive': ['완벽', '최고', '훌륭', '대단히', '매우 만족', '탁월'],
            'positive': ['좋다', '만족', '우수', '효과적', '성공적', '도움'],
            'neutral': ['보통', '그저그런', '무난', '적당'],
            'negative': ['아쉽다', '부족', '개선', '문제', '불편', '어려움'],
            'very_negative': ['실망', '최악', '불만', '심각', '매우 나쁘다', '실패']
        }
        
        self.logger.info("TmsFeedbackProcessor initialized")
    
    def process_feedback(self, conversation_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """피드백 처리 및 분석"""
        try:
            feedback_id = str(uuid.uuid4())
            
            # 기본 피드백 분석
            analysis_result = self._analyze_feedback_content(feedback_data)
            
            # 컨텍스트 기반 분석
            context_analysis = self._analyze_feedback_context(conversation_id, feedback_data)
            
            # 패턴 매칭 및 업데이트
            pattern_matches = self._match_and_update_patterns(conversation_id, feedback_data, analysis_result)
            
            # 학습 인사이트 생성
            insights = self._generate_learning_insights(conversation_id, feedback_data, analysis_result)
            
            # 최적화 제안 생성
            optimization_suggestions = self._generate_optimization_suggestions(
                feedback_data, analysis_result, context_analysis
            )
            
            # 종합 분석 결과
            processed_feedback = {
                'feedback_id': feedback_id,
                'conversation_id': conversation_id,
                'processed_at': datetime.now().isoformat(),
                'original_feedback': feedback_data,
                'content_analysis': analysis_result,
                'context_analysis': context_analysis,
                'pattern_matches': pattern_matches,
                'learning_insights': [insight.to_dict() for insight in insights],
                'optimization_suggestions': optimization_suggestions,
                'processing_metadata': {
                    'version': '1.0',
                    'processor': 'TmsFeedbackProcessor'
                }
            }
            
            # Redis에 저장
            self._save_processed_feedback(feedback_id, processed_feedback)
            
            # 피드백 통계 업데이트
            self._update_feedback_statistics(conversation_id, analysis_result)
            
            self.logger.info("Feedback processed successfully", extra={
                'conversation_id': conversation_id,
                'feedback_id': feedback_id,
                'sentiment': analysis_result.get('sentiment_level'),
                'topic_count': len(analysis_result.get('detected_topics', [])),
                'insight_count': len(insights)
            })
            
            return processed_feedback
            
        except Exception as e:
            self.logger.error("Failed to process feedback", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            raise MemoryRepositoryError(f"Feedback processing failed: {e}")
    
    def get_learning_insights(self, conversation_id: Optional[str] = None, 
                            days: int = 30) -> List[LearningInsight]:
        """학습 인사이트 조회"""
        try:
            # 피드백 분석 데이터 조회
            analytics = self.memory_repository.get_feedback_analytics(conversation_id, days)
            
            insights = []
            
            # 선호도 패턴 인사이트
            preference_insights = self._analyze_preference_patterns(analytics)
            insights.extend(preference_insights)
            
            # 만족도 트렌드 인사이트
            satisfaction_insights = self._analyze_satisfaction_trends(analytics)
            insights.extend(satisfaction_insights)
            
            # 문제 패턴 인사이트
            problem_insights = self._analyze_problem_patterns(analytics)
            insights.extend(problem_insights)
            
            # 개선 기회 인사이트
            improvement_insights = self._analyze_improvement_opportunities(analytics)
            insights.extend(improvement_insights)
            
            # 신뢰도 기준으로 정렬
            insights.sort(key=lambda x: x.confidence_score, reverse=True)
            
            self.logger.info("Generated learning insights", extra={
                'conversation_id': conversation_id,
                'days': days,
                'total_insights': len(insights)
            })
            
            return insights[:10]  # 상위 10개 인사이트
            
        except Exception as e:
            self.logger.error("Failed to get learning insights", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            return []
    
    def get_feedback_patterns(self, conversation_id: Optional[str] = None) -> List[FeedbackPattern]:
        """피드백 패턴 조회"""
        try:
            patterns_key = f"feedback_patterns:{conversation_id}" if conversation_id else "feedback_patterns:global"
            
            stored_patterns = self.memory_repository.redis_client.hgetall(patterns_key)
            
            patterns = []
            for pattern_id, pattern_data in stored_patterns.items():
                pattern_dict = json.loads(pattern_data)
                pattern = FeedbackPattern(
                    pattern_id=pattern_dict['pattern_id'],
                    pattern_type=pattern_dict['pattern_type'],
                    keywords=pattern_dict['keywords'],
                    context_conditions=pattern_dict['context_conditions'],
                    success_rate=pattern_dict['success_rate'],
                    occurrence_count=pattern_dict['occurrence_count'],
                    last_seen=datetime.fromisoformat(pattern_dict['last_seen'])
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error("Failed to get feedback patterns", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            return []
    
    def get_optimization_recommendations(self, conversation_id: str) -> Dict[str, Any]:
        """최적화 추천사항 조회"""
        try:
            # 최근 피드백 분석
            recent_feedback = self.memory_repository.get_conversation_messages(
                conversation_id, limit=50
            )
            feedback_messages = [
                msg for msg in recent_feedback 
                if msg.get('message_type') == 'feedback'
            ]
            
            if not feedback_messages:
                return {
                    'conversation_id': conversation_id,
                    'recommendations': [],
                    'confidence': 0.0,
                    'analysis_date': datetime.now().isoformat()
                }
            
            # 피드백 패턴 분석
            patterns = self._extract_feedback_patterns(feedback_messages)
            
            # 추천사항 생성
            recommendations = []
            
            # 1. 선호도 기반 추천
            preference_recommendations = self._generate_preference_recommendations(patterns)
            recommendations.extend(preference_recommendations)
            
            # 2. 문제점 개선 추천
            problem_recommendations = self._generate_problem_solution_recommendations(patterns)
            recommendations.extend(problem_recommendations)
            
            # 3. 성능 향상 추천
            performance_recommendations = self._generate_performance_recommendations(patterns)
            recommendations.extend(performance_recommendations)
            
            # 신뢰도 계산
            total_feedback = len(feedback_messages)
            confidence = min(total_feedback / 10.0, 1.0)  # 10개 피드백에서 최대 신뢰도
            
            result = {
                'conversation_id': conversation_id,
                'recommendations': recommendations[:5],  # 상위 5개
                'confidence': round(confidence, 2),
                'analysis_date': datetime.now().isoformat(),
                'feedback_count': total_feedback,
                'patterns_found': len(patterns)
            }
            
            self.logger.info("Generated optimization recommendations", extra={
                'conversation_id': conversation_id,
                'recommendation_count': len(recommendations),
                'confidence': confidence
            })
            
            return result
            
        except Exception as e:
            self.logger.error("Failed to get optimization recommendations", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            return {
                'conversation_id': conversation_id,
                'recommendations': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    # Private helper methods
    
    def _analyze_feedback_content(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """피드백 내용 분석"""
        content = feedback_data.get('feedback_content', '')
        rating = feedback_data.get('rating', 0)
        
        # 감정 분석
        sentiment_level = self._detect_sentiment(content)
        sentiment_score = self._calculate_sentiment_score(content, sentiment_level)
        
        # 토픽 감지
        detected_topics = self._detect_topics(content)
        
        # 키워드 추출
        keywords = self._extract_keywords(content)
        
        # 긴급도 평가
        urgency = self._assess_urgency(content, rating)
        
        # 실행 가능성 평가
        actionability = self._assess_actionability(content, detected_topics)
        
        return {
            'sentiment_level': sentiment_level.value,
            'sentiment_score': sentiment_score,
            'detected_topics': detected_topics,
            'keywords': keywords,
            'urgency': urgency,
            'actionability': actionability,
            'content_length': len(content),
            'has_specific_details': self._has_specific_details(content)
        }
    
    def _analyze_feedback_context(self, conversation_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """피드백 컨텍스트 분석"""
        # 최근 최적화 결과 조회
        recent_messages = self.memory_repository.get_conversation_messages(conversation_id, limit=10)
        
        optimization_messages = [
            msg for msg in recent_messages 
            if msg.get('message_type') == 'assistant' and 
               msg.get('metadata', {}).get('optimization_result')
        ]
        
        context_analysis = {
            'recent_optimization_count': len(optimization_messages),
            'feedback_timing': 'immediate' if len(recent_messages) <= 2 else 'delayed',
            'conversation_length': len(recent_messages),
            'has_previous_feedback': any(
                msg.get('message_type') == 'feedback' for msg in recent_messages[1:]
            )
        }
        
        # 최근 최적화 결과와 피드백 연관성 분석
        if optimization_messages:
            latest_optimization = optimization_messages[0]
            optimization_metadata = latest_optimization.get('metadata', {}).get('optimization_result', {})
            
            context_analysis.update({
                'optimization_confidence': optimization_metadata.get('confidence_score', 0),
                'optimization_scenario': optimization_metadata.get('scenario_type', 'unknown'),
                'optimization_complexity': self._assess_optimization_complexity(optimization_metadata)
            })
        
        return context_analysis
    
    def _detect_sentiment(self, content: str) -> SentimentLevel:
        """감정 감지"""
        content_lower = content.lower()
        
        # 각 감정 레벨별 점수 계산
        sentiment_scores = {}
        for level, keywords in self.sentiment_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            sentiment_scores[level] = score
        
        # 가장 높은 점수의 감정 레벨 반환
        if max(sentiment_scores.values()) == 0:
            return SentimentLevel.NEUTRAL
        
        max_sentiment = max(sentiment_scores.keys(), key=lambda x: sentiment_scores[x])
        return SentimentLevel(max_sentiment)
    
    def _calculate_sentiment_score(self, content: str, sentiment_level: SentimentLevel) -> float:
        """감정 점수 계산 (-1.0 ~ 1.0)"""
        sentiment_mapping = {
            SentimentLevel.VERY_NEGATIVE: -1.0,
            SentimentLevel.NEGATIVE: -0.5,
            SentimentLevel.NEUTRAL: 0.0,
            SentimentLevel.POSITIVE: 0.5,
            SentimentLevel.VERY_POSITIVE: 1.0
        }
        return sentiment_mapping.get(sentiment_level, 0.0)
    
    def _detect_topics(self, content: str) -> List[str]:
        """토픽 감지"""
        content_lower = content.lower()
        detected_topics = []
        
        for topic, keywords in self.tms_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    def _extract_keywords(self, content: str) -> List[str]:
        """키워드 추출"""
        # 단순한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
        words = re.findall(r'\b\w+\b', content.lower())
        
        # TMS 관련 키워드 필터링
        tms_relevant_words = []
        all_tms_keywords = []
        for keywords in self.tms_keywords.values():
            all_tms_keywords.extend(keywords)
        
        for word in words:
            if word in all_tms_keywords or len(word) >= 3:
                tms_relevant_words.append(word)
        
        # 빈도 기반 상위 키워드 반환
        word_counts = Counter(tms_relevant_words)
        return [word for word, count in word_counts.most_common(10)]
    
    def _assess_urgency(self, content: str, rating: int) -> str:
        """긴급도 평가"""
        urgency_keywords = {
            'high': ['긴급', '즉시', '심각', '중요', '빨리', '시급'],
            'medium': ['필요', '개선', '요청', '바라'],
            'low': ['나중', '언제', '여유', '참고']
        }
        
        content_lower = content.lower()
        
        # 키워드 기반 긴급도
        for level, keywords in urgency_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                if level == 'high':
                    return 'high'
        
        # 평점 기반 긴급도
        if rating <= 2:
            return 'high'
        elif rating <= 3:
            return 'medium'
        else:
            return 'low'
    
    def _assess_actionability(self, content: str, topics: List[str]) -> str:
        """실행 가능성 평가"""
        # 구체적인 토픽이 많을수록 실행 가능성 높음
        if len(topics) >= 3:
            return 'high'
        elif len(topics) >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _has_specific_details(self, content: str) -> bool:
        """구체적 세부사항 포함 여부"""
        # 숫자, 구체적 요청 등이 포함되어 있는지 확인
        has_numbers = bool(re.search(r'\d+', content))
        has_specific_requests = any(keyword in content.lower() for keyword in [
            '더', '덜', '줄여', '늘려', '개선', '수정', '변경'
        ])
        
        return has_numbers or has_specific_requests or len(content.split()) >= 10
    
    def _match_and_update_patterns(self, conversation_id: str, feedback_data: Dict[str, Any], 
                                  analysis: Dict[str, Any]) -> List[str]:
        """패턴 매칭 및 업데이트"""
        matched_patterns = []
        
        # 토픽 기반 패턴
        for topic in analysis['detected_topics']:
            pattern_id = f"topic_{topic}"
            self._update_pattern(conversation_id, pattern_id, 'topic', [topic], feedback_data)
            matched_patterns.append(pattern_id)
        
        # 감정 기반 패턴
        sentiment = analysis['sentiment_level']
        if sentiment != 'neutral':
            pattern_id = f"sentiment_{sentiment}"
            self._update_pattern(conversation_id, pattern_id, 'sentiment', [sentiment], feedback_data)
            matched_patterns.append(pattern_id)
        
        return matched_patterns
    
    def _update_pattern(self, conversation_id: str, pattern_id: str, pattern_type: str, 
                       keywords: List[str], feedback_data: Dict[str, Any]):
        """패턴 업데이트"""
        patterns_key = f"feedback_patterns:{conversation_id}"
        
        # 기존 패턴 조회
        existing_pattern = self.memory_repository.redis_client.hget(patterns_key, pattern_id)
        
        if existing_pattern:
            pattern_dict = json.loads(existing_pattern)
            pattern_dict['occurrence_count'] += 1
            pattern_dict['last_seen'] = datetime.now().isoformat()
            
            # 성공률 업데이트 (평점 기반)
            rating = feedback_data.get('rating', 0)
            current_success = pattern_dict['success_rate'] * (pattern_dict['occurrence_count'] - 1)
            new_success = 1 if rating >= 4 else 0
            pattern_dict['success_rate'] = (current_success + new_success) / pattern_dict['occurrence_count']
        else:
            # 새 패턴 생성
            rating = feedback_data.get('rating', 0)
            pattern_dict = {
                'pattern_id': pattern_id,
                'pattern_type': pattern_type,
                'keywords': keywords,
                'context_conditions': {},
                'success_rate': 1.0 if rating >= 4 else 0.0,
                'occurrence_count': 1,
                'last_seen': datetime.now().isoformat()
            }
        
        # Redis에 저장
        self.memory_repository.redis_client.hset(
            patterns_key, 
            pattern_id, 
            json.dumps(pattern_dict)
        )
    
    def _generate_learning_insights(self, conversation_id: str, feedback_data: Dict[str, Any], 
                                  analysis: Dict[str, Any]) -> List[LearningInsight]:
        """학습 인사이트 생성"""
        insights = []
        
        # 선호도 인사이트
        if analysis['sentiment_score'] > 0.5 and analysis['detected_topics']:
            for topic in analysis['detected_topics']:
                insight = LearningInsight(
                    insight_id=str(uuid.uuid4()),
                    insight_type='preference',
                    title=f"{topic.title()} 최적화 선호",
                    description=f"사용자가 {topic} 관련 최적화에 높은 만족도를 보입니다",
                    confidence_score=min(analysis['sentiment_score'] + 0.3, 1.0),
                    supporting_feedback_count=1,
                    actionable_recommendations=[
                        f"{topic} 가중치를 높여 최적화하세요",
                        f"{topic} 관련 상세 정보를 더 제공하세요"
                    ],
                    created_at=datetime.now()
                )
                insights.append(insight)
        
        # 문제점 인사이트
        if analysis['sentiment_score'] < -0.3 and analysis['urgency'] == 'high':
            insight = LearningInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='pattern',
                title="긴급 개선 필요 패턴",
                description="높은 긴급도의 부정적 피드백 패턴이 감지되었습니다",
                confidence_score=abs(analysis['sentiment_score']),
                supporting_feedback_count=1,
                actionable_recommendations=[
                    "최적화 알고리즘 재검토 필요",
                    "추가 제약조건 고려 필요",
                    "사용자 요구사항 재확인 필요"
                ],
                created_at=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    def _generate_optimization_suggestions(self, feedback_data: Dict[str, Any], 
                                         content_analysis: Dict[str, Any],
                                         context_analysis: Dict[str, Any]) -> List[str]:
        """최적화 제안 생성"""
        suggestions = []
        
        rating = feedback_data.get('rating', 0)
        topics = content_analysis.get('detected_topics', [])
        sentiment_score = content_analysis.get('sentiment_score', 0)
        
        # 평점 기반 제안
        if rating <= 2:
            suggestions.append("최적화 알고리즘 전반적 재검토 필요")
            if sentiment_score < -0.5:
                suggestions.append("사용자 요구사항과 현재 최적화 방향성 재확인 필요")
        
        # 토픽 기반 제안
        if 'cost' in topics:
            if rating >= 4:
                suggestions.append("비용 최적화 가중치 유지 또는 증대")
            else:
                suggestions.append("비용 계산 방식 재검토 및 개선")
        
        if 'time' in topics:
            if rating >= 4:
                suggestions.append("시간 효율성 최적화 방식 유지")
            else:
                suggestions.append("시간 예측 정확도 향상 필요")
        
        if 'distance' in topics:
            if rating >= 4:
                suggestions.append("거리 기반 경로 선택 알고리즘 효과적")
            else:
                suggestions.append("경로 계산 방식 개선 필요")
        
        # 컨텍스트 기반 제안
        if context_analysis.get('optimization_confidence', 0) < 0.7:
            suggestions.append("최적화 신뢰도 향상을 위한 추가 데이터 고려")
        
        if context_analysis.get('feedback_timing') == 'immediate' and rating <= 3:
            suggestions.append("즉각적인 결과 개선을 위한 실시간 조정 필요")
        
        return suggestions[:5]  # 상위 5개 제안
    
    def _save_processed_feedback(self, feedback_id: str, processed_data: Dict[str, Any]):
        """처리된 피드백 저장"""
        key = f"processed_feedback:{feedback_id}"
        self.memory_repository.redis_client.setex(
            key,
            timedelta(days=90),  # 90일 보관
            json.dumps(processed_data, ensure_ascii=False)
        )
    
    def _update_feedback_statistics(self, conversation_id: str, analysis: Dict[str, Any]):
        """피드백 통계 업데이트"""
        stats_key = f"feedback_stats:{conversation_id}"
        
        # 기존 통계 조회
        existing_stats = self.memory_repository.redis_client.get(stats_key)
        if existing_stats:
            stats = json.loads(existing_stats)
        else:
            stats = {
                'total_feedback': 0,
                'sentiment_distribution': defaultdict(int),
                'topic_distribution': defaultdict(int),
                'average_sentiment': 0.0,
                'last_updated': datetime.now().isoformat()
            }
        
        # 통계 업데이트
        stats['total_feedback'] += 1
        stats['sentiment_distribution'][analysis['sentiment_level']] += 1
        
        for topic in analysis['detected_topics']:
            stats['topic_distribution'][topic] += 1
        
        # 평균 감정 점수 업데이트
        current_avg = stats['average_sentiment']
        total_count = stats['total_feedback']
        new_sentiment = analysis['sentiment_score']
        stats['average_sentiment'] = ((current_avg * (total_count - 1)) + new_sentiment) / total_count
        
        stats['last_updated'] = datetime.now().isoformat()
        
        # Redis에 저장
        self.memory_repository.redis_client.setex(
            stats_key,
            timedelta(days=365),  # 1년 보관
            json.dumps(stats, ensure_ascii=False)
        )
    
    def _analyze_preference_patterns(self, analytics: Dict[str, Any]) -> List[LearningInsight]:
        """선호도 패턴 분석"""
        insights = []
        
        # 구현 예시 - 실제로는 더 복잡한 분석
        if analytics.get('total_feedback', 0) >= 5:
            insight = LearningInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='preference',
                title="사용자 선호도 패턴 감지",
                description="충분한 피드백이 수집되어 선호도 패턴을 분석할 수 있습니다",
                confidence_score=0.8,
                supporting_feedback_count=analytics.get('total_feedback', 0),
                actionable_recommendations=[
                    "감지된 선호도 패턴을 최적화에 반영",
                    "선호하는 최적화 방식의 가중치 증대"
                ],
                created_at=datetime.now()
            )
            insights.append(insight)
        
        return insights
    
    def _analyze_satisfaction_trends(self, analytics: Dict[str, Any]) -> List[LearningInsight]:
        """만족도 트렌드 분석"""
        # 구현 생략 (실제로는 시간대별 만족도 변화 분석)
        return []
    
    def _analyze_problem_patterns(self, analytics: Dict[str, Any]) -> List[LearningInsight]:
        """문제 패턴 분석"""
        # 구현 생략 (실제로는 반복되는 문제 패턴 감지)
        return []
    
    def _analyze_improvement_opportunities(self, analytics: Dict[str, Any]) -> List[LearningInsight]:
        """개선 기회 분석"""
        # 구현 생략 (실제로는 개선 가능 영역 식별)
        return []
    
    def _extract_feedback_patterns(self, feedback_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """피드백 패턴 추출"""
        # 구현 생략
        return {}
    
    def _generate_preference_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """선호도 기반 추천 생성"""
        # 구현 생략
        return []
    
    def _generate_problem_solution_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """문제 해결 추천 생성"""
        # 구현 생략
        return []
    
    def _generate_performance_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """성능 향상 추천 생성"""
        # 구현 생략
        return []
    
    def _assess_optimization_complexity(self, optimization_metadata: Dict[str, Any]) -> str:
        """최적화 복잡도 평가"""
        route_count = len(optimization_metadata.get('routes', []))
        
        if route_count <= 2:
            return 'simple'
        elif route_count <= 5:
            return 'medium'
        else:
            return 'complex' 