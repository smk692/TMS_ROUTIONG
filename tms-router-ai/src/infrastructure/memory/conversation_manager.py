"""
대화 메모리 및 컨텍스트 관리자

LangChain Memory 컴포넌트와 Redis를 통합하여 
TMS 배차 최적화 대화의 컨텍스트를 지능적으로 관리합니다.
"""
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory.chat_memory import BaseChatMemory

from src.shared.logging_config import TmsLoggerMixin
from src.shared.exceptions import MemoryRepositoryError
from src.shared.constants import MemoryConstants
from src.infrastructure.memory.redis_memory_repository import RedisMemoryRepository


@dataclass
class ConversationContext:
    """대화 컨텍스트 데이터 클래스"""
    conversation_id: str
    user_preferences: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    feedback_history: List[Dict[str, Any]]
    learned_patterns: Dict[str, Any]
    session_metadata: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """딕셔너리에서 생성"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


@dataclass
class FeedbackAnalysis:
    """피드백 분석 결과"""
    feedback_id: str
    conversation_id: str
    feedback_type: str
    rating: int
    content: str
    sentiment_score: float
    key_topics: List[str]
    improvement_suggestions: List[str]
    pattern_updates: Dict[str, Any]
    analyzed_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['analyzed_at'] = self.analyzed_at.isoformat()
        return data


class TmsConversationManager(TmsLoggerMixin):
    """TMS 전용 대화 메모리 관리자"""
    
    def __init__(self, 
                 memory_repository: RedisMemoryRepository,
                 window_size: int = 20,
                 max_token_limit: int = 2000):
        """
        Args:
            memory_repository: Redis 메모리 저장소
            window_size: 대화 윈도우 크기
            max_token_limit: 요약 메모리 토큰 제한
        """
        super().__init__()  # TmsLoggerMixin 초기화
        self.memory_repository = memory_repository
        self.window_size = window_size
        self.max_token_limit = max_token_limit
        
        # LangChain 메모리 인스턴스 캐시
        self._memory_cache: Dict[str, BaseChatMemory] = {}
        
        self.logger.info("TmsConversationManager initialized", extra={
            'window_size': window_size,
            'max_token_limit': max_token_limit
        })
    
    def get_or_create_conversation_context(self, conversation_id: str) -> ConversationContext:
        """대화 컨텍스트 조회 또는 생성"""
        try:
            # 기존 컨텍스트 조회
            memory_data = self.memory_repository.get_conversation_memory(conversation_id)
            
            if memory_data and 'context' in memory_data:
                context = ConversationContext.from_dict(memory_data['context'])
                self.logger.debug("Retrieved existing conversation context", extra={
                    'conversation_id': conversation_id,
                    'created_at': context.created_at
                })
                return context
            
            # 새 컨텍스트 생성
            context = ConversationContext(
                conversation_id=conversation_id,
                user_preferences={
                    'optimization_priority': 'balanced',  # distance, time, cost, balanced
                    'preferred_algorithms': [],
                    'risk_tolerance': 'medium',  # low, medium, high
                    'feedback_style': 'detailed'  # brief, detailed, technical
                },
                optimization_history=[],
                feedback_history=[],
                learned_patterns={
                    'successful_scenarios': [],
                    'problem_patterns': [],
                    'preference_weights': {
                        'distance': 0.33,
                        'time': 0.33,
                        'cost': 0.34
                    }
                },
                session_metadata={
                    'total_requests': 0,
                    'successful_optimizations': 0,
                    'average_satisfaction': 0.0,
                    'common_scenarios': []
                },
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            # Redis에 저장
            self._save_context(context)
            
            self.logger.info("Created new conversation context", extra={
                'conversation_id': conversation_id
            })
            
            return context
            
        except Exception as e:
            self.logger.error("Failed to get or create conversation context", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            raise MemoryRepositoryError(f"Context management failed: {e}")
    
    def get_langchain_memory(self, conversation_id: str) -> BaseChatMemory:
        """LangChain 메모리 인스턴스 조회"""
        if conversation_id in self._memory_cache:
            return self._memory_cache[conversation_id]
        
        # 대화 메시지 조회
        messages = self.memory_repository.get_conversation_messages(
            conversation_id, 
            limit=self.window_size * 2  # 여유분 포함
        )
        
        # ConversationBufferWindowMemory 생성
        memory = ConversationBufferWindowMemory(
            k=self.window_size,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # 기존 메시지 복원
        for msg_data in reversed(messages):  # 시간 순서로 복원
            if msg_data['message_type'] == 'user':
                memory.chat_memory.add_user_message(msg_data['content'])
            elif msg_data['message_type'] == 'assistant':
                memory.chat_memory.add_ai_message(msg_data['content'])
        
        # 캐시에 저장
        self._memory_cache[conversation_id] = memory
        
        self.logger.debug("Created LangChain memory instance", extra={
            'conversation_id': conversation_id,
            'message_count': len(messages)
        })
        
        return memory
    
    def add_user_message(self, conversation_id: str, content: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """사용자 메시지 추가"""
        message_id = str(uuid.uuid4())
        
        message_data = {
            'id': message_id,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'user',
            'content': content,
            'metadata': metadata or {}
        }
        
        # Redis에 저장
        saved_id = self.memory_repository.save_conversation_message(message_data)
        
        # LangChain 메모리 업데이트
        if conversation_id in self._memory_cache:
            self._memory_cache[conversation_id].chat_memory.add_user_message(content)
        
        # 컨텍스트 업데이트
        context = self.get_or_create_conversation_context(conversation_id)
        context.session_metadata['total_requests'] += 1
        context.last_updated = datetime.now()
        self._save_context(context)
        
        self.logger.debug("Added user message", extra={
            'conversation_id': conversation_id,
            'message_id': saved_id
        })
        
        return saved_id
    
    def add_ai_message(self, conversation_id: str, content: str,
                      optimization_result: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """AI 응답 메시지 추가"""
        message_id = str(uuid.uuid4())
        
        # 메타데이터 구성
        full_metadata = metadata or {}
        if optimization_result:
            full_metadata['optimization_result'] = optimization_result
            full_metadata['confidence_score'] = optimization_result.get('confidence_score', 0.0)
            full_metadata['scenario_type'] = optimization_result.get('scenario_type', 'unknown')
        
        message_data = {
            'id': message_id,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'assistant',
            'content': content,
            'metadata': full_metadata
        }
        
        # Redis에 저장
        saved_id = self.memory_repository.save_conversation_message(message_data)
        
        # LangChain 메모리 업데이트
        if conversation_id in self._memory_cache:
            self._memory_cache[conversation_id].chat_memory.add_ai_message(content)
        
        # 컨텍스트 업데이트
        if optimization_result:
            context = self.get_or_create_conversation_context(conversation_id)
            context.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'scenario_type': optimization_result.get('scenario_type', 'unknown'),
                'confidence_score': optimization_result.get('confidence_score', 0.0),
                'result_summary': {
                    'routes_count': len(optimization_result.get('routes', [])),
                    'total_distance': optimization_result.get('total_distance_km', 0),
                    'total_cost': optimization_result.get('total_cost', 0)
                }
            })
            
            if optimization_result.get('confidence_score', 0) >= 0.8:
                context.session_metadata['successful_optimizations'] += 1
            
            context.last_updated = datetime.now()
            self._save_context(context)
        
        self.logger.debug("Added AI message", extra={
            'conversation_id': conversation_id,
            'message_id': saved_id,
            'has_optimization_result': optimization_result is not None
        })
        
        return saved_id
    
    def process_feedback(self, conversation_id: str, feedback_data: Dict[str, Any]) -> FeedbackAnalysis:
        """피드백 처리 및 분석"""
        feedback_id = str(uuid.uuid4())
        
        # 감정 분석 (단순화된 버전)
        sentiment_score = self._analyze_sentiment(feedback_data.get('feedback_content', ''))
        
        # 키 토픽 추출
        key_topics = self._extract_key_topics(feedback_data.get('feedback_content', ''))
        
        # 개선 제안 생성
        improvement_suggestions = self._generate_improvement_suggestions(
            feedback_data, sentiment_score, key_topics
        )
        
        # 패턴 업데이트 계산
        pattern_updates = self._calculate_pattern_updates(conversation_id, feedback_data)
        
        # 피드백 분석 결과
        analysis = FeedbackAnalysis(
            feedback_id=feedback_id,
            conversation_id=conversation_id,
            feedback_type=feedback_data.get('feedback_type', 'general'),
            rating=feedback_data.get('rating', 0),
            content=feedback_data.get('feedback_content', ''),
            sentiment_score=sentiment_score,
            key_topics=key_topics,
            improvement_suggestions=improvement_suggestions,
            pattern_updates=pattern_updates,
            analyzed_at=datetime.now()
        )
        
        # 피드백 메시지로 저장
        feedback_message_data = {
            'id': feedback_id,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'feedback',
            'content': feedback_data.get('feedback_content', ''),
            'metadata': {
                'feedback_type': feedback_data.get('feedback_type'),
                'rating': feedback_data.get('rating'),
                'analysis': analysis.to_dict()
            }
        }
        
        self.memory_repository.save_conversation_message(feedback_message_data)
        
        # 컨텍스트 업데이트
        self._update_context_with_feedback(conversation_id, analysis)
        
        self.logger.info("Processed feedback", extra={
            'conversation_id': conversation_id,
            'feedback_id': feedback_id,
            'feedback_type': feedback_data.get('feedback_type'),
            'rating': feedback_data.get('rating'),
            'sentiment_score': sentiment_score
        })
        
        return analysis
    
    def get_context_for_optimization(self, conversation_id: str) -> Dict[str, Any]:
        """최적화를 위한 컨텍스트 정보 조합"""
        context = self.get_or_create_conversation_context(conversation_id)
        langchain_memory = self.get_langchain_memory(conversation_id)
        
        # LangChain 메모리에서 최근 대화 조회
        recent_messages = []
        if hasattr(langchain_memory, 'chat_memory') and langchain_memory.chat_memory.messages:
            for message in langchain_memory.chat_memory.messages[-6:]:  # 최근 6개 메시지
                if isinstance(message, HumanMessage):
                    recent_messages.append({'role': 'user', 'content': message.content})
                elif isinstance(message, AIMessage):
                    recent_messages.append({'role': 'assistant', 'content': message.content})
        
        # 학습된 선호도 가중치 계산
        preference_weights = self._calculate_dynamic_preferences(context)
        
        # 최적화 컨텍스트 구성
        optimization_context = {
            'conversation_id': conversation_id,
            'user_preferences': context.user_preferences,
            'recent_messages': recent_messages,
            'optimization_history': context.optimization_history[-5:],  # 최근 5개
            'feedback_summary': self._summarize_feedback(context.feedback_history),
            'learned_patterns': context.learned_patterns,
            'preference_weights': preference_weights,
            'session_metadata': context.session_metadata,
            'context_hints': self._generate_context_hints(context)
        }
        
        self.logger.debug("Generated optimization context", extra={
            'conversation_id': conversation_id,
            'recent_messages_count': len(recent_messages),
            'optimization_history_count': len(context.optimization_history),
            'feedback_count': len(context.feedback_history)
        })
        
        return optimization_context
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """대화 초기화 (선택적)"""
        try:
            # LangChain 메모리 캐시 제거
            if conversation_id in self._memory_cache:
                del self._memory_cache[conversation_id]
            
            # Redis 데이터는 TTL로 자동 관리되므로 컨텍스트만 리셋
            context = self.get_or_create_conversation_context(conversation_id)
            context.optimization_history = []
            context.feedback_history = []
            context.session_metadata['total_requests'] = 0
            context.session_metadata['successful_optimizations'] = 0
            context.last_updated = datetime.now()
            
            self._save_context(context)
            
            self.logger.info("Cleared conversation context", extra={
                'conversation_id': conversation_id
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to clear conversation", extra={
                'conversation_id': conversation_id,
                'error': str(e)
            })
            return False
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """대화 요약 정보 조회"""
        context = self.get_or_create_conversation_context(conversation_id)
        messages = self.memory_repository.get_conversation_messages(conversation_id, limit=100)
        
        # 메시지 유형별 카운트
        message_counts = {'user': 0, 'assistant': 0, 'feedback': 0}
        for msg in messages:
            msg_type = msg.get('message_type', 'unknown')
            if msg_type in message_counts:
                message_counts[msg_type] += 1
        
        # 만족도 평균 계산
        feedback_ratings = [
            fb.get('rating', 0) for fb in context.feedback_history 
            if fb.get('rating', 0) > 0
        ]
        avg_satisfaction = sum(feedback_ratings) / len(feedback_ratings) if feedback_ratings else 0.0
        
        summary = {
            'conversation_id': conversation_id,
            'created_at': context.created_at.isoformat(),
            'last_updated': context.last_updated.isoformat(),
            'duration': (context.last_updated - context.created_at).total_seconds(),
            'message_counts': message_counts,
            'optimization_count': len(context.optimization_history),
            'feedback_count': len(context.feedback_history),
            'average_satisfaction': round(avg_satisfaction, 2),
            'successful_optimization_rate': (
                context.session_metadata['successful_optimizations'] / 
                max(context.session_metadata['total_requests'], 1) * 100
            ),
            'user_preferences': context.user_preferences,
            'learned_patterns_count': len(context.learned_patterns.get('successful_scenarios', []))
        }
        
        return summary
    
    # Private helper methods
    
    def _save_context(self, context: ConversationContext):
        """컨텍스트를 Redis에 저장"""
        self.memory_repository.update_conversation_summary(
            context.conversation_id,
            {'context': context.to_dict()}
        )
    
    def _analyze_sentiment(self, text: str) -> float:
        """감정 분석 (단순화된 키워드 기반)"""
        positive_keywords = [
            '좋', '훌륭', '완벽', '만족', '효율적', '빠른', '정확', '우수', '최고', '감사'
        ]
        negative_keywords = [
            '나쁘', '실망', '느린', '비효율', '문제', '오류', '불만', '개선', '수정'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0  # 중립
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """키 토픽 추출 (TMS 도메인 특화)"""
        tms_topics = {
            '경로': ['경로', '루트', '길'],
            '비용': ['비용', '가격', '요금', '돈'],
            '시간': ['시간', '속도', '빠르', '느린'],
            '차량': ['차량', '트럭', '운송수단'],
            '배송': ['배송', '배달', '운송'],
            '최적화': ['최적화', '효율', '개선'],
            '거리': ['거리', '킬로미터', 'km'],
            '연료': ['연료', '기름', '연비']
        }
        
        found_topics = []
        text_lower = text.lower()
        
        for topic, keywords in tms_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def _generate_improvement_suggestions(self, feedback_data: Dict[str, Any], 
                                        sentiment_score: float, 
                                        key_topics: List[str]) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        rating = feedback_data.get('rating', 0)
        
        if rating <= 2 or sentiment_score < -0.3:
            suggestions.append("더 정확한 최적화를 위해 추가 제약조건을 고려하겠습니다")
            
        if '비용' in key_topics and rating < 4:
            suggestions.append("비용 최적화 가중치를 높여 더 경제적인 경로를 제안하겠습니다")
            
        if '시간' in key_topics and rating < 4:
            suggestions.append("시간 효율성을 우선시하는 최적화 방식을 적용하겠습니다")
            
        if '경로' in key_topics and sentiment_score < 0:
            suggestions.append("경로 계산 알고리즘을 개선하여 더 실용적인 경로를 제안하겠습니다")
            
        if not suggestions:
            suggestions.append("현재 최적화 방식을 유지하면서 세부사항을 개선하겠습니다")
        
        return suggestions
    
    def _calculate_pattern_updates(self, conversation_id: str, 
                                 feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """패턴 업데이트 계산"""
        context = self.get_or_create_conversation_context(conversation_id)
        rating = feedback_data.get('rating', 0)
        
        updates = {}
        
        # 선호도 가중치 조정
        if rating >= 4:  # 긍정적 피드백
            if '비용' in feedback_data.get('feedback_content', '').lower():
                updates['increase_cost_weight'] = 0.1
            if '시간' in feedback_data.get('feedback_content', '').lower():
                updates['increase_time_weight'] = 0.1
            if '거리' in feedback_data.get('feedback_content', '').lower():
                updates['increase_distance_weight'] = 0.1
        elif rating <= 2:  # 부정적 피드백
            if '비용' in feedback_data.get('feedback_content', '').lower():
                updates['decrease_cost_weight'] = -0.1
            if '시간' in feedback_data.get('feedback_content', '').lower():
                updates['decrease_time_weight'] = -0.1
            if '거리' in feedback_data.get('feedback_content', '').lower():
                updates['decrease_distance_weight'] = -0.1
        
        # 최적화 우선순위 조정
        if rating >= 4:
            updates['successful_pattern'] = True
        elif rating <= 2:
            updates['problematic_pattern'] = True
        
        return updates
    
    def _update_context_with_feedback(self, conversation_id: str, analysis: FeedbackAnalysis):
        """피드백 분석 결과로 컨텍스트 업데이트"""
        context = self.get_or_create_conversation_context(conversation_id)
        
        # 피드백 히스토리 추가
        context.feedback_history.append({
            'feedback_id': analysis.feedback_id,
            'timestamp': analysis.analyzed_at.isoformat(),
            'rating': analysis.rating,
            'sentiment_score': analysis.sentiment_score,
            'key_topics': analysis.key_topics
        })
        
        # 학습된 패턴 업데이트
        if analysis.rating >= 4:
            context.learned_patterns['successful_scenarios'].append({
                'timestamp': analysis.analyzed_at.isoformat(),
                'topics': analysis.key_topics,
                'rating': analysis.rating
            })
        elif analysis.rating <= 2:
            context.learned_patterns['problem_patterns'].append({
                'timestamp': analysis.analyzed_at.isoformat(),
                'topics': analysis.key_topics,
                'rating': analysis.rating
            })
        
        # 선호도 가중치 조정
        for update_key, update_value in analysis.pattern_updates.items():
            if 'cost_weight' in update_key:
                current = context.learned_patterns['preference_weights']['cost']
                context.learned_patterns['preference_weights']['cost'] = max(0.1, min(0.8, current + update_value))
            elif 'time_weight' in update_key:
                current = context.learned_patterns['preference_weights']['time']
                context.learned_patterns['preference_weights']['time'] = max(0.1, min(0.8, current + update_value))
            elif 'distance_weight' in update_key:
                current = context.learned_patterns['preference_weights']['distance']
                context.learned_patterns['preference_weights']['distance'] = max(0.1, min(0.8, current + update_value))
        
        # 가중치 정규화
        total_weight = sum(context.learned_patterns['preference_weights'].values())
        if total_weight > 0:
            for key in context.learned_patterns['preference_weights']:
                context.learned_patterns['preference_weights'][key] /= total_weight
        
        # 평균 만족도 업데이트
        ratings = [fb['rating'] for fb in context.feedback_history if fb.get('rating', 0) > 0]
        if ratings:
            context.session_metadata['average_satisfaction'] = sum(ratings) / len(ratings)
        
        context.last_updated = datetime.now()
        self._save_context(context)
    
    def _calculate_dynamic_preferences(self, context: ConversationContext) -> Dict[str, float]:
        """동적 선호도 가중치 계산"""
        base_weights = context.learned_patterns['preference_weights'].copy()
        
        # 최근 피드백 기반 조정
        recent_feedback = context.feedback_history[-5:]  # 최근 5개
        if recent_feedback:
            positive_feedback = [fb for fb in recent_feedback if fb.get('rating', 0) >= 4]
            if len(positive_feedback) / len(recent_feedback) > 0.8:  # 80% 이상 긍정적
                # 현재 가중치 유지 (성공적인 패턴)
                pass
            elif len(positive_feedback) / len(recent_feedback) < 0.4:  # 40% 미만 긍정적
                # 가중치 균등화 (다른 접근 시도)
                base_weights = {'distance': 0.33, 'time': 0.33, 'cost': 0.34}
        
        return base_weights
    
    def _summarize_feedback(self, feedback_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """피드백 요약"""
        if not feedback_history:
            return {'total_count': 0, 'average_rating': 0.0, 'common_topics': []}
        
        ratings = [fb.get('rating', 0) for fb in feedback_history if fb.get('rating', 0) > 0]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        # 공통 토픽 추출
        all_topics = []
        for fb in feedback_history:
            all_topics.extend(fb.get('key_topics', []))
        
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        common_topics = sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:3]
        
        return {
            'total_count': len(feedback_history),
            'average_rating': round(avg_rating, 2),
            'common_topics': common_topics,
            'recent_trend': 'positive' if avg_rating >= 3.5 else 'negative'
        }
    
    def _generate_context_hints(self, context: ConversationContext) -> List[str]:
        """컨텍스트 힌트 생성"""
        hints = []
        
        # 성공 패턴 기반 힌트
        successful_scenarios = context.learned_patterns.get('successful_scenarios', [])
        if len(successful_scenarios) >= 3:
            common_topics = {}
            for scenario in successful_scenarios[-5:]:
                for topic in scenario.get('topics', []):
                    common_topics[topic] = common_topics.get(topic, 0) + 1
            
            if common_topics:
                most_common = max(common_topics.keys(), key=lambda x: common_topics[x])
                hints.append(f"사용자는 '{most_common}' 관련 최적화를 선호합니다")
        
        # 선호도 가중치 기반 힌트
        weights = context.learned_patterns['preference_weights']
        max_weight_key = max(weights.keys(), key=lambda x: weights[x])
        if weights[max_weight_key] > 0.4:
            hints.append(f"{max_weight_key} 우선 최적화를 선호합니다")
        
        # 평균 만족도 기반 힌트
        avg_satisfaction = context.session_metadata.get('average_satisfaction', 0)
        if avg_satisfaction >= 4.0:
            hints.append("현재 최적화 방식에 높은 만족도를 보입니다")
        elif avg_satisfaction <= 2.5:
            hints.append("최적화 방식 개선이 필요할 수 있습니다")
        
        return hints 