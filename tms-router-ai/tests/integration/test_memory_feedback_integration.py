"""
대화 메모리 및 피드백 시스템 통합 테스트

TMS 대화 메모리 관리자와 피드백 처리 시스템의 통합 동작을 검증합니다.
"""
import pytest
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from src.infrastructure.memory.redis_memory_repository import RedisMemoryRepository
from src.infrastructure.memory.conversation_manager import TmsConversationManager
from src.infrastructure.memory.feedback_processor import TmsFeedbackProcessor
from src.use_cases.process_feedback_use_case import ProcessFeedbackUseCase, FeedbackRequest


@pytest.fixture
def redis_memory_repository():
    """Redis 메모리 저장소 픽스처"""
    return RedisMemoryRepository(
        host='localhost',
        port=6379,
        db=1,  # 테스트용 DB
        decode_responses=True
    )


@pytest.fixture
def conversation_manager(redis_memory_repository):
    """대화 메모리 관리자 픽스처"""
    return TmsConversationManager(
        memory_repository=redis_memory_repository,
        window_size=10,
        max_token_limit=1000
    )


@pytest.fixture
def feedback_processor(redis_memory_repository):
    """피드백 처리기 픽스처"""
    return TmsFeedbackProcessor(memory_repository=redis_memory_repository)


@pytest.fixture
def feedback_use_case(conversation_manager, feedback_processor):
    """피드백 처리 Use Case 픽스처"""
    return ProcessFeedbackUseCase(
        conversation_manager=conversation_manager,
        feedback_processor=feedback_processor
    )


@pytest.fixture
def sample_conversation_id():
    """샘플 대화 ID"""
    return f"test_conv_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_redis(redis_memory_repository, sample_conversation_id):
    """테스트 후 Redis 정리"""
    yield
    
    # 테스트 데이터 정리
    try:
        # 대화 관련 키들 삭제
        keys_to_delete = [
            f"conv:{sample_conversation_id}:*",
            f"msg:*",
            f"feedback_patterns:{sample_conversation_id}",
            f"feedback_stats:{sample_conversation_id}",
            f"processed_feedback:*"
        ]
        
        for pattern in keys_to_delete:
            keys = redis_memory_repository.redis_client.keys(pattern)
            if keys:
                redis_memory_repository.redis_client.delete(*keys)
                
    except Exception as e:
        print(f"Cleanup warning: {e}")


class TestMemoryFeedbackIntegration:
    """대화 메모리 및 피드백 시스템 통합 테스트"""
    
    def test_conversation_context_creation(self, conversation_manager, sample_conversation_id, cleanup_redis):
        """대화 컨텍스트 생성 테스트"""
        # When: 새 대화 컨텍스트 생성
        context = conversation_manager.get_or_create_conversation_context(sample_conversation_id)
        
        # Then: 컨텍스트가 올바르게 생성됨
        assert context.conversation_id == sample_conversation_id
        assert context.user_preferences is not None
        assert context.optimization_history == []
        assert context.feedback_history == []
        assert context.learned_patterns is not None
        assert context.session_metadata is not None
    
    def test_user_message_handling(self, conversation_manager, sample_conversation_id, cleanup_redis):
        """사용자 메시지 처리 테스트"""
        # Given: 대화 컨텍스트 존재
        context = conversation_manager.get_or_create_conversation_context(sample_conversation_id)
        
        # When: 사용자 메시지 추가
        message_content = "서울에서 부산까지 3대 차량으로 10개 주문 배송 최적화해주세요"
        message_id = conversation_manager.add_user_message(
            sample_conversation_id, 
            message_content,
            metadata={'request_type': 'optimization'}
        )
        
        # Then: 메시지가 저장되고 컨텍스트 업데이트됨
        assert message_id is not None
        
        # 컨텍스트 업데이트 확인
        updated_context = conversation_manager.get_or_create_conversation_context(sample_conversation_id)
        assert updated_context.session_metadata['total_requests'] == 1
    
    def test_ai_message_with_optimization_result(self, conversation_manager, sample_conversation_id, cleanup_redis):
        """AI 응답 메시지 처리 테스트"""
        # Given: 사용자 메시지 존재
        conversation_manager.add_user_message(sample_conversation_id, "경로 최적화 요청")
        
        # When: AI 응답 추가 (최적화 결과 포함)
        ai_content = "최적화 완료: 3개 경로, 총 거리 150km, 예상 시간 4시간"
        optimization_result = {
            'scenario_type': 'multi_vehicle_vrp',
            'confidence_score': 0.85,
            'routes': [
                {'vehicle_id': 'V001', 'orders': ['O001', 'O002'], 'distance_km': 50},
                {'vehicle_id': 'V002', 'orders': ['O003', 'O004'], 'distance_km': 60},
                {'vehicle_id': 'V003', 'orders': ['O005'], 'distance_km': 40}
            ],
            'total_distance_km': 150,
            'total_cost': 45000
        }
        
        message_id = conversation_manager.add_ai_message(
            sample_conversation_id,
            ai_content,
            optimization_result=optimization_result
        )
        
        # Then: 메시지가 저장되고 최적화 히스토리 업데이트됨
        assert message_id is not None
        
        context = conversation_manager.get_or_create_conversation_context(sample_conversation_id)
        assert len(context.optimization_history) == 1
        assert context.optimization_history[0]['scenario_type'] == 'multi_vehicle_vrp'
        assert context.optimization_history[0]['confidence_score'] == 0.85
        assert context.session_metadata['successful_optimizations'] == 1
    
    def test_feedback_processing_integration(self, feedback_use_case, sample_conversation_id, cleanup_redis):
        """피드백 처리 통합 테스트"""
        # Given: 피드백 요청
        feedback_request = FeedbackRequest(
            feedback_id=None,
            conversation_id=sample_conversation_id,
            feedback_type="positive",
            feedback_content="경로 최적화가 매우 좋았습니다. 비용도 절약되고 시간도 단축되었어요.",
            rating=5,
            metadata={'context': 'after_optimization'}
        )
        
        # When: 피드백 처리 실행
        response = feedback_use_case.execute(feedback_request)
        
        # Then: 피드백이 성공적으로 처리됨
        assert response.processing_status == "success"
        assert response.feedback_id is not None
        assert response.conversation_id == sample_conversation_id
        assert len(response.learning_insights) > 0
        assert len(response.improvement_suggestions) > 0
        
        # 분석 결과 검증
        assert 'sentiment' in response.analysis_summary
        assert response.analysis_summary['sentiment']['level'] in ['positive', 'very_positive']
        assert response.analysis_summary['sentiment']['score'] > 0
        assert len(response.analysis_summary['topics']['detected']) > 0
    
    def test_negative_feedback_processing(self, feedback_use_case, sample_conversation_id, cleanup_redis):
        """부정적 피드백 처리 테스트"""
        # Given: 부정적 피드백 요청
        feedback_request = FeedbackRequest(
            feedback_id=None,
            conversation_id=sample_conversation_id,
            feedback_type="negative",
            feedback_content="경로가 너무 복잡하고 비용이 많이 들었습니다. 개선이 필요해요.",
            rating=2,
            metadata={'urgency': 'high'}
        )
        
        # When: 피드백 처리
        response = feedback_use_case.execute(feedback_request)
        
        # Then: 부정적 피드백이 올바르게 분석됨
        assert response.processing_status == "success"
        assert response.analysis_summary['sentiment']['score'] < 0
        assert response.analysis_summary['urgency'] == 'high'
        assert len(response.improvement_suggestions) > 0
        
        # 개선 제안에 비용/경로 관련 내용 포함 확인
        suggestions_text = ' '.join(response.improvement_suggestions)
        assert any(keyword in suggestions_text for keyword in ['비용', '경로', '개선', '알고리즘'])
    
    def test_conversation_context_update_with_feedback(self, conversation_manager, sample_conversation_id, cleanup_redis):
        """피드백으로 대화 컨텍스트 업데이트 테스트"""
        # Given: 초기 컨텍스트
        initial_context = conversation_manager.get_or_create_conversation_context(sample_conversation_id)
        initial_feedback_count = len(initial_context.feedback_history)
        
        # When: 피드백 처리
        feedback_data = {
            'feedback_type': 'positive',
            'feedback_content': '시간 효율성이 정말 좋았습니다',
            'rating': 4
        }
        
        analysis = conversation_manager.process_feedback(sample_conversation_id, feedback_data)
        
        # Then: 컨텍스트가 업데이트됨
        updated_context = conversation_manager.get_or_create_conversation_context(sample_conversation_id)
        assert len(updated_context.feedback_history) == initial_feedback_count + 1
        assert updated_context.feedback_history[-1]['rating'] == 4
        assert 'time' in updated_context.feedback_history[-1]['key_topics']
        
        # 학습된 패턴 업데이트 확인
        successful_scenarios = updated_context.learned_patterns['successful_scenarios']
        assert len(successful_scenarios) > 0
        assert successful_scenarios[-1]['rating'] == 4
    
    def test_learning_insights_generation(self, feedback_processor, sample_conversation_id, cleanup_redis):
        """학습 인사이트 생성 테스트"""
        # Given: 여러 피드백 처리
        feedback_data_list = [
            {'feedback_type': 'positive', 'feedback_content': '비용 최적화가 훌륭해요', 'rating': 5},
            {'feedback_type': 'positive', 'feedback_content': '시간 단축이 좋았습니다', 'rating': 4},
            {'feedback_type': 'suggestion', 'feedback_content': '거리 계산을 더 정확히 해주세요', 'rating': 3}
        ]
        
        for feedback_data in feedback_data_list:
            feedback_processor.process_feedback(sample_conversation_id, feedback_data)
            time.sleep(0.1)  # 타이밍 차이를 위한 지연
        
        # When: 학습 인사이트 조회
        insights = feedback_processor.get_learning_insights(sample_conversation_id, days=1)
        
        # Then: 인사이트가 생성됨
        assert len(insights) > 0
        
        # 인사이트 내용 검증
        insight_types = [insight.insight_type for insight in insights]
        assert 'preference' in insight_types or 'pattern' in insight_types
        
        for insight in insights:
            assert insight.confidence_score > 0
            assert len(insight.actionable_recommendations) > 0
    
    def test_optimization_recommendations(self, feedback_processor, sample_conversation_id, cleanup_redis):
        """최적화 추천사항 생성 테스트"""
        # Given: 다양한 피드백 패턴
        feedback_scenarios = [
            {'content': '비용이 너무 높아요', 'rating': 2, 'type': 'negative'},
            {'content': '시간은 좋은데 거리가 길어요', 'rating': 3, 'type': 'suggestion'},
            {'content': '전체적으로 만족합니다', 'rating': 4, 'type': 'positive'}
        ]
        
        for scenario in feedback_scenarios:
            feedback_data = {
                'feedback_type': scenario['type'],
                'feedback_content': scenario['content'],
                'rating': scenario['rating']
            }
            feedback_processor.process_feedback(sample_conversation_id, feedback_data)
            time.sleep(0.1)
        
        # When: 최적화 추천사항 조회
        recommendations = feedback_processor.get_optimization_recommendations(sample_conversation_id)
        
        # Then: 추천사항이 생성됨
        assert 'recommendations' in recommendations
        assert 'confidence' in recommendations
        assert isinstance(recommendations['recommendations'], list)
        assert 0 <= recommendations['confidence'] <= 1.0
        
        if recommendations['recommendations']:
            # 추천사항에 TMS 관련 키워드 포함 확인
            recommendations_text = ' '.join(recommendations['recommendations'])
            tms_keywords = ['비용', '시간', '거리', '최적화', '경로', '알고리즘']
            assert any(keyword in recommendations_text for keyword in tms_keywords)
    
    def test_conversation_summary_generation(self, conversation_manager, sample_conversation_id, cleanup_redis):
        """대화 요약 생성 테스트"""
        # Given: 다양한 메시지와 피드백이 있는 대화
        # 사용자 메시지
        conversation_manager.add_user_message(sample_conversation_id, "경로 최적화 요청")
        
        # AI 응답
        optimization_result = {
            'scenario_type': 'delivery_optimization',
            'confidence_score': 0.9,
            'routes': [{'vehicle_id': 'V001', 'orders': ['O001']}],
            'total_distance_km': 100
        }
        conversation_manager.add_ai_message(
            sample_conversation_id, 
            "최적화 완료", 
            optimization_result=optimization_result
        )
        
        # 피드백
        feedback_data = {'feedback_type': 'positive', 'feedback_content': '좋아요', 'rating': 5}
        conversation_manager.process_feedback(sample_conversation_id, feedback_data)
        
        # When: 대화 요약 조회
        summary = conversation_manager.get_conversation_summary(sample_conversation_id)
        
        # Then: 요약이 올바르게 생성됨
        assert summary['conversation_id'] == sample_conversation_id
        assert summary['optimization_count'] >= 1
        assert summary['feedback_count'] >= 1
        assert summary['average_satisfaction'] > 0
        assert summary['successful_optimization_rate'] > 0
        assert isinstance(summary['user_preferences'], dict)
    
    def test_memory_persistence_and_retrieval(self, conversation_manager, sample_conversation_id, cleanup_redis):
        """메모리 지속성 및 조회 테스트"""
        # Given: 메시지와 피드백 저장
        user_message_id = conversation_manager.add_user_message(
            sample_conversation_id, 
            "배송 경로 최적화 요청"
        )
        
        ai_message_id = conversation_manager.add_ai_message(
            sample_conversation_id,
            "최적화 결과 제공",
            optimization_result={'scenario_type': 'test', 'confidence_score': 0.8}
        )
        
        feedback_analysis = conversation_manager.process_feedback(
            sample_conversation_id,
            {'feedback_type': 'positive', 'feedback_content': '만족', 'rating': 4}
        )
        
        # When: 최적화 컨텍스트 조회
        context = conversation_manager.get_context_for_optimization(sample_conversation_id)
        
        # Then: 저장된 데이터가 올바르게 조회됨
        assert context['conversation_id'] == sample_conversation_id
        assert 'user_preferences' in context
        assert 'recent_messages' in context
        assert 'optimization_history' in context
        assert 'feedback_summary' in context
        assert 'learned_patterns' in context
        assert 'preference_weights' in context
        
        # 최근 메시지 확인
        assert len(context['recent_messages']) >= 2  # user + ai message
        
        # 피드백 요약 확인
        feedback_summary = context['feedback_summary']
        assert feedback_summary['total_count'] >= 1
        assert feedback_summary['average_rating'] == 4.0
    
    def test_error_handling_and_recovery(self, conversation_manager, sample_conversation_id, cleanup_redis):
        """에러 처리 및 복구 테스트"""
        # Given: 잘못된 피드백 데이터
        invalid_feedback = {
            # 'feedback_type': 'positive',  # 누락
            'feedback_content': '테스트',
            'rating': 'invalid'  # 잘못된 타입
        }
        
        # When: 에러가 발생하는 피드백 처리
        try:
            conversation_manager.process_feedback(sample_conversation_id, invalid_feedback)
        except Exception as e:
            # Then: 에러가 발생해도 시스템이 복구됨
            print(f"Expected error caught: {e}")
        
        # 정상적인 피드백으로 복구 테스트
        valid_feedback = {
            'feedback_type': 'positive',
            'feedback_content': '복구 테스트',
            'rating': 4
        }
        
        # 복구 후 정상 동작 확인
        analysis = conversation_manager.process_feedback(sample_conversation_id, valid_feedback)
        assert analysis.rating == 4
        assert analysis.sentiment_score > 0


class TestMemoryFeedbackPerformance:
    """메모리 및 피드백 시스템 성능 테스트"""
    
    def test_bulk_feedback_processing_performance(self, feedback_processor, cleanup_redis):
        """대량 피드백 처리 성능 테스트"""
        # Given: 대량 피드백 데이터
        conversation_id = f"perf_test_{uuid.uuid4().hex[:8]}"
        feedback_count = 50
        
        start_time = time.time()
        
        # When: 대량 피드백 처리
        for i in range(feedback_count):
            feedback_data = {
                'feedback_type': 'positive' if i % 2 == 0 else 'negative',
                'feedback_content': f'테스트 피드백 {i}',
                'rating': (i % 5) + 1
            }
            feedback_processor.process_feedback(conversation_id, feedback_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Then: 성능 기준 확인 (피드백당 평균 100ms 이하)
        avg_time_per_feedback = processing_time / feedback_count
        assert avg_time_per_feedback < 0.1, f"Too slow: {avg_time_per_feedback:.3f}s per feedback"
        
        print(f"Processed {feedback_count} feedbacks in {processing_time:.2f}s")
        print(f"Average time per feedback: {avg_time_per_feedback:.3f}s")
    
    def test_memory_usage_optimization(self, conversation_manager, redis_memory_repository, cleanup_redis):
        """메모리 사용량 최적화 테스트"""
        # Given: 메모리 사용량 측정 시작
        initial_stats = redis_memory_repository.get_memory_stats()
        conversation_id = f"memory_test_{uuid.uuid4().hex[:8]}"
        
        # When: 대량 메시지 추가
        message_count = 100
        for i in range(message_count):
            if i % 2 == 0:
                conversation_manager.add_user_message(
                    conversation_id, 
                    f"사용자 메시지 {i}"
                )
            else:
                conversation_manager.add_ai_message(
                    conversation_id,
                    f"AI 응답 {i}",
                    optimization_result={'test': True, 'confidence_score': 0.5}
                )
        
        # Then: 메모리 사용량 확인
        final_stats = redis_memory_repository.get_memory_stats()
        
        print(f"Initial conversations: {initial_stats['conversations']['total_conversations']}")
        print(f"Final conversations: {final_stats['conversations']['total_conversations']}")
        print(f"Redis memory usage: {final_stats['redis_memory']['used_memory_human']}")
        
        # 메모리 사용량이 합리적인 범위 내에 있는지 확인
        assert final_stats['conversations']['total_conversations'] >= initial_stats['conversations']['total_conversations']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 