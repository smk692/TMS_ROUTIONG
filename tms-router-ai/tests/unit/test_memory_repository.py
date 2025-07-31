"""
Redis 메모리 저장소 단위 테스트

RedisMemoryRepository의 개별 기능을 테스트합니다.
Redis 연결이 필요하므로 테스트 환경에서 Redis가 실행되어야 합니다.
"""
import pytest
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.infrastructure.memory.redis_memory_repository import RedisMemoryRepository
from src.shared.exceptions import MemoryRepositoryError


class TestRedisMemoryRepository:
    """Redis 메모리 저장소 테스트"""
    
    @pytest.fixture
    def redis_repo(self):
        """테스트용 Redis 저장소 픽스처"""
        try:
            # 테스트용 Redis DB (일반적으로 DB 15 사용)
            repo = RedisMemoryRepository(
                host='localhost',
                port=6379,
                db=15,  # 테스트 전용 DB
                decode_responses=True
            )
            
            # 테스트 전 DB 정리
            repo.redis_client.flushdb()
            yield repo
            
            # 테스트 후 정리
            repo.redis_client.flushdb()
            
        except Exception as e:
            pytest.skip(f"Redis not available for testing: {e}")
    
    def test_redis_connection(self, redis_repo):
        """Redis 연결 테스트"""
        assert redis_repo.redis_client.ping() is True
    
    def test_save_conversation_message(self, redis_repo):
        """대화 메시지 저장 테스트"""
        conversation_id = f"test_conv_{uuid.uuid4().hex[:8]}"
        message_data = {
            'id': 'msg_001',
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'user',
            'content': 'Test message content',
            'metadata': {'test': True}
        }
        
        # 메시지 저장
        message_id = redis_repo.save_conversation_message(message_data)
        assert message_id == 'msg_001'
        
        # 저장된 데이터 확인
        messages_key = f"conv:{conversation_id}:messages"
        saved_messages = redis_repo.redis_client.lrange(messages_key, 0, -1)
        assert len(saved_messages) == 1
        
        saved_message = json.loads(saved_messages[0])
        assert saved_message['id'] == 'msg_001'
        assert saved_message['content'] == 'Test message content'
        
        # 메타데이터 확인
        meta_key = f"conv:{conversation_id}:meta"
        message_count = redis_repo.redis_client.hget(meta_key, 'message_count')
        assert int(message_count) == 1
    
    def test_get_conversation_messages(self, redis_repo):
        """대화 메시지 조회 테스트"""
        conversation_id = f"test_conv_{uuid.uuid4().hex[:8]}"
        
        # 여러 메시지 저장
        messages = []
        for i in range(3):
            message_data = {
                'id': f'msg_{i+1:03d}',
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'user' if i % 2 == 0 else 'assistant',
                'content': f'Test message {i+1}',
                'metadata': {'order': i+1}
            }
            messages.append(message_data)
            redis_repo.save_conversation_message(message_data)
        
        # 메시지 조회
        retrieved_messages = redis_repo.get_conversation_messages(conversation_id)
        assert len(retrieved_messages) == 3
        
        # 최신순으로 정렬되어 있는지 확인 (LPUSH로 저장하므로 역순)
        assert retrieved_messages[0]['id'] == 'msg_003'
        assert retrieved_messages[1]['id'] == 'msg_002'
        assert retrieved_messages[2]['id'] == 'msg_001'
        
        # 제한된 수량 조회
        limited_messages = redis_repo.get_conversation_messages(conversation_id, limit=2)
        assert len(limited_messages) == 2
    
    def test_get_conversation_memory(self, redis_repo):
        """대화 메모리 조회 테스트"""
        conversation_id = f"test_conv_{uuid.uuid4().hex[:8]}"
        
        # 메시지 저장으로 메타데이터 생성
        message_data = {
            'id': 'msg_001',
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'user',
            'content': 'Test message',
            'metadata': {}
        }
        redis_repo.save_conversation_message(message_data)
        
        # 대화 요약 저장
        summary_data = {
            'key_topics': ['VRP', 'optimization'],
            'satisfaction_score': 4.5
        }
        redis_repo.update_conversation_summary(conversation_id, summary_data)
        
        # 메모리 조회
        memory = redis_repo.get_conversation_memory(conversation_id)
        assert memory is not None
        assert memory['conversation_id'] == conversation_id
        assert memory['message_count'] == 1
        assert memory['summary']['key_topics'] == ['VRP', 'optimization']
        assert memory['summary']['satisfaction_score'] == 4.5
    
    def test_update_conversation_summary(self, redis_repo):
        """대화 요약 업데이트 테스트"""
        conversation_id = f"test_conv_{uuid.uuid4().hex[:8]}"
        summary_data = {
            'key_topics': ['TSP', 'route_optimization'],
            'optimization_count': 3,
            'average_rating': 4.2
        }
        
        # 요약 업데이트
        redis_repo.update_conversation_summary(conversation_id, summary_data)
        
        # 저장된 요약 확인
        summary_key = f"conv:{conversation_id}:summary"
        stored_summary = redis_repo.redis_client.get(summary_key)
        assert stored_summary is not None
        
        parsed_summary = json.loads(stored_summary)
        assert parsed_summary['key_topics'] == ['TSP', 'route_optimization']
        assert parsed_summary['optimization_count'] == 3
        assert parsed_summary['average_rating'] == 4.2
        
        # 메타데이터 업데이트 확인
        meta_key = f"conv:{conversation_id}:meta"
        last_updated = redis_repo.redis_client.hget(meta_key, 'last_updated')
        assert last_updated is not None
    
    def test_feedback_analytics(self, redis_repo):
        """피드백 분석 테스트"""
        conversation_id = f"test_conv_{uuid.uuid4().hex[:8]}"
        
        # 피드백 메시지들 저장
        feedback_messages = [
            {
                'id': 'feedback_001',
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'feedback',
                'content': 'Great optimization!',
                'metadata': {'feedback_type': 'positive', 'rating': 5}
            },
            {
                'id': 'feedback_002',
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'feedback',
                'content': 'Could be better',
                'metadata': {'feedback_type': 'negative', 'rating': 2}
            }
        ]
        
        for feedback in feedback_messages:
            redis_repo.save_conversation_message(feedback)
        
        # 피드백 분석
        analytics = redis_repo.get_feedback_analytics(conversation_id)
        
        assert analytics['total_feedback'] == 2
        assert analytics['conversation_id'] == conversation_id
        assert 'analyzed_at' in analytics
    
    def test_health_check(self, redis_repo):
        """헬스 체크 테스트"""
        health_status = redis_repo.health_check()
        
        assert health_status['status'] == 'healthy'
        assert health_status['redis_ping'] is True
        assert health_status['set_get_test'] is True
        assert 'response_time_ms' in health_status
        assert 'redis_version' in health_status
        assert 'used_memory_human' in health_status
    
    def test_ttl_functionality(self, redis_repo):
        """TTL 기능 테스트"""
        conversation_id = f"test_conv_{uuid.uuid4().hex[:8]}"
        message_data = {
            'id': 'msg_ttl_test',
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'user',
            'content': 'TTL test message',
            'metadata': {}
        }
        
        # 메시지 저장
        redis_repo.save_conversation_message(message_data)
        
        # TTL 확인
        messages_key = f"conv:{conversation_id}:messages"
        ttl = redis_repo.redis_client.ttl(messages_key)
        
        # TTL이 설정되어 있는지 확인 (30일 = 2,592,000초)
        assert ttl > 0
        assert ttl <= 30 * 24 * 3600  # 30일 이하
    
    def test_memory_stats(self, redis_repo):
        """메모리 통계 테스트"""
        # 테스트 데이터 생성
        conversation_id = f"test_conv_{uuid.uuid4().hex[:8]}"
        message_data = {
            'id': 'msg_stats_test',
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'user',
            'content': 'Stats test message',
            'metadata': {}
        }
        redis_repo.save_conversation_message(message_data)
        
        # 통계 조회
        stats = redis_repo.get_memory_stats()
        
        assert 'conversations' in stats
        assert 'redis_memory' in stats
        assert 'collected_at' in stats
        
        # 대화 통계 확인
        conv_stats = stats['conversations']
        assert conv_stats['total_conversations'] >= 1
        assert conv_stats['total_message_lists'] >= 1
    
    @patch('redis.Redis')
    def test_connection_error_handling(self, mock_redis):
        """Redis 연결 오류 처리 테스트"""
        # Redis 연결 실패 시뮬레이션
        mock_redis.side_effect = Exception("Connection failed")
        
        with pytest.raises(MemoryRepositoryError):
            RedisMemoryRepository()
    
    def test_invalid_message_data(self, redis_repo):
        """잘못된 메시지 데이터 처리 테스트"""
        # 필수 필드 누락
        invalid_message = {
            'id': 'invalid_msg',
            # conversation_id 누락
            'timestamp': datetime.now().isoformat(),
            'message_type': 'user',
            'content': 'Invalid message'
        }
        
        with pytest.raises(Exception):  # KeyError 또는 MemoryRepositoryError
            redis_repo.save_conversation_message(invalid_message)
    
    def test_large_message_handling(self, redis_repo):
        """대용량 메시지 처리 테스트"""
        conversation_id = f"test_conv_{uuid.uuid4().hex[:8]}"
        
        # 큰 메시지 생성 (1MB)
        large_content = "x" * (1024 * 1024)
        large_message = {
            'id': 'large_msg',
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'user',
            'content': large_content,
            'metadata': {}
        }
        
        # 대용량 메시지 저장 및 조회
        message_id = redis_repo.save_conversation_message(large_message)
        assert message_id == 'large_msg'
        
        retrieved_messages = redis_repo.get_conversation_messages(conversation_id)
        assert len(retrieved_messages) == 1
        assert len(retrieved_messages[0]['content']) == 1024 * 1024 