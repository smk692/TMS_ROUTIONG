"""
RedisMemoryRepository - Redis 기반 메모리 저장소

Redis를 사용하여 대화 기록과 피드백을 저장합니다.
로컬 개발: Docker Compose Redis
프로덕션: ElastiCache Redis
"""
import json
import redis
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

from src.shared.exceptions import MemoryRepositoryError
from src.shared.constants import MemoryConstants
from src.shared.logging_config import TmsLoggerMixin


class RedisMemoryRepository(TmsLoggerMixin):
    """Redis 기반 메모리 저장소"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: Optional[str] = None, 
                 decode_responses: bool = True):
        """
        Redis 메모리 저장소 초기화
        
        Args:
            host: Redis 호스트
            port: Redis 포트
            db: Redis 데이터베이스 번호
            password: Redis 비밀번호
            decode_responses: 응답 자동 디코딩 여부
        """
        super().__init__()
        
        try:
            # Redis 클라이언트 초기화
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # 연결 테스트
            self.redis_client.ping()
            
            self.logger.info("Redis memory repository initialized", extra={
                'host': host,
                'port': port,
                'db': db
            })
            
        except redis.ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise MemoryRepositoryError(f"Redis connection failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Unexpected error initializing Redis: {e}")
            raise MemoryRepositoryError(f"Failed to initialize Redis repository: {e}")
    
    def save_conversation_message(self, message_data: Dict[str, Any]) -> str:
        """
        대화 메시지 저장
        
        Args:
            message_data: 메시지 데이터
            
        Returns:
            메시지 ID
        """
        try:
            conversation_id = message_data['conversation_id']
            message_id = message_data['id']
            
            # Redis 키 구성
            messages_key = f"conv:{conversation_id}:messages"
            message_key = f"msg:{message_id}"
            meta_key = f"conv:{conversation_id}:meta"
            
            # 메시지 데이터 직렬화
            message_json = json.dumps({
                'id': message_data['id'],
                'timestamp': message_data['timestamp'],
                'message_type': message_data['message_type'],
                'content': message_data['content'],
                'metadata': message_data.get('metadata', {})
            })
            
            # Redis Pipeline 사용 (원자적 연산)
            pipe = self.redis_client.pipeline()
            
            # 1. 메시지를 리스트에 추가 (최신순)
            pipe.lpush(messages_key, message_json)
            
            # 2. 메시지 개별 저장 (빠른 조회용)
            pipe.hset(message_key, mapping={
                'conversation_id': conversation_id,
                'data': message_json
            })
            
            # 3. 대화 메타데이터 업데이트
            pipe.hincrby(meta_key, 'message_count', 1)
            pipe.hset(meta_key, 'last_updated', datetime.now().isoformat())
            
            # 4. TTL 설정 (30일)
            ttl_seconds = MemoryConstants.MESSAGE_TTL_DAYS * 24 * 3600
            pipe.expire(messages_key, ttl_seconds)
            pipe.expire(message_key, ttl_seconds)
            pipe.expire(meta_key, MemoryConstants.SUMMARY_TTL_DAYS * 24 * 3600)
            
            # 5. 메시지 수 제한 (최대 1000개)
            pipe.ltrim(messages_key, 0, 999)
            
            # Pipeline 실행
            pipe.execute()
            
            self.logger.debug("Message saved to Redis", extra={
                'message_id': message_id,
                'conversation_id': conversation_id,
                'message_type': message_data['message_type']
            })
            
            return message_id
            
        except redis.RedisError as e:
            self.logger.error("Failed to save message to Redis", extra={
                'error': str(e),
                'message_id': message_data.get('id')
            })
            raise MemoryRepositoryError(f"Failed to save message: {e}")
        
        except Exception as e:
            self.logger.error("Unexpected error saving message", extra={
                'error': str(e),
                'message_id': message_data.get('id')
            })
            raise MemoryRepositoryError(f"Failed to save message: {e}")
    
    def get_conversation_messages(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        대화 메시지 조회
        
        Args:
            conversation_id: 대화 ID
            limit: 조회할 메시지 수 제한
            
        Returns:
            메시지 리스트 (최신순)
        """
        try:
            messages_key = f"conv:{conversation_id}:messages"
            
            # Redis에서 최신 메시지 조회 (LRANGE는 최신순으로 저장된 것을 가져옴)
            message_jsons = self.redis_client.lrange(messages_key, 0, limit - 1)
            
            messages = []
            for message_json in message_jsons:
                try:
                    message_data = json.loads(message_json)
                    messages.append(message_data)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse message JSON: {e}")
                    continue
            
            self.logger.debug("Messages retrieved from Redis", extra={
                'conversation_id': conversation_id,
                'message_count': len(messages),
                'requested_limit': limit
            })
            
            return messages
            
        except redis.RedisError as e:
            self.logger.error("Failed to get messages from Redis", extra={
                'error': str(e),
                'conversation_id': conversation_id
            })
            raise MemoryRepositoryError(f"Failed to get messages: {e}")
        
        except Exception as e:
            self.logger.error("Unexpected error getting messages", extra={
                'error': str(e),
                'conversation_id': conversation_id
            })
            raise MemoryRepositoryError(f"Failed to get messages: {e}")
    
    def get_conversation_memory(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        대화 메모리 조회 (요약된 컨텍스트)
        
        Args:
            conversation_id: 대화 ID
            
        Returns:
            대화 메모리 또는 None
        """
        try:
            meta_key = f"conv:{conversation_id}:meta"
            
            # 대화 메타데이터 조회
            meta_data = self.redis_client.hgetall(meta_key)
            
            if not meta_data:
                return None
            
            # 요약 정보 조회 (별도 키에 저장된 경우)
            summary_key = f"conv:{conversation_id}:summary"
            summary_data = self.redis_client.get(summary_key)
            
            memory = {
                'conversation_id': conversation_id,
                'message_count': int(meta_data.get('message_count', 0)),
                'feedback_count': int(meta_data.get('feedback_count', 0)),
                'last_updated': meta_data.get('last_updated'),
                'summary': json.loads(summary_data) if summary_data else {}
            }
            
            self.logger.debug("Conversation memory retrieved", extra={
                'conversation_id': conversation_id,
                'message_count': memory['message_count']
            })
            
            return memory
            
        except redis.RedisError as e:
            self.logger.error("Failed to get conversation memory", extra={
                'error': str(e),
                'conversation_id': conversation_id
            })
            return None
        
        except Exception as e:
            self.logger.error("Unexpected error getting conversation memory", extra={
                'error': str(e),
                'conversation_id': conversation_id
            })
            return None
    
    def update_conversation_summary(self, conversation_id: str, summary_data: Dict[str, Any]):
        """
        대화 요약 업데이트
        
        Args:
            conversation_id: 대화 ID
            summary_data: 요약 데이터
        """
        try:
            summary_key = f"conv:{conversation_id}:summary"
            meta_key = f"conv:{conversation_id}:meta"
            
            # Pipeline으로 원자적 업데이트
            pipe = self.redis_client.pipeline()
            
            # 요약 데이터 저장
            pipe.set(summary_key, json.dumps(summary_data))
            
            # 메타데이터 업데이트
            pipe.hset(meta_key, mapping={
                'last_updated': datetime.now().isoformat(),
                'summary_updated': datetime.now().isoformat()
            })
            
            # TTL 설정
            summary_ttl = MemoryConstants.SUMMARY_TTL_DAYS * 24 * 3600
            pipe.expire(summary_key, summary_ttl)
            pipe.expire(meta_key, summary_ttl)
            
            pipe.execute()
            
            self.logger.debug("Conversation summary updated", extra={
                'conversation_id': conversation_id
            })
            
        except redis.RedisError as e:
            self.logger.error("Failed to update conversation summary", extra={
                'error': str(e),
                'conversation_id': conversation_id
            })
            raise MemoryRepositoryError(f"Failed to update summary: {e}")
        
        except Exception as e:
            self.logger.error("Unexpected error updating summary", extra={
                'error': str(e),
                'conversation_id': conversation_id
            })
            raise MemoryRepositoryError(f"Failed to update summary: {e}")
    
    def delete_old_conversations(self, cutoff_date: datetime) -> int:
        """
        오래된 대화 삭제
        
        Args:
            cutoff_date: 삭제 기준 날짜
            
        Returns:
            삭제된 대화 수
        """
        try:
            # Redis는 TTL로 자동 만료되므로 수동 삭제는 최소화
            # 필요시 패턴 검색으로 오래된 키 찾아서 삭제
            
            cutoff_timestamp = cutoff_date.isoformat()
            deleted_count = 0
            
            # 모든 대화 메타데이터 키 검색
            meta_keys = self.redis_client.keys("conv:*:meta")
            
            for meta_key in meta_keys:
                try:
                    last_updated = self.redis_client.hget(meta_key, 'last_updated')
                    
                    if last_updated and last_updated < cutoff_timestamp:
                        # 해당 대화의 모든 관련 키 삭제
                        conversation_id = meta_key.split(':')[1]
                        
                        keys_to_delete = [
                            f"conv:{conversation_id}:messages",
                            f"conv:{conversation_id}:meta", 
                            f"conv:{conversation_id}:summary"
                        ]
                        
                        # 관련 메시지 키들도 찾아서 삭제
                        msg_keys = self.redis_client.keys(f"msg:*")
                        for msg_key in msg_keys:
                            msg_conv_id = self.redis_client.hget(msg_key, 'conversation_id')
                            if msg_conv_id == conversation_id:
                                keys_to_delete.append(msg_key)
                        
                        # 배치 삭제
                        if keys_to_delete:
                            self.redis_client.delete(*keys_to_delete)
                            deleted_count += 1
                            
                except Exception as e:
                    self.logger.warning(f"Failed to check/delete conversation {meta_key}: {e}")
                    continue
            
            self.logger.info("Old conversations deleted", extra={
                'deleted_count': deleted_count,
                'cutoff_date': cutoff_timestamp
            })
            
            return deleted_count
            
        except redis.RedisError as e:
            self.logger.error("Failed to delete old conversations", extra={
                'error': str(e),
                'cutoff_date': cutoff_date.isoformat()
            })
            return 0
        
        except Exception as e:
            self.logger.error("Unexpected error deleting conversations", extra={
                'error': str(e)
            })
            return 0
    
    def get_feedback_analytics(self, conversation_id: Optional[str] = None, 
                             days: int = 30) -> Dict[str, Any]:
        """
        피드백 분석 데이터 조회
        
        Args:
            conversation_id: 특정 대화 ID (None이면 전체)
            days: 분석 기간 (일)
            
        Returns:
            피드백 분석 결과
        """
        try:
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            if conversation_id:
                # 특정 대화의 피드백만 조회
                messages = self.get_conversation_messages(conversation_id, 1000)
                feedback_items = [
                    msg for msg in messages 
                    if msg.get('message_type') == 'feedback' and msg.get('timestamp', '') >= start_date
                ]
            else:
                # 전체 피드백 조회 (비효율적이지만 간단 구현)
                feedback_items = []
                feedback_keys = self.redis_client.keys("feedback:*")
                
                for key in feedback_keys:
                    feedback_data = self.redis_client.hgetall(key)
                    if feedback_data.get('timestamp', '') >= start_date:
                        feedback_items.append(feedback_data)
            
            # 피드백 통계 계산
            total_feedback = len(feedback_items)
            feedback_types = {}
            ratings = []
            
            for item in feedback_items:
                if isinstance(item, dict):
                    feedback_type = item.get('metadata', {}).get('feedback_type', 'unknown')
                    if isinstance(item.get('metadata'), str):
                        metadata = json.loads(item['metadata'])
                        feedback_type = metadata.get('feedback_type', 'unknown')
                    
                    feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1
                    
                    rating = item.get('metadata', {}).get('rating')
                    if isinstance(item.get('metadata'), str):
                        metadata = json.loads(item['metadata'])
                        rating = metadata.get('rating')
                    
                    if rating:
                        ratings.append(int(rating))
            
            analytics = {
                'total_feedback': total_feedback,
                'feedback_types': feedback_types,
                'average_rating': sum(ratings) / len(ratings) if ratings else None,
                'rating_distribution': {i: ratings.count(i) for i in range(1, 6)} if ratings else {},
                'period_days': days,
                'conversation_id': conversation_id,
                'analyzed_at': datetime.now().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error("Failed to get feedback analytics", extra={
                'error': str(e),
                'conversation_id': conversation_id
            })
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Redis 헬스 체크
        
        Returns:
            헬스 체크 결과
        """
        try:
            # Redis 연결 및 기본 동작 테스트
            start_time = datetime.now()
            
            # 1. PING 테스트
            pong = self.redis_client.ping()
            
            # 2. 간단한 SET/GET 테스트
            test_key = f"health_check:{datetime.now().timestamp()}"
            test_value = "health_test"
            
            self.redis_client.set(test_key, test_value, ex=10)  # 10초 후 만료
            retrieved_value = self.redis_client.get(test_key)
            
            # 3. 테스트 키 삭제
            self.redis_client.delete(test_key)
            
            # 4. INFO 명령으로 Redis 상태 정보 조회
            redis_info = self.redis_client.info()
            
            end_time = datetime.now()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            health_status = {
                'status': 'healthy',
                'redis_ping': pong,
                'set_get_test': retrieved_value == test_value,
                'response_time_ms': response_time_ms,
                'redis_version': redis_info.get('redis_version'),
                'connected_clients': redis_info.get('connected_clients'),
                'used_memory_human': redis_info.get('used_memory_human'),
                'keyspace': {
                    db_key: db_info for db_key, db_info in redis_info.items() 
                    if db_key.startswith('db')
                }
            }
            
            return health_status
            
        except redis.RedisError as e:
            return {
                'status': 'unhealthy',
                'error': f'Redis error: {str(e)}',
                'error_type': 'redis_error'
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy', 
                'error': f'Unexpected error: {str(e)}',
                'error_type': 'unknown_error'
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        메모리 사용량 통계 조회
        
        Returns:
            메모리 통계 정보
        """
        try:
            # 대화별 통계
            conversation_keys = self.redis_client.keys("conv:*:meta")
            total_conversations = len(conversation_keys)
            
            # 메시지 통계
            message_keys = self.redis_client.keys("conv:*:messages")
            total_message_lists = len(message_keys)
            
            # 피드백 통계
            feedback_keys = self.redis_client.keys("feedback:*")
            total_feedbacks = len(feedback_keys)
            
            # Redis 메모리 정보
            redis_info = self.redis_client.info('memory')
            
            stats = {
                'conversations': {
                    'total_conversations': total_conversations,
                    'total_message_lists': total_message_lists,
                    'total_feedbacks': total_feedbacks
                },
                'redis_memory': {
                    'used_memory': redis_info.get('used_memory'),
                    'used_memory_human': redis_info.get('used_memory_human'),
                    'used_memory_peak': redis_info.get('used_memory_peak'),
                    'used_memory_peak_human': redis_info.get('used_memory_peak_human')
                },
                'collected_at': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get memory stats", extra={'error': str(e)})
            return {'error': str(e)} 