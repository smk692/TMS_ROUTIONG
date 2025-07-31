#!/usr/bin/env python3
"""
Redis 연결 및 TMS 메모리 저장소 테스트 스크립트
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.memory.redis_memory_repository import RedisMemoryRepository
from src.shared.logging_config import setup_logging

def test_redis_connection():
    """Redis 연결 테스트"""
    print("🔌 Redis 연결 테스트...")
    
    try:
        # 환경 변수에서 Redis 설정 읽기
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        redis_db = int(os.environ.get('REDIS_DB', 0))
        
        # Redis 저장소 초기화
        repo = RedisMemoryRepository(
            host=redis_host,
            port=redis_port,
            db=redis_db
        )
        
        print(f"✅ Redis 연결 성공! ({redis_host}:{redis_port}/{redis_db})")
        return repo
        
    except Exception as e:
        print(f"❌ Redis 연결 실패: {e}")
        print("💡 해결 방법:")
        print("   1. Redis가 실행 중인지 확인: docker-compose up -d redis")
        print("   2. 환경 변수 확인: REDIS_HOST, REDIS_PORT")
        return None

def test_memory_operations(repo: RedisMemoryRepository):
    """메모리 저장소 기능 테스트"""
    print("\n📝 메모리 저장소 기능 테스트...")
    
    try:
        # 테스트 대화 ID
        conversation_id = f"test_conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 1. 메시지 저장 테스트
        print("1️⃣ 메시지 저장 테스트...")
        test_messages = [
            {
                'id': 'msg_001',
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'user',
                'content': 'VRP 배차 최적화를 요청합니다',
                'metadata': {'scenario_type': 'vrp'}
            },
            {
                'id': 'msg_002', 
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'assistant',
                'content': '최적화된 배차 계획을 생성했습니다',
                'metadata': {'confidence_score': 0.95}
            },
            {
                'id': 'msg_003',
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'message_type': 'feedback',
                'content': '경로가 효율적입니다!',
                'metadata': {'feedback_type': 'positive', 'rating': 5}
            }
        ]
        
        for message in test_messages:
            message_id = repo.save_conversation_message(message)
            print(f"   ✅ 메시지 저장됨: {message_id}")
        
        # 2. 메시지 조회 테스트
        print("2️⃣ 메시지 조회 테스트...")
        retrieved_messages = repo.get_conversation_messages(conversation_id)
        print(f"   ✅ {len(retrieved_messages)}개 메시지 조회됨")
        
        for msg in retrieved_messages:
            print(f"   - {msg['message_type']}: {msg['content'][:50]}...")
        
        # 3. 대화 메모리 조회 테스트
        print("3️⃣ 대화 메모리 조회 테스트...")
        memory = repo.get_conversation_memory(conversation_id)
        if memory:
            print(f"   ✅ 메모리 조회됨: {memory['message_count']}개 메시지")
            print(f"   - 마지막 업데이트: {memory['last_updated']}")
        
        # 4. 대화 요약 업데이트 테스트
        print("4️⃣ 대화 요약 업데이트 테스트...")
        summary_data = {
            'key_topics': ['VRP', '배차 최적화'],
            'satisfaction_score': 5,
            'optimization_count': 1
        }
        repo.update_conversation_summary(conversation_id, summary_data)
        print("   ✅ 대화 요약 업데이트됨")
        
        # 5. 피드백 분석 테스트
        print("5️⃣ 피드백 분석 테스트...")
        analytics = repo.get_feedback_analytics(conversation_id)
        print(f"   ✅ 피드백 분석 완료: {analytics['total_feedback']}개 피드백")
        
        print(f"\n🎉 모든 테스트 통과! (대화 ID: {conversation_id})")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def test_health_check(repo: RedisMemoryRepository):
    """헬스 체크 테스트"""
    print("\n🏥 헬스 체크 테스트...")
    
    try:
        health_status = repo.health_check()
        print(f"상태: {health_status['status']}")
        print(f"응답 시간: {health_status.get('response_time_ms', 'N/A')}ms")
        print(f"Redis 버전: {health_status.get('redis_version', 'N/A')}")
        print(f"메모리 사용량: {health_status.get('used_memory_human', 'N/A')}")
        
        if health_status['status'] == 'healthy':
            print("✅ 헬스 체크 통과!")
            return True
        else:
            print("❌ 헬스 체크 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 헬스 체크 에러: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🧪 TMS Router AI - Redis 메모리 저장소 테스트")
    print("=" * 50)
    
    # 로깅 설정
    setup_logging()
    
    # Redis 연결 테스트
    repo = test_redis_connection()
    if not repo:
        sys.exit(1)
    
    # 헬스 체크
    if not test_health_check(repo):
        sys.exit(1)
    
    # 메모리 저장소 기능 테스트
    if not test_memory_operations(repo):
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
    print("🚀 이제 TMS Router AI를 안전하게 사용할 수 있습니다!")

if __name__ == "__main__":
    main() 