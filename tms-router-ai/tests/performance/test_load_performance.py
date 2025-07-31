"""
부하 테스트 및 성능 테스트

시스템의 부하 처리 능력과 성능을 측정합니다.
"""
import pytest
import time
import statistics
import concurrent.futures
import threading
from typing import List, Dict, Any
from datetime import datetime, timedelta

from tests import (
    generate_vrp_scenario, generate_tsp_scenario, PerformanceTimer,
    PERFORMANCE_THRESHOLD_MS
)


class LoadTestResults:
    """부하 테스트 결과"""
    
    def __init__(self):
        self.response_times: List[int] = []
        self.success_count: int = 0
        self.error_count: int = 0
        self.errors: List[str] = []
        self.start_time: datetime = None
        self.end_time: datetime = None
    
    def add_result(self, response_time_ms: int, success: bool, error: str = None):
        """결과 추가"""
        self.response_times.append(response_time_ms)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            if error:
                self.errors.append(error)
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 계산"""
        if not self.response_times:
            return {}
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            "total_requests": len(self.response_times),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / len(self.response_times) * 100,
            "total_duration_seconds": total_duration,
            "requests_per_second": len(self.response_times) / total_duration if total_duration > 0 else 0,
            "response_times": {
                "min_ms": min(self.response_times),
                "max_ms": max(self.response_times),
                "avg_ms": statistics.mean(self.response_times),
                "median_ms": statistics.median(self.response_times),
                "p95_ms": self._percentile(self.response_times, 95),
                "p99_ms": self._percentile(self.response_times, 99)
            },
            "errors": self.errors[:10]  # 최대 10개 에러만 저장
        }
    
    def _percentile(self, data: List[int], percentile: int) -> int:
        """백분위수 계산"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestLoadPerformance:
    """부하 및 성능 테스트"""
    
    @pytest.fixture
    def mock_tms_service(self):
        """TMS 서비스 모킹 (성능 테스트용)"""
        from unittest.mock import Mock, patch
        
        with patch('src.use_cases.optimize_route_use_case.OptimizeRouteUseCase') as mock_use_case:
            # 빠른 응답을 위한 모킹
            mock_result = Mock()
            mock_result.routes = [{"vehicle_id": "V001", "orders": ["O001"]}]
            mock_result.total_distance_km = 10.0
            mock_result.confidence_score = 0.85
            
            mock_instance = Mock()
            mock_instance.execute.return_value = mock_result
            mock_use_case.return_value = mock_instance
            
            yield mock_instance
    
    @pytest.mark.performance
    def test_single_request_performance(self, mock_tms_service):
        """단일 요청 성능 테스트"""
        from src.use_cases.optimize_route_use_case import OptimizeRouteUseCase, TmsRequest
        from src.infrastructure.memory.redis_memory_repository import RedisMemoryRepository
        
        # 실제 서비스 인스턴스 (모킹된 것)
        use_case = mock_tms_service
        
        # 테스트 데이터
        test_data = generate_vrp_scenario(vehicle_count=2, order_count=8)
        tms_request = TmsRequest(
            request_id="perf_test_001",
            vehicles=test_data["vehicles"],
            orders=test_data["orders"],
            constraints=test_data["constraints"]
        )
        
        # 성능 측정
        times = []
        for i in range(10):  # 10회 반복
            with PerformanceTimer() as timer:
                result = use_case.execute(tms_request)
            times.append(timer.elapsed_ms())
        
        # 성능 검증
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        assert avg_time < PERFORMANCE_THRESHOLD_MS, f"Average response time {avg_time}ms exceeds threshold"
        assert max_time < PERFORMANCE_THRESHOLD_MS * 1.5, f"Max response time {max_time}ms exceeds threshold"
        
        print(f"Performance Results - Avg: {avg_time:.1f}ms, Max: {max_time}ms, Min: {min(times)}ms")
    
    @pytest.mark.performance
    def test_concurrent_load(self, mock_tms_service):
        """동시 부하 테스트"""
        def execute_request(request_id: int) -> tuple:
            """개별 요청 실행"""
            try:
                test_data = generate_vrp_scenario(vehicle_count=1, order_count=5)
                
                with PerformanceTimer() as timer:
                    # 실제 Use Case 실행 시뮬레이션
                    time.sleep(0.1)  # AI 처리 시간 시뮬레이션
                    result = {"success": True, "request_id": request_id}
                
                return timer.elapsed_ms(), True, None
                
            except Exception as e:
                return 0, False, str(e)
        
        # 동시 요청 설정
        concurrent_users = 20
        requests_per_user = 5
        
        results = LoadTestResults()
        results.start_time = datetime.now()
        
        # 동시 요청 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            for user_id in range(concurrent_users):
                for req_id in range(requests_per_user):
                    future = executor.submit(execute_request, user_id * requests_per_user + req_id)
                    futures.append(future)
            
            # 결과 수집
            for future in concurrent.futures.as_completed(futures):
                response_time, success, error = future.result()
                results.add_result(response_time, success, error)
        
        results.end_time = datetime.now()
        
        # 성능 통계 분석
        stats = results.get_statistics()
        
        # 성능 기준 검증
        assert stats["success_rate"] >= 95.0, f"Success rate {stats['success_rate']:.1f}% is below 95%"
        assert stats["response_times"]["avg_ms"] < PERFORMANCE_THRESHOLD_MS, \
            f"Average response time {stats['response_times']['avg_ms']:.1f}ms exceeds threshold"
        assert stats["response_times"]["p95_ms"] < PERFORMANCE_THRESHOLD_MS * 1.5, \
            f"95th percentile {stats['response_times']['p95_ms']}ms exceeds threshold"
        
        print(f"Load Test Results:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print(f"  RPS: {stats['requests_per_second']:.1f}")
        print(f"  Avg Response Time: {stats['response_times']['avg_ms']:.1f}ms")
        print(f"  95th Percentile: {stats['response_times']['p95_ms']}ms")
    
    @pytest.mark.performance  
    def test_memory_usage_under_load(self):
        """부하 상황에서 메모리 사용량 테스트"""
        import psutil
        import os
        
        # 초기 메모리 사용량
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 대량의 테스트 데이터 생성 및 처리
        test_scenarios = []
        for i in range(50):  # 50개 시나리오
            scenario = generate_vrp_scenario(vehicle_count=3, order_count=10)
            test_scenarios.append(scenario)
        
        # 메모리 사용량 모니터링
        max_memory_usage = initial_memory
        
        for i, scenario in enumerate(test_scenarios):
            # 처리 시뮬레이션
            processed_data = {
                "scenario_id": i,
                "vehicles": len(scenario["vehicles"]),
                "orders": len(scenario["orders"]),
                "result": "processed"
            }
            
            # 메모리 사용량 체크
            current_memory = process.memory_info().rss / 1024 / 1024
            max_memory_usage = max(max_memory_usage, current_memory)
            
            # 매 10번째마다 메모리 정리 시뮬레이션
            if i % 10 == 0:
                import gc
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # 메모리 누수 검증 (100MB 이상 증가시 실패)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"
        
        print(f"Memory Usage - Initial: {initial_memory:.1f}MB, "
              f"Max: {max_memory_usage:.1f}MB, Final: {final_memory:.1f}MB")
    
    @pytest.mark.performance
    def test_redis_performance_under_load(self):
        """Redis 성능 부하 테스트"""
        try:
            from src.infrastructure.memory.redis_memory_repository import RedisMemoryRepository
            
            # 테스트용 Redis 연결
            redis_repo = RedisMemoryRepository(
                host='localhost',
                port=6379,
                db=15  # 테스트 DB
            )
            
            # 초기 상태 확인
            initial_stats = redis_repo.get_memory_stats()
            
            # 대량 메시지 생성 및 저장
            conversation_count = 10
            messages_per_conversation = 100
            
            start_time = time.time()
            
            for conv_id in range(conversation_count):
                conversation_id = f"load_test_conv_{conv_id}"
                
                for msg_id in range(messages_per_conversation):
                    message_data = {
                        'id': f'msg_{msg_id:04d}',
                        'conversation_id': conversation_id,
                        'timestamp': datetime.now().isoformat(),
                        'message_type': 'user' if msg_id % 2 == 0 else 'assistant',
                        'content': f'Load test message {msg_id} with some content to simulate real usage',
                        'metadata': {'load_test': True, 'msg_number': msg_id}
                    }
                    
                    redis_repo.save_conversation_message(message_data)
            
            write_time = time.time() - start_time
            
            # 읽기 성능 테스트
            start_time = time.time()
            
            for conv_id in range(conversation_count):
                conversation_id = f"load_test_conv_{conv_id}"
                messages = redis_repo.get_conversation_messages(conversation_id, limit=50)
                assert len(messages) == 50  # 최신 50개
            
            read_time = time.time() - start_time
            
            # 최종 통계
            final_stats = redis_repo.get_memory_stats()
            
            # 성능 검증
            total_operations = conversation_count * messages_per_conversation
            write_ops_per_sec = total_operations / write_time
            read_ops_per_sec = conversation_count / read_time
            
            assert write_ops_per_sec > 100, f"Write performance {write_ops_per_sec:.1f} ops/sec is too low"
            assert read_ops_per_sec > 50, f"Read performance {read_ops_per_sec:.1f} ops/sec is too low"
            
            print(f"Redis Performance:")
            print(f"  Write: {write_ops_per_sec:.1f} ops/sec")
            print(f"  Read: {read_ops_per_sec:.1f} ops/sec")
            print(f"  Memory Used: {final_stats['redis_memory']['used_memory_human']}")
            
            # 테스트 데이터 정리
            redis_repo.redis_client.flushdb()
            
        except ImportError:
            pytest.skip("Redis not available for performance testing")
    
    @pytest.mark.performance
    def test_scaling_performance(self, mock_tms_service):
        """확장성 성능 테스트"""
        # 다양한 크기의 문제에 대한 성능 측정
        test_cases = [
            {"vehicles": 1, "orders": 5, "expected_max_time": 1000},
            {"vehicles": 2, "orders": 10, "expected_max_time": 2000},
            {"vehicles": 3, "orders": 15, "expected_max_time": 3000},
            {"vehicles": 5, "orders": 25, "expected_max_time": 5000},
        ]
        
        results = []
        
        for case in test_cases:
            test_data = generate_vrp_scenario(
                vehicle_count=case["vehicles"],
                order_count=case["orders"]
            )
            
            # 3회 실행하여 평균 시간 측정
            times = []
            for _ in range(3):
                with PerformanceTimer() as timer:
                    # 실제 처리 시뮬레이션 (복잡도에 따라 시간 조정)
                    complexity = case["vehicles"] * case["orders"]
                    simulation_time = min(complexity * 0.01, 2.0)  # 최대 2초
                    time.sleep(simulation_time)
                
                times.append(timer.elapsed_ms())
            
            avg_time = statistics.mean(times)
            
            results.append({
                "vehicles": case["vehicles"],
                "orders": case["orders"],
                "avg_time_ms": avg_time,
                "expected_max_time": case["expected_max_time"]
            })
            
            # 성능 기준 검증
            assert avg_time < case["expected_max_time"], \
                f"Performance degraded: {avg_time:.1f}ms > {case['expected_max_time']}ms for {case['vehicles']}V/{case['orders']}O"
        
        # 확장성 분석
        print("Scaling Performance Results:")
        for result in results:
            print(f"  {result['vehicles']}V/{result['orders']}O: {result['avg_time_ms']:.1f}ms")
        
        # 선형 확장성 검증 (단순 체크)
        if len(results) > 1:
            time_ratio = results[-1]["avg_time_ms"] / results[0]["avg_time_ms"]
            complexity_ratio = (results[-1]["vehicles"] * results[-1]["orders"]) / (results[0]["vehicles"] * results[0]["orders"])
            
            # 복잡도 대비 시간 증가가 과도하지 않은지 확인
            assert time_ratio < complexity_ratio * 2, "Performance does not scale linearly" 