"""
API 엔드포인트 통합 테스트

실제 Chalice 애플리케이션의 API 엔드포인트를 테스트합니다.
"""
import pytest
import json
import requests
import time
from typing import Dict, Any
from unittest.mock import patch, Mock

from tests import (
    generate_vrp_scenario, generate_tsp_scenario, generate_test_vehicle,
    generate_test_order, validate_optimization_response, validate_feedback_response,
    PerformanceTimer, PERFORMANCE_THRESHOLD_MS
)


class TestAPIEndpoints:
    """API 엔드포인트 통합 테스트"""
    
    @pytest.fixture
    def api_base_url(self):
        """API 베이스 URL"""
        # 로컬 테스트 환경
        return "http://localhost:8000"
    
    @pytest.fixture
    def mock_ai_service(self):
        """AI 서비스 모킹"""
        with patch('src.infrastructure.ai.langchain_service.LangChainAIService') as mock:
            # 성공적인 AI 응답 모킹
            mock_response = {
                "success": True,
                "analysis": "테스트 분석 결과",
                "solution": {
                    "routes": [
                        {
                            "vehicle_id": "V001",
                            "orders": ["O001", "O002"],
                            "waypoints": [
                                {
                                    "location": {"lat": 37.5665, "lng": 126.9780},
                                    "type": "start",
                                    "estimated_arrival": "2024-01-01T09:00:00",
                                    "estimated_duration_minutes": 0
                                },
                                {
                                    "location": {"lat": 37.5547, "lng": 126.9706},
                                    "type": "pickup",
                                    "order_id": "O001",
                                    "estimated_arrival": "2024-01-01T09:30:00",
                                    "estimated_duration_minutes": 30
                                }
                            ],
                            "total_distance_km": 15.5,
                            "total_duration_hours": 2.5,
                            "estimated_cost": 50000,
                            "polyline": "test_polyline_encoded_string",
                            "efficiency_score": 0.85
                        }
                    ],
                    "summary": {
                        "total_vehicles_used": 1,
                        "total_orders_assigned": 2,
                        "total_distance_km": 15.5,
                        "total_cost": 50000,
                        "average_efficiency": 0.85
                    }
                },
                "reasoning": "테스트 추론 과정",
                "confidence_score": 0.9,
                "recommendations": ["테스트 추천사항"],
                "warnings": []
            }
            
            mock_instance = Mock()
            mock_instance.process_tms_request.return_value = Mock(
                success=True,
                response_data=mock_response,
                raw_response=json.dumps(mock_response),
                scenario_type="vrp",
                confidence_score=0.9,
                token_usage={"total_tokens": 1500},
                processing_time_ms=2000,
                validation_report={"status": "valid"}
            )
            
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_health_endpoint(self, api_base_url):
        """헬스 체크 엔드포인트 테스트"""
        response = requests.get(f"{api_base_url}/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "tms-router-ai"
        assert "timestamp" in data
        assert "components" in data
    
    @pytest.mark.integration
    def test_optimize_route_vrp_scenario(self, api_base_url, mock_ai_service):
        """VRP 시나리오 경로 최적화 테스트"""
        # VRP 테스트 시나리오 생성
        vrp_data = generate_vrp_scenario(vehicle_count=2, order_count=5)
        
        with PerformanceTimer() as timer:
            response = requests.post(
                f"{api_base_url}/optimize-route",
                json=vrp_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
        
        # 응답 검증
        assert response.status_code == 200
        assert timer.elapsed_ms() < PERFORMANCE_THRESHOLD_MS
        
        data = response.json()
        assert validate_optimization_response(data)
        
        # 응답 구조 검증
        solution = data["data"]["solution"]
        assert len(solution["routes"]) >= 1
        assert solution["summary"]["total_vehicles_used"] >= 1
        assert solution["summary"]["total_orders_assigned"] == 5
        
        # 각 경로에 폴리라인 포함 확인
        for route in solution["routes"]:
            assert "polyline" in route
            assert route["polyline"] != ""
    
    @pytest.mark.integration
    def test_optimize_route_tsp_scenario(self, api_base_url, mock_ai_service):
        """TSP 시나리오 경로 최적화 테스트"""
        # TSP 테스트 시나리오 생성
        tsp_data = generate_tsp_scenario(order_count=8)
        
        response = requests.post(
            f"{api_base_url}/optimize-route",
            json=tsp_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert validate_optimization_response(data)
        
        # TSP 특성 검증 (단일 차량)
        solution = data["data"]["solution"]
        assert len(solution["routes"]) == 1  # TSP는 단일 경로
        assert solution["summary"]["total_vehicles_used"] == 1
        assert solution["summary"]["total_orders_assigned"] == 8
    
    @pytest.mark.integration
    def test_feedback_submission(self, api_base_url):
        """피드백 제출 테스트"""
        feedback_data = {
            "conversation_id": "test_conv_123",
            "feedback_type": "positive",
            "feedback_content": "경로 최적화가 매우 효율적이었습니다!",
            "rating": 5
        }
        
        response = requests.post(
            f"{api_base_url}/feedback",
            json=feedback_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert validate_feedback_response(data)
        
        # 피드백 ID 생성 확인
        feedback_result = data["data"]
        assert "feedback_id" in feedback_result
        assert feedback_result["status"] == "success"
    
    def test_invalid_request_validation(self, api_base_url):
        """잘못된 요청 검증 테스트"""
        # 필수 필드 누락
        invalid_data = {
            "vehicles": [],  # 빈 배열
            "orders": [generate_test_order()]
        }
        
        response = requests.post(
            f"{api_base_url}/optimize-route",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400  # Bad Request
        
        error_data = response.json()
        assert error_data["status"] == "error"
        assert "error" in error_data
    
    def test_capacity_constraint_validation(self, api_base_url):
        """용량 제약 검증 테스트"""
        # 용량 초과 시나리오
        small_vehicle = generate_test_vehicle(capacity_tons=1.0)
        heavy_orders = [
            generate_test_order(weight_tons=2.0),
            generate_test_order(weight_tons=3.0)
        ]
        
        overweight_data = {
            "vehicles": [small_vehicle],
            "orders": heavy_orders,
            "constraints": {}
        }
        
        response = requests.post(
            f"{api_base_url}/optimize-route",
            json=overweight_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400  # Validation Error
        
        error_data = response.json()
        assert "capacity" in error_data["error"]["message"].lower()
    
    def test_coordinate_validation(self, api_base_url):
        """좌표 검증 테스트"""
        # 잘못된 좌표
        invalid_vehicle = generate_test_vehicle()
        invalid_vehicle["current_location"] = {"lat": 200.0, "lng": 300.0}  # 범위 초과
        
        invalid_data = {
            "vehicles": [invalid_vehicle],
            "orders": [generate_test_order()],
            "constraints": {}
        }
        
        response = requests.post(
            f"{api_base_url}/optimize-route",
            json=invalid_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 400
        
        error_data = response.json()
        assert "lat" in error_data["error"]["message"].lower() or "coordinate" in error_data["error"]["message"].lower()
    
    @pytest.mark.integration
    def test_conversation_continuity(self, api_base_url, mock_ai_service):
        """대화 연속성 테스트"""
        conversation_id = "test_conversation_continuity"
        
        # 첫 번째 최적화 요청
        first_request = generate_vrp_scenario(vehicle_count=1, order_count=3)
        first_request["conversation_id"] = conversation_id
        
        response1 = requests.post(
            f"{api_base_url}/optimize-route",
            json=first_request,
            headers={"Content-Type": "application/json"}
        )
        
        assert response1.status_code == 200
        
        # 피드백 제출
        feedback_data = {
            "conversation_id": conversation_id,
            "feedback_type": "suggestion",
            "feedback_content": "더 짧은 경로를 제안해주세요",
            "rating": 3
        }
        
        feedback_response = requests.post(
            f"{api_base_url}/feedback",
            json=feedback_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert feedback_response.status_code == 200
        
        # 두 번째 최적화 요청 (같은 대화)
        second_request = generate_vrp_scenario(vehicle_count=1, order_count=3)
        second_request["conversation_id"] = conversation_id
        
        response2 = requests.post(
            f"{api_base_url}/optimize-route",
            json=second_request,
            headers={"Content-Type": "application/json"}
        )
        
        assert response2.status_code == 200
        
        # 두 번째 응답이 피드백을 고려했는지 확인 (메타데이터에서)
        data2 = response2.json()
        assert "metadata" in data2["data"]
    
    def test_concurrent_requests(self, api_base_url, mock_ai_service):
        """동시 요청 처리 테스트"""
        import concurrent.futures
        import threading
        
        def make_request(request_id):
            """개별 요청 실행"""
            test_data = generate_vrp_scenario(vehicle_count=1, order_count=2)
            test_data["conversation_id"] = f"concurrent_test_{request_id}"
            
            response = requests.post(
                f"{api_base_url}/optimize-route",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            return response.status_code, response.json()
        
        # 5개의 동시 요청
        concurrent_count = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_count) as executor:
            futures = [executor.submit(make_request, i) for i in range(concurrent_count)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 모든 요청이 성공했는지 확인
        for status_code, response_data in results:
            assert status_code == 200
            assert validate_optimization_response(response_data)
    
    @pytest.mark.integration
    def test_error_handling_robustness(self, api_base_url):
        """에러 처리 견고성 테스트"""
        test_cases = [
            # JSON 형식 오류
            ("invalid json", 400),
            # 빈 요청
            ({}, 400),
            # 필수 필드 누락
            ({"vehicles": []}, 400),
            # 타입 오류
            ({"vehicles": "not_array", "orders": []}, 400)
        ]
        
        for test_data, expected_status in test_cases:
            if isinstance(test_data, str):
                # 잘못된 JSON
                response = requests.post(
                    f"{api_base_url}/optimize-route",
                    data=test_data,
                    headers={"Content-Type": "application/json"}
                )
            else:
                response = requests.post(
                    f"{api_base_url}/optimize-route",
                    json=test_data,
                    headers={"Content-Type": "application/json"}
                )
            
            assert response.status_code == expected_status
            
            # 에러 응답 구조 확인
            if response.status_code != 200:
                error_data = response.json()
                assert "status" in error_data
                assert error_data["status"] == "error"
                assert "error" in error_data
    
    @pytest.mark.performance
    def test_response_time_performance(self, api_base_url, mock_ai_service):
        """응답 시간 성능 테스트"""
        # 다양한 크기의 시나리오 테스트
        test_scenarios = [
            {"vehicles": 1, "orders": 5, "max_time_ms": 3000},
            {"vehicles": 2, "orders": 10, "max_time_ms": 4000},
            {"vehicles": 3, "orders": 15, "max_time_ms": 5000}
        ]
        
        for scenario in test_scenarios:
            test_data = generate_vrp_scenario(
                vehicle_count=scenario["vehicles"],
                order_count=scenario["orders"]
            )
            
            with PerformanceTimer() as timer:
                response = requests.post(
                    f"{api_base_url}/optimize-route",
                    json=test_data,
                    headers={"Content-Type": "application/json"}
                )
            
            assert response.status_code == 200
            assert timer.elapsed_ms() < scenario["max_time_ms"], \
                f"Response time {timer.elapsed_ms()}ms exceeded limit {scenario['max_time_ms']}ms"
    
    @pytest.mark.integration
    def test_memory_usage_monitoring(self, api_base_url):
        """메모리 사용량 모니터링 테스트"""
        # 헬스 체크를 통한 메모리 상태 확인
        response = requests.get(f"{api_base_url}/health")
        
        assert response.status_code == 200
        
        health_data = response.json()
        assert "components" in health_data
        assert "memory" in health_data["components"]
        
        # Redis 메모리 정보가 포함되어 있는지 확인
        memory_status = health_data["components"]["memory"]
        assert memory_status in ["operational", "healthy"] 