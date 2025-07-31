"""
TMS Router AI 테스트 패키지

단위 테스트, 통합 테스트, 성능 테스트를 포함합니다.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import uuid

# 테스트 공통 설정
TEST_TIMEOUT = 30  # 테스트 타임아웃 (초)
PERFORMANCE_THRESHOLD_MS = 5000  # 성능 임계값 (밀리초)

# 테스트 데이터 생성 헬퍼
def generate_test_vehicle(vehicle_id: str = None, **kwargs) -> Dict[str, Any]:
    """테스트용 차량 데이터 생성"""
    vehicle_id = vehicle_id or f"test_vehicle_{uuid.uuid4().hex[:8]}"
    
    default_vehicle = {
        "vehicle_id": vehicle_id,
        "capacity_tons": 5.0,
        "current_location": {"lat": 37.5665, "lng": 126.9780},  # 서울시청
        "driver_id": f"driver_{vehicle_id}",
        "fuel_efficiency_kmpl": 12.0,
        "hourly_cost": 25000.0,
        "fuel_cost_per_liter": 1600.0,
        "special_capabilities": []
    }
    
    default_vehicle.update(kwargs)
    return default_vehicle

def generate_test_order(order_id: str = None, **kwargs) -> Dict[str, Any]:
    """테스트용 주문 데이터 생성"""
    order_id = order_id or f"test_order_{uuid.uuid4().hex[:8]}"
    
    default_order = {
        "order_id": order_id,
        "pickup_location": {"lat": 37.5547, "lng": 126.9706},  # 용산구
        "delivery_location": {"lat": 37.5172, "lng": 127.0473},  # 강남구
        "weight_tons": 2.0,
        "volume_cbm": 1.5,
        "priority": "MEDIUM",
        "customer_id": f"customer_{order_id}",
        "special_requirements": []
    }
    
    default_order.update(kwargs)
    return default_order

def generate_test_constraints(**kwargs) -> Dict[str, Any]:
    """테스트용 제약조건 생성"""
    default_constraints = {
        "max_distance_km": 100,
        "max_duration_hours": 8,
        "working_hours": {
            "start": "09:00",
            "end": "18:00"
        },
        "avoid_tolls": False,
        "optimize_for": "distance"  # distance, time, cost
    }
    
    default_constraints.update(kwargs)
    return default_constraints

def generate_vrp_scenario(vehicle_count: int = 2, order_count: int = 5) -> Dict[str, Any]:
    """VRP 테스트 시나리오 생성"""
    vehicles = [generate_test_vehicle(f"V{i+1:03d}") for i in range(vehicle_count)]
    orders = [generate_test_order(f"O{i+1:03d}") for i in range(order_count)]
    constraints = generate_test_constraints()
    
    return {
        "scenario_type": "vrp",
        "vehicles": vehicles,
        "orders": orders,
        "constraints": constraints
    }

def generate_tsp_scenario(order_count: int = 8) -> Dict[str, Any]:
    """TSP 테스트 시나리오 생성"""
    vehicles = [generate_test_vehicle("TSP_VEHICLE")]
    orders = [generate_test_order(f"TSP_O{i+1:03d}") for i in range(order_count)]
    constraints = generate_test_constraints()
    
    return {
        "scenario_type": "tsp",
        "vehicles": vehicles,
        "orders": orders,
        "constraints": constraints
    }

def generate_seoul_locations(count: int = 10) -> List[Dict[str, float]]:
    """서울 지역 테스트 위치 생성"""
    # 서울 주요 지역 좌표
    seoul_locations = [
        {"lat": 37.5665, "lng": 126.9780},  # 중구 (시청)
        {"lat": 37.5172, "lng": 127.0473},  # 강남구
        {"lat": 37.5547, "lng": 126.9706},  # 용산구
        {"lat": 37.5636, "lng": 126.9976},  # 종로구
        {"lat": 37.5133, "lng": 127.0592},  # 서초구
        {"lat": 37.5319, "lng": 126.9918},  # 동작구
        {"lat": 37.5208, "lng": 126.9745},  # 영등포구
        {"lat": 37.5735, "lng": 126.9788},  # 마포구
        {"lat": 37.6022, "lng": 127.0163},  # 성북구
        {"lat": 37.5732, "lng": 127.0469},  # 성동구
        {"lat": 37.5435, "lng": 127.0982},  # 송파구
        {"lat": 37.4979, "lng": 127.0628},  # 강동구
    ]
    
    # 요청된 수만큼 반복하여 반환
    return (seoul_locations * ((count // len(seoul_locations)) + 1))[:count]

# 테스트 결과 검증 헬퍼
def validate_optimization_response(response: Dict[str, Any]) -> bool:
    """최적화 응답 형식 검증"""
    required_fields = ["status", "data", "timestamp"]
    if not all(field in response for field in required_fields):
        return False
    
    if response["status"] != "success":
        return False
    
    data = response["data"]
    required_data_fields = ["solution", "analysis", "confidence_score"]
    if not all(field in data for field in required_data_fields):
        return False
    
    solution = data["solution"]
    if "routes" not in solution or "summary" not in solution:
        return False
    
    # 각 경로에 폴리라인이 있는지 확인
    for route in solution["routes"]:
        if "polyline" not in route:
            return False
        if route["polyline"] == "":  # 빈 폴리라인은 실패로 간주
            return False
    
    return True

def validate_feedback_response(response: Dict[str, Any]) -> bool:
    """피드백 응답 형식 검증"""
    required_fields = ["status", "data", "timestamp"]
    if not all(field in response for field in response):
        return False
    
    if response["status"] != "success":
        return False
    
    data = response["data"]
    required_data_fields = ["feedback_id", "status", "message"]
    return all(field in data for field in required_data_fields)

# 성능 측정 헬퍼
class PerformanceTimer:
    """성능 측정 타이머"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = datetime.now()
    
    def stop(self):
        self.end_time = datetime.now()
    
    def elapsed_ms(self) -> int:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# 테스트 설정
pytest_plugins = ["pytest_asyncio"] 