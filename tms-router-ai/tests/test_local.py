#!/usr/bin/env python3
"""
TMS Router AI 로컬 테스트 스크립트

실제 API 서버와 Redis를 사용한 통합 테스트를 수행합니다.
"""
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

from tests import (
    generate_vrp_scenario, generate_tsp_scenario, 
    validate_optimization_response, validate_feedback_response,
    PerformanceTimer
)


def test_health_endpoint():
    """헬스 체크 테스트"""
    print("🏥 헬스 체크 테스트...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 헬스 체크 성공: {data['status']}")
            print(f"   서비스: {data['service']}")
            print(f"   타임스탬프: {data['timestamp']}")
            
            if "components" in data:
                print("   컴포넌트 상태:")
                for component, status in data["components"].items():
                    print(f"     - {component}: {status}")
            
            return True
        else:
            print(f"❌ 헬스 체크 실패: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 헬스 체크 연결 실패: {e}")
        print("   chalice local이 실행되어 있는지 확인하세요.")
        return False


def test_optimize_route_vrp():
    """VRP 경로 최적화 테스트"""
    print("\n🚛 VRP 경로 최적화 테스트...")
    
    # 테스트 데이터 생성
    vrp_data = generate_vrp_scenario(vehicle_count=2, order_count=5)
    vrp_data["conversation_id"] = f"local_test_vrp_{int(time.time())}"
    
    print(f"   차량 수: {len(vrp_data['vehicles'])}")
    print(f"   주문 수: {len(vrp_data['orders'])}")
    
    try:
        with PerformanceTimer() as timer:
            response = requests.post(
                "http://localhost:8000/optimize-route",
                json=vrp_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            
            if validate_optimization_response(data):
                solution = data["data"]["solution"]
                print(f"✅ VRP 최적화 성공 ({timer.elapsed_ms()}ms)")
                print(f"   생성된 경로: {len(solution['routes'])}개")
                print(f"   총 거리: {solution['summary']['total_distance_km']:.1f}km")
                print(f"   총 비용: {solution['summary']['total_cost']:,}원")
                print(f"   신뢰도: {data['data']['confidence_score']:.2f}")
                
                # 폴리라인 확인
                for i, route in enumerate(solution["routes"]):
                    polyline_status = "✅ 포함" if route.get("polyline") else "❌ 누락"
                    print(f"   경로 {i+1} 폴리라인: {polyline_status}")
                
                return True
            else:
                print(f"❌ VRP 응답 형식 오류")
                return False
        else:
            print(f"❌ VRP 최적화 실패: HTTP {response.status_code}")
            print(f"   응답: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ VRP 요청 실패: {e}")
        return False


def test_optimize_route_tsp():
    """TSP 경로 최적화 테스트"""
    print("\n🎯 TSP 경로 최적화 테스트...")
    
    # 테스트 데이터 생성
    tsp_data = generate_tsp_scenario(order_count=8)
    tsp_data["conversation_id"] = f"local_test_tsp_{int(time.time())}"
    
    print(f"   단일 차량 주문: {len(tsp_data['orders'])}개")
    
    try:
        with PerformanceTimer() as timer:
            response = requests.post(
                "http://localhost:8000/optimize-route",
                json=tsp_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            
            if validate_optimization_response(data):
                solution = data["data"]["solution"]
                print(f"✅ TSP 최적화 성공 ({timer.elapsed_ms()}ms)")
                print(f"   단일 경로 거리: {solution['summary']['total_distance_km']:.1f}km")
                print(f"   예상 소요시간: {solution['routes'][0]['total_duration_hours']:.1f}시간")
                print(f"   신뢰도: {data['data']['confidence_score']:.2f}")
                
                return True
            else:
                print(f"❌ TSP 응답 형식 오류")
                return False
        else:
            print(f"❌ TSP 최적화 실패: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ TSP 요청 실패: {e}")
        return False


def test_feedback_submission():
    """피드백 제출 테스트"""
    print("\n💬 피드백 제출 테스트...")
    
    feedback_data = {
        "conversation_id": f"local_test_feedback_{int(time.time())}",
        "feedback_type": "positive",
        "feedback_content": "로컬 테스트에서 최적화 결과가 우수했습니다!",
        "rating": 5
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/feedback",
            json=feedback_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if validate_feedback_response(data):
                feedback_result = data["data"]
                print(f"✅ 피드백 제출 성공")
                print(f"   피드백 ID: {feedback_result['feedback_id']}")
                print(f"   상태: {feedback_result['status']}")
                
                return True
            else:
                print(f"❌ 피드백 응답 형식 오류")
                return False
        else:
            print(f"❌ 피드백 제출 실패: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 피드백 요청 실패: {e}")
        return False


def test_conversation_continuity():
    """대화 연속성 테스트"""
    print("\n🔄 대화 연속성 테스트...")
    
    conversation_id = f"local_test_continuity_{int(time.time())}"
    
    # 첫 번째 요청
    first_request = generate_vrp_scenario(vehicle_count=1, order_count=3)
    first_request["conversation_id"] = conversation_id
    
    try:
        response1 = requests.post(
            "http://localhost:8000/optimize-route",
            json=first_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response1.status_code != 200:
            print(f"❌ 첫 번째 요청 실패: HTTP {response1.status_code}")
            return False
        
        # 피드백 제출
        feedback_data = {
            "conversation_id": conversation_id,
            "feedback_type": "suggestion",
            "feedback_content": "더 효율적인 경로를 제안해주세요",
            "rating": 3
        }
        
        feedback_response = requests.post(
            "http://localhost:8000/feedback",
            json=feedback_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if feedback_response.status_code != 200:
            print(f"❌ 피드백 제출 실패: HTTP {feedback_response.status_code}")
            return False
        
        # 두 번째 요청 (같은 대화)
        second_request = generate_vrp_scenario(vehicle_count=1, order_count=3)
        second_request["conversation_id"] = conversation_id
        
        response2 = requests.post(
            "http://localhost:8000/optimize-route",
            json=second_request,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response2.status_code == 200:
            print(f"✅ 대화 연속성 테스트 성공")
            print(f"   대화 ID: {conversation_id}")
            print(f"   요청 순서: 최적화 → 피드백 → 재최적화")
            
            return True
        else:
            print(f"❌ 두 번째 요청 실패: HTTP {response2.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 대화 연속성 테스트 실패: {e}")
        return False


def test_error_handling():
    """에러 처리 테스트"""
    print("\n⚠️ 에러 처리 테스트...")
    
    # 잘못된 요청 데이터
    invalid_data = {
        "vehicles": [],  # 빈 배열
        "orders": [{"invalid": "data"}],  # 잘못된 구조
        "constraints": {}
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/optimize-route",
            json=invalid_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 400:  # Bad Request 기대
            error_data = response.json()
            print(f"✅ 에러 처리 정상 작동")
            print(f"   상태 코드: {response.status_code}")
            print(f"   에러 상태: {error_data.get('status', 'unknown')}")
            print(f"   에러 메시지 포함: {'error' in error_data}")
            
            return True
        else:
            print(f"❌ 예상과 다른 응답: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 에러 처리 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 실행"""
    print("=" * 50)
    print("🚀 TMS Router AI 로컬 테스트 시작")
    print("=" * 50)
    
    # 시작 시간
    start_time = datetime.now()
    
    # 테스트 실행
    tests = [
        ("헬스 체크", test_health_endpoint),
        ("VRP 최적화", test_optimize_route_vrp),
        ("TSP 최적화", test_optimize_route_tsp),
        ("피드백 제출", test_feedback_submission),
        ("대화 연속성", test_conversation_continuity),
        ("에러 처리", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # 결과 요약
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 통과" if success else "❌ 실패"
        print(f"   {test_name}: {status}")
    
    print(f"\n📈 전체 결과: {passed}/{total} 통과 ({passed/total*100:.1f}%)")
    print(f"⏱️ 실행 시간: {duration:.1f}초")
    
    if passed == total:
        print("🎉 모든 테스트가 성공했습니다!")
        return 0
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        return 1


if __name__ == "__main__":
    exit(main()) 