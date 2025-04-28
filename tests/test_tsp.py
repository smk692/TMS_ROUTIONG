import pytest
from src.algorithm.tsp import solve_tsp
from src.model.delivery_point import DeliveryPoint
from datetime import datetime, timedelta
import random
import json
from pathlib import Path
import numpy as np
import time

def create_test_points(num_points: int = 20) -> list:
    """현실적인 테스트 배송지점 생성"""
    points = []
    base_time = datetime.now()
    
    # 수원시 주요 지역 좌표
    areas = [
        (37.2636, 127.0286),  # 수원역
        (37.2849, 127.0169),  # 장안구
        (37.2858, 127.0875),  # 영통구
        (37.2489, 127.0571),  # 팔달구
        (37.2703, 127.0286),  # 권선구
    ]
    
    for i in range(num_points):
        # 실제 지역 기반 좌표 생성
        base_area = random.choice(areas)
        lat = base_area[0] + (random.random() - 0.5) * 0.02  # ±0.01도 (약 1km 반경)
        lon = base_area[1] + (random.random() - 0.5) * 0.02
        
        # 현실적인 시간 제약 설정
        start_hour = random.randint(9, 16)  # 9AM - 4PM
        time_window_start = base_time.replace(hour=start_hour, minute=0)
        time_window_end = time_window_start + timedelta(hours=2)
        
        points.append(DeliveryPoint(
            id=i,
            latitude=lat,
            longitude=lon,
            address1=f"테스트 주소 {i}",
            address2="",
            time_window=(time_window_start, time_window_end),
            service_time=random.randint(5, 15),  # 5-15분 서비스 시간
            volume=random.uniform(0.1, 0.5),     # 0.1-0.5m³
            weight=random.uniform(5, 20),        # 5-20kg
            special_requirements=[],
            priority=random.randint(1, 5)
        ))
    
    return points

def test_tsp_small_scale():
    """소규모 데이터셋(20개 지점)에 대한 TSP 테스트"""
    points = create_test_points(20)
    
    start_time = time.time()
    route, distance = solve_tsp(points)
    execution_time = time.time() - start_time
    
    assert len(route) == len(points), "모든 지점이 경로에 포함되어야 함"
    assert len(set(p.id for p in route)) == len(points), "중복 방문이 없어야 함"
    assert distance > 0, "총 거리는 0보다 커야 함"
    assert execution_time < 1.0, "1초 이내에 완료되어야 함"
    
    print(f"\n소규모 테스트 완료:")
    print(f"- 실행 시간: {execution_time:.2f}초")
    print(f"- 총 거리: {distance:.2f}km")

@pytest.mark.slow
def test_tsp_medium_scale():
    """중규모 데이터셋(50개 지점)에 대한 TSP 테스트"""
    points = create_test_points(50)
    route, distance = solve_tsp(points)
    
    assert len(route) == len(points), "모든 지점이 경로에 포함되어야 함"
    assert len(set(p.id for p in route)) == len(points), "중복 방문이 없어야 함"
    assert distance > 0, "총 거리는 0보다 커야 함"
    
    print(f"중규모 테스트 완료: {len(points)}개 지점, 총 거리 = {distance:.2f}km")

def test_tsp_priority_consideration():
    """우선순위가 반영되는지 테스트"""
    points = create_test_points(30)
    
    # 일부 지점에 높은 우선순위 설정
    high_priority_indices = random.sample(range(len(points)), 5)
    for idx in high_priority_indices:
        points[idx].priority = 1  # 최우선순위
    
    start_time = time.time()
    route, distance = solve_tsp(points)
    execution_time = time.time() - start_time
    
    # 우선순위가 높은 지점들의 방문 순서 확인
    high_priority_positions = [
        route.index(points[idx]) for idx in high_priority_indices
    ]
    avg_position = sum(high_priority_positions) / len(high_priority_positions)
    
    print(f"\n우선순위 테스트 결과:")
    print(f"- 실행 시간: {execution_time:.2f}초")
    print(f"- 고우선순위 지점 평균 방문 순서: {avg_position:.1f}")
    print(f"- 총 거리: {distance:.2f}km")
    
    # 우선순위가 높은 지점들이 평균적으로 더 일찍 방문되어야 함
    assert avg_position < len(points) / 2, "우선순위가 높은 지점들이 더 일찍 방문되어야 함"
    assert execution_time < 2.0, "2초 이내에 완료되어야 함"

def test_tsp_edge_cases():
    """엣지 케이스 테스트"""
    # 빈 리스트 케이스
    start_time = time.time()
    route, distance = solve_tsp([])
    execution_time = time.time() - start_time
    
    assert len(route) == 0 and distance == 0.0, "빈 입력에 대해 빈 경로를 반환해야 함"
    assert execution_time < 0.1, "0.1초 이내에 완료되어야 함"
    
    # 단일 지점 케이스
    single_point = create_test_points(1)
    start_time = time.time()
    route, distance = solve_tsp(single_point)
    execution_time = time.time() - start_time
    
    assert len(route) == 1 and distance == 0.0, "단일 지점에 대해 올바른 결과를 반환해야 함"
    assert execution_time < 0.1, "0.1초 이내에 완료되어야 함"
    
    print("\n엣지 케이스 테스트 완료")

@pytest.mark.slow
def test_tsp_performance():
    """성능 테스트 개선"""
    test_sizes = [20, 50, 100, 200]
    results = []
    
    for size in test_sizes:
        points = create_test_points(size)
        
        # 여러 번 실행하여 평균 시간 측정
        num_runs = 3
        times = []
        distances = []
        
        for run in range(num_runs):
            start_time = time.time()
            route, distance = solve_tsp(points)
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            distances.append(distance)
            
            # 기본 검증
            assert len(route) == size, f"크기 {size}: 모든 지점이 경로에 포함되어야 함"
            assert len(set(p.id for p in route)) == size, f"크기 {size}: 중복 방문이 없어야 함"
        
        avg_time = sum(times) / num_runs
        avg_distance = sum(distances) / num_runs
        std_dev = np.std(times)
        
        results.append({
            'size': size,
            'avg_time': avg_time,
            'min_time': min(times),
            'max_time': max(times),
            'std_dev': std_dev,
            'avg_distance': avg_distance
        })
        
        print(f"\n크기 {size} 테스트 결과:")
        print(f"- 평균 실행 시간: {avg_time:.2f}초")
        print(f"- 최소/최대 시간: {min(times):.2f}s / {max(times):.2f}s")
        print(f"- 표준 편차: {std_dev:.2f}s")
        print(f"- 평균 경로 거리: {avg_distance:.2f}km")
        
        # 성능 요구사항 검증
        time_limits = {
            20: 1.0,   # 1초
            50: 3.0,   # 3초
            100: 10.0, # 10초
            200: 30.0  # 30초
        }
        assert avg_time < time_limits[size], f"크기 {size}: {time_limits[size]}초 이내에 완료되어야 함"
    
    return results

def test_solution_quality():
    """해의 품질 테스트"""
    size = 50
    points = create_test_points(size)
    
    # 여러 번 실행하여 해의 일관성 확인
    num_runs = 5
    solutions = []
    
    for _ in range(num_runs):
        route, distance = solve_tsp(points)
        solutions.append(distance)
    
    # 해의 변동성 검사
    avg_distance = sum(solutions) / num_runs
    max_deviation = max(abs(d - avg_distance) for d in solutions)
    relative_deviation = max_deviation / avg_distance
    
    print(f"\n해의 품질 테스트 결과:")
    print(f"- 평균 거리: {avg_distance:.2f}km")
    print(f"- 최대 편차: {max_deviation:.2f}km ({relative_deviation*100:.1f}%)")
    
    # 해의 안정성 검증
    assert relative_deviation < 0.1, "해의 변동성이 10% 이내여야 함"

if __name__ == "__main__":
    pytest.main(["-v", __file__])