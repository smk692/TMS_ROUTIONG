import pytest
import numpy as np
from datetime import datetime, timedelta
from src.model.delivery_point import DeliveryPoint
from src.model.vehicle import Vehicle, VehicleCapacity
from src.algorithm.cluster import (
    cluster_points,
    ClusteringStrategy,
    EnhancedKMeansStrategy,
    EnhancedDBSCANStrategy,
    HDBSCANStrategy
)
from src.model.time_window import TimeWindow
from typing import List, Optional
from sklearn.cluster import KMeans
import math
import logging
import random
import copy
import cProfile
import pstats
import io
from pstats import SortKey
import psutil
import os
import time

def create_test_points(n: int) -> List[DeliveryPoint]:
    """테스트용 배송 포인트 생성"""
    points = []
    base_time = datetime.now()
    
    for i in range(n):
        point = DeliveryPoint(
            id=f'DP{i+1}',
            latitude=37.2636 + random.uniform(-0.02, 0.02),
            longitude=127.0286 + random.uniform(-0.02, 0.02),
            address1=f'Test Address 수원역 {i+1}',
            address2=f'상세주소 {i+1}',
            time_window=TimeWindow(
                start=base_time,
                end=base_time + timedelta(hours=8)
            ),
            service_time=15,
            volume=random.uniform(0.5, 2.0),
            weight=random.uniform(10, 50),
            special_requirements=[],
            priority=random.randint(1, 3)
        )
        points.append(point)
    return points

def create_test_vehicles(n: int) -> List[Vehicle]:
    """테스트용 차량 생성"""
    now = datetime.now()
    vehicles = []
    
    for i in range(n):
        vehicle = Vehicle(
            id=f'V{i+1}',
            type='TRUCK_1TON',
            capacity=VehicleCapacity(
                volume=30.0,
                weight=300.0
            ),
            features=['STANDARD'],
            cost_per_km=500.0,
            start_time=now,
            end_time=now + timedelta(hours=8),
            current_location=(37.2636, 127.0286)
        )
        vehicles.append(vehicle)
    return vehicles

def create_test_points_with_requirements(n: int) -> List[DeliveryPoint]:
    """특수 요구사항이 있는 테스트 포인트 생성"""
    points = []
    requirements_options = [
        ['REFRIGERATED'],
        ['LIFT'],
        ['TAIL_LIFT'],
        ['REFRIGERATED', 'LIFT'],
        ['REFRIGERATED', 'TAIL_LIFT'],
        []  # 일반 배송
    ]
    
    base_time = datetime.now()
    
    for i in range(n):
        volume = random.uniform(1.0, 2.0) if random.random() < 0.3 else random.uniform(0.3, 1.0)
        weight = random.uniform(50, 100) if random.random() < 0.3 else random.uniform(10, 50)
        priority = 3 if random.random() < 0.2 else random.randint(1, 2)
        
        start_time = base_time + timedelta(hours=random.randint(0, 4))
        end_time = start_time + timedelta(hours=4)
        
        point = DeliveryPoint(
            id=f'DP{i+1}',
            latitude=37.2636 + random.uniform(-0.05, 0.05),
            longitude=127.0286 + random.uniform(-0.05, 0.05),
            address1=f'Test Address {i+1}',
            address2=f'상세주소 {i+1}',
            time_window=TimeWindow(
                start=start_time,
                end=end_time
            ),
            service_time=15,
            volume=volume,
            weight=weight,
            special_requirements=random.choice(requirements_options),
            priority=priority
        )
        points.append(point)
    
    return points

def create_special_vehicles(n: int) -> List[Vehicle]:
    """특수 기능을 가진 차량 생성"""
    vehicles = []
    features_combinations = [
        ['STANDARD', 'REFRIGERATED', 'LIFT'],
        ['STANDARD', 'LIFT', 'TAIL_LIFT'],
        ['STANDARD', 'REFRIGERATED', 'LIFT', 'TAIL_LIFT']
    ]
    
    now = datetime.now()
    
    for i in range(n):
        vehicle = Vehicle(
            id=f'SV{i+1}',
            type='TRUCK_1TON',
            capacity=VehicleCapacity(
                volume=20.0,
                weight=1500.0
            ),
            features=random.choice(features_combinations),
            cost_per_km=500.0,
            start_time=now,
            end_time=now + timedelta(hours=8),
            current_location=(37.2636, 127.0286)
        )
        vehicles.append(vehicle)
    
    return vehicles

def create_high_priority_points(n: int) -> List[DeliveryPoint]:
    """우선순위가 높은 테스트 포인트 생성"""
    points = create_test_points(n)
    for point in points:
        point.priority = 3
    return points

def test_cluster_small_scale():
    """소규모 클러스터링 테스트 (20개 지점)"""
    points = create_test_points(20)
    vehicles = create_test_vehicles(3)
    
    clusters = cluster_points(points, vehicles, 'enhanced_kmeans')
    
    assert clusters is not None
    assert len(clusters) == 3
    assert sum(len(cluster) for cluster in clusters) == 20
    
    # 각 클러스터의 용량 제약 검증
    for cluster, vehicle in zip(clusters, vehicles):
        total_volume = sum(p.volume for p in cluster)
        total_weight = sum(p.weight for p in cluster)
        assert total_volume <= vehicle.capacity.volume
        assert total_weight <= vehicle.capacity.weight

def test_cluster_medium_scale():
    """중규모 클러스터링 테스트 (50개 지점)"""
    points = create_test_points(50)
    vehicles = create_test_vehicles(5)
    
    clusters = cluster_points(points, vehicles, 'enhanced_kmeans')
    
    assert clusters is not None
    assert len(clusters) == 5
    assert sum(len(cluster) for cluster in clusters) == 50
    
    # 클러스터 크기의 균형 검증
    cluster_sizes = [len(cluster) for cluster in clusters]
    size_diff = max(cluster_sizes) - min(cluster_sizes)
    assert size_diff <= 5  # 클러스터 간 크기 차이가 5 이하여야 함

def test_cluster_large_scale():
    """대규모 클러스터링 테스트 (100개 지점)"""
    points = create_test_points(100)
    vehicles = create_test_vehicles(8)
    
    clusters = cluster_points(points, vehicles, 'enhanced_kmeans')
    
    assert clusters is not None
    assert len(clusters) == 8
    assert sum(len(cluster) for cluster in clusters) == 100

def test_cluster_priority_consideration():
    """우선순위 고려 테스트"""
    points = create_test_points(30)
    vehicles = create_test_vehicles(3)
    
    # 일부 포인트의 우선순위를 높게 설정
    high_priority_indices = [0, 5, 10, 15, 20, 25]
    for idx in high_priority_indices:
        points[idx].priority = 3
    
    clusters = cluster_points(points, vehicles, 'enhanced_kmeans')
    
    # 우선순위가 높은 포인트들이 잘 분산되었는지 확인
    high_priority_distribution = [
        sum(1 for p in cluster if p.priority == 3)
        for cluster in clusters
    ]
    
    # 각 클러스터에 최소 1개 이상의 우선순위 높은 포인트가 있어야 함
    assert all(count > 0 for count in high_priority_distribution)

def test_cluster_time_window_consideration():
    """시간 윈도우 고려 테스트"""
    points = create_test_points(40)
    vehicles = create_test_vehicles(4)
    
    clusters = cluster_points(points, vehicles, 'enhanced_kmeans')
    
    for cluster in clusters:
        # 각 클러스터의 시간 윈도우 범위 계산
        start_times = [p.time_window.start for p in cluster]
        end_times = [p.time_window.end for p in cluster]
        
        time_span = max(end_times) - min(start_times)
        assert time_span.total_seconds() <= 8 * 3600  # 8시간 이내

def test_cluster_strategy_comparison():
    """클러스터링 전략 비교 테스트"""
    points = create_test_points(60)
    vehicles = create_test_vehicles(5)
    
    strategies = ['enhanced_kmeans', 'enhanced_dbscan', 'hdbscan']
    results = {}
    
    for strategy in strategies:
        start_time = datetime.now()
        clusters = cluster_points(points, vehicles, strategy)
        end_time = datetime.now()
        
        assert clusters is not None
        
        # 성능 메트릭 계산
        execution_time = (end_time - start_time).total_seconds()
        cluster_sizes = [len(cluster) for cluster in clusters]
        size_variance = np.var(cluster_sizes)
        
        results[strategy] = {
            'execution_time': execution_time,
            'size_variance': size_variance,
            'num_clusters': len(clusters)
        }
    
    # 결과 출력
    print("\n클러스터링 전략 비교 결과:")
    for strategy, metrics in results.items():
        print(f"\n{strategy}:")
        print(f"- 실행 시간: {metrics['execution_time']:.3f}초")
        print(f"- 클러스터 크기 분산: {metrics['size_variance']:.2f}")
        print(f"- 클러스터 수: {metrics['num_clusters']}")
    
    # 모든 전략이 요구사항을 만족하는지 확인
    for metrics in results.values():
        assert metrics['execution_time'] < 5.0  # 5초 이내 실행
        assert metrics['num_clusters'] == 5     # 정확한 클러스터 수

def test_cluster_edge_cases():
    """엣지 케이스 테스트"""
    # 빈 입력
    assert cluster_points([], create_test_vehicles(1)) is None
    assert cluster_points(create_test_points(10), []) is None
    
    # 단일 포인트
    single_point = create_test_points(1)
    single_vehicle = create_test_vehicles(1)
    clusters = cluster_points(single_point, single_vehicle)
    assert len(clusters) == 1
    assert len(clusters[0]) == 1
    
    # 차량보다 적은 포인트
    points = create_test_points(3)
    vehicles = create_test_vehicles(5)
    clusters = cluster_points(points, vehicles)
    assert len(clusters) <= len(points)

def test_cluster_performance():
    """성능 테스트"""
    test_cases = [
        {'size': 20, 'vehicles': 3},
        {'size': 50, 'vehicles': 5},
        {'size': 100, 'vehicles': 8},
        {'size': 200, 'vehicles': 15}
    ]
    
    results = []
    for case in test_cases:
        points = create_test_points(case['size'])
        vehicles = create_test_vehicles(case['vehicles'])
        
        # 여러 번 실행하여 평균 성능 측정
        times = []
        for _ in range(3):
            start_time = datetime.now()
            clusters = cluster_points(points, vehicles)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            times.append(execution_time)
            
            # 기본적인 유효성 검사
            assert clusters is not None
            assert len(clusters) == case['vehicles']
            assert sum(len(cluster) for cluster in clusters) == case['size']
        
        avg_time = np.mean(times)
        std_dev = np.std(times)
        
        results.append({
            'size': case['size'],
            'avg_time': avg_time,
            'std_dev': std_dev
        })
        
        # 성능 요구사항 검증
        assert avg_time < case['size'] / 10  # 포인트 10개당 1초 이하
    
    # 결과 출력
    print("\n성능 테스트 결과:")
    for result in results:
        print(f"\n{result['size']}개 포인트:")
        print(f"- 평균 실행 시간: {result['avg_time']:.3f}초")
        print(f"- 표준 편차: {result['std_dev']:.3f}초")

def test_cluster_solution_quality():
    """해의 품질 테스트"""
    points = create_test_points(50)
    vehicles = create_test_vehicles(5)
    
    # 여러 번 실행하여 해의 일관성 확인
    num_runs = 5
    cluster_metrics = []
    
    for _ in range(num_runs):
        clusters = cluster_points(points, vehicles)
        
        # 클러스터링 품질 메트릭 계산
        total_intra_distance = 0
        total_priority_score = 0
        
        for cluster in clusters:
            # 클러스터 중심점 계산
            center = np.mean([[p.latitude, p.longitude] for p in cluster], axis=0)
            
            # 클러스터 내 거리 합계
            intra_distances = sum(
                np.sqrt(
                    (p.latitude - center[0])**2 +
                    (p.longitude - center[1])**2
                )
                for p in cluster
            )
            
            # 우선순위 점수
            priority_score = sum(p.get_priority_weight() for p in cluster)
            
            total_intra_distance += intra_distances
            total_priority_score += priority_score
        
        cluster_metrics.append({
            'intra_distance': total_intra_distance,
            'priority_score': total_priority_score
        })
    
    # 해의 안정성 검증
    intra_distances = [m['intra_distance'] for m in cluster_metrics]
    priority_scores = [m['priority_score'] for m in cluster_metrics]
    
    intra_distance_std = np.std(intra_distances) / np.mean(intra_distances)
    priority_score_std = np.std(priority_scores) / np.mean(priority_scores)
    
    print("\n해의 품질 테스트 결과:")
    print(f"- 클러스터 내 거리 변동계수: {intra_distance_std:.3f}")
    print(f"- 우선순위 점수 변동계수: {priority_score_std:.3f}")
    
    # 변동계수가 10% 이내여야 함
    assert intra_distance_std < 0.1
    assert priority_score_std < 0.1

def test_cluster_edge_cases_advanced():
    """고급 엣지 케이스 테스트"""
    # 특수 차량이 필요한 케이스
    special_points = create_test_points_with_requirements(3)
    regular_vehicles = create_test_vehicles(5)
    
    # Vehicle 클래스의 현재 인터페이스에 맞게 수정
    now = datetime.now()
    special_vehicles = [
        Vehicle(
            id=f'SV{i+1}',
            type='TRUCK_1TON',
            capacity=VehicleCapacity(
                volume=20.0,
                weight=1500.0
            ),
            features=random.choice([
                ['STANDARD', 'REFRIGERATED', 'LIFT'],
                ['STANDARD', 'LIFT', 'TAIL_LIFT'],
                ['STANDARD', 'REFRIGERATED', 'LIFT', 'TAIL_LIFT']
            ]),
            cost_per_km=500.0,
            start_time=now,
            end_time=now + timedelta(hours=8),
            current_location=(37.2636, 127.0286)
        ) for i in range(2)
    ]
    
    clusters = cluster_points(special_points, regular_vehicles + special_vehicles)
    assert len(clusters) >= 2  # 특수 차량이 필요한 포인트들은 반드시 별도 클러스터

    # 우선순위가 높은 포인트들
    high_priority_points = create_high_priority_points(3)
    vehicles = create_test_vehicles(5)
    clusters = cluster_points(high_priority_points, vehicles)
    assert len(clusters) >= 3  # 우선순위 포인트는 분산되어야 함

def test_clustering_basic_performance():
    """기본적인 성능 테스트"""
    logger = logging.getLogger(__name__)
    
    points = create_test_points(100)
    vehicles = create_test_vehicles(8)
    
    start_time = time.time()
    clusters = cluster_points(points, vehicles, 'enhanced_kmeans')
    execution_time = time.time() - start_time
    
    assert clusters is not None
    assert len(clusters) == 8
    assert sum(len(cluster) for cluster in clusters) == 100
    assert execution_time < 3.0, f"실행 시간이 너무 깁니다: {execution_time:.2f}초"
    
    logger.info(f"클러스터링 실행 시간: {execution_time:.2f}초")

def test_clustering_memory_efficiency():
    """메모리 효율성 테스트"""
    logger = logging.getLogger(__name__)
    
    try:
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        points = create_test_points(200)
        vehicles = create_test_vehicles(15)
        clusters = cluster_points(points, vehicles, 'enhanced_kmeans')
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        assert clusters is not None
        assert len(clusters) == 15
        assert sum(len(cluster) for cluster in clusters) == 200
        assert memory_increase < 50, f"메모리 사용량 증가가 너무 큽니다: {memory_increase:.2f}MB"
        
        logger.info(f"메모리 사용량 증가: {memory_increase:.2f}MB")
        
    except ImportError:
        pytest.skip("psutil 패키지가 설치되지 않았습니다.")

def test_clustering_stability():
    """안정성 테스트"""
    logger = logging.getLogger(__name__)
    
    # 여러 번 실행하여 결과의 일관성 확인
    num_runs = 5
    execution_times = []
    cluster_sizes = []
    
    points = create_test_points(50)
    vehicles = create_test_vehicles(5)
    
    for i in range(num_runs):
        start_time = time.time()
        clusters = cluster_points(points, vehicles, 'enhanced_kmeans')
        end_time = time.time()
        
        execution_times.append(end_time - start_time)
        cluster_sizes.append([len(cluster) for cluster in clusters])
        
        # 기본 검증
        assert clusters is not None
        assert len(clusters) == 5
        assert sum(len(cluster) for cluster in clusters) == 50
    
    # 실행 시간 일관성 검증
    time_variance = np.var(execution_times)
    assert time_variance < 0.1, f"실행 시간이 불안정합니다. 분산: {time_variance:.3f}"
    
    # 클러스터 크기 일관성 검증
    size_variance = np.var([np.var(sizes) for sizes in cluster_sizes])
    assert size_variance < 2.0, f"클러스터 크기가 불안정합니다. 분산: {size_variance:.3f}"
    
    logger.info(f"평균 실행 시간: {np.mean(execution_times):.2f}초")
    logger.info(f"실행 시간 분산: {time_variance:.3f}")
    logger.info(f"클러스터 크기 분산: {size_variance:.3f}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main(['-v']) 