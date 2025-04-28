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

def create_test_points(size: int) -> list[DeliveryPoint]:
    """테스트용 배송 지점 생성"""
    points = []
    now = datetime.now()
    
    # 주요 상권 지역 (실제 수원시 좌표 기반)
    areas = [
        (37.2800, 127.0388, "수원역"),    # 수원역 부근
        (37.2849, 127.0178, "아주대"),    # 아주대학교 부근
        (37.2537, 127.0554, "영통"),      # 영통 신도시
        (37.2864, 127.0555, "광교"),      # 광교 신도시
    ]
    
    for i in range(size):
        # 랜덤하게 지역 선택
        area = areas[i % len(areas)]
        
        # 선택된 지역 주변에 포인트 생성
        lat = area[0] + np.random.normal(0, 0.005)  # 약 500m 반경
        lon = area[1] + np.random.normal(0, 0.005)
        
        # 현실적인 시간 윈도우 생성
        start_time = now + timedelta(hours=np.random.randint(0, 4))
        end_time = start_time + timedelta(hours=np.random.randint(2, 6))
        
        # 현실적인 물량 생성
        volume = np.random.uniform(0.5, 2.0)  # m³
        weight = np.random.uniform(10, 50)    # kg
        
        # 우선순위는 시간 윈도우에 따라 설정
        priority = 3 if (end_time - now).total_seconds() < 7200 else \
                  2 if (end_time - now).total_seconds() < 14400 else 1
        
        point = DeliveryPoint(
            id=f"DP{i+1}",
            latitude=lat,
            longitude=lon,
            address1=f"Test Address {area[2]} {i+1}",
            address2="",
            time_window=TimeWindow(start_time, end_time),
            service_time=15,  # 15분 고정
            volume=volume,
            weight=weight,
            special_requirements=[],
            priority=priority
        )
        points.append(point)
    
    return points

def create_test_vehicles(count: int) -> list[Vehicle]:
    """테스트용 차량 생성"""
    vehicles = []
    now = datetime.now()
    
    for i in range(count):
        capacity = VehicleCapacity(
            volume=15.0,  # 15m³
            weight=1000.0  # 1톤
        )
        
        vehicle = Vehicle(
            id=f"V{i+1}",
            type="TRUCK_1TON",
            capacity=capacity,
            features=["STANDARD"],
            cost_per_km=500.0,
            start_time=now,
            end_time=now + timedelta(hours=8),
            current_location=(37.2636, 127.0286)  # 수원시 중심
        )
        vehicles.append(vehicle)
    
    return vehicles

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

def test_stability():
    """안정성 테스트: 100회 반복"""
    success_count = 0
    failures = []
    
    for i in range(100):
        try:
            # 기본 테스트
            points = create_test_points(30)
            vehicles = create_test_vehicles(3)
            
            # 일부 포인트의 우선순위를 높게 설정
            high_priority_indices = [0, 5, 10, 15, 20, 25]
            for idx in high_priority_indices:
                points[idx].priority = 3
            
            clusters = cluster_points(points, vehicles, 'enhanced_kmeans')
            
            # 검증
            if clusters is None:
                failures.append(f"반복 {i+1}: clusters is None")
                continue
                
            # 1. 클러스터 수 확인
            if len(clusters) != 3:
                failures.append(f"반복 {i+1}: 잘못된 클러스터 수 - {len(clusters)}")
                continue
                
            # 2. 포인트 수 확인
            total_points = sum(len(cluster) for cluster in clusters)
            if total_points != 30:
                failures.append(f"반복 {i+1}: 잘못된 총 포인트 수 - {total_points}")
                continue
                
            # 3. 클러스터 크기 차이 확인
            cluster_sizes = [len(cluster) for cluster in clusters]
            size_diff = max(cluster_sizes) - min(cluster_sizes)
            if size_diff > 5:
                failures.append(f"반복 {i+1}: 클러스터 크기 차이 초과 - {size_diff}")
                continue
                
            # 4. 우선순위 분배 확인
            high_priority_distribution = [
                sum(1 for p in cluster if p.priority == 3)
                for cluster in clusters
            ]
            if not all(count > 0 for count in high_priority_distribution):
                failures.append(f"반복 {i+1}: 우선순위 분배 실패 - {high_priority_distribution}")
                continue
                
            # 5. 용량 제약 확인
            capacity_violated = False
            for cluster, vehicle in zip(clusters, vehicles):
                total_volume = sum(p.volume for p in cluster)
                total_weight = sum(p.weight for p in cluster)
                if (total_volume > vehicle.capacity.volume or
                    total_weight > vehicle.capacity.weight):
                    failures.append(f"반복 {i+1}: 용량 제약 위반")
                    capacity_violated = True
                    break
            
            if capacity_violated:
                continue
                
            success_count += 1
            
        except Exception as e:
            failures.append(f"반복 {i+1}: 예외 발생 - {str(e)}")
    
    # 결과 출력
    print(f"\n안정성 테스트 결과:")
    print(f"성공: {success_count}/100")
    if failures:
        print("\n실패 케이스:")
        for failure in failures:
            print(f"- {failure}")
    
    assert success_count == 100, f"안정성 테스트 실패: {100-success_count}개의 테스트 실패"

if __name__ == '__main__':
    pytest.main(['-v']) 