import pytest
import os
from datetime import datetime
from src.monitoring.clustering_monitor import ClusteringMonitor
from src.algorithm.cluster import cluster_points
from tests.test_cluster import create_test_points, create_test_vehicles

@pytest.fixture
def monitor():
    monitor = ClusteringMonitor()
    monitor.clear_metrics()
    return monitor

def test_clustering_monitoring_basic(monitor):
    """기본 모니터링 기능 테스트"""
    points = create_test_points(50)
    vehicles = create_test_vehicles(5)
    
    clusters = cluster_points(points, vehicles)
    
    assert len(monitor.metrics_history) > 0
    latest_metrics = monitor.metrics_history[-1]
    
    assert latest_metrics.success
    assert latest_metrics.num_points == 50
    assert latest_metrics.num_vehicles == 5
    assert latest_metrics.num_clusters == 5
    assert latest_metrics.execution_time > 0
    assert latest_metrics.memory_usage > 0

def test_clustering_monitoring_performance_summary(monitor):
    """성능 요약 기능 테스트"""
    points = create_test_points(50)
    vehicles = create_test_vehicles(5)
    
    # 여러 번 실행
    for _ in range(3):
        cluster_points(points, vehicles)
    
    summary = monitor.get_performance_summary()
    
    assert summary["total_runs"] == 3
    assert 0 <= summary["success_rate"] <= 1
    assert summary["average_execution_time"] > 0
    assert summary["average_memory_usage"] > 0
    assert "enhanced_kmeans" in summary["strategy_distribution"]

def test_clustering_monitoring_save_load(monitor, tmp_path):
    """메트릭스 저장/로드 테스트"""
    points = create_test_points(50)
    vehicles = create_test_vehicles(5)
    
    cluster_points(points, vehicles)
    
    # 임시 파일에 저장
    metrics_file = tmp_path / "test_metrics.json"
    monitor.save_metrics(str(metrics_file))
    
    # 메트릭스 초기화 후 다시 로드
    original_metrics = monitor.metrics_history.copy()
    monitor.clear_metrics()
    assert len(monitor.metrics_history) == 0
    
    monitor.load_metrics(str(metrics_file))
    assert len(monitor.metrics_history) == len(original_metrics)
