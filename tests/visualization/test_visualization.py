# tests/visualization/test_visualization.py

import pytest
import os
from src.visualization.clustering_visualizer import ClusteringVisualizer
from src.monitoring.clustering_monitor import ClusteringMonitor
from tests.test_cluster import create_test_points, create_test_vehicles
from src.algorithm.cluster import cluster_points

@pytest.fixture
def visualizer():
    return ClusteringVisualizer()

@pytest.fixture
def monitor():
    monitor = ClusteringMonitor()
    monitor.clear_metrics()
    return monitor

def test_plot_clusters(visualizer):
    """클러스터 시각화 테스트"""
    points = create_test_points(50)
    vehicles = create_test_vehicles(5)
    clusters = cluster_points(points, vehicles)
    
    save_path = "visualizations/clusters.png"
    visualizer.plot_clusters(clusters, save_path)
    assert os.path.exists(save_path)
    print(f"\n클러스터 시각화가 저장됨: {save_path}")

def test_plot_performance_metrics(visualizer, monitor):
    """성능 메트릭스 시각화 테스트"""
    points = create_test_points(50)
    vehicles = create_test_vehicles(5)
    
    for _ in range(3):
        cluster_points(points, vehicles)
    
    save_dir = "visualizations"
    visualizer.plot_performance_metrics(monitor.metrics_history, save_dir)
    
    expected_files = [
        'execution_time_trend.png',
        'memory_usage_trend.png',
        'cluster_size_distribution.png',
        'capacity_usage_heatmap.png'
    ]
    
    for file in expected_files:
        file_path = os.path.join(save_dir, file)
        assert os.path.exists(file_path)
        print(f"\n성능 메트릭스 시각화가 저장됨: {file_path}")

def test_create_performance_report(visualizer, monitor):
    """성능 리포트 생성 테스트"""
    points = create_test_points(50)
    vehicles = create_test_vehicles(5)
    
    for _ in range(3):
        cluster_points(points, vehicles)
    
    report_path = "visualizations/performance_report.txt"
    visualizer.create_performance_report(monitor.metrics_history, report_path)
    
    assert os.path.exists(report_path)
    print(f"\n성능 리포트가 저장됨: {report_path}")