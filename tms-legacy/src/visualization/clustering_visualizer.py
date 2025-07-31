import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
from src.model.delivery_point import DeliveryPoint
from src.monitoring.clustering_monitor import ClusteringMetrics

class ClusteringVisualizer:
    def __init__(self):
        # 기본 스타일 설정
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def plot_clusters(self, clusters: List[List[DeliveryPoint]], save_path: str = None):
        """클러스터 시각화"""
        plt.figure(figsize=(12, 8))
        
        for i, cluster in enumerate(clusters):
            lats = [p.latitude for p in cluster]
            lons = [p.longitude for p in cluster]
            priorities = [p.priority for p in cluster]
            
            # 우선순위에 따른 마커 크기
            sizes = [50 * p for p in priorities]
            
            plt.scatter(lons, lats, s=sizes, c=[self.colors[i]], 
                       alpha=0.6, label=f'Cluster {i+1}')
            
            # 클러스터 중심점
            center_lat = np.mean(lats)
            center_lon = np.mean(lons)
            plt.plot(center_lon, center_lat, 'k+', markersize=10)

        plt.title('Clustering Results')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_performance_metrics(self, metrics_history: List[ClusteringMetrics], 
                               save_dir: str = None):
        """성능 메트릭스 시각화"""
        # 데이터 준비
        data = pd.DataFrame([m.to_dict() for m in metrics_history])
        data['start_time'] = pd.to_datetime(data['start_time'])
        
        # 1. 실행 시간 트렌드
        plt.figure(figsize=(12, 6))
        plt.plot(data['start_time'], data['execution_time'], 'b-')
        plt.title('Execution Time Trend')
        plt.xlabel('Time')
        plt.ylabel('Execution Time (s)')
        if save_dir:
            plt.savefig(f'{save_dir}/execution_time_trend.png')
        plt.close()
        
        # 2. 메모리 사용량 트렌드
        plt.figure(figsize=(12, 6))
        plt.plot(data['start_time'], data['memory_usage'], 'g-')
        plt.title('Memory Usage Trend')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (MB)')
        if save_dir:
            plt.savefig(f'{save_dir}/memory_usage_trend.png')
        plt.close()
        
        # 3. 클러스터 크기 분포
        plt.figure(figsize=(12, 6))
        all_sizes = [size for sizes in data['cluster_sizes'] for size in sizes]
        sns.histplot(all_sizes, bins=20)
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster Size')
        plt.ylabel('Count')
        if save_dir:
            plt.savefig(f'{save_dir}/cluster_size_distribution.png')
        plt.close()
        
        # 4. 용량 사용률 히트맵
        plt.figure(figsize=(10, 6))
        volume_ratios = [usage['volume_ratio'] for usages in data['capacity_usage'] 
                        for usage in usages]
        weight_ratios = [usage['weight_ratio'] for usages in data['capacity_usage'] 
                        for usage in usages]
        
        heatmap_data = np.histogram2d(volume_ratios, weight_ratios, bins=20)[0]
        sns.heatmap(heatmap_data, cmap='YlOrRd')
        plt.title('Capacity Usage Distribution')
        plt.xlabel('Volume Ratio')
        plt.ylabel('Weight Ratio')
        if save_dir:
            plt.savefig(f'{save_dir}/capacity_usage_heatmap.png')
        plt.close()

    def create_performance_report(self, metrics_history: List[ClusteringMetrics], 
                                save_path: str):
        """성능 리포트 생성"""
        data = pd.DataFrame([m.to_dict() for m in metrics_history])
        
        report = f"""
        Clustering Performance Report
        Generated at: {datetime.now()}
        
        Overall Statistics:
        ------------------
        Total Runs: {len(data)}
        Success Rate: {(data['success'].mean() * 100):.2f}%
        Average Execution Time: {data['execution_time'].mean():.2f}s
        Average Memory Usage: {data['memory_usage'].mean():.2f}MB
        
        Strategy Distribution:
        --------------------
        {data['strategy_name'].value_counts().to_string()}
        
        Performance Metrics:
        ------------------
        Execution Time:
          Mean: {data['execution_time'].mean():.2f}s
          Std: {data['execution_time'].std():.2f}s
          Min: {data['execution_time'].min():.2f}s
          Max: {data['execution_time'].max():.2f}s
        
        Memory Usage:
          Mean: {data['memory_usage'].mean():.2f}MB
          Std: {data['memory_usage'].std():.2f}MB
          Min: {data['memory_usage'].min():.2f}MB
          Max: {data['memory_usage'].max():.2f}MB
        
        Cluster Statistics:
        -----------------
        Average Number of Clusters: {data['num_clusters'].mean():.2f}
        Average Points per Cluster: {data['cluster_sizes'].apply(lambda x: sum(x)/len(x) if len(x) > 0 else 0).mean():.2f}
        
        Recent Performance (Last 5 runs):
        ------------------------------
        {data.tail(5)[['strategy_name', 'execution_time', 'memory_usage', 'success']].to_string()}
        """
        
        with open(save_path, 'w') as f:
            f.write(report)
