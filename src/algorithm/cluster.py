# src/algorithm/cluster.py
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from src.model.delivery_point import DeliveryPoint
from src.model.vehicle import Vehicle
from src.core.logger import setup_logger
from src.core.distance_matrix import DistanceMatrix
import numpy.typing as npt
from abc import ABC, abstractmethod
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from src.utils.distance_calculator import calculate_distance
import logging
from dataclasses import dataclass
import math
import json

logger = logging.getLogger(__name__)
distance_matrix = DistanceMatrix()

@dataclass
class TimeWindow:
    """배송 가능 시간대를 나타내는 클래스"""
    start: datetime
    end: datetime

    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end = end

    def is_valid(self, current_time: datetime) -> bool:
        """주어진 시간이 시간 윈도우 내에 있는지 확인"""
        return self.start <= current_time <= self.end

    def get_duration(self) -> float:
        """시간 윈도우의 길이를 시간 단위로 반환"""
        return (self.end - self.start).total_seconds() / 3600

class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, points: List[DeliveryPoint], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        pass

    def _adjust_cluster_count(
        self,
        clusters: List[List[DeliveryPoint]],
        target_num: int,
        vehicles: List[Vehicle]
    ) -> List[List[DeliveryPoint]]:
        """클러스터 수를 목표 수에 맞게 조정"""
        current_num = len(clusters)
        
        if current_num == target_num:
            return clusters
            
        if current_num > target_num:
            # 클러스터 병합
            while len(clusters) > target_num:
                # 가장 가까운 두 클러스터 찾기
                min_dist = float('inf')
                merge_pair = (0, 1)
                
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        dist = self._calculate_cluster_distance(clusters[i], clusters[j])
                        if dist < min_dist:
                            min_dist = dist
                            merge_pair = (i, j)
                
                # 클러스터 병합
                i, j = merge_pair
                clusters[i].extend(clusters[j])
                clusters.pop(j)
        else:
            # 클러스터 분할
            while len(clusters) < target_num:
                # 가장 큰 클러스터 찾기
                largest_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
                cluster = clusters[largest_idx]
                
                if len(cluster) < 2:
                    break
                    
                # 클러스터를 2개로 분할
                coords = np.array([[p.latitude, p.longitude] for p in cluster])
                kmeans = KMeans(n_clusters=2, random_state=42)
                labels = kmeans.fit_predict(coords)
                
                new_clusters = [[], []]
                for point, label in zip(cluster, labels):
                    new_clusters[label].append(point)
                
                clusters[largest_idx] = new_clusters[0]
                clusters.append(new_clusters[1])
        
        return clusters

    def _calculate_cluster_distance(self, cluster1: List[DeliveryPoint], cluster2: List[DeliveryPoint]) -> float:
        """두 클러스터 간의 거리 계산"""
        center1 = np.mean([[p.latitude, p.longitude] for p in cluster1], axis=0)
        center2 = np.mean([[p.latitude, p.longitude] for p in cluster2], axis=0)
        return np.sqrt(np.sum((center1 - center2) ** 2))

    def _adjust_clusters(self, clusters: List[List[DeliveryPoint]], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """클러스터 크기와 우선순위 균형 조정"""
        max_diff = 5  # 최대 허용 크기 차이
        
        while True:
            sizes = [len(cluster) for cluster in clusters]
            size_diff = max(sizes) - min(sizes)
            
            # 우선순위 분포 확인
            priority_distribution = [
                sum(1 for p in cluster if p.priority > 2)
                for cluster in clusters
            ]
            
            # 크기와 우선순위 모두 균형이 맞으면 종료
            if size_diff <= max_diff and all(count > 0 for count in priority_distribution):
                break
                
            # 가장 큰/작은 클러스터 찾기
            largest_idx = sizes.index(max(sizes))
            smallest_idx = sizes.index(min(sizes))
            
            # 우선순위가 없는 클러스터 찾기
            no_priority_clusters = [
                i for i, count in enumerate(priority_distribution)
                if count == 0
            ]
            
            if no_priority_clusters:
                # 우선순위 높은 포인트 재분배
                for source_idx in range(len(clusters)):
                    if source_idx in no_priority_clusters:
                        continue
                    high_priority_points = [
                        p for p in clusters[source_idx]
                        if p.priority > 2
                    ]
                    if len(high_priority_points) > 1:
                        for target_idx in no_priority_clusters:
                            point = high_priority_points[0]
                            clusters[source_idx].remove(point)
                            clusters[target_idx].append(point)
                            break
            
            # 크기 균형 조정
            if size_diff > max_diff:
                best_point = None
                min_cost = float('inf')
                
                for point in clusters[largest_idx]:
                    # 우선순위가 높은 포인트는 이동하지 않음
                    if point.priority > 2 and len([p for p in clusters[largest_idx] if p.priority > 2]) <= 1:
                        continue
                    cost = self._calculate_move_cost(point, clusters[largest_idx], clusters[smallest_idx])
                    if cost < min_cost:
                        min_cost = cost
                        best_point = point
                
                if best_point:
                    clusters[largest_idx].remove(best_point)
                    clusters[smallest_idx].append(best_point)
        
        return clusters

    def _balance_capacity(self, clusters: List[List[DeliveryPoint]], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        for i, (cluster, vehicle) in enumerate(zip(clusters, vehicles)):
            while True:
                total_volume = sum(p.volume for p in cluster)
                total_weight = sum(p.weight for p in cluster)
                
                if total_volume <= vehicle.capacity.volume and total_weight <= vehicle.capacity.weight:
                    break
                    
                if not cluster:
                    break
                
                # 용량이 큰 포인트들을 먼저 이동
                points_with_size = [(idx, p, p.volume + p.weight) 
                                  for idx, p in enumerate(cluster)]
                points_with_size.sort(key=lambda x: (-x[2], -x[1].priority))  # 용량 내림차순, 우선순위 내림차순
                
                moved = False
                for idx, point, _ in points_with_size:
                    # 가장 여유 있는 클러스터 찾기
                    best_cluster_idx = None
                    min_utilization = float('inf')
                    
                    for j, (other_cluster, other_vehicle) in enumerate(zip(clusters, vehicles)):
                        if j == i:
                            continue
                            
                        other_volume = sum(p.volume for p in other_cluster)
                        other_weight = sum(p.weight for p in other_cluster)
                        
                        if (other_volume + point.volume <= other_vehicle.capacity.volume and
                            other_weight + point.weight <= other_vehicle.capacity.weight):
                            # 용량 활용도 계산
                            utilization = max(
                                (other_volume + point.volume) / other_vehicle.capacity.volume,
                                (other_weight + point.weight) / other_vehicle.capacity.weight
                            )
                            if utilization < min_utilization:
                                min_utilization = utilization
                                best_cluster_idx = j
                    
                    if best_cluster_idx is not None:
                        # 포인트 이동
                        point = cluster.pop(idx)
                        clusters[best_cluster_idx].append(point)
                        moved = True
                        break
                
                if not moved:
                    # 이동할 수 없는 경우 가장 작은 포인트 제거
                    if cluster:
                        min_idx = min(range(len(cluster)), 
                                    key=lambda x: (cluster[x].priority, cluster[x].volume + cluster[x].weight))
                        cluster.pop(min_idx)
                    else:
                        break
        
        return clusters
    
class EnhancedKMeansStrategy(ClusteringStrategy):
    def __init__(self):
        self.kmeans = None
        self.min_cluster_size = 2
        
    def _validate_clusters(self, clusters, vehicles):
        """클러스터 유효성 검사"""
        if not clusters or len(clusters) == 0:
            return False
            
        for cluster, vehicle in zip(clusters, vehicles):
            total_volume = sum(point.volume for point in cluster)
            total_weight = sum(point.weight for point in cluster)
            
            # 95% 용량 제한 적용
            if total_volume > vehicle.capacity.volume * 0.95 or \
               total_weight > vehicle.capacity.weight * 0.95:
                return False
                
        return True
        
    def cluster(self, points, vehicles):
        """메인 클러스터링 함수"""
        if not points or not vehicles:
            return []
            
        # 포인트 수가 차량 수보다 적은 경우
        if len(points) <= len(vehicles):
            return [[point] for point in points]
            
        # 필요한 차량 수 계산
        total_volume = sum(point.volume for point in points)
        total_weight = sum(point.weight for point in points)
        min_vehicles_needed = max(
            math.ceil(total_volume / (vehicles[0].capacity.volume * 0.95)),
            math.ceil(total_weight / (vehicles[0].capacity.weight * 0.95))
        )
        
        n_clusters = min(len(vehicles), min_vehicles_needed)
        
        # 특수 케이스: 포인트가 10개 이하인 경우
        if len(points) <= 10:
            return self._handle_small_case(points, vehicles[:n_clusters])
            
        # 좌표 추출
        coords = np.array([[p.latitude, p.longitude] for p in points])
        
        # K-means 클러스터링 시도 (최대 3번)
        for attempt in range(3):
            try:
        kmeans = KMeans(
                    n_clusters=n_clusters,
                    init='k-means++',
                    n_init=1,
                    max_iter=20,
                    tol=1e-4,
                    random_state=attempt
                )
                
                labels = kmeans.fit_predict(coords)
                clusters = [[] for _ in range(n_clusters)]
                
                # 우선순위 기반 정렬
                sorted_points = sorted(
                    enumerate(points),
                    key=lambda x: (x[1].priority, -x[1].volume),
                    reverse=True
                )
                
                # 포인트 할당
                for idx, point in sorted_points:
                    cluster_idx = labels[idx]
                    clusters[cluster_idx].append(point)
                    
                # 클러스터 밸런싱
                self._balance_clusters(clusters, vehicles[:n_clusters])
                
                # 유효성 검사
                if self._validate_clusters(clusters, vehicles[:n_clusters]):
                    return clusters
                    
            except Exception as e:
                logging.error(f"클러스터링 시도 {attempt + 1} 실패: {str(e)}")
                continue
                
        # 모든 시도가 실패한 경우 보수적인 방법 사용
        return self._conservative_clustering(points, vehicles[:n_clusters])
        
    def _handle_small_case(self, points, vehicles):
        """소규모 케이스 처리"""
        n_clusters = min(len(points), len(vehicles))
        clusters = [[] for _ in range(n_clusters)]
        
        sorted_points = sorted(
            points,
            key=lambda x: (x.priority, x.volume),
            reverse=True
        )
        
        for i, point in enumerate(sorted_points):
            clusters[i % n_clusters].append(point)
            
        return clusters

    def _balance_clusters(self, clusters, vehicles):
        """클러스터 밸런싱"""
        for _ in range(3):  # 최대 3번 시도
            for i, (cluster, vehicle) in enumerate(zip(clusters, vehicles)):
                while cluster:
                    total_volume = sum(p.volume for p in cluster)
                    total_weight = sum(p.weight for p in cluster)
                    
                    if total_volume <= vehicle.capacity.volume * 0.95 and \
                       total_weight <= vehicle.capacity.weight * 0.95:
                        break
                        
                    # 가장 큰 포인트 제거
                    largest_point = max(cluster, key=lambda x: x.volume)
                    cluster.remove(largest_point)
                    
                    # 다른 클러스터에 재할당
                    for j, (other_cluster, other_vehicle) in enumerate(zip(clusters, vehicles)):
                        if i != j:
                            other_volume = sum(p.volume for p in other_cluster)
                            other_weight = sum(p.weight for p in other_cluster)
                            
                            if other_volume + largest_point.volume <= other_vehicle.capacity.volume * 0.95 and \
                               other_weight + largest_point.weight <= other_vehicle.capacity.weight * 0.95:
                                other_cluster.append(largest_point)
                                break
                                
    def _conservative_clustering(self, points, vehicles):
        """보수적인 클러스터링 방법"""
        n_clusters = len(vehicles)
        clusters = [[] for _ in range(n_clusters)]
        
        sorted_points = sorted(
            points,
            key=lambda x: (x.priority, -x.volume),
            reverse=True
        )
        
        for point in sorted_points:
            # 가장 여유 있는 클러스터 찾기
            best_fit = -1
            min_usage = float('inf')
            
            for i, (cluster, vehicle) in enumerate(zip(clusters, vehicles)):
                total_volume = sum(p.volume for p in cluster)
                total_weight = sum(p.weight for p in cluster)
                
                usage = max(
                    total_volume / vehicle.capacity.volume,
                    total_weight / vehicle.capacity.weight
                )
                
                if usage < min_usage and \
                   total_volume + point.volume <= vehicle.capacity.volume * 0.95 and \
                   total_weight + point.weight <= vehicle.capacity.weight * 0.95:
                    min_usage = usage
                    best_fit = i
                    
            if best_fit >= 0:
                clusters[best_fit].append(point)
            else:
                # 어떤 클러스터에도 들어갈 수 없는 경우
                # 가장 큰 클러스터에 추가
                largest_cluster = max(range(len(clusters)), key=lambda i: len(clusters[i]))
                clusters[largest_cluster].append(point)
                
        return clusters

class EnhancedDBSCANStrategy(ClusteringStrategy):
    def cluster(self, points: List[DeliveryPoint], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        if len(points) <= len(vehicles):
            return [[p] for p in points]
        
        n_clusters = len(vehicles)
        
        # 좌표 행렬 생성
        coords = np.array([[p.latitude, p.longitude] for p in points])
        
        # DBSCAN 파라미터 동적 계산
        distances = pdist(coords)
        eps = np.percentile(distances, 20)  # 상위 20% 거리를 eps로 사용
        min_samples = max(3, len(points) // (n_clusters * 2))
        
        # DBSCAN 클러스터링
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(coords)
        
        # 클러스터 병합/분할하여 원하는 수의 클러스터 생성
        unique_labels = np.unique(labels[labels >= 0])
        clusters = [[] for _ in range(n_clusters)]
        
        # 포인트 할당
        current_cluster = 0
        for label in unique_labels:
            points_in_label = [p for p, l in zip(points, labels) if l == label]
            points_per_cluster = len(points_in_label) // (n_clusters // len(unique_labels))
            
            for i, point in enumerate(points_in_label):
                target_cluster = (current_cluster + i // points_per_cluster) % n_clusters
                clusters[target_cluster].append(point)
            
            current_cluster = (current_cluster + len(points_in_label) // points_per_cluster) % n_clusters
        
        # 노이즈 포인트 처리
        noise_points = [p for p, l in zip(points, labels) if l == -1]
        for point in noise_points:
            # 가장 크기가 작은 클러스터에 할당
            smallest_cluster = min(range(len(clusters)), key=lambda i: len(clusters[i]))
            clusters[smallest_cluster].append(point)
        
        # 용량 제약 조정
        clusters = self._balance_capacity(clusters, vehicles)
        
        return clusters

class HDBSCANStrategy(ClusteringStrategy):
    """HDBSCAN 클러스터링 전략"""
    def cluster(
        self,
        points: List[DeliveryPoint],
        num_clusters: int,
        vehicles: List[Vehicle]
    ) -> List[List[DeliveryPoint]]:
        coords = np.array([[p.latitude, p.longitude] for p in points])
        
        # 시간 윈도우와 우선순위를 고려한 추가 특성
        time_features = np.array([
            [
                p.time_window.start.hour + p.time_window.start.minute / 60,
                p.time_window.end.hour + p.time_window.end.minute / 60,
                p.get_priority_weight()
            ]
            for p in points
        ])
        
        # 특성 결합 및 정규화
        features = np.hstack([
            coords,
            time_features * 0.3  # 시간과 우선순위에 가중치 부여
        ])
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, len(points) // num_clusters),
            min_samples=3,
            cluster_selection_epsilon=0.5,
            prediction_data=True
        )
        labels = clusterer.fit_predict(features)
            
        # 클러스터 조정
        return self._adjust_clusters(points, labels, num_clusters, vehicles)
    
    def _adjust_clusters(
        self,
        points: List[DeliveryPoint],
        labels: np.ndarray,
        target_num: int,
        vehicles: List[Vehicle]
    ) -> List[List[DeliveryPoint]]:
        # DBSCAN의 조정 로직과 유사하게 구현
        ...

def cluster_points(
    delivery_points: List[DeliveryPoint],
    vehicles: List[Vehicle],
    strategy: str = 'enhanced_kmeans'
) -> Optional[List[List[DeliveryPoint]]]:
    """개선된 클러스터링 함수"""
    try:
        if not delivery_points or not vehicles:
            logger.error("배송지점 또는 차량 데이터가 없습니다.")
            return None
            
        logger.info(f"클러스터링 시작: {len(delivery_points)}개 지점, {len(vehicles)}대 차량")
        
        # 클러스터링 전략 선택
        strategies = {
            'enhanced_kmeans': EnhancedKMeansStrategy(),
            'enhanced_dbscan': EnhancedDBSCANStrategy()
        }
        clustering_strategy = strategies.get(strategy, EnhancedKMeansStrategy())
        
        # 클러스터링 수행
        clusters = clustering_strategy.cluster(delivery_points, vehicles)
        
        if clusters:
            logger.info(f"클러스터링 완료: {len(clusters)}개 클러스터 생성")
        
        # 클러스터 유효성 검사 및 상세 로깅
        for i, (cluster, vehicle) in enumerate(zip(clusters, vehicles)):
            is_valid = clustering_strategy._validate_clusters(cluster, vehicle)
            if not is_valid:
                logger.error(f"""
클러스터 {i+1} 용량 제약 위반:
- 클러스터 크기: {len(cluster)} 포인트
- 총 부피: {sum(p.volume for p in cluster):.2f} / {vehicle.capacity.volume:.2f}
- 총 무게: {sum(p.weight for p in cluster):.2f} / {vehicle.capacity.weight:.2f}
""")
                return None
        
        return clusters
        
    except Exception as e:
        logger.error(f"클러스터링 중 오류 발생: {str(e)}")
        return None

def calculate_cluster_metrics(cluster: List[DeliveryPoint]) -> Dict:
    """클러스터의 메트릭 계산"""
    total_volume = sum(point.volume for point in cluster)
    total_weight = sum(point.weight for point in cluster)
    total_priority = sum(point.get_priority_weight() for point in cluster)
    
    # 시간 윈도우 범위 계산
    time_windows = [(p.time_window[0], p.time_window[1]) for p in cluster]
    earliest = min(tw[0] for tw in time_windows)
    latest = max(tw[1] for tw in time_windows)
    time_span = (latest - earliest).total_seconds() / 3600  # 시간 단위
    
    return {
        'volume': total_volume,
        'weight': total_weight,
        'priority': total_priority,
        'time_span': time_span,
        'size': len(cluster)
    }

def balance_clusters(
    clusters: List[List[DeliveryPoint]],
    vehicles: List[Vehicle]
) -> List[List[DeliveryPoint]]:
    """개선된 클러스터 밸런싱"""
    if not clusters or not vehicles:
        return clusters

        logger.info("클러스터 밸런싱 시작")
    max_iterations = 100
    iteration = 0
    balanced = False

    try:
        # 클러스터 메트릭 계산
        with ProcessPoolExecutor() as executor:
            cluster_metrics = list(executor.map(calculate_cluster_metrics, clusters))
        
        while not balanced and iteration < max_iterations:
            balanced = True
            iteration += 1
            
        for i, (cluster, vehicle) in enumerate(zip(clusters, vehicles)):
            metrics = cluster_metrics[i]
            
                # 용량 또는 시간 제약 위반 확인
                if (metrics['volume'] > vehicle.capacity.volume or
                    metrics['weight'] > vehicle.capacity.weight or
                    metrics['time_span'] > 8):  # 8시간 제한
                    
                    # 이동할 포인트 선택
                    move_candidates = []
                    for point in cluster:
                        # 다른 클러스터로 이동 가능성 검사
                        for j, (target_cluster, target_vehicle) in enumerate(zip(clusters, vehicles)):
                    if i == j:
                        continue
                        
                            target_metrics = cluster_metrics[j]
                            
                            # 이동 가능성 검사
                            if (target_metrics['volume'] + point.volume <= target_vehicle.capacity.volume and
                                target_metrics['weight'] + point.weight <= target_vehicle.capacity.weight):
                        
                                # 이동 비용 계산
                                cost = self._calculate_move_cost(point, cluster, target_cluster)
                                move_candidates.append((point, j, cost))
                        
                    if move_candidates:
                        # 가장 비용이 낮은 이동 선택
                        point, target_idx, _ = min(move_candidates, key=lambda x: x[2])
                        
                    # 포인트 이동
                        clusters[i].remove(point)
                        clusters[target_idx].append(point)
                    
                    # 메트릭 업데이트
                        cluster_metrics[i] = calculate_cluster_metrics(clusters[i])
                        cluster_metrics[target_idx] = calculate_cluster_metrics(clusters[target_idx])
                        
                        balanced = False
        
    except Exception as e:
        logger.error(f"클러스터 밸런싱 중 오류 발생: {str(e)}")
        return clusters

    return clusters

def _calculate_move_cost(
    point: DeliveryPoint,
    source_cluster: List[DeliveryPoint],
    target_cluster: List[DeliveryPoint]
) -> float:
    """포인트 이동 비용 계산"""
    if not target_cluster:
        return float('inf')
    
    # 거리 기반 비용
    target_center = np.mean([[p.latitude, p.longitude] for p in target_cluster], axis=0)
    distance_cost = np.sqrt(
        (point.latitude - target_center[0])**2 +
        (point.longitude - target_center[1])**2
    )
    
    # 시간 윈도우 기반 비용
    target_times = [p.time_window for p in target_cluster]
    target_start = min(tw[0] for tw in target_times)
    target_end = max(tw[1] for tw in target_times)
    time_compatibility = (
        1 if target_start <= point.time_window[0] <= target_end and
           target_start <= point.time_window[1] <= target_end
        else 2
    )
    
    # 우선순위 기반 비용
    priority_diff = abs(
        np.mean([p.priority for p in target_cluster]) -
        point.priority
    )
    
    return distance_cost * time_compatibility * (1 + priority_diff * 0.1)

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
            for j, (cluster, vehicle) in enumerate(zip(clusters, vehicles)):
                total_volume = sum(p.volume for p in cluster)
                total_weight = sum(p.weight for p in cluster)
                if (total_volume > vehicle.capacity.volume or
                    total_weight > vehicle.capacity.weight):
                    failures.append(
                        f"반복 {i+1}, 클러스터 {j+1} 용량 제약 위반:\n"
                        f"- 부피: {total_volume:.2f}/{vehicle.capacity.volume:.2f}\n"
                        f"- 무게: {total_weight:.2f}/{vehicle.capacity.weight:.2f}"
                    )
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