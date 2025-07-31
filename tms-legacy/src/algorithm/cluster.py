# src/algorithm/cluster.py
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from src.model.delivery_point import DeliveryPoint
from src.model.vehicle import Vehicle, VehicleCapacity
from src.core.logger import setup_logger
from src.core.distance_matrix import DistanceMatrix
import numpy.typing as npt
from abc import ABC, abstractmethod
from scipy.spatial.distance import pdist, cdist
from sklearn.preprocessing import StandardScaler
from src.utils.distance_calculator import calculate_distance
import logging
from dataclasses import dataclass
import math
import json
import random
import copy
from src.model.time_window import TimeWindow
from src.monitoring.clustering_monitor import ClusteringMonitor

logger = logging.getLogger(__name__)
distance_matrix = DistanceMatrix()

__all__ = ['cluster_points']

@dataclass
class ClusteringMetrics:
    """클러스터링 성능 메트릭"""
    total_distance: float
    balance_score: float
    efficiency_score: float
    constraint_violations: int

class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, points: List[DeliveryPoint], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """포인트를 클러스터링하는 기본 메서드"""
        pass
        
    def _validate_clusters(self, clusters: List[List[DeliveryPoint]], vehicles: List[Vehicle]) -> bool:
        """모든 클러스터가 유효한지 확인"""
        if len(clusters) != len(vehicles):
            return False
            
        for cluster, vehicle in zip(clusters, vehicles):
            total_volume = sum(p.volume for p in cluster)
            total_weight = sum(p.weight for p in cluster)
            
            # 품목 부피 정보가 있는 경우에만 부피 제약 확인
            volume_exceeded = False
            if any(p.volume > 0 for p in cluster):
                volume_exceeded = total_volume > vehicle.capacity.volume
                
            # 품목 무게 정보가 있는 경우에만 무게 제약 확인
            weight_exceeded = False
            if any(p.weight > 0 for p in cluster):
                weight_exceeded = total_weight > vehicle.capacity.weight
                
            if volume_exceeded or weight_exceeded:
                return False
                
        return True
        
    def _calculate_cluster_distance(self, cluster1: List[DeliveryPoint], cluster2: List[DeliveryPoint]) -> float:
        """두 클러스터 간의 거리 계산"""
        if not cluster1 or not cluster2:
            return float('inf')
            
        center1 = np.mean([[p.latitude, p.longitude] for p in cluster1], axis=0)
        center2 = np.mean([[p.latitude, p.longitude] for p in cluster2], axis=0)
        return np.sqrt(np.sum((center1 - center2) ** 2))
        
    def _ensure_priority_distribution(self, clusters: List[List[DeliveryPoint]]) -> List[List[DeliveryPoint]]:
        """우선순위 높은 포인트를 모든 클러스터에 고르게 분배"""
        # 각 클러스터의 우선순위 포인트 수 계산
        priority_count = [sum(1 for p in cluster if p.priority > 2) for cluster in clusters]
        
        # 우선순위 포인트가 없는 클러스터 찾기
        empty_clusters = [i for i, count in enumerate(priority_count) if count == 0]
        
        if not empty_clusters:
            return clusters
            
        # 우선순위 포인트가 여러 개 있는 클러스터 찾기
        surplus_clusters = [i for i, count in enumerate(priority_count) if count > 1]
        
        # 재분배를 위한 클러스터 복사
        result_clusters = copy.deepcopy(clusters)
        
        for empty_idx in empty_clusters:
            if not surplus_clusters:
                break
                
            # 가장 많은 우선순위 포인트를 가진 클러스터 선택
            source_idx = max(surplus_clusters, key=lambda i: priority_count[i])
            
            # 이동할 우선순위 포인트 찾기
            high_priority_points = [(i, p) for i, p in enumerate(result_clusters[source_idx]) if p.priority > 2]
            
            if high_priority_points:
                idx, point = high_priority_points[0]
                # 포인트 이동
                result_clusters[source_idx].pop(idx)
                result_clusters[empty_idx].append(point)
                
                # 상태 업데이트
                priority_count[source_idx] -= 1
                priority_count[empty_idx] += 1
                
                if priority_count[source_idx] <= 1:
                    surplus_clusters.remove(source_idx)
                    
        return result_clusters
        
    def _balance_cluster_sizes(self, clusters: List[List[DeliveryPoint]], max_diff: int = 5) -> List[List[DeliveryPoint]]:
        """클러스터 크기 균형 조정"""
        result_clusters = copy.deepcopy(clusters)
        
        for _ in range(10):  # 최대 10번 시도
            sizes = [len(cluster) for cluster in result_clusters]
            size_diff = max(sizes) - min(sizes)
            
            if size_diff <= max_diff:
                break
                
            largest_idx = sizes.index(max(sizes))
            smallest_idx = sizes.index(min(sizes))
            
            # 이동할 점 찾기
            best_point_idx = -1
            min_cost = float('inf')
            
            for i, point in enumerate(result_clusters[largest_idx]):
                # 우선순위 고려: 유일한 고우선순위 포인트는 이동하지 않음
                if point.priority > 2 and sum(1 for p in result_clusters[largest_idx] if p.priority > 2) <= 1:
                    continue
                    
                # 가장 작은 클러스터와의 거리 계산
                point_coords = np.array([[point.latitude, point.longitude]])
                if result_clusters[smallest_idx]:
                    cluster_coords = np.array([[p.latitude, p.longitude] for p in result_clusters[smallest_idx]])
                    distances = cdist(point_coords, cluster_coords).min()
                else:
                    distances = 0.0
                    
                if distances < min_cost:
                    min_cost = distances
                    best_point_idx = i
            
            if best_point_idx >= 0:
                # 포인트 이동
                point = result_clusters[largest_idx].pop(best_point_idx)
                result_clusters[smallest_idx].append(point)
            else:
                # 적합한 포인트가 없으면 탈출
                break
                
        return result_clusters
        
    def _adjust_cluster_count(self, clusters: List[List[DeliveryPoint]], target_count: int, vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """클러스터 수를 목표 수에 맞게 조정"""
        if len(clusters) == target_count:
            return clusters
            
        result_clusters = copy.deepcopy(clusters)
        
        if len(result_clusters) > target_count:
            # 클러스터 병합
            while len(result_clusters) > target_count:
                min_dist = float('inf')
                merge_indices = (0, 1)
                
                # 가장 가까운 두 클러스터 찾기
                for i in range(len(result_clusters)):
                    for j in range(i + 1, len(result_clusters)):
                        dist = self._calculate_cluster_distance(result_clusters[i], result_clusters[j])
                        if dist < min_dist:
                            min_dist = dist
                            merge_indices = (i, j)
                
                # 병합
                i, j = merge_indices
                result_clusters[i].extend(result_clusters[j])
                result_clusters.pop(j)
                
        elif len(result_clusters) < target_count:
            # 클러스터 분할
            while len(result_clusters) < target_count:
                # 가장 큰 클러스터 찾기
                sizes = [len(cluster) for cluster in result_clusters]
                largest_idx = sizes.index(max(sizes))
                
                if sizes[largest_idx] < 2:
                    break  # 더 이상 분할할 수 없음
                
                # K-means를 사용한 분할
                points = result_clusters[largest_idx]
                coords = np.array([[p.latitude, p.longitude] for p in points])
                
                kmeans = KMeans(n_clusters=2, random_state=42)
                labels = kmeans.fit_predict(coords)
                
                # 분할된 두 클러스터 생성
                new_clusters = [[], []]
                for idx, label in enumerate(labels):
                    new_clusters[label].append(points[idx])
                
                # 원래 클러스터 대체 및 새 클러스터 추가
                result_clusters[largest_idx] = new_clusters[0]
                result_clusters.append(new_clusters[1])
                
        return result_clusters

class BalancedKMeansStrategy(ClusteringStrategy):
    """균형 잡힌 K-means 클러스터링 전략"""
    
    def __init__(self):
        self.max_iterations = 50
        self.tolerance = 1e-4
        
    def cluster(self, points: List[DeliveryPoint], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """균형 잡힌 클러스터링 실행"""
        if not points or not vehicles:
            return []
            
        n_clusters = len(vehicles)
        logger.info(f"균형 잡힌 K-means 클러스터링 시작: {len(points)}개 포인트 → {n_clusters}개 클러스터")
        
        # 1. 초기 중심점 설정 (지리적으로 분산된 점들 선택)
        centers = self._initialize_balanced_centers(points, n_clusters)
        
        # 2. 반복적으로 클러스터 할당 및 중심점 업데이트
        for iteration in range(self.max_iterations):
            # 균형 잡힌 할당
            clusters = self._balanced_assignment(points, centers, vehicles)
            
            # 새로운 중심점 계산
            new_centers = self._update_centers(clusters)
            
            # 수렴 확인
            if self._has_converged(centers, new_centers):
                logger.info(f"클러스터링 수렴: {iteration + 1}회 반복")
                break
                
            centers = new_centers
        
        # 3. 최종 균형 조정
        balanced_clusters = self._final_balance_adjustment(clusters, vehicles)
        
        # 4. 검증 (완화된 조건)
        if self._validate_clusters_relaxed(balanced_clusters, vehicles):
            logger.info("클러스터링 검증 성공")
            return balanced_clusters
        else:
            logger.warning("클러스터링 검증 실패, 기본 분할 사용")
            return self._fallback_equal_division(points, vehicles)
    
    def _initialize_balanced_centers(self, points: List[DeliveryPoint], n_clusters: int) -> List[Tuple[float, float]]:
        """지리적으로 균등하게 분산된 초기 중심점 설정"""
        # 모든 점의 좌표 수집
        coords = np.array([[p.latitude, p.longitude] for p in points])
        
        # K-means++와 유사하지만 더 균등한 분포를 위한 초기화
        centers = []
        
        # 첫 번째 중심점: 전체 중심
        first_center = np.mean(coords, axis=0)
        centers.append((first_center[0], first_center[1]))
        
        # 나머지 중심점들: 기존 중심점들로부터 최대한 멀리
        for _ in range(n_clusters - 1):
            max_min_distance = -1
            best_center = None
            
            for point in points:
                # 이 점에서 가장 가까운 기존 중심점까지의 거리
                min_distance = min(
                    np.sqrt((point.latitude - center[0])**2 + (point.longitude - center[1])**2)
                    for center in centers
                )
                
                # 최대 최소 거리 업데이트
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_center = (point.latitude, point.longitude)
            
            if best_center:
                centers.append(best_center)
        
        return centers
    
    def _balanced_assignment(self, points: List[DeliveryPoint], centers: List[Tuple[float, float]], 
                           vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """균형 잡힌 클러스터 할당"""
        n_clusters = len(centers)
        target_size = len(points) // n_clusters
        max_size = target_size + (len(points) % n_clusters)
        
        # 각 점에서 각 중심점까지의 거리 계산
        distances = []
        for point in points:
            point_distances = []
            for center in centers:
                dist = np.sqrt((point.latitude - center[0])**2 + (point.longitude - center[1])**2)
                point_distances.append(dist)
            distances.append(point_distances)
        
        # 초기 클러스터 생성
        clusters = [[] for _ in range(n_clusters)]
        assigned = [False] * len(points)
        
        # 1단계: 각 클러스터에 최소한의 점들 할당
        for cluster_idx in range(n_clusters):
            # 이 클러스터에 가장 가까운 미할당 점 찾기
            best_point_idx = -1
            min_distance = float('inf')
            
            for point_idx, point_assigned in enumerate(assigned):
                if not point_assigned and distances[point_idx][cluster_idx] < min_distance:
                    min_distance = distances[point_idx][cluster_idx]
                    best_point_idx = point_idx
            
            if best_point_idx >= 0:
                clusters[cluster_idx].append(points[best_point_idx])
                assigned[best_point_idx] = True
        
        # 2단계: 나머지 점들을 균형 있게 할당
        for point_idx, point in enumerate(points):
            if assigned[point_idx]:
                continue
                
            # 현재 클러스터 크기 확인
            cluster_sizes = [len(cluster) for cluster in clusters]
            
            # 가장 작은 클러스터들 중에서 가장 가까운 클러스터 선택
            min_size = min(cluster_sizes)
            candidate_clusters = [i for i, size in enumerate(cluster_sizes) if size == min_size]
            
            # 후보 클러스터들 중 가장 가까운 것 선택
            best_cluster = min(candidate_clusters, key=lambda i: distances[point_idx][i])
            
            # 용량 제약 확인
            if self._can_add_to_cluster_balanced(point, clusters[best_cluster], vehicles[best_cluster]):
                clusters[best_cluster].append(point)
                assigned[point_idx] = True
            else:
                # 용량 제약으로 인해 할당 불가능한 경우, 다른 클러스터 시도
                for cluster_idx in sorted(range(n_clusters), key=lambda i: distances[point_idx][i]):
                    if (cluster_idx != best_cluster and 
                        len(clusters[cluster_idx]) < max_size and
                        self._can_add_to_cluster_balanced(point, clusters[cluster_idx], vehicles[cluster_idx])):
                        clusters[cluster_idx].append(point)
                        assigned[point_idx] = True
                        break
        
        return clusters
    
    def _can_add_to_cluster_balanced(self, point: DeliveryPoint, cluster: List[DeliveryPoint], 
                                   vehicle: Vehicle) -> bool:
        """클러스터에 점을 추가할 수 있는지 확인 (균형 고려)"""
        # 부피 제약 (품목 부피가 있는 경우에만)
        if any(p.volume > 0 for p in cluster + [point]):
            total_volume = sum(p.volume for p in cluster) + point.volume
            if total_volume > vehicle.capacity.volume * 0.95:  # 95% 제한
                return False
        
        # 무게 제약 (품목 무게가 있는 경우에만)
        if any(p.weight > 0 for p in cluster + [point]):
            total_weight = sum(p.weight for p in cluster) + point.weight
            if total_weight > vehicle.capacity.weight * 0.95:  # 95% 제한
                return False
        
        return True
    
    def _update_centers(self, clusters: List[List[DeliveryPoint]]) -> List[Tuple[float, float]]:
        """클러스터 중심점 업데이트"""
        new_centers = []
        for cluster in clusters:
            if cluster:
                center_lat = np.mean([p.latitude for p in cluster])
                center_lng = np.mean([p.longitude for p in cluster])
                new_centers.append((center_lat, center_lng))
            else:
                # 빈 클러스터의 경우 이전 중심점 유지
                new_centers.append((0.0, 0.0))  # 임시값
        return new_centers
    
    def _has_converged(self, old_centers: List[Tuple[float, float]], 
                      new_centers: List[Tuple[float, float]]) -> bool:
        """수렴 여부 확인"""
        if len(old_centers) != len(new_centers):
            return False
            
        for old, new in zip(old_centers, new_centers):
            distance = np.sqrt((old[0] - new[0])**2 + (old[1] - new[1])**2)
            if distance > self.tolerance:
                return False
        return True
    
    def _final_balance_adjustment(self, clusters: List[List[DeliveryPoint]], 
                                vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """최종 균형 조정"""
        # 크기 균형 조정
        clusters = self._balance_cluster_sizes(clusters, max_diff=3)
        
        # 우선순위 분배 조정
        clusters = self._ensure_priority_distribution(clusters)
        
        return clusters

    def _fallback_equal_division(self, points: List[DeliveryPoint], 
                               vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """균등 분할 fallback 방법"""
        n_clusters = len(vehicles)
        clusters = [[] for _ in range(n_clusters)]
        
        # 단순히 순서대로 균등 분할
        for i, point in enumerate(points):
            cluster_idx = i % n_clusters
            clusters[cluster_idx].append(point)
        
        return clusters

    def _validate_clusters_relaxed(self, clusters: List[List[DeliveryPoint]], vehicles: List[Vehicle]) -> bool:
        """완화된 클러스터 검증 (지리적 클러스터링 우선)"""
        if len(clusters) != len(vehicles):
            return False
            
        # 빈 클러스터 확인
        if any(len(cluster) == 0 for cluster in clusters):
            return False
            
        # 기본적인 용량 제약만 확인 (완화된 조건)
        for cluster, vehicle in zip(clusters, vehicles):
            # 품목 부피 정보가 있는 경우에만 부피 제약 확인
            if any(p.volume > 0 for p in cluster):
                total_volume = sum(p.volume for p in cluster)
                if total_volume > vehicle.capacity.volume * 1.2:  # 120% 허용 (완화)
                    return False
                    
            # 품목 무게 정보가 있는 경우에만 무게 제약 확인
            if any(p.weight > 0 for p in cluster):
                total_weight = sum(p.weight for p in cluster)
                if total_weight > vehicle.capacity.weight * 1.2:  # 120% 허용 (완화)
                    return False
                    
        return True

class EnhancedDBSCANStrategy(ClusteringStrategy):
    def __init__(self):
        self._coords_cache = {}
        self._eps_cache = None
        self._min_samples_cache = None

    def cluster(self, points: List[DeliveryPoint], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """개선된 DBSCAN 기반 클러스터링"""
        if len(points) <= len(vehicles):
            return [[p] for p in points[:len(vehicles)]]
            
        n_clusters = len(vehicles)
        
        try:
            # 1. 특성 준비 및 정규화
            coords = self._prepare_coordinates(points)
            
            # 2. DBSCAN 파라미터 최적화
            eps, min_samples = self._optimize_parameters(coords, n_clusters)
            
            # 3. DBSCAN 클러스터링 with 예외 처리
            clusters = self._perform_dbscan(coords, points, eps, min_samples, vehicles)
            
            # 4. 클러스터 수 조정 및 최적화
            clusters = self._optimize_clusters(clusters, n_clusters, vehicles)
            
            # 5. 제약 조건 검증 및 조정
            clusters = self._adjust_constraints(clusters, vehicles)
            
            # 6. 최종 검증
            if self._validate_final_clusters(clusters, vehicles):
                return clusters
            else:
                logger.warning("HDBSCAN 클러스터링 검증 실패, 기본 전략으로 fallback")
                return self._fallback_equal_division(points, vehicles)
                
        except Exception as e:
            logger.error(f"HDBSCAN 클러스터링 오류: {str(e)}")
            return self._fallback_equal_division(points, vehicles)

    def _prepare_coordinates(self, points: List[DeliveryPoint]) -> np.ndarray:
        """좌표 준비 및 정규화"""
        # 캐시 확인
        cache_key = tuple(p.id for p in points)
        if cache_key in self._coords_cache:
            return self._coords_cache[cache_key]

        # 좌표 추출 및 정규화
        coords = np.array([[p.latitude, p.longitude] for p in points])
        priority_weights = np.array([[p.priority] for p in points])
        
        # 특성 결합
        features = np.hstack([coords, priority_weights])
        
        # 정규화
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        # 캐시 저장
        self._coords_cache[cache_key] = normalized_features
        return normalized_features

    def _optimize_parameters(self, coords: np.ndarray, n_clusters: int) -> Tuple[float, int]:
        """DBSCAN 파라미터 최적화"""
        if self._eps_cache is not None and self._min_samples_cache is not None:
            return self._eps_cache, self._min_samples_cache

        # 거리 행렬 계산
        distances = pdist(coords)
        
        # eps 값 최적화
        eps_candidates = np.percentile(distances, [15, 20, 25, 30])
        min_samples_candidates = [max(3, len(coords) // (n_clusters * x)) for x in [2, 3, 4]]
        
        best_params = None
        best_score = float('-inf')
        
        for eps in eps_candidates:
            for min_samples in min_samples_candidates:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(coords)
                
                # 클러스터링 품질 평가
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = np.sum(labels == -1) / len(labels)
                
                # 점수 계산
                score = self._calculate_clustering_score(n_clusters_found, n_clusters, noise_ratio)
                
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
        
        self._eps_cache = best_params[0]
        self._min_samples_cache = best_params[1]
        return best_params

    def _calculate_clustering_score(self, n_clusters_found: int, target_clusters: int, noise_ratio: float) -> float:
        """클러스터링 품질 점수 계산"""
        cluster_diff_penalty = abs(n_clusters_found - target_clusters) * 0.2
        noise_penalty = noise_ratio * 0.5
        return 1.0 - cluster_diff_penalty - noise_penalty

    def _perform_dbscan(self, coords: np.ndarray, points: List[DeliveryPoint], 
                       eps: float, min_samples: int, vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """DBSCAN 실행 및 결과 처리"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(coords)
        
        # 클러스터 및 노이즈 포인트 분리
        clusters = []
        noise_points = []
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                noise_points.extend([points[i] for i, l in enumerate(labels) if l == -1])
            else:
                cluster = [points[i] for i, l in enumerate(labels) if l == label]
                clusters.append(cluster)
        
        return self._handle_noise_points(clusters, noise_points, vehicles)

    def _handle_noise_points(self, clusters: List[List[DeliveryPoint]], 
                           noise_points: List[DeliveryPoint], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """노이즈 포인트 처리 개선"""
        if not noise_points:
            return clusters
        
        result_clusters = copy.deepcopy(clusters)
        remaining_points = []
        
        for point in noise_points:
            assigned = False
            distances = []
            
            # 각 클러스터와의 거리 계산
            for i, cluster in enumerate(result_clusters):
                if not cluster:
                    continue
                cluster_coords = np.array([[p.latitude, p.longitude] for p in cluster])
                point_coords = np.array([[point.latitude, point.longitude]])
                min_dist = cdist(point_coords, cluster_coords).min()
                distances.append((min_dist, i))
            
            # 거리순으로 정렬
            distances.sort()
            
            # 가장 가까운 클러스터부터 시도
            for dist, cluster_idx in distances:
                if self._can_add_to_cluster(point, result_clusters[cluster_idx], vehicles[cluster_idx % len(vehicles)]):
                    result_clusters[cluster_idx].append(point)
                    assigned = True
                    break
            
            if not assigned:
                remaining_points.append(point)
        
        # 남은 포인트들을 위한 새로운 클러스터 생성
        if remaining_points:
            result_clusters.append(remaining_points)
        
        return result_clusters

    def _optimize_clusters(self, clusters: List[List[DeliveryPoint]], 
                         target_count: int, vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """클러스터 최적화"""
        # 클러스터 수 조정
        clusters = self._adjust_cluster_count(clusters, target_count, vehicles)
        
        # 우선순위 분배
        clusters = self._ensure_priority_distribution(clusters)
        
        # 크기 균형 조정
        clusters = self._balance_cluster_sizes(clusters)
        
        return clusters
    
    def _adjust_constraints(self, clusters: List[List[DeliveryPoint]], 
                          vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """제약 조건에 맞게 클러스터 조정"""
        result_clusters = clusters.copy()
        
        for i, (cluster, vehicle) in enumerate(zip(result_clusters, vehicles)):
            # 용량 제약 확인 및 조정
            while not self._validate_capacity(cluster, vehicle):
                if not cluster:
                    break
                    
                # 가장 큰 용량을 차지하는 포인트 제거
                point_to_move = max(cluster, 
                                  key=lambda p: max(p.volume/vehicle.capacity.volume,
                                                   p.weight/vehicle.capacity.weight))
                cluster.remove(point_to_move)
                
                # 다른 클러스터에 재할당 시도
                assigned = False
                for j, (other_cluster, other_vehicle) in enumerate(zip(result_clusters, vehicles)):
                    if i != j and self._can_add_to_cluster(point_to_move, other_cluster, other_vehicle):
                        other_cluster.append(point_to_move)
                        assigned = True
                        break
                
                if not assigned:
                    # 새 클러스터 생성 (차량이 남아있는 경우)
                    if len(result_clusters) < len(vehicles):
                        result_clusters.append([point_to_move])
        
        return result_clusters
    
    def _validate_capacity(self, cluster: List[DeliveryPoint], vehicle: Vehicle) -> bool:
        """클러스터가 차량 용량을 초과하지 않는지 확인"""
        total_volume = sum(point.volume for point in cluster)
        total_weight = sum(point.weight for point in cluster)
        
        return (total_volume <= vehicle.capacity.volume and 
                total_weight <= vehicle.capacity.weight)
    
    def _can_add_to_cluster(self, point: DeliveryPoint, cluster: List[DeliveryPoint], 
                           vehicle: Vehicle) -> bool:
        """클러스터에 포인트를 추가할 수 있는지 확인"""
        total_volume = sum(p.volume for p in cluster) + point.volume
        total_weight = sum(p.weight for p in cluster) + point.weight
        
        return (total_volume <= vehicle.capacity.volume and 
                total_weight <= vehicle.capacity.weight)
    
    def _fallback_equal_division(self, points: List[DeliveryPoint], 
                               vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """균등 분할 fallback 방법"""
        clusters = [[] for _ in vehicles]
        points_per_cluster = len(points) // len(vehicles)
        
        for i, point in enumerate(points):
            cluster_idx = min(i // points_per_cluster, len(clusters) - 1)
            clusters[cluster_idx].append(point)
        
        return clusters

    def _validate_final_clusters(self, clusters: List[List[DeliveryPoint]], 
                               vehicles: List[Vehicle]) -> bool:
        """최종 클러스터 검증"""
        if not self._validate_clusters(clusters, vehicles):
            return False
            
        # 추가 검증
        for cluster, vehicle in zip(clusters, vehicles):
            # 시간 윈도우 검증
            time_span = self._calculate_time_span(cluster)
            if time_span > 8:  # 8시간 초과
                return False
                
            # 우선순위 분포 검증
            if not self._validate_priority_distribution(cluster):
                return False
        
        return True

    def _calculate_time_span(self, cluster: List[DeliveryPoint]) -> float:
        """클러스터의 시간 범위 계산"""
        if not cluster:
            return 0.0
            
        # 시간 윈도우 범위 계산 - 튜플 형태로 통일
        time_windows = []
        for p in cluster:
            if hasattr(p, 'time_window') and p.time_window:
                # 튜플인 경우 (DeliveryPoint 정의에 맞춤)
                if isinstance(p.time_window, (tuple, list)) and len(p.time_window) == 2:
                    time_windows.append((p.time_window[0], p.time_window[1]))
        
        if time_windows:
            earliest = min(tw[0] for tw in time_windows)
            latest = max(tw[1] for tw in time_windows)
            time_span = (latest - earliest).total_seconds() / 3600  # 시간 단위
        else:
            time_span = 0.0
        
        return time_span

    def _validate_priority_distribution(self, cluster: List[DeliveryPoint]) -> bool:
        """우선순위 분포 검증"""
        if not cluster:
            return True
            
        priority_counts = {1: 0, 2: 0, 3: 0}
        for point in cluster:
            priority_counts[point.priority] += 1
            
        # 우선순위 3인 포인트가 너무 많지 않은지 확인
        high_priority_ratio = priority_counts[3] / len(cluster)
        return high_priority_ratio <= 0.4  # 최대 40%까지 허용

    def _optimize_priority_distribution(self, clusters: List[List[DeliveryPoint]]) -> List[List[DeliveryPoint]]:
        """우선순위 분배 최적화"""
        # 각 클러스터의 평균 우선순위 계산
        cluster_priorities = []
        for cluster in clusters:
            if cluster:
                avg_priority = sum(p.priority for p in cluster) / len(cluster)
                cluster_priorities.append(avg_priority)
            else:
                cluster_priorities.append(0)
        
        # 우선순위 불균형이 큰 경우 조정
        max_priority = max(cluster_priorities) if cluster_priorities else 0
        min_priority = min(cluster_priorities) if cluster_priorities else 0
        
        if max_priority - min_priority > 2:  # 우선순위 차이가 2 이상인 경우
            # 고우선순위 포인트를 저우선순위 클러스터로 이동
            for i, cluster in enumerate(clusters):
                if cluster_priorities[i] > max_priority - 0.5:
                    # 고우선순위 클러스터에서 저우선순위 포인트 찾기
                    low_priority_points = [p for p in cluster if p.priority <= 2]
                    for point in low_priority_points[:1]:  # 최대 1개만 이동
                        # 저우선순위 클러스터 찾기
                        target_idx = cluster_priorities.index(min_priority)
                        if target_idx != i:
                            cluster.remove(point)
                            clusters[target_idx].append(point)
                            break
        
        return clusters

monitor = ClusteringMonitor()

@monitor.monitor("balanced_kmeans")
def cluster_points(points: List[DeliveryPoint], vehicles: List[Vehicle], 
                  strategy: str = 'balanced_kmeans') -> Optional[List[List[DeliveryPoint]]]:
    if not points or not vehicles:
        return None
        
    available_vehicles = _filter_suitable_vehicles(points, vehicles)
    if not available_vehicles:
        return None
        
    # 최적 클러스터 수 결정
    n_clusters = _determine_optimal_cluster_count(points, available_vehicles)
    
    # 클러스터링 전략 선택
    strategies = {
        'balanced_kmeans': BalancedKMeansStrategy(),
        'enhanced_dbscan': EnhancedDBSCANStrategy()
    }
    
    clustering_strategy = strategies.get(strategy, BalancedKMeansStrategy())
    return clustering_strategy.cluster(points, available_vehicles[:n_clusters])

def _filter_suitable_vehicles(points: List[DeliveryPoint], vehicles: List[Vehicle]) -> List[Vehicle]:
    """포인트 특성에 맞는 차량만 필터링"""
    suitable_vehicles = []
    for vehicle in vehicles:
        # 차량의 기본 요구사항만 확인
        if _check_basic_vehicle_requirements(vehicle):
            suitable_vehicles.append(vehicle)
    return suitable_vehicles

def _check_basic_vehicle_requirements(vehicle: Vehicle) -> bool:
    """차량의 기본 요구사항 확인"""
    # 기본적인 운영 가능 여부만 체크
    return (vehicle.capacity.volume > 0 and 
            vehicle.capacity.weight > 0 and 
            'STANDARD' in vehicle.features)

def _determine_optimal_cluster_count(points: List[DeliveryPoint], vehicles: List[Vehicle]) -> int:
    """최적의 클러스터 수 결정"""
    # 요청된 차량 수를 우선적으로 사용
    requested_clusters = len(vehicles)
    
    # 포인트 특성 분석
    total_volume = sum(p.volume for p in points)
    total_weight = sum(p.weight for p in points)
    priority_points = sum(1 for p in points if p.priority > 2)
    
    # 차량 용량 분석
    avg_vehicle_volume = sum(v.capacity.volume for v in vehicles) / len(vehicles)
    avg_vehicle_weight = sum(v.capacity.weight for v in vehicles) / len(vehicles)
    
    # 필요한 최소 차량 수 계산
    min_vehicles_by_volume = math.ceil(total_volume / (avg_vehicle_volume * 0.95))  # 95% 용량 제한
    min_vehicles_by_weight = math.ceil(total_weight / (avg_vehicle_weight * 0.95))
    min_vehicles_by_priority = math.ceil(priority_points / 2)
    
    # 최소 필요 클러스터 수
    min_required = max(
        min_vehicles_by_volume,
        min_vehicles_by_weight,
        min_vehicles_by_priority,
        1  # 최소 1개
    )
    
    # 요청된 클러스터 수와 실제 필요한 수 중 큰 값 사용
    return min(max(requested_clusters, min_required), len(points))

def calculate_cluster_metrics(cluster: List[DeliveryPoint]) -> Dict:
    """클러스터의 메트릭 계산"""
    total_volume = sum(point.volume for point in cluster)
    total_weight = sum(point.weight for point in cluster)
    total_priority = sum(point.get_priority_weight() for point in cluster)
    
    # 시간 윈도우 범위 계산 - 튜플 형태로 통일
    time_windows = []
    for p in cluster:
        if hasattr(p, 'time_window') and p.time_window:
            # 튜플인 경우 (DeliveryPoint 정의에 맞춤)
            if isinstance(p.time_window, (tuple, list)) and len(p.time_window) == 2:
                time_windows.append((p.time_window[0], p.time_window[1]))
    
    if time_windows:
        earliest = min(tw[0] for tw in time_windows)
        latest = max(tw[1] for tw in time_windows)
        time_span = (latest - earliest).total_seconds() / 3600  # 시간 단위
    else:
        time_span = 0.0
    
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
                                cost = _calculate_move_cost(point, cluster, target_cluster)
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
    
    # 시간 윈도우 기반 비용 - 튜플 형태로 통일
    target_times = []
    for p in target_cluster:
        if hasattr(p, 'time_window') and p.time_window:
            # 튜플인 경우 (DeliveryPoint 정의에 맞춤)
            if isinstance(p.time_window, (tuple, list)) and len(p.time_window) == 2:
                target_times.append((p.time_window[0], p.time_window[1]))
    
    time_compatibility = 1  # 기본값
    if target_times and hasattr(point, 'time_window') and point.time_window:
        target_start = min(tw[0] for tw in target_times)
        target_end = max(tw[1] for tw in target_times)
        
        # point의 time_window 처리 - 튜플 방식으로 통일
        if isinstance(point.time_window, (tuple, list)) and len(point.time_window) == 2:
            point_start, point_end = point.time_window[0], point.time_window[1]
        else:
            point_start, point_end = None, None
        
        if point_start and point_end:
            time_compatibility = (
                1 if target_start <= point_start <= target_end and
                   target_start <= point_end <= target_end
                else 2
            )
    
    # 우선순위 기반 비용
    priority_diff = abs(
        np.mean([p.priority for p in target_cluster]) -
        point.priority
    )
    
    return distance_cost * time_compatibility * (1 + priority_diff * 0.1)