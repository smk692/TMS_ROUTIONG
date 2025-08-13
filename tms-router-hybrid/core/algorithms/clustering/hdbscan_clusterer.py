"""
HDBSCAN 기반 지리적 클러스터링
"""

import hdbscan
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from geopy.distance import geodesic

from ...models import Order, Coordinates


@dataclass 
class OrderCluster:
    """주문 클러스터"""
    orders: List[Order]
    centroid: Coordinates
    confidence: float = 1.0
    is_noise_cluster: bool = False
    
    @property
    def size(self) -> int:
        return len(self.orders)
    
    @property
    def order_ids(self) -> List[str]:
        return [order.id for order in self.orders]


class HDBSCANGeographicClusterer:
    """HDBSCAN 기반 지리적 클러스터링"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        
        # HDBSCAN 파라미터
        self.min_cluster_size = config.get('min_cluster_size', 8)  # 최소 8개 주문
        self.min_samples = config.get('min_samples', 3)           # 최소 3개 샘플
        self.max_cluster_size = config.get('max_cluster_size', 35) # 최대 35개 주문
        self.epsilon = config.get('epsilon', 0.005)               # ~500m 반경 (위경도)
        self.outlier_threshold = config.get('outlier_threshold', 0.7)  # 이상치 판정 임계값
        
    def cluster_orders(self, orders: List[Order]) -> Dict[str, List[OrderCluster]]:
        """권역별 HDBSCAN 클러스터링 실행"""
        
        self.logger.info(f"HDBSCAN 클러스터링 시작: {len(orders)}개 주문")
        
        results = {}
        
        # 1. 권역별 분할
        region_orders = self._group_by_region(orders)
        
        for region_id, region_order_list in region_orders.items():
            self.logger.info(f"권역 {region_id}: {len(region_order_list)}개 주문 클러스터링")
            
            try:
                # 2. 지리적 좌표 추출
                coordinates = self._extract_coordinates(region_order_list)
                
                if len(coordinates) < self.min_cluster_size:
                    # 주문이 너무 적으면 단일 클러스터로 처리
                    cluster = OrderCluster(
                        orders=region_order_list,
                        centroid=self._calculate_centroid(region_order_list),
                        confidence=0.8,
                        is_noise_cluster=True
                    )
                    results[region_id] = [cluster]
                    continue
                
                # 3. HDBSCAN 클러스터링 실행
                clusters = self._perform_hdbscan_clustering(region_order_list, coordinates)
                
                # 4. 클러스터 후처리
                optimized_clusters = self._post_process_clusters(clusters)
                
                results[region_id] = optimized_clusters
                
                self.logger.info(f"권역 {region_id} 클러스터링 완료: {len(optimized_clusters)}개 클러스터")
                
            except Exception as e:
                self.logger.error(f"권역 {region_id} 클러스터링 오류: {str(e)}")
                # 폴백: 단일 클러스터로 처리
                fallback_cluster = OrderCluster(
                    orders=region_order_list,
                    centroid=self._calculate_centroid(region_order_list),
                    confidence=0.5,
                    is_noise_cluster=True
                )
                results[region_id] = [fallback_cluster]
        
        total_clusters = sum(len(clusters) for clusters in results.values())
        self.logger.info(f"HDBSCAN 클러스터링 완료: 총 {total_clusters}개 클러스터 생성")
        
        return results
    
    def _group_by_region(self, orders: List[Order]) -> Dict[str, List[Order]]:
        """권역별 주문 그룹화"""
        region_groups = {}
        
        for order in orders:
            region_id = order.region_id
            if region_id not in region_groups:
                region_groups[region_id] = []
            region_groups[region_id].append(order)
            
        return region_groups
    
    def _extract_coordinates(self, orders: List[Order]) -> np.ndarray:
        """주문들의 지리적 좌표 추출"""
        coordinates = []
        
        for order in orders:
            lat = order.coordinates.latitude
            lng = order.coordinates.longitude
            coordinates.append([lat, lng])
            
        return np.array(coordinates)
    
    def _perform_hdbscan_clustering(self, orders: List[Order], coordinates: np.ndarray) -> List[OrderCluster]:
        """HDBSCAN 클러스터링 수행"""
        
        # HDBSCAN 클러스터러 초기화 (epsilon 파라미터는 cluster_selection_epsilon으로 대체)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='haversine',  # 구면 거리 계산
            cluster_selection_epsilon=self.epsilon,
            cluster_selection_method='eom',  # Excess of Mass
            allow_single_cluster=True
        )
        
        # 라디안 변환 (Haversine metric 요구사항)
        coords_radians = np.radians(coordinates)
        
        # 클러스터링 실행
        cluster_labels = clusterer.fit_predict(coords_radians)
        
        # 클러스터 생성
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # 노이즈는 별도 처리
                continue
                
            # 해당 라벨의 주문들 추출
            cluster_orders = [orders[i] for i in range(len(orders)) if cluster_labels[i] == label]
            
            # 클러스터 신뢰도 계산
            confidence = self._calculate_cluster_confidence(cluster_orders, clusterer, label)
            
            cluster = OrderCluster(
                orders=cluster_orders,
                centroid=self._calculate_centroid(cluster_orders),
                confidence=confidence
            )
            clusters.append(cluster)
        
        # 노이즈 주문 처리
        noise_orders = [orders[i] for i in range(len(orders)) if cluster_labels[i] == -1]
        if noise_orders:
            noise_clusters = self._handle_noise_orders(noise_orders, clusters)
            clusters.extend(noise_clusters)
            
        return clusters
    
    def _post_process_clusters(self, clusters: List[OrderCluster]) -> List[OrderCluster]:
        """클러스터 후처리 및 최적화"""
        
        optimized_clusters = []
        
        for cluster in clusters:
            # 클러스터 크기 검증 및 조정
            if cluster.size > self.max_cluster_size:
                # 큰 클러스터는 재분할
                sub_clusters = self._split_large_cluster(cluster)
                optimized_clusters.extend(sub_clusters)
            elif cluster.size < 3 and not cluster.is_noise_cluster:
                # 너무 작은 클러스터는 노이즈로 재분류
                cluster.is_noise_cluster = True
                cluster.confidence *= 0.7
                optimized_clusters.append(cluster)
            else:
                optimized_clusters.append(cluster)
        
        return optimized_clusters
    
    def _split_large_cluster(self, large_cluster: OrderCluster) -> List[OrderCluster]:
        """큰 클러스터를 차량 용량에 맞게 분할"""
        
        orders = large_cluster.orders
        
        if len(orders) <= self.max_cluster_size:
            return [large_cluster]
        
        # 하위 HDBSCAN으로 재분할
        try:
            coordinates = self._extract_coordinates(orders)
            
            sub_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(5, len(orders) // 3),
                min_samples=2,
                epsilon=self.epsilon / 2,  # 더 세밀한 분할
                metric='haversine',
                allow_single_cluster=False
            )
            
            coords_radians = np.radians(coordinates)
            sub_labels = sub_clusterer.fit_predict(coords_radians)
            
            sub_clusters = []
            for label in set(sub_labels):
                if label == -1:
                    continue
                    
                sub_orders = [orders[i] for i in range(len(orders)) if sub_labels[i] == label]
                
                if len(sub_orders) >= 3:  # 최소 크기 보장
                    sub_cluster = OrderCluster(
                        orders=sub_orders,
                        centroid=self._calculate_centroid(sub_orders),
                        confidence=large_cluster.confidence * 0.9
                    )
                    sub_clusters.append(sub_cluster)
            
            return sub_clusters if sub_clusters else [large_cluster]
            
        except Exception as e:
            self.logger.warning(f"클러스터 분할 실패: {str(e)}")
            # 간단한 거리 기반 분할으로 폴백
            return self._simple_distance_split(large_cluster)
    
    def _simple_distance_split(self, large_cluster: OrderCluster) -> List[OrderCluster]:
        """간단한 거리 기반 클러스터 분할"""
        
        orders = large_cluster.orders
        target_size = self.max_cluster_size
        
        if len(orders) <= target_size:
            return [large_cluster]
        
        # 중심점에서 가장 가까운 순으로 정렬
        centroid = large_cluster.centroid
        
        def distance_to_centroid(order):
            return geodesic(
                (centroid.latitude, centroid.longitude),
                (order.coordinates.latitude, order.coordinates.longitude)
            ).kilometers
        
        sorted_orders = sorted(orders, key=distance_to_centroid)
        
        # target_size 크기로 분할
        sub_clusters = []
        for i in range(0, len(sorted_orders), target_size):
            chunk_orders = sorted_orders[i:i + target_size]
            
            if len(chunk_orders) >= 3:  # 최소 크기 보장
                sub_cluster = OrderCluster(
                    orders=chunk_orders,
                    centroid=self._calculate_centroid(chunk_orders),
                    confidence=large_cluster.confidence * 0.8
                )
                sub_clusters.append(sub_cluster)
        
        return sub_clusters if sub_clusters else [large_cluster]
    
    def _handle_noise_orders(self, noise_orders: List[Order], 
                           main_clusters: List[OrderCluster]) -> List[OrderCluster]:
        """노이즈 주문을 기존 클러스터에 통합 또는 독립 클러스터 생성"""
        
        noise_clusters = []
        remaining_noise = noise_orders.copy()
        
        # 1. 기존 클러스터와의 거리 기반 통합
        for cluster in main_clusters:
            if cluster.size >= self.max_cluster_size:
                continue
                
            nearby_noise = self._find_nearby_orders(cluster.centroid, remaining_noise, max_distance_km=2.0)
            
            # 용량 허용 범위 내에서 노이즈 주문 추가
            capacity_left = self.max_cluster_size - cluster.size
            if nearby_noise and capacity_left > 0:
                add_count = min(len(nearby_noise), capacity_left)
                cluster.orders.extend(nearby_noise[:add_count])
                
                # 중심점 재계산
                cluster.centroid = self._calculate_centroid(cluster.orders)
                cluster.confidence *= 0.9  # 신뢰도 약간 감소
                
                # 추가된 주문들 제거
                for added_order in nearby_noise[:add_count]:
                    remaining_noise.remove(added_order)
        
        # 2. 남은 노이즈로 독립 클러스터 생성
        if remaining_noise:
            noise_groups = self._group_noise_by_distance(remaining_noise)
            for group in noise_groups:
                if len(group) >= 3:  # 최소 크기 보장
                    noise_cluster = OrderCluster(
                        orders=group,
                        centroid=self._calculate_centroid(group),
                        confidence=0.7,  # 노이즈 클러스터는 낮은 신뢰도
                        is_noise_cluster=True
                    )
                    noise_clusters.append(noise_cluster)
        
        return noise_clusters
    
    def _find_nearby_orders(self, centroid: Coordinates, orders: List[Order], 
                           max_distance_km: float) -> List[Order]:
        """중심점 근처의 주문들 찾기"""
        
        nearby_orders = []
        
        for order in orders:
            distance = geodesic(
                (centroid.latitude, centroid.longitude),
                (order.coordinates.latitude, order.coordinates.longitude)
            ).kilometers
            
            if distance <= max_distance_km:
                nearby_orders.append((order, distance))
        
        # 거리순으로 정렬하여 주문만 반환
        nearby_orders.sort(key=lambda x: x[1])
        return [order for order, _ in nearby_orders]
    
    def _group_noise_by_distance(self, noise_orders: List[Order]) -> List[List[Order]]:
        """거리 기반으로 노이즈 주문들 그룹화"""
        
        if not noise_orders:
            return []
        
        groups = []
        remaining = noise_orders.copy()
        
        while remaining:
            # 첫 번째 주문을 그룹 시드로 선택
            seed_order = remaining.pop(0)
            current_group = [seed_order]
            
            # 시드 주문 근처의 주문들 찾기
            nearby = self._find_nearby_orders(
                seed_order.coordinates, 
                remaining, 
                max_distance_km=1.5
            )
            
            # 최대 크기까지 그룹에 추가
            max_group_size = min(self.max_cluster_size, len(nearby) + 1)
            for i in range(min(len(nearby), max_group_size - 1)):
                order = nearby[i]
                if order in remaining:
                    current_group.append(order)
                    remaining.remove(order)
            
            groups.append(current_group)
        
        return groups
    
    def _calculate_centroid(self, orders: List[Order]) -> Coordinates:
        """주문들의 중심점 계산"""
        
        if not orders:
            return Coordinates(latitude=0.0, longitude=0.0)
        
        total_lat = sum(order.coordinates.latitude for order in orders)
        total_lng = sum(order.coordinates.longitude for order in orders)
        
        return Coordinates(
            latitude=total_lat / len(orders),
            longitude=total_lng / len(orders)
        )
    
    def _calculate_cluster_confidence(self, orders: List[Order], clusterer, label: int = None) -> float:
        """클러스터 신뢰도 계산"""
        
        try:
            # HDBSCAN의 클러스터 확률 사용
            if hasattr(clusterer, 'probabilities_') and len(clusterer.probabilities_) > 0:
                cluster_probs = [clusterer.probabilities_[i] for i in range(len(clusterer.labels_)) 
                               if clusterer.labels_[i] == label]
                if cluster_probs:
                    avg_probability = sum(cluster_probs) / len(cluster_probs)
                    return min(1.0, avg_probability + 0.2)  # 약간의 보정
            
            # 폴백: 클러스터 크기 기반 신뢰도
            cluster_size = len(orders)
            if cluster_size >= self.min_cluster_size * 2:
                return 0.95
            elif cluster_size >= self.min_cluster_size:
                return 0.85
            else:
                return 0.7
                
        except Exception:
            return 0.8  # 기본 신뢰도