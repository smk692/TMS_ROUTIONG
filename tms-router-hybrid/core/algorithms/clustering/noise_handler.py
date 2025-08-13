"""
노이즈 주문 처리 모듈
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from geopy.distance import geodesic

from ...models import Order, Coordinates
from .hdbscan_clusterer import OrderCluster


class NoiseHandler:
    """HDBSCAN 클러스터링에서 발생한 노이즈 주문 처리"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 노이즈 처리 설정
        self.max_noise_distance_km = config.get('max_noise_distance_km', 3.0)
        self.min_noise_cluster_size = config.get('min_noise_cluster_size', 3)
        self.noise_integration_threshold = config.get('noise_integration_threshold', 0.7)
        self.max_single_noise_group = config.get('max_single_noise_group', 8)
    
    def process_noise_orders(self, noise_orders: List[Order], 
                           main_clusters: List[OrderCluster]) -> Tuple[List[OrderCluster], List[OrderCluster]]:
        """노이즈 주문 처리 메인 프로세스"""
        
        self.logger.info(f"노이즈 주문 처리 시작: {len(noise_orders)}개 노이즈 주문")
        
        if not noise_orders:
            return main_clusters, []
        
        # 1. 기존 클러스터 근처 노이즈들을 통합
        updated_clusters, remaining_noise = self._integrate_noise_to_clusters(
            noise_orders, main_clusters
        )
        
        # 2. 남은 노이즈들로 독립 클러스터 생성
        noise_clusters = self._create_noise_clusters(remaining_noise)
        
        self.logger.info(f"노이즈 처리 완료: {len(remaining_noise)}개 → {len(noise_clusters)}개 노이즈 클러스터")
        
        return updated_clusters, noise_clusters
    
    def _integrate_noise_to_clusters(self, noise_orders: List[Order], 
                                   main_clusters: List[OrderCluster]) -> Tuple[List[OrderCluster], List[Order]]:
        """노이즈 주문을 기존 클러스터에 통합"""
        
        updated_clusters = []
        remaining_noise = noise_orders.copy()
        
        for cluster in main_clusters:
            updated_cluster, integrated_orders = self._try_integrate_noise_to_cluster(
                cluster, remaining_noise
            )
            
            updated_clusters.append(updated_cluster)
            
            # 통합된 주문들을 남은 노이즈에서 제거
            for order in integrated_orders:
                if order in remaining_noise:
                    remaining_noise.remove(order)
        
        return updated_clusters, remaining_noise
    
    def _try_integrate_noise_to_cluster(self, cluster: OrderCluster, 
                                      noise_orders: List[Order]) -> Tuple[OrderCluster, List[Order]]:
        """단일 클러스터에 노이즈 주문 통합 시도"""
        
        # 클러스터가 이미 최대 크기에 가까우면 통합하지 않음
        max_cluster_size = self.config.get('max_cluster_size', 35)
        if cluster.size >= max_cluster_size * 0.9:
            return cluster, []
        
        # 클러스터 중심점 근처의 노이즈 주문들 찾기
        nearby_noise = self._find_nearby_noise_orders(cluster.centroid, noise_orders)
        
        if not nearby_noise:
            return cluster, []
        
        # 통합 가능한 주문 수 계산
        available_capacity = max_cluster_size - cluster.size
        integrable_count = min(len(nearby_noise), available_capacity)
        
        if integrable_count == 0:
            return cluster, []
        
        # 거리 순으로 정렬하여 가장 가까운 주문들부터 통합
        nearby_with_distances = []
        for order in nearby_noise:
            distance = geodesic(
                (cluster.centroid.latitude, cluster.centroid.longitude),
                (order.coordinates.latitude, order.coordinates.longitude)
            ).kilometers
            nearby_with_distances.append((order, distance))
        
        nearby_with_distances.sort(key=lambda x: x[1])
        orders_to_integrate = [order for order, _ in nearby_with_distances[:integrable_count]]
        
        # 새로운 통합 클러스터 생성
        integrated_orders = cluster.orders + orders_to_integrate
        updated_cluster = OrderCluster(
            orders=integrated_orders,
            centroid=self._calculate_centroid(integrated_orders),
            confidence=cluster.confidence * 0.95,  # 노이즈 통합으로 약간 신뢰도 감소
            is_noise_cluster=cluster.is_noise_cluster
        )
        
        self.logger.debug(f"클러스터에 노이즈 {len(orders_to_integrate)}개 통합: {cluster.size} → {updated_cluster.size}")
        
        return updated_cluster, orders_to_integrate
    
    def _find_nearby_noise_orders(self, centroid: Coordinates, 
                                 noise_orders: List[Order]) -> List[Order]:
        """중심점 근처의 노이즈 주문들 찾기"""
        
        nearby_orders = []
        
        for order in noise_orders:
            distance = geodesic(
                (centroid.latitude, centroid.longitude),
                (order.coordinates.latitude, order.coordinates.longitude)
            ).kilometers
            
            if distance <= self.max_noise_distance_km:
                nearby_orders.append(order)
        
        return nearby_orders
    
    def _create_noise_clusters(self, remaining_noise: List[Order]) -> List[OrderCluster]:
        """남은 노이즈 주문들로 독립 클러스터 생성"""
        
        if not remaining_noise:
            return []
        
        noise_clusters = []
        
        # 1. 지리적으로 가까운 노이즈들끼리 그룹화
        noise_groups = self._group_noise_by_proximity(remaining_noise)
        
        # 2. 각 그룹을 클러스터로 변환
        for group in noise_groups:
            if len(group) >= self.min_noise_cluster_size:
                noise_cluster = OrderCluster(
                    orders=group,
                    centroid=self._calculate_centroid(group),
                    confidence=0.6,  # 노이즈 클러스터는 낮은 신뢰도
                    is_noise_cluster=True
                )
                noise_clusters.append(noise_cluster)
                
                self.logger.debug(f"노이즈 클러스터 생성: {len(group)}개 주문")
        
        return noise_clusters
    
    def _group_noise_by_proximity(self, noise_orders: List[Order]) -> List[List[Order]]:
        """지리적 근접도를 기반으로 노이즈 주문들 그룹화"""
        
        if not noise_orders:
            return []
        
        groups = []
        remaining_orders = noise_orders.copy()
        
        while remaining_orders:
            # 새로운 그룹 시작
            seed_order = remaining_orders.pop(0)
            current_group = [seed_order]
            
            # 그룹 확장 (반복적으로 근처 주문들 추가)
            group_expanded = True
            while group_expanded and len(current_group) < self.max_single_noise_group:
                group_expanded = False
                
                # 현재 그룹의 중심점 계산
                group_centroid = self._calculate_centroid(current_group)
                
                # 근처 주문들 찾기
                nearby_orders = self._find_nearby_noise_orders(group_centroid, remaining_orders)
                
                if nearby_orders:
                    # 가장 가까운 주문 하나 추가
                    distances = []
                    for order in nearby_orders:
                        distance = geodesic(
                            (group_centroid.latitude, group_centroid.longitude),
                            (order.coordinates.latitude, order.coordinates.longitude)
                        ).kilometers
                        distances.append((order, distance))
                    
                    # 가장 가까운 주문 선택
                    distances.sort(key=lambda x: x[1])
                    closest_order = distances[0][0]
                    
                    current_group.append(closest_order)
                    remaining_orders.remove(closest_order)
                    group_expanded = True
            
            groups.append(current_group)
        
        return groups
    
    def handle_isolated_orders(self, isolated_orders: List[Order]) -> List[OrderCluster]:
        """완전히 격리된 주문들을 처리"""
        
        if not isolated_orders:
            return []
        
        isolated_clusters = []
        
        # 격리된 주문들을 작은 그룹으로 묶어 처리
        for i in range(0, len(isolated_orders), self.min_noise_cluster_size):
            group = isolated_orders[i:i + self.min_noise_cluster_size]
            
            if len(group) >= 2:  # 최소 2개 이상만 클러스터로 처리
                isolated_cluster = OrderCluster(
                    orders=group,
                    centroid=self._calculate_centroid(group),
                    confidence=0.5,  # 매우 낮은 신뢰도
                    is_noise_cluster=True
                )
                isolated_clusters.append(isolated_cluster)
                
                self.logger.debug(f"격리 주문 클러스터 생성: {len(group)}개 주문")
        
        return isolated_clusters
    
    def optimize_noise_cluster_distribution(self, noise_clusters: List[OrderCluster]) -> List[OrderCluster]:
        """노이즈 클러스터들의 분배 최적화"""
        
        if len(noise_clusters) <= 1:
            return noise_clusters
        
        optimized_clusters = []
        
        # 너무 작은 클러스터들을 합치거나 재분배
        small_clusters = [c for c in noise_clusters if c.size < self.min_noise_cluster_size]
        normal_clusters = [c for c in noise_clusters if c.size >= self.min_noise_cluster_size]
        
        # 작은 클러스터들을 근처의 큰 클러스터에 합치기
        for small_cluster in small_clusters:
            merged = False
            
            for normal_cluster in normal_clusters:
                if self._can_merge_clusters(small_cluster, normal_cluster):
                    # 합치기
                    merged_orders = normal_cluster.orders + small_cluster.orders
                    normal_cluster.orders = merged_orders
                    normal_cluster.centroid = self._calculate_centroid(merged_orders)
                    normal_cluster.confidence *= 0.9  # 합치면서 신뢰도 약간 감소
                    
                    merged = True
                    self.logger.debug(f"작은 노이즈 클러스터 합병: {small_cluster.size} + {len(normal_cluster.orders) - small_cluster.size}")
                    break
            
            if not merged:
                # 합칠 수 없으면 그대로 유지
                normal_clusters.append(small_cluster)
        
        return normal_clusters
    
    def _can_merge_clusters(self, cluster1: OrderCluster, cluster2: OrderCluster) -> bool:
        """두 클러스터가 합칠 수 있는지 확인"""
        
        max_cluster_size = self.config.get('max_cluster_size', 35)
        
        # 1. 크기 제한 확인
        if cluster1.size + cluster2.size > max_cluster_size:
            return False
        
        # 2. 거리 확인
        distance = geodesic(
            (cluster1.centroid.latitude, cluster1.centroid.longitude),
            (cluster2.centroid.latitude, cluster2.centroid.longitude)
        ).kilometers
        
        return distance <= self.max_noise_distance_km * 1.5
    
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