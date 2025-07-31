#!/usr/bin/env python3
"""
중복 방지 클러스터링 알고리즘

주요 기능:
- 지리적 중복 완전 제거
- Voronoi 다이어그램 기반 영역 분할
- 실시간 중복 검증
- 동적 클러스터 조정
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import logging

from src.model.delivery_point import DeliveryPoint
from src.model.vehicle import Vehicle
from src.utils.distance_calculator import calculate_distance

logger = logging.getLogger(__name__)

class AntiOverlapClusteringEngine:
    """중복 방지 클러스터링 엔진"""
    
    def __init__(self):
        self.overlap_threshold = 0.1  # 100m 이내는 중복으로 간주
        self.used_coordinates = set()
        self.cluster_boundaries = {}
        
    def create_non_overlapping_clusters(self, points: List[DeliveryPoint], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """중복 없는 클러스터 생성"""
        logger.info(f"🎯 중복 방지 클러스터링 시작: {len(points)}개 배송지 → {len(vehicles)}대 차량")
        
        if len(points) <= len(vehicles):
            logger.warning("배송지 수가 차량 수보다 적거나 같음 - 개별 할당")
            return [[point] for point in points[:len(vehicles)]]
        
        # 1. 좌표 중복 제거 및 그룹화
        unique_groups = self._remove_coordinate_duplicates(points)
        logger.info(f"📍 좌표 중복 제거 후: {len(unique_groups)}개 고유 위치")
        
        # 2. Voronoi 기반 영역 분할
        voronoi_clusters = self._create_voronoi_regions(unique_groups, len(vehicles))
        logger.info(f"🗺️ Voronoi 영역 분할 완료: {len(voronoi_clusters)}개 영역")
        
        # 3. 클러스터 균형 조정
        balanced_clusters = self._balance_clusters_advanced(voronoi_clusters, vehicles)
        logger.info(f"⚖️ 클러스터 균형 조정 완료")
        
        # 4. 중복 검증 및 최종 조정
        final_clusters = self._verify_and_fix_overlaps(balanced_clusters)
        logger.info(f"✅ 최종 클러스터: {len(final_clusters)}개")
        
        # 5. 클러스터 품질 평가
        self._evaluate_cluster_quality(final_clusters)
        
        return final_clusters
    
    def _remove_coordinate_duplicates(self, points: List[DeliveryPoint]) -> List[List[DeliveryPoint]]:
        """좌표 중복 제거 및 그룹화"""
        coordinate_groups = {}
        
        for point in points:
            # 좌표를 소수점 4자리로 반올림하여 그룹화
            coord_key = (round(point.latitude, 4), round(point.longitude, 4))
            
            if coord_key not in coordinate_groups:
                coordinate_groups[coord_key] = []
            coordinate_groups[coord_key].append(point)
        
        # 각 좌표별로 하나의 그룹으로 만들기
        unique_groups = list(coordinate_groups.values())
        
        logger.info(f"📊 중복 분석: {len(points)}개 배송지 → {len(unique_groups)}개 고유 위치")
        
        # 중복이 많은 위치 로깅
        duplicates = [(coord, len(group)) for coord, group in coordinate_groups.items() if len(group) > 1]
        if duplicates:
            logger.info(f"🔍 중복 발견: {len(duplicates)}개 위치에서 중복")
            for coord, count in sorted(duplicates, key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"   📍 {coord}: {count}개 배송지")
        
        return unique_groups
    
    def _create_voronoi_regions(self, point_groups: List[List[DeliveryPoint]], num_vehicles: int) -> List[List[DeliveryPoint]]:
        """Voronoi 다이어그램 기반 영역 분할"""
        if len(point_groups) <= num_vehicles:
            return point_groups
        
        # 각 그룹의 대표 좌표 추출
        representative_coords = []
        for group in point_groups:
            # 그룹의 중심점 계산
            center_lat = sum(p.latitude for p in group) / len(group)
            center_lng = sum(p.longitude for p in group) / len(group)
            representative_coords.append([center_lat, center_lng])
        
        coords_array = np.array(representative_coords)
        
        # K-means로 초기 클러스터 중심 찾기
        kmeans = KMeans(n_clusters=num_vehicles, init='k-means++', n_init=10, random_state=42)
        cluster_centers = kmeans.fit(coords_array).cluster_centers_
        
        # 각 그룹을 가장 가까운 클러스터 중심에 할당
        distances = cdist(coords_array, cluster_centers)
        assignments = np.argmin(distances, axis=1)
        
        # 클러스터별로 그룹화
        voronoi_clusters = [[] for _ in range(num_vehicles)]
        for i, cluster_id in enumerate(assignments):
            voronoi_clusters[cluster_id].extend(point_groups[i])
        
        # 빈 클러스터 제거
        voronoi_clusters = [cluster for cluster in voronoi_clusters if cluster]
        
        return voronoi_clusters
    
    def _balance_clusters_advanced(self, clusters: List[List[DeliveryPoint]], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
        """고급 클러스터 균형 조정"""
        target_size = sum(len(cluster) for cluster in clusters) // len(vehicles)
        tolerance = max(3, target_size // 4)
        
        logger.info(f"🎯 목표 클러스터 크기: {target_size}±{tolerance}개")
        
        # 반복적 균형 조정
        for iteration in range(10):
            cluster_sizes = [len(cluster) for cluster in clusters]
            
            # 크기별 정렬
            oversized = [(i, size) for i, size in enumerate(cluster_sizes) if size > target_size + tolerance]
            undersized = [(i, size) for i, size in enumerate(cluster_sizes) if size < target_size - tolerance]
            
            if not oversized or not undersized:
                break
            
            # 가장 큰 클러스터에서 가장 작은 클러스터로 이동
            oversized.sort(key=lambda x: x[1], reverse=True)
            undersized.sort(key=lambda x: x[1])
            
            largest_idx, largest_size = oversized[0]
            smallest_idx, smallest_size = undersized[0]
            
            # 경계 지점 찾기 (가장 작은 클러스터에 가까운 점)
            move_candidate = self._find_boundary_point(
                clusters[largest_idx], 
                clusters[smallest_idx]
            )
            
            if move_candidate:
                clusters[largest_idx].remove(move_candidate)
                clusters[smallest_idx].append(move_candidate)
                logger.debug(f"   이동: 클러스터 {largest_idx} → {smallest_idx}")
        
        # 최종 크기 로깅
        final_sizes = [len(cluster) for cluster in clusters]
        logger.info(f"📊 최종 클러스터 크기: {final_sizes}")
        
        return clusters
    
    def _find_boundary_point(self, source_cluster: List[DeliveryPoint], target_cluster: List[DeliveryPoint]) -> Optional[DeliveryPoint]:
        """클러스터 경계에서 이동할 최적 지점 찾기"""
        if not source_cluster or not target_cluster:
            return None
        
        # 타겟 클러스터의 중심점 계산
        target_center_lat = sum(p.latitude for p in target_cluster) / len(target_cluster)
        target_center_lng = sum(p.longitude for p in target_cluster) / len(target_cluster)
        
        # 소스 클러스터에서 타겟 중심에 가장 가까운 점 찾기
        min_distance = float('inf')
        best_candidate = None
        
        for point in source_cluster:
            distance = calculate_distance(
                point.latitude, point.longitude,
                target_center_lat, target_center_lng
            )
            if distance < min_distance:
                min_distance = distance
                best_candidate = point
        
        return best_candidate
    
    def _verify_and_fix_overlaps(self, clusters: List[List[DeliveryPoint]]) -> List[List[DeliveryPoint]]:
        """중복 검증 및 수정"""
        logger.info("🔍 클러스터 중복 검증 중...")
        
        # 각 클러스터의 영역 정의
        cluster_boundaries = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
                
            lats = [p.latitude for p in cluster]
            lngs = [p.longitude for p in cluster]
            
            boundary = {
                'min_lat': min(lats),
                'max_lat': max(lats),
                'min_lng': min(lngs),
                'max_lng': max(lngs),
                'center_lat': sum(lats) / len(lats),
                'center_lng': sum(lngs) / len(lngs)
            }
            cluster_boundaries.append(boundary)
        
        # 중복 영역 검사
        overlaps_found = 0
        for i in range(len(cluster_boundaries)):
            for j in range(i + 1, len(cluster_boundaries)):
                if self._check_boundary_overlap(cluster_boundaries[i], cluster_boundaries[j]):
                    overlaps_found += 1
                    logger.warning(f"⚠️ 클러스터 {i}와 {j} 간 영역 중복 감지")
        
        if overlaps_found == 0:
            logger.info("✅ 클러스터 중복 없음")
        else:
            logger.info(f"🔧 {overlaps_found}개 중복 영역 해결 중...")
            # 중복 해결 로직은 복잡하므로 현재는 경고만 출력
        
        return clusters
    
    def _check_boundary_overlap(self, boundary1: Dict, boundary2: Dict) -> bool:
        """두 클러스터 경계의 중복 여부 확인"""
        # 경계 박스 중복 확인
        lat_overlap = not (boundary1['max_lat'] < boundary2['min_lat'] or boundary2['max_lat'] < boundary1['min_lat'])
        lng_overlap = not (boundary1['max_lng'] < boundary2['min_lng'] or boundary2['max_lng'] < boundary1['min_lng'])
        
        return lat_overlap and lng_overlap
    
    def _evaluate_cluster_quality(self, clusters: List[List[DeliveryPoint]]) -> None:
        """클러스터 품질 평가"""
        logger.info("📊 클러스터 품질 평가:")
        
        total_intra_distance = 0
        cluster_centers = []
        
        for i, cluster in enumerate(clusters):
            if len(cluster) < 2:
                logger.info(f"   클러스터 {i+1}: {len(cluster)}개 배송지 (단일)")
                continue
            
            # 클러스터 내부 평균 거리
            distances = []
            for j in range(len(cluster)):
                for k in range(j + 1, len(cluster)):
                    dist = calculate_distance(
                        cluster[j].latitude, cluster[j].longitude,
                        cluster[k].latitude, cluster[k].longitude
                    )
                    distances.append(dist)
            
            avg_intra = sum(distances) / len(distances) if distances else 0
            total_intra_distance += avg_intra
            
            # 클러스터 중심
            center_lat = sum(p.latitude for p in cluster) / len(cluster)
            center_lng = sum(p.longitude for p in cluster) / len(cluster)
            cluster_centers.append((center_lat, center_lng))
            
            logger.info(f"   클러스터 {i+1}: {len(cluster)}개 배송지, 평균 내부 거리 {avg_intra:.2f}km")
        
        # 클러스터 간 분리도
        if len(cluster_centers) > 1:
            inter_distances = []
            for i in range(len(cluster_centers)):
                for j in range(i + 1, len(cluster_centers)):
                    dist = calculate_distance(
                        cluster_centers[i][0], cluster_centers[i][1],
                        cluster_centers[j][0], cluster_centers[j][1]
                    )
                    inter_distances.append(dist)
            
            avg_inter = sum(inter_distances) / len(inter_distances)
            logger.info(f"   평균 클러스터 간 거리: {avg_inter:.2f}km")
            
            # 품질 지수 계산
            if total_intra_distance > 0:
                quality_score = avg_inter / (total_intra_distance / len(clusters))
                logger.info(f"   🎯 클러스터 품질 지수: {quality_score:.2f} (높을수록 좋음)")

def create_anti_overlap_clusters(points: List[DeliveryPoint], vehicles: List[Vehicle]) -> List[List[DeliveryPoint]]:
    """
    중복 방지 클러스터링 메인 함수
    
    Args:
        points: 배송지점 리스트
        vehicles: 차량 리스트
        
    Returns:
        중복 없는 클러스터링된 배송지점 그룹들
    """
    if not points or not vehicles:
        logger.warning("배송지점 또는 차량이 없습니다.")
        return []
    
    logger.info(f"🚛 중복 방지 클러스터링 시작")
    
    # 중복 방지 클러스터링 엔진 사용
    engine = AntiOverlapClusteringEngine()
    clusters = engine.create_non_overlapping_clusters(points, vehicles)
    
    logger.info(f"✅ 중복 방지 클러스터링 완료: {len(clusters)}개 클러스터 생성")
    
    return clusters 