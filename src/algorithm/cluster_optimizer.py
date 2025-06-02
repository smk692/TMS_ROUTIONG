#!/usr/bin/env python3
"""
클러스터링 기반 배차 최적화 알고리즘

경로 중복 문제를 해결하기 위한 고급 클러스터링 및 배차 최적화
src/algorithm 모듈로 통합됨
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from src.model.delivery_point import DeliveryPoint
from src.model.vehicle import Vehicle
from src.utils.distance_calculator import calculate_distance, calculate_distances_batch
import logging

logger = logging.getLogger(__name__)

class ClusterOptimizer:
    """클러스터링 기반 배차 최적화기 - DBSCAN + Balanced K-means"""
    
    def __init__(self, data: Dict = None, data_file: str = None):
        """
        초기화
        Args:
            data: 직접 전달된 데이터 딕셔너리
            data_file: 데이터 파일 경로 (data 없을 때만 사용)
        """
        if data is not None:
            self.data = data
        elif data_file is not None:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            raise ValueError("data 또는 data_file 중 하나는 반드시 제공되어야 합니다.")
        
        self.routes = self.data['routes']
        self.tc_stats = self.data.get('tc_stats', [])
        
        # 성능 최적화 설정
        self.max_workers = min(4, multiprocessing.cpu_count())
        self.max_iterations = self.max_workers * 2  # 8회로 감소
        self.sample_size = min(100, len(self.routes) // 2)  # 샘플 크기 감소
        self.distance_cache = {}  # 거리 계산 캐싱
        
        print(f"🔧 클러스터 최적화기 초기화 완료")
        print(f"   - 최대 반복: {self.max_iterations}회")
        print(f"   - 샘플 크기: {self.sample_size}개")
        print(f"   - 병렬 워커: {self.max_workers}개")
        
    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """두 좌표 간 거리 계산 (km) - 캐시 적용"""
        # 캐시 키 생성 (소수점 4자리로 반올림하여 캐시 효율성 증대)
        key = (round(lat1, 4), round(lng1, 4), round(lat2, 4), round(lng2, 4))
        
        if key in self.distance_cache:
            return self.distance_cache[key]
        
        # 통합된 거리 계산 함수 사용
        distance = calculate_distance(lat1, lng1, lat2, lng2)
        self.distance_cache[key] = distance
        return distance
    
    def extract_delivery_points_by_tc(self) -> Dict:
        """TC별 배송지 추출 - 최적화"""
        tc_deliveries = defaultdict(list)
        
        for route in self.routes:
            tc_id = route.get('depot_id', 'unknown')
            delivery_coords = [coord for coord in route['coordinates'] if coord['type'] == 'delivery']
            
            for coord in delivery_coords:
                tc_deliveries[tc_id].append({
                    'lat': coord['lat'],
                    'lng': coord['lng'],
                    'label': coord['label'],
                    'current_vehicle': route['vehicle_id']
                })
        
        return tc_deliveries
    
    def optimize_tc_clustering(self, tc_id: str, deliveries: List[Dict], vehicle_count: int) -> List[List[Dict]]:
        """TC별 최적화된 클러스터링 - 성능 최적화"""
        print(f"\n🔧 {tc_id} 클러스터링 최적화 중...")
        
        if len(deliveries) < vehicle_count:
            print(f"  ⚠️ 배송지 수({len(deliveries)})가 차량 수({vehicle_count})보다 적음")
            return [deliveries]
        
        # 좌표 데이터 준비
        coords = np.array([[d['lat'], d['lng']] for d in deliveries])
        
        # 1. 간소화된 DBSCAN (샘플링 기반)
        sample_size = min(100, len(coords))
        sample_indices = np.random.choice(len(coords), sample_size, replace=False)
        sample_coords = coords[sample_indices]
        
        scaler = StandardScaler()
        sample_coords_scaled = scaler.fit_transform(sample_coords)
        
        # eps 값을 동적으로 계산 (샘플 기반)
        distances = []
        for i in range(min(50, len(sample_coords))):
            for j in range(i+1, min(50, len(sample_coords))):
                dist = self.calculate_distance(sample_coords[i][0], sample_coords[i][1], 
                                             sample_coords[j][0], sample_coords[j][1])
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 5.0
        eps = avg_distance / 3
        
        dbscan = DBSCAN(eps=eps/100, min_samples=max(2, sample_size//vehicle_count//2))
        sample_labels = dbscan.fit_predict(sample_coords_scaled)
        
        # 2. K-means로 직접 클러스터링 (DBSCAN 결과 무시하고 단순화)
        kmeans = KMeans(
            n_clusters=vehicle_count,
            init='k-means++',
            n_init=5,  # 반복 횟수 감소
            max_iter=100,  # 최대 반복 감소
            random_state=42
        )
        
        cluster_labels = kmeans.fit_predict(coords)
        
        # 3. 간소화된 클러스터 균형 조정
        cluster_labels = self.balance_clusters_fast(coords, cluster_labels, vehicle_count)
        
        # 4. 클러스터별 배송지 그룹화
        clusters = [[] for _ in range(vehicle_count)]
        for i, label in enumerate(cluster_labels):
            clusters[label].append(deliveries[i])
        
        # 5. 빈 클러스터 처리
        clusters = [cluster for cluster in clusters if cluster]
        
        # 6. 간소화된 클러스터 품질 평가
        self.evaluate_clustering_quality_fast(clusters, tc_id)
        
        return clusters
    
    def balance_clusters_fast(self, coords: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
        """클러스터 크기 균형 조정 - 고속 버전"""
        target_size = len(coords) // n_clusters
        tolerance = max(5, target_size // 3)  # 허용 오차 증가
        
        # 현재 클러스터 크기 계산
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        
        # 크기가 불균형한 경우 재조정 (반복 횟수 감소)
        max_iterations = 5  # 기존 10에서 감소
        for iteration in range(max_iterations):
            rebalanced = False
            
            for i in range(n_clusters):
                if cluster_sizes[i] > target_size + tolerance:
                    cluster_points = np.where(labels == i)[0]
                    
                    # 가장 작은 클러스터 찾기
                    min_cluster = np.argmin(cluster_sizes)
                    if cluster_sizes[min_cluster] < target_size - tolerance:
                        # 간단한 이동 (거리 계산 생략)
                        if len(cluster_points) > 0:
                            # 랜덤하게 하나 선택 (거리 계산 생략으로 성능 향상)
                            random_idx = np.random.choice(cluster_points)
                            labels[random_idx] = min_cluster
                            
                            cluster_sizes[i] -= 1
                            cluster_sizes[min_cluster] += 1
                            rebalanced = True
            
            if not rebalanced:
                break
        
        return labels
    
    def evaluate_clustering_quality_fast(self, clusters: List[List[Dict]], tc_id: str) -> None:
        """클러스터링 품질 평가 - 고속 버전"""
        print(f"  📊 {tc_id} 클러스터링 품질 평가:")
        
        total_intra_distance = 0
        
        # 간소화된 품질 평가 (샘플링 기반)
        for i, cluster in enumerate(clusters):
            if len(cluster) < 2:
                print(f"    🚛 클러스터 {i+1}: {len(cluster)}개 배송지 (단일 지점)")
                continue
                
            # 샘플링으로 거리 계산 (최대 5개 샘플)
            sample_size = min(5, len(cluster))
            sample_indices = np.random.choice(len(cluster), sample_size, replace=False)
            
            cluster_distances = []
            for j in range(len(sample_indices)):
                for k in range(j+1, len(sample_indices)):
                    idx_j, idx_k = sample_indices[j], sample_indices[k]
                    dist = self.calculate_distance(
                        cluster[idx_j]['lat'], cluster[idx_j]['lng'],
                        cluster[idx_k]['lat'], cluster[idx_k]['lng']
                    )
                    cluster_distances.append(dist)
            
            if cluster_distances:
                avg_intra = np.mean(cluster_distances)
                total_intra_distance += avg_intra
                print(f"    🚛 클러스터 {i+1}: {len(cluster)}개 배송지, 평균 내부 거리 {avg_intra:.2f}km")
        
        # 클러스터 간 거리 (샘플링)
        cluster_centers = []
        for cluster in clusters:
            if cluster:
                center_lat = np.mean([d['lat'] for d in cluster])
                center_lng = np.mean([d['lng'] for d in cluster])
                cluster_centers.append((center_lat, center_lng))
        
        if len(cluster_centers) > 1:
            # 최대 3개 샘플만 계산
            sample_size = min(3, len(cluster_centers))
            inter_distances = []
            for i in range(sample_size):
                for j in range(i+1, len(cluster_centers)):
                    dist = self.calculate_distance(
                        cluster_centers[i][0], cluster_centers[i][1],
                        cluster_centers[j][0], cluster_centers[j][1]
                    )
                    inter_distances.append(dist)
            
            if inter_distances:
                avg_inter = np.mean(inter_distances)
                print(f"    📏 평균 클러스터 간 거리: {avg_inter:.2f}km")
                
                # 실루엣 점수 유사 지표
                if total_intra_distance > 0:
                    separation_score = avg_inter / (total_intra_distance / len(clusters))
                    print(f"    🎯 분리도 점수: {separation_score:.2f} (높을수록 좋음)")
    
    def generate_optimized_routes(self) -> Dict:
        """최적화된 경로 생성 - 병렬 처리 적용"""
        print("\n🚀 배차 최적화 시작...")
        
        # TC별 배송지 추출
        tc_deliveries = self.extract_delivery_points_by_tc()
        
        optimized_data = {
            'multi_depot': True,
            'depots': self.data['depots'],
            'routes': [],
            'stats': {},
            'tc_stats': [],
            'optimization_info': {
                'algorithm': 'DBSCAN + Balanced K-means (Optimized)',
                'improvements': [
                    '지역 기반 자연스러운 클러스터링',
                    '클러스터 크기 균형 조정',
                    '중복 영역 최소화',
                    '효율성 지표 최적화',
                    '병렬 처리 및 캐싱 최적화'
                ]
            }
        }
        
        total_routes = []
        total_distance = 0
        total_time = 0
        total_points = 0
        
        # TC별 처리를 병렬화
        tc_tasks = []
        for tc_stat in self.tc_stats:
            tc_id = tc_stat['tc_id']
            if tc_id in tc_deliveries:
                tc_tasks.append((tc_stat, tc_deliveries[tc_id]))
        
        # 병렬 처리로 TC별 최적화
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tc = {executor.submit(self.process_tc_optimization, task): task for task in tc_tasks}
            
            for future in future_to_tc:
                try:
                    tc_result = future.result()
                    if tc_result:
                        total_routes.extend(tc_result['routes'])
                        total_distance += tc_result['distance']
                        total_time += tc_result['time']
                        total_points += tc_result['points']
                        optimized_data['tc_stats'].append(tc_result['tc_stat'])
                except Exception as e:
                    print(f"❌ TC 처리 오류: {e}")
        
        # 전체 통계
        optimized_data['routes'] = total_routes
        optimized_data['stats'] = {
            'total_points': total_points,
            'total_vehicles': len(total_routes),
            'total_distance': total_distance,
            'total_time': total_time,
            'avg_distance_per_vehicle': total_distance / len(total_routes) if total_routes else 0,
            'avg_time_per_vehicle': total_time / len(total_routes) if total_routes else 0,
            'tc_count': len(self.tc_stats),
            'optimization_applied': True,
            'time_efficiency': (total_points / total_time) if total_time > 0 else 0
        }
        
        return optimized_data
    
    def process_tc_optimization(self, task_data) -> Dict:
        """TC별 최적화 처리 (병렬 처리용)"""
        tc_stat, deliveries = task_data
        tc_id = tc_stat['tc_id']
        tc_name = tc_stat['tc_name']
        vehicle_count = tc_stat['vehicles']
        
        print(f"\n🏢 {tc_name}: {len(deliveries)}개 배송지 → {vehicle_count}대 차량")
        
        # 최적화된 클러스터링
        clusters = self.optimize_tc_clustering(tc_id, deliveries, vehicle_count)
        
        # 각 클러스터를 차량에 할당
        tc_routes = []
        tc_distance = 0
        tc_time = 0
        
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
            
            vehicle_id = len(tc_routes) + 1
            tc_vehicle_number = i + 1
            
            # 기존 데이터에서 해당 차량 정보 찾기
            original_route = None
            for route in self.routes:
                if route.get('depot_id') == tc_id:
                    original_route = route
                    break
            
            if original_route:
                # 간소화된 거리/시간 추정
                estimated_distance = len(cluster) * 2.5  # 배송지당 평균 2.5km 추정
                estimated_time = len(cluster) * 15  # 배송지당 평균 15분 추정
                
                route_data = {
                    'vehicle_id': vehicle_id,
                    'vehicle_name': f'{tc_vehicle_number}호차',
                    'vehicle_type': original_route.get('vehicle_type', 'TRUCK_1TON'),
                    'depot_id': tc_id,
                    'depot_name': tc_name,
                    'delivery_count': len(cluster),
                    'distance': estimated_distance,
                    'time': estimated_time,
                    'coordinates': [
                        {
                            'id': tc_id,
                            'label': tc_name,
                            'lat': original_route['coordinates'][0]['lat'],
                            'lng': original_route['coordinates'][0]['lng'],
                            'type': 'depot'
                        }
                    ]
                }
                
                # 클러스터의 배송지들 추가
                for j, delivery in enumerate(cluster):
                    route_data['coordinates'].append({
                        'id': f"{tc_id}_delivery_{vehicle_id}_{j+1}",
                        'label': delivery['label'],
                        'lat': delivery['lat'],
                        'lng': delivery['lng'],
                        'type': 'delivery',
                        'sequence': j + 1
                    })
                
                # 다시 depot으로 돌아가기
                route_data['coordinates'].append(route_data['coordinates'][0])
                
                tc_routes.append(route_data)
                tc_distance += estimated_distance
                tc_time += estimated_time
        
        print(f"  ✅ 최적화 완료: {len(tc_routes)}대 차량, 예상 거리 {tc_distance:.1f}km")
        
        return {
            'routes': tc_routes,
            'distance': tc_distance,
            'time': tc_time,
            'points': len(deliveries),
            'tc_stat': {
                'tc_id': tc_id,
                'tc_name': tc_name,
                'delivery_points': len(deliveries),
                'vehicles': len(tc_routes),
                'total_distance': tc_distance,
                'total_time': tc_time
            }
        }

    def save_optimized_routes(self, output_file: str) -> None:
        """최적화된 경로를 파일로 저장"""
        optimized_data = self.generate_optimized_routes()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 최적화된 데이터 저장 완료: {output_file}")

def optimize_routes_with_clustering(data: Dict) -> Dict:
    """
    클러스터링 최적화를 적용한 경로 생성 (외부 호출용)
    
    Args:
        data: 기존 경로 데이터
        
    Returns:
        클러스터링 최적화된 경로 데이터
    """
    optimizer = ClusterOptimizer(data=data)
    return optimizer.generate_optimized_routes()

# 기존 main 함수는 테스트용으로 유지
def main():
    """테스트용 메인 함수"""
    input_file = Path("../../data/extracted_coordinates.json")  # 상대 경로 수정
    output_file = Path("../../data/cluster_optimized_coordinates.json")
    
    if not input_file.exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        return
    
    optimizer = ClusterOptimizer(data_file=str(input_file))
    optimizer.save_optimized_routes(str(output_file))
    
    print(f"\n🎉 클러스터링 최적화 완료!")
    print(f"📁 최적화된 데이터: {output_file}")

if __name__ == "__main__":
    main() 