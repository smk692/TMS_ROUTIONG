#!/usr/bin/env python3
"""
Voronoi 기반 배차 최적화 알고리즘

차량 간 경로 중복을 근본적으로 해결하는 고급 알고리즘
src/algorithm 모듈로 통합됨
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from scipy.spatial import Voronoi, voronoi_plot_2d
from src.utils.distance_calculator import calculate_distance, calculate_distances_batch, assign_points_to_nearest_centers
import logging

logger = logging.getLogger(__name__)

class VoronoiOptimizer:
    """Voronoi 기반 배차 최적화기 - 영역 분할 알고리즘"""
    
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
        
        # 성능 최적화: 거리 계산 캐시
        self._distance_cache = {}
        
        # 병렬 처리 설정
        self.max_workers = min(4, multiprocessing.cpu_count())
        
        # 성능 최적화: 반복 횟수 감소
        self.max_iterations = self.max_workers * 2  # 12회로 감소 (이전 20회)
        self.sample_size = min(100, len(self.routes) // 2)  # 샘플 크기 감소
        self.distance_cache = {}  # 거리 계산 캐싱
        
        print(f"🎯 Voronoi 최적화기 초기화 완료")
        print(f"   - 최대 반복: {self.max_iterations}회")
        print(f"   - 샘플 크기: {self.sample_size}개")
        print(f"   - 병렬 워커: {self.max_workers}개")
        
    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """두 좌표 간 거리 계산 (km) - 캐시 적용"""
        # 캐시 키 생성 (소수점 4자리로 반올림하여 캐시 효율성 증대)
        key = (round(lat1, 4), round(lng1, 4), round(lat2, 4), round(lng2, 4))
        
        if key in self._distance_cache:
            return self._distance_cache[key]
        
        # 통합된 거리 계산 함수 사용
        distance = calculate_distance(lat1, lng1, lat2, lng2)
        self._distance_cache[key] = distance
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
    
    def create_voronoi_regions(self, tc_id: str, deliveries: List[Dict], vehicle_count: int) -> List[List[Dict]]:
        """Voronoi 다이어그램 기반 영역 분할 - 성능 최적화"""
        print(f"\n🗺️ {tc_id} Voronoi 영역 분할 중...")
        
        if len(deliveries) < vehicle_count:
            print(f"  ⚠️ 배송지 수({len(deliveries)})가 차량 수({vehicle_count})보다 적음")
            return [deliveries]
        
        # 좌표 데이터 준비
        coords = np.array([[d['lat'], d['lng']] for d in deliveries])
        
        # 1. 전략적 시드 포인트 생성 (최적화된 버전)
        seed_points = self.generate_strategic_seeds_optimized(coords, vehicle_count)
        
        # 2. 각 배송지를 가장 가까운 시드에 할당 (벡터화 연산)
        distances = cdist(coords, seed_points)
        assignments = np.argmin(distances, axis=1)
        
        # 3. 영역별 배송지 그룹화 (최적화)
        regions = [[] for _ in range(vehicle_count)]
        for i, region_id in enumerate(assignments):
            regions[region_id].append(deliveries[i])
        
        # 4. 빈 영역 처리 및 균형 조정 (간소화)
        regions = self.balance_voronoi_regions_fast(regions, deliveries, vehicle_count)
        
        # 5. 영역 품질 평가 (간소화)
        self.evaluate_voronoi_quality_fast(regions, tc_id)
        
        return regions
    
    def generate_strategic_seeds_optimized(self, coords: np.ndarray, vehicle_count: int) -> np.ndarray:
        """전략적 시드 포인트 생성 - 최적화 버전"""
        print(f"  🎯 {vehicle_count}개 전략적 시드 포인트 생성 중...")
        
        # 1. 경계 박스 계산
        min_lat, max_lat = coords[:, 0].min(), coords[:, 0].max()
        min_lng, max_lng = coords[:, 1].min(), coords[:, 1].max()
        
        # 2. K-means++로 직접 최적 시드 선택 (중간 단계 생략)
        kmeans = KMeans(n_clusters=vehicle_count, init='k-means++', n_init=5, max_iter=100, random_state=42)
        kmeans.fit(coords)
        seeds = kmeans.cluster_centers_
        
        # 3. 시드 포인트 미세 조정 (반복 횟수 감소)
        seeds = self.refine_seeds_fast(coords, seeds)
        
        print(f"  ✅ 시드 포인트 생성 완료: {len(seeds)}개")
        return seeds
    
    def refine_seeds_fast(self, coords: np.ndarray, initial_seeds: np.ndarray) -> np.ndarray:
        """시드 포인트 미세 조정 - 고속 버전"""
        refined_seeds = initial_seeds.copy()
        
        # 반복 횟수 감소 (5 → 2)
        for iteration in range(2):
            distances = cdist(coords, refined_seeds)
            assignments = np.argmin(distances, axis=1)
            
            new_seeds = []
            for i in range(len(refined_seeds)):
                assigned_points = coords[assignments == i]
                if len(assigned_points) > 0:
                    new_center = np.mean(assigned_points, axis=0)
                    new_seeds.append(new_center)
                else:
                    new_seeds.append(refined_seeds[i])
            
            refined_seeds = np.array(new_seeds)
        
        return refined_seeds
    
    def balance_voronoi_regions_fast(self, regions: List[List[Dict]], all_deliveries: List[Dict], vehicle_count: int) -> List[List[Dict]]:
        """Voronoi 영역 균형 조정 - 고속 버전"""
        print(f"  ⚖️ 영역 균형 조정 중...")
        
        # 빈 영역 제거
        non_empty_regions = [region for region in regions if region]
        
        if len(non_empty_regions) < vehicle_count:
            # 빈 영역이 있는 경우, 가장 큰 영역을 분할
            while len(non_empty_regions) < vehicle_count:
                largest_idx = max(range(len(non_empty_regions)), key=lambda i: len(non_empty_regions[i]))
                largest_region = non_empty_regions[largest_idx]
                
                if len(largest_region) < 2:
                    break
                
                # 간단한 분할 (K-means 대신 중간점 기준)
                mid_point = len(largest_region) // 2
                region1 = largest_region[:mid_point]
                region2 = largest_region[mid_point:]
                
                non_empty_regions[largest_idx] = region1
                non_empty_regions.append(region2)
        
        # 크기 균형 조정 (반복 횟수 감소)
        target_size = len(all_deliveries) // vehicle_count
        tolerance = max(3, target_size // 3)  # 허용 오차 증가
        
        # 최대 5회 반복 (기존 10회에서 감소)
        for iteration in range(5):
            region_sizes = [len(region) for region in non_empty_regions]
            
            oversized = [i for i, size in enumerate(region_sizes) if size > target_size + tolerance]
            undersized = [i for i, size in enumerate(region_sizes) if size < target_size - tolerance]
            
            if not oversized or not undersized:
                break
            
            # 간단한 이동 (거리 계산 생략)
            largest_region_idx = max(oversized, key=lambda i: region_sizes[i])
            smallest_region_idx = min(undersized, key=lambda i: region_sizes[i])
            
            largest_region = non_empty_regions[largest_region_idx]
            smallest_region = non_empty_regions[smallest_region_idx]
            
            if largest_region:
                moved_delivery = largest_region.pop()
                smallest_region.append(moved_delivery)
        
        final_regions = [region for region in non_empty_regions if region]
        
        print(f"  ✅ 균형 조정 완료: {len(final_regions)}개 영역")
        for i, region in enumerate(final_regions):
            print(f"    영역 {i+1}: {len(region)}개 배송지")
        
        return final_regions
    
    def evaluate_voronoi_quality_fast(self, regions: List[List[Dict]], tc_id: str) -> None:
        """Voronoi 영역 품질 평가 - 고속 버전"""
        print(f"  📊 {tc_id} Voronoi 영역 품질 평가:")
        
        if not regions:
            print("    ❌ 영역이 없습니다.")
            return
        
        # 간소화된 품질 평가 (샘플링 기반)
        total_intra_distance = 0
        region_centers = []
        
        for i, region in enumerate(regions):
            if len(region) < 2:
                print(f"    🚛 영역 {i+1}: {len(region)}개 배송지 (단일 지점)")
                if region:
                    region_centers.append((region[0]['lat'], region[0]['lng']))
                continue
            
            # 샘플링으로 거리 계산 (모든 조합 대신 최대 10개 샘플)
            sample_size = min(10, len(region))
            sample_indices = np.random.choice(len(region), sample_size, replace=False)
            
            region_distances = []
            for j in range(len(sample_indices)):
                for k in range(j+1, len(sample_indices)):
                    idx_j, idx_k = sample_indices[j], sample_indices[k]
                    dist = self.calculate_distance(
                        region[idx_j]['lat'], region[idx_j]['lng'],
                        region[idx_k]['lat'], region[idx_k]['lng']
                    )
                    region_distances.append(dist)
            
            if region_distances:
                avg_intra = np.mean(region_distances)
                total_intra_distance += avg_intra
                print(f"    🚛 영역 {i+1}: {len(region)}개 배송지, 평균 내부 거리 {avg_intra:.2f}km")
            
            # 영역 중심 계산
            center_lat = np.mean([d['lat'] for d in region])
            center_lng = np.mean([d['lng'] for d in region])
            region_centers.append((center_lat, center_lng))
        
        # 영역 간 분리도 (샘플링)
        if len(region_centers) > 1:
            inter_distances = []
            for i in range(min(5, len(region_centers))):  # 최대 5개 샘플
                for j in range(i+1, len(region_centers)):
                    dist = self.calculate_distance(
                        region_centers[i][0], region_centers[i][1],
                        region_centers[j][0], region_centers[j][1]
                    )
                    inter_distances.append(dist)
            
            if inter_distances:
                avg_inter = np.mean(inter_distances)
                print(f"    📏 평균 영역 간 거리: {avg_inter:.2f}km")
                
                # Voronoi 품질 지수
                if total_intra_distance > 0:
                    voronoi_quality = avg_inter / (total_intra_distance / len(regions))
                    print(f"    🎯 Voronoi 품질 지수: {voronoi_quality:.2f} (높을수록 좋음)")
                    
                    if voronoi_quality > 3.0:
                        print(f"    ✅ 우수한 영역 분할")
                    elif voronoi_quality > 2.0:
                        print(f"    👍 양호한 영역 분할")
                    else:
                        print(f"    ⚠️ 개선 필요한 영역 분할")
    
    def generate_optimized_routes(self) -> Dict:
        """Voronoi 기반 최적화된 경로 생성 - 병렬 처리 적용"""
        print("\n🚀 Voronoi 기반 배차 최적화 시작...")
        
        # TC별 배송지 추출
        tc_deliveries = self.extract_delivery_points_by_tc()
        
        optimized_data = {
            'multi_depot': True,
            'depots': self.data['depots'],
            'routes': [],
            'stats': {},
            'tc_stats': [],
            'optimization_info': {
                'algorithm': 'Voronoi Diagram + Strategic Seeding (Optimized)',
                'improvements': [
                    '영역 기반 완전 분할',
                    '차량 간 중복 영역 제거',
                    '전략적 시드 포인트 배치',
                    'Voronoi 품질 지수 최적화',
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
            'voronoi_optimized': True,
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
        
        # Voronoi 영역 분할
        voronoi_regions = self.create_voronoi_regions(tc_id, deliveries, vehicle_count)
        
        # 각 영역을 차량에 할당
        tc_routes = []
        tc_distance = 0
        tc_time = 0
        
        for i, region in enumerate(voronoi_regions):
            if not region:
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
                # 영역 기반 거리/시간 추정 (간소화)
                estimated_distance = self.estimate_region_distance_fast(region)
                estimated_time = self.estimate_region_time_fast(region)
                
                route_data = {
                    'vehicle_id': vehicle_id,
                    'vehicle_name': f'{tc_vehicle_number}호차',
                    'vehicle_type': original_route.get('vehicle_type', 'TRUCK_1TON'),
                    'depot_id': tc_id,
                    'depot_name': tc_name,
                    'delivery_count': len(region),
                    'distance': estimated_distance,
                    'time': estimated_time,
                    'voronoi_region': i + 1,
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
                
                # 영역의 배송지들 추가 (간소화된 순서 정렬)
                sorted_region = self.optimize_region_sequence_fast(region, route_data['coordinates'][0])
                
                for j, delivery in enumerate(sorted_region):
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
        
        print(f"  ✅ Voronoi 최적화 완료: {len(tc_routes)}대 차량, 예상 거리 {tc_distance:.1f}km")
        
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
                'total_time': tc_time,
                'voronoi_optimized': True
            }
        }
    
    def estimate_region_distance_fast(self, region: List[Dict]) -> float:
        """영역 기반 거리 추정 - 고속 버전"""
        if len(region) <= 1:
            return 5.0
        
        # 간소화된 추정 (TSP 근사 대신 평균 거리 기반)
        coords = [(d['lat'], d['lng']) for d in region]
        
        # 중심점 계산
        center_lat = sum(coord[0] for coord in coords) / len(coords)
        center_lng = sum(coord[1] for coord in coords) / len(coords)
        
        # 중심점에서 각 점까지의 평균 거리 * 2 (왕복)
        total_distance = 0
        for lat, lng in coords:
            distance = self.calculate_distance(center_lat, center_lng, lat, lng)
            total_distance += distance * 2  # 왕복
        
        # depot 거리 추가
        depot_distance = len(region) * 1.5
        
        return total_distance + depot_distance
    
    def estimate_region_time_fast(self, region: List[Dict]) -> int:
        """영역 기반 시간 추정 - 고속 버전"""
        if len(region) <= 1:
            return 30
        
        # 간소화된 시간 추정
        delivery_time = len(region) * 10  # 배송지당 10분
        travel_time = len(region) * 8     # 배송지당 평균 8분 이동
        
        return delivery_time + travel_time
    
    def optimize_region_sequence_fast(self, region: List[Dict], depot: Dict) -> List[Dict]:
        """영역 내 배송지 순서 최적화 - 고속 버전"""
        if len(region) <= 1:
            return region
        
        # 간소화된 순서 정렬 (nearest neighbor 대신 거리 기반 정렬)
        depot_pos = (depot['lat'], depot['lng'])
        
        # depot에서 각 배송지까지의 거리 계산
        distances = []
        for delivery in region:
            dist = self.calculate_distance(
                depot_pos[0], depot_pos[1],
                delivery['lat'], delivery['lng']
            )
            distances.append((dist, delivery))
        
        # 거리순으로 정렬
        distances.sort(key=lambda x: x[0])
        
        return [delivery for _, delivery in distances]

    def save_optimized_routes(self, output_file: str) -> None:
        """최적화된 경로를 파일로 저장"""
        optimized_data = self.generate_optimized_routes()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 최적화된 데이터 저장 완료: {output_file}")

def optimize_routes_with_voronoi(data: Dict) -> Dict:
    """
    Voronoi 최적화를 적용한 경로 생성 (외부 호출용)
    
    Args:
        data: 기존 경로 데이터
        
    Returns:
        Voronoi 최적화된 경로 데이터
    """
    optimizer = VoronoiOptimizer(data=data)
    return optimizer.generate_optimized_routes()

# 기존 main 함수는 테스트용으로 유지
def main():
    """테스트용 메인 함수"""
    input_file = Path("../../data/extracted_coordinates.json")  # 상대 경로 수정
    output_file = Path("../../data/voronoi_optimized_coordinates.json")
    
    if not input_file.exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        return
    
    optimizer = VoronoiOptimizer(data_file=str(input_file))
    optimizer.save_optimized_routes(str(output_file))
    
    print(f"\n🎉 Voronoi 최적화 완료!")
    print(f"📁 최적화된 데이터: {output_file}")

if __name__ == "__main__":
    main() 