#!/usr/bin/env python3
"""
경로 최적화 분석기

경로 중복, 효율성, 성능 지표를 종합적으로 분석
src/analysis 모듈로 통합됨
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class RouteAnalyzer:
    """경로 최적화 종합 분석기"""
    
    def __init__(self, data: Dict = None, data_file: str = None):
        """
        초기화
        Args:
            data: 직접 전달된 데이터 딕셔너리
            data_file: 데이터 파일 경로 (data 없을 때만 사용)
        """
        if data is not None:
            self.data = data
        else:
            # 기본 파일 경로 설정
            if data_file is None:
                data_file = "../../data/extracted_coordinates.json"
            
            self.data = self._load_data(data_file)
        
        # 분석 결과 저장
        self.analysis_results = {}
    
    def _load_data(self, file_path: str) -> Optional[Dict]:
        """데이터 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ 파일을 찾을 수 없습니다: {file_path}")
            return None
        except Exception as e:
            print(f"❌ 파일 로드 오류: {e}")
            return None
    
    def analyze_route_optimization(self) -> Dict:
        """경로 최적화 종합 분석"""
        if not self.data:
            print("❌ 분석할 데이터가 없습니다.")
            return {}
        
        print("🔍 경로 최적화 종합 분석 시작...")
        
        # 기본 통계 분석
        basic_stats = self._analyze_basic_statistics()
        
        # 경로 중복 분석
        overlap_analysis = self._analyze_route_overlaps()
        
        # 효율성 지표 분석
        efficiency_analysis = self._analyze_efficiency_metrics()
        
        # TC별 성능 분석
        tc_analysis = self._analyze_tc_performance()
        
        # 차량별 성능 분석
        vehicle_analysis = self._analyze_vehicle_performance()
        
        # 지역별 분포 분석
        distribution_analysis = self._analyze_geographical_distribution()
        
        # 결과 통합
        self.analysis_results = {
            'basic_stats': basic_stats,
            'overlap_analysis': overlap_analysis,
            'efficiency_analysis': efficiency_analysis,
            'tc_analysis': tc_analysis,
            'vehicle_analysis': vehicle_analysis,
            'distribution_analysis': distribution_analysis,
            'recommendations': self._generate_recommendations()
        }
        
        return self.analysis_results
    
    def _analyze_basic_statistics(self) -> Dict:
        """기본 통계 분석"""
        stats = self.data.get('stats', {})
        routes = self.data.get('routes', {})
        
        return {
            'total_vehicles': len(routes),
            'total_points': stats.get('total_points', 0),
            'total_distance': stats.get('total_distance', 0),
            'total_time': stats.get('total_time', 0),
            'avg_distance_per_vehicle': stats.get('total_distance', 0) / max(len(routes), 1),
            'avg_points_per_vehicle': stats.get('total_points', 0) / max(len(routes), 1),
            'avg_time_per_vehicle': stats.get('total_time', 0) / max(len(routes), 1)
        }
    
    def _analyze_route_overlaps(self) -> Dict:
        """경로 중복 분석"""
        routes = self.data.get('routes', {})
        
        # TC별 그룹화
        tc_routes = defaultdict(list)
        for vehicle_id, route_data in routes.items():
            tc_id = route_data.get('tc_id', 'unknown')
            tc_routes[tc_id].append((vehicle_id, route_data))
        
        total_overlaps = 0
        tc_overlap_details = {}
        
        for tc_id, tc_route_list in tc_routes.items():
            if len(tc_route_list) < 2:
                tc_overlap_details[tc_id] = {'overlaps': 0, 'overlap_pairs': []}
                continue
            
            tc_overlaps = 0
            overlap_pairs = []
            
            for i in range(len(tc_route_list)):
                for j in range(i + 1, len(tc_route_list)):
                    vehicle1_id, route1 = tc_route_list[i]
                    vehicle2_id, route2 = tc_route_list[j]
                    
                    # 경로 중복 계산
                    overlap_count = self._calculate_route_overlap(route1, route2)
                    
                    if overlap_count > 0:
                        tc_overlaps += overlap_count
                        total_overlaps += overlap_count
                        overlap_pairs.append({
                            'vehicle1': vehicle1_id,
                            'vehicle2': vehicle2_id,
                            'overlap_count': overlap_count
                        })
            
            tc_overlap_details[tc_id] = {
                'overlaps': tc_overlaps,
                'overlap_pairs': overlap_pairs
            }
        
        return {
            'total_overlaps': total_overlaps,
            'tc_overlap_details': tc_overlap_details,
            'overlap_rate': self._calculate_overlap_rate(total_overlaps, len(routes))
        }
    
    def _calculate_route_overlap(self, route1: Dict, route2: Dict) -> int:
        """두 경로 간 중복 배송지 계산"""
        route1_points = set()
        route2_points = set()
        
        # 경로 1의 배송지 수집
        for point in route1.get('route', []):
            if point.get('type') == 'delivery':
                route1_points.add((point.get('lat'), point.get('lng')))
        
        # 경로 2의 배송지 수집
        for point in route2.get('route', []):
            if point.get('type') == 'delivery':
                route2_points.add((point.get('lat'), point.get('lng')))
        
        # 중복 배송지 계산
        return len(route1_points.intersection(route2_points))
    
    def _calculate_overlap_rate(self, total_overlaps: int, total_vehicles: int) -> float:
        """중복률 계산"""
        if total_vehicles <= 1:
            return 0.0
        
        max_possible_pairs = total_vehicles * (total_vehicles - 1) / 2
        return (total_overlaps / max_possible_pairs) * 100 if max_possible_pairs > 0 else 0.0
    
    def _analyze_efficiency_metrics(self) -> Dict:
        """효율성 지표 분석"""
        routes = self.data.get('routes', {})
        stats = self.data.get('stats', {})
        
        # 차량별 효율성 계산
        vehicle_efficiencies = []
        for vehicle_id, route_data in routes.items():
            delivery_count = len([p for p in route_data.get('route', []) if p.get('type') == 'delivery'])
            distance = route_data.get('total_distance', 0)
            time = route_data.get('total_time', 0)
            
            efficiency = {
                'vehicle_id': vehicle_id,
                'delivery_count': delivery_count,
                'distance': distance,
                'time': time,
                'distance_per_delivery': distance / max(delivery_count, 1),
                'time_per_delivery': time / max(delivery_count, 1)
            }
            vehicle_efficiencies.append(efficiency)
        
        # 통계 계산
        distances = [v['distance'] for v in vehicle_efficiencies]
        times = [v['time'] for v in vehicle_efficiencies]
        deliveries = [v['delivery_count'] for v in vehicle_efficiencies]
        
        return {
            'vehicle_efficiencies': vehicle_efficiencies,
            'distance_stats': {
                'mean': np.mean(distances) if distances else 0,
                'std': np.std(distances) if distances else 0,
                'min': min(distances) if distances else 0,
                'max': max(distances) if distances else 0
            },
            'time_stats': {
                'mean': np.mean(times) if times else 0,
                'std': np.std(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0
            },
            'delivery_stats': {
                'mean': np.mean(deliveries) if deliveries else 0,
                'std': np.std(deliveries) if deliveries else 0,
                'min': min(deliveries) if deliveries else 0,
                'max': max(deliveries) if deliveries else 0
            }
        }
    
    def _analyze_tc_performance(self) -> Dict:
        """TC별 성능 분석"""
        routes = self.data.get('routes', {})
        
        tc_performance = defaultdict(lambda: {
            'vehicles': 0,
            'total_points': 0,
            'total_distance': 0,
            'total_time': 0
        })
        
        for vehicle_id, route_data in routes.items():
            tc_id = route_data.get('tc_id', 'unknown')
            delivery_count = len([p for p in route_data.get('route', []) if p.get('type') == 'delivery'])
            
            tc_performance[tc_id]['vehicles'] += 1
            tc_performance[tc_id]['total_points'] += delivery_count
            tc_performance[tc_id]['total_distance'] += route_data.get('total_distance', 0)
            tc_performance[tc_id]['total_time'] += route_data.get('total_time', 0)
        
        # 평균 계산
        for tc_id, perf in tc_performance.items():
            vehicles = perf['vehicles']
            if vehicles > 0:
                perf['avg_distance_per_vehicle'] = perf['total_distance'] / vehicles
                perf['avg_points_per_vehicle'] = perf['total_points'] / vehicles
                perf['avg_time_per_vehicle'] = perf['total_time'] / vehicles
            else:
                perf['avg_distance_per_vehicle'] = 0
                perf['avg_points_per_vehicle'] = 0
                perf['avg_time_per_vehicle'] = 0
        
        return dict(tc_performance)
    
    def _analyze_vehicle_performance(self) -> Dict:
        """차량별 성능 분석"""
        routes = self.data.get('routes', {})
        
        vehicle_performance = []
        for vehicle_id, route_data in routes.items():
            delivery_count = len([p for p in route_data.get('route', []) if p.get('type') == 'delivery'])
            distance = route_data.get('total_distance', 0)
            time = route_data.get('total_time', 0)
            
            performance = {
                'vehicle_id': vehicle_id,
                'tc_id': route_data.get('tc_id', 'unknown'),
                'delivery_count': delivery_count,
                'distance': distance,
                'time': time,
                'efficiency_score': self._calculate_efficiency_score(delivery_count, distance, time)
            }
            vehicle_performance.append(performance)
        
        # 성능 순위 계산
        vehicle_performance.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        return {
            'vehicle_rankings': vehicle_performance,
            'top_performers': vehicle_performance[:5],
            'bottom_performers': vehicle_performance[-5:] if len(vehicle_performance) >= 5 else []
        }
    
    def _calculate_efficiency_score(self, delivery_count: int, distance: float, time: float) -> float:
        """효율성 점수 계산"""
        if distance == 0 or time == 0:
            return 0.0
        
        # 배송지 수 대비 거리와 시간의 효율성
        distance_efficiency = delivery_count / distance if distance > 0 else 0
        time_efficiency = delivery_count / time if time > 0 else 0
        
        # 가중 평균 (거리 60%, 시간 40%)
        return (distance_efficiency * 0.6 + time_efficiency * 0.4) * 100
    
    def _analyze_geographical_distribution(self) -> Dict:
        """지역별 분포 분석"""
        routes = self.data.get('routes', {})
        
        # 배송지 좌표 수집
        all_coordinates = []
        tc_coordinates = defaultdict(list)
        
        for vehicle_id, route_data in routes.items():
            tc_id = route_data.get('tc_id', 'unknown')
            
            for point in route_data.get('route', []):
                if point.get('type') == 'delivery':
                    coord = (point.get('lat'), point.get('lng'))
                    all_coordinates.append(coord)
                    tc_coordinates[tc_id].append(coord)
        
        # 분포 통계 계산
        distribution_stats = {}
        
        if all_coordinates:
            lats = [coord[0] for coord in all_coordinates]
            lngs = [coord[1] for coord in all_coordinates]
            
            distribution_stats['overall'] = {
                'center_lat': np.mean(lats),
                'center_lng': np.mean(lngs),
                'lat_range': max(lats) - min(lats),
                'lng_range': max(lngs) - min(lngs),
                'spread_score': np.std(lats) + np.std(lngs)
            }
        
        # TC별 분포
        for tc_id, coords in tc_coordinates.items():
            if coords:
                lats = [coord[0] for coord in coords]
                lngs = [coord[1] for coord in coords]
                
                distribution_stats[tc_id] = {
                    'center_lat': np.mean(lats),
                    'center_lng': np.mean(lngs),
                    'lat_range': max(lats) - min(lats),
                    'lng_range': max(lngs) - min(lngs),
                    'spread_score': np.std(lats) + np.std(lngs)
                }
        
        return distribution_stats
    
    def _generate_recommendations(self) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if not self.analysis_results:
            return recommendations
        
        # 중복 분석 기반 권장사항
        overlap_analysis = self.analysis_results.get('overlap_analysis', {})
        total_overlaps = overlap_analysis.get('total_overlaps', 0)
        
        if total_overlaps > 50:
            recommendations.append("🔥 높은 경로 중복 감지: Voronoi 다이어그램 기반 최적화 적용 권장")
        elif total_overlaps > 20:
            recommendations.append("⚠️ 중간 수준 경로 중복: 클러스터링 기반 최적화 검토 권장")
        
        # 효율성 분석 기반 권장사항
        efficiency_analysis = self.analysis_results.get('efficiency_analysis', {})
        distance_std = efficiency_analysis.get('distance_stats', {}).get('std', 0)
        
        if distance_std > 50:
            recommendations.append("📊 차량 간 거리 편차 큼: 배송지 재분배 검토 필요")
        
        # TC별 성능 기반 권장사항
        tc_analysis = self.analysis_results.get('tc_analysis', {})
        tc_distances = [tc_data.get('avg_distance_per_vehicle', 0) for tc_data in tc_analysis.values()]
        
        if tc_distances and max(tc_distances) / min(tc_distances) > 2:
            recommendations.append("🏢 TC 간 부하 불균형: 차량 재배치 또는 영역 조정 권장")
        
        if not recommendations:
            recommendations.append("✅ 현재 최적화 상태 양호: 정기적인 모니터링 유지")
        
        return recommendations
    
    def print_analysis_report(self):
        """분석 결과 리포트 출력"""
        if not self.analysis_results:
            self.analyze_route_optimization()
        
        print("\n" + "="*60)
        print("📊 경로 최적화 종합 분석 리포트")
        print("="*60)
        
        # 기본 통계
        basic_stats = self.analysis_results.get('basic_stats', {})
        print(f"\n📈 기본 통계:")
        print(f"   총 차량: {basic_stats.get('total_vehicles', 0)}대")
        print(f"   총 배송지: {basic_stats.get('total_points', 0)}개")
        print(f"   총 거리: {basic_stats.get('total_distance', 0):.1f}km")
        print(f"   총 시간: {basic_stats.get('total_time', 0):.0f}분")
        print(f"   차량당 평균 거리: {basic_stats.get('avg_distance_per_vehicle', 0):.1f}km")
        print(f"   차량당 평균 배송지: {basic_stats.get('avg_points_per_vehicle', 0):.1f}개")
        
        # 중복 분석
        overlap_analysis = self.analysis_results.get('overlap_analysis', {})
        print(f"\n🔄 경로 중복 분석:")
        print(f"   총 중복 건수: {overlap_analysis.get('total_overlaps', 0)}건")
        print(f"   중복률: {overlap_analysis.get('overlap_rate', 0):.1f}%")
        
        # TC별 중복 상세
        tc_overlap_details = overlap_analysis.get('tc_overlap_details', {})
        for tc_id, details in tc_overlap_details.items():
            if details['overlaps'] > 0:
                print(f"   {tc_id}: {details['overlaps']}건 중복")
        
        # 효율성 분석
        efficiency_analysis = self.analysis_results.get('efficiency_analysis', {})
        distance_stats = efficiency_analysis.get('distance_stats', {})
        print(f"\n⚡ 효율성 분석:")
        print(f"   거리 편차: {distance_stats.get('std', 0):.1f}km")
        print(f"   최대 거리: {distance_stats.get('max', 0):.1f}km")
        print(f"   최소 거리: {distance_stats.get('min', 0):.1f}km")
        
        # 권장사항
        recommendations = self.analysis_results.get('recommendations', [])
        print(f"\n💡 개선 권장사항:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print("="*60)


def analyze_route_optimization(data: Dict = None, data_file: str = None) -> Dict:
    """
    경로 최적화 종합 분석 (외부 호출용 함수)
    
    Args:
        data: 직접 전달된 데이터 딕셔너리
        data_file: 데이터 파일 경로
    
    Returns:
        분석 결과 딕셔너리
    """
    analyzer = RouteAnalyzer(data, data_file)
    results = analyzer.analyze_route_optimization()
    analyzer.print_analysis_report()
    return results


if __name__ == "__main__":
    # 독립 실행 시 기본 분석 수행
    analyzer = RouteAnalyzer()
    analyzer.analyze_route_optimization()
    analyzer.print_analysis_report() 