#!/usr/bin/env python3
"""
Voronoi 최적화 분석기

Voronoi 기반 최적화와 기존 알고리즘 간의 성능 비교 분석
src/analysis 모듈로 통합됨
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

class VoronoiAnalyzer:
    """Voronoi 최적화 효과 분석기"""
    
    def __init__(self, original_data: Dict = None, optimized_data: Dict = None, 
                 original_file: str = None, optimized_file: str = None):
        """
        초기화
        Args:
            original_data: 원본 데이터 딕셔너리
            optimized_data: 최적화된 데이터 딕셔너리
            original_file: 원본 데이터 파일 경로
            optimized_file: 최적화된 데이터 파일 경로
        """
        # 데이터 로드
        if original_data is not None and optimized_data is not None:
            self.original_data = original_data
            self.optimized_data = optimized_data
        else:
            # 기본 파일 경로 설정
            if original_file is None:
                original_file = "../../data/extracted_coordinates.json"
            if optimized_file is None:
                optimized_file = "../../data/voronoi_optimized_coordinates.json"
            
            self.original_data = self._load_data(original_file)
            self.optimized_data = self._load_data(optimized_file)
        
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
    
    def analyze_optimization_effects(self) -> Dict:
        """최적화 효과 종합 분석"""
        if not self.original_data or not self.optimized_data:
            print("❌ 분석할 데이터가 없습니다.")
            return {}
        
        print("🔍 Voronoi 최적화 효과 분석 시작...")
        
        # 기본 통계 비교
        original_stats = self.original_data.get('stats', {})
        optimized_stats = self.optimized_data.get('stats', {})
        
        # 거리 분석
        distance_analysis = self._analyze_distance_improvement(original_stats, optimized_stats)
        
        # 경로 중복 분석
        overlap_analysis = self._analyze_route_overlaps()
        
        # 차량 효율성 분석
        efficiency_analysis = self._analyze_vehicle_efficiency()
        
        # 지역별 분석
        regional_analysis = self._analyze_regional_distribution()
        
        # 결과 통합
        self.analysis_results = {
            'distance_analysis': distance_analysis,
            'overlap_analysis': overlap_analysis,
            'efficiency_analysis': efficiency_analysis,
            'regional_analysis': regional_analysis,
            'summary': self._generate_summary()
        }
        
        return self.analysis_results
    
    def _analyze_distance_improvement(self, original_stats: Dict, optimized_stats: Dict) -> Dict:
        """거리 개선 효과 분석"""
        original_distance = original_stats.get('total_distance', 0)
        optimized_distance = optimized_stats.get('total_distance', 0)
        
        if original_distance == 0:
            return {'error': '원본 데이터에 거리 정보가 없습니다.'}
        
        distance_change = optimized_distance - original_distance
        distance_change_percent = (distance_change / original_distance) * 100
        
        return {
            'original_distance': original_distance,
            'optimized_distance': optimized_distance,
            'distance_change': distance_change,
            'distance_change_percent': distance_change_percent,
            'improvement': distance_change < 0
        }
    
    def _analyze_route_overlaps(self) -> Dict:
        """경로 중복 분석"""
        # 원본 데이터 중복 계산
        original_overlaps = self._calculate_route_overlaps(self.original_data)
        
        # 최적화된 데이터 중복 계산
        optimized_overlaps = self._calculate_route_overlaps(self.optimized_data)
        
        overlap_reduction = original_overlaps - optimized_overlaps
        overlap_reduction_percent = (overlap_reduction / max(original_overlaps, 1)) * 100
        
        return {
            'original_overlaps': original_overlaps,
            'optimized_overlaps': optimized_overlaps,
            'overlap_reduction': overlap_reduction,
            'overlap_reduction_percent': overlap_reduction_percent
        }
    
    def _calculate_route_overlaps(self, data: Dict) -> int:
        """경로 중복 계산"""
        overlaps = 0
        routes = data.get('routes', {})
        
        # 모든 차량 쌍에 대해 중복 검사
        vehicle_routes = list(routes.values())
        for i in range(len(vehicle_routes)):
            for j in range(i + 1, len(vehicle_routes)):
                route1_points = set()
                route2_points = set()
                
                # 경로 1의 배송지 수집
                for point in vehicle_routes[i].get('route', []):
                    if point.get('type') == 'delivery':
                        route1_points.add((point.get('lat'), point.get('lng')))
                
                # 경로 2의 배송지 수집
                for point in vehicle_routes[j].get('route', []):
                    if point.get('type') == 'delivery':
                        route2_points.add((point.get('lat'), point.get('lng')))
                
                # 중복 배송지 계산
                overlaps += len(route1_points.intersection(route2_points))
        
        return overlaps
    
    def _analyze_vehicle_efficiency(self) -> Dict:
        """차량 효율성 분석"""
        original_vehicles = len(self.original_data.get('routes', {}))
        optimized_vehicles = len(self.optimized_data.get('routes', {}))
        
        original_distance = self.original_data.get('stats', {}).get('total_distance', 0)
        optimized_distance = self.optimized_data.get('stats', {}).get('total_distance', 0)
        
        original_avg_distance = original_distance / max(original_vehicles, 1)
        optimized_avg_distance = optimized_distance / max(optimized_vehicles, 1)
        
        efficiency_improvement = ((original_avg_distance - optimized_avg_distance) / max(original_avg_distance, 1)) * 100
        
        return {
            'original_vehicles': original_vehicles,
            'optimized_vehicles': optimized_vehicles,
            'original_avg_distance': original_avg_distance,
            'optimized_avg_distance': optimized_avg_distance,
            'efficiency_improvement_percent': efficiency_improvement
        }
    
    def _analyze_regional_distribution(self) -> Dict:
        """지역별 분포 분석"""
        # TC별 분석
        original_tcs = self._get_tc_distribution(self.original_data)
        optimized_tcs = self._get_tc_distribution(self.optimized_data)
        
        return {
            'original_tc_distribution': original_tcs,
            'optimized_tc_distribution': optimized_tcs,
            'tc_balance_improvement': self._calculate_balance_improvement(original_tcs, optimized_tcs)
        }
    
    def _get_tc_distribution(self, data: Dict) -> Dict:
        """TC별 분포 계산"""
        tc_distribution = defaultdict(lambda: {'vehicles': 0, 'points': 0, 'distance': 0})
        
        for vehicle_id, route_data in data.get('routes', {}).items():
            tc_id = route_data.get('tc_id', 'unknown')
            tc_distribution[tc_id]['vehicles'] += 1
            tc_distribution[tc_id]['points'] += len([p for p in route_data.get('route', []) if p.get('type') == 'delivery'])
            tc_distribution[tc_id]['distance'] += route_data.get('total_distance', 0)
        
        return dict(tc_distribution)
    
    def _calculate_balance_improvement(self, original: Dict, optimized: Dict) -> float:
        """균형 개선도 계산"""
        # 표준편차를 이용한 균형도 측정
        original_distances = [tc_data['distance'] for tc_data in original.values()]
        optimized_distances = [tc_data['distance'] for tc_data in optimized.values()]
        
        if not original_distances or not optimized_distances:
            return 0.0
        
        original_std = np.std(original_distances)
        optimized_std = np.std(optimized_distances)
        
        if original_std == 0:
            return 0.0
        
        return ((original_std - optimized_std) / original_std) * 100
    
    def _generate_summary(self) -> Dict:
        """분석 결과 요약 생성"""
        distance_analysis = self.analysis_results.get('distance_analysis', {})
        overlap_analysis = self.analysis_results.get('overlap_analysis', {})
        efficiency_analysis = self.analysis_results.get('efficiency_analysis', {})
        
        return {
            'total_distance_change_percent': distance_analysis.get('distance_change_percent', 0),
            'overlap_reduction_percent': overlap_analysis.get('overlap_reduction_percent', 0),
            'efficiency_improvement_percent': efficiency_analysis.get('efficiency_improvement_percent', 0),
            'overall_improvement': self._calculate_overall_improvement()
        }
    
    def _calculate_overall_improvement(self) -> str:
        """전체적인 개선도 평가"""
        distance_analysis = self.analysis_results.get('distance_analysis', {})
        overlap_analysis = self.analysis_results.get('overlap_analysis', {})
        
        distance_improved = distance_analysis.get('distance_change_percent', 0) < 0
        overlap_reduced = overlap_analysis.get('overlap_reduction_percent', 0) > 50
        
        if distance_improved and overlap_reduced:
            return "우수"
        elif overlap_reduced:
            return "양호"
        else:
            return "보통"
    
    def print_analysis_report(self):
        """분석 결과 리포트 출력"""
        if not self.analysis_results:
            self.analyze_optimization_effects()
        
        print("\n" + "="*60)
        print("🔍 VORONOI 최적화 효과 분석 리포트")
        print("="*60)
        
        # 거리 분석
        distance_analysis = self.analysis_results.get('distance_analysis', {})
        print(f"\n📏 거리 분석:")
        print(f"   원본: {distance_analysis.get('original_distance', 0):.1f}km")
        print(f"   최적화: {distance_analysis.get('optimized_distance', 0):.1f}km")
        print(f"   변화: {distance_analysis.get('distance_change_percent', 0):+.1f}%")
        
        # 중복 분석
        overlap_analysis = self.analysis_results.get('overlap_analysis', {})
        print(f"\n🔄 경로 중복 분석:")
        print(f"   원본 중복: {overlap_analysis.get('original_overlaps', 0)}개")
        print(f"   최적화 중복: {overlap_analysis.get('optimized_overlaps', 0)}개")
        print(f"   중복 감소: {overlap_analysis.get('overlap_reduction_percent', 0):.1f}%")
        
        # 효율성 분석
        efficiency_analysis = self.analysis_results.get('efficiency_analysis', {})
        print(f"\n⚡ 차량 효율성 분석:")
        print(f"   원본 평균 거리: {efficiency_analysis.get('original_avg_distance', 0):.1f}km/차량")
        print(f"   최적화 평균 거리: {efficiency_analysis.get('optimized_avg_distance', 0):.1f}km/차량")
        print(f"   효율성 개선: {efficiency_analysis.get('efficiency_improvement_percent', 0):+.1f}%")
        
        # 종합 평가
        summary = self.analysis_results.get('summary', {})
        print(f"\n🎯 종합 평가: {summary.get('overall_improvement', '보통')}")
        print("="*60)


def analyze_voronoi_optimization(original_data: Dict = None, optimized_data: Dict = None,
                                original_file: str = None, optimized_file: str = None) -> Dict:
    """
    Voronoi 최적화 효과 분석 (외부 호출용 함수)
    
    Args:
        original_data: 원본 데이터 딕셔너리
        optimized_data: 최적화된 데이터 딕셔너리
        original_file: 원본 데이터 파일 경로
        optimized_file: 최적화된 데이터 파일 경로
    
    Returns:
        분석 결과 딕셔너리
    """
    analyzer = VoronoiAnalyzer(original_data, optimized_data, original_file, optimized_file)
    results = analyzer.analyze_optimization_effects()
    analyzer.print_analysis_report()
    return results


if __name__ == "__main__":
    # 독립 실행 시 기본 분석 수행
    analyzer = VoronoiAnalyzer()
    analyzer.analyze_optimization_effects()
    analyzer.print_analysis_report() 