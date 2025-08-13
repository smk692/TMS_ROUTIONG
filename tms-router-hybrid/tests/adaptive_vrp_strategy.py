#!/usr/bin/env python3
"""
적응형 VRP 전략 - 주문 규모와 차량 수에 따라 최적의 설정 자동 선택
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from typing import Dict, Tuple
from dataclasses import dataclass
from core.algorithms.ortools_vrp_algorithm import ORToolsVRPConfig


@dataclass
class VRPScenario:
    """VRP 시나리오 정의"""
    name: str
    order_range: Tuple[int, int]
    min_vehicles: int
    config: ORToolsVRPConfig


class AdaptiveVRPStrategy:
    """적응형 VRP 전략"""
    
    def __init__(self):
        # 시나리오별 최적 설정 정의
        self.scenarios = [
            VRPScenario(
                name="소규모_최적화",
                order_range=(0, 50),
                min_vehicles=2,
                config=ORToolsVRPConfig(
                    max_solve_time_seconds=60,
                    use_clustering=False,
                    unassigned_penalty=100000,
                    distance_weight=1.0,
                    vehicle_fixed_cost=5000,
                    distance_api={'api_priority': ['haversine']}
                )
            ),
            VRPScenario(
                name="중규모_균형",
                order_range=(50, 100),
                min_vehicles=5,
                config=ORToolsVRPConfig(
                    max_solve_time_seconds=120,
                    use_clustering=True,
                    min_cluster_size=8,
                    max_cluster_size=35,
                    epsilon=0.005,
                    unassigned_penalty=100000,
                    distance_weight=1.0,
                    vehicle_fixed_cost=5000,
                    distance_api={'api_priority': ['haversine']}
                )
            ),
            VRPScenario(
                name="대규모_처리량",
                order_range=(100, 200),
                min_vehicles=10,
                config=ORToolsVRPConfig(
                    max_solve_time_seconds=180,
                    use_clustering=True,
                    min_cluster_size=15,
                    max_cluster_size=50,
                    epsilon=0.008,
                    unassigned_penalty=50000,
                    distance_weight=0.8,
                    vehicle_fixed_cost=3000,
                    distance_api={'api_priority': ['haversine']}
                )
            ),
            VRPScenario(
                name="초대규모_분할",
                order_range=(200, 1000),
                min_vehicles=20,
                config=ORToolsVRPConfig(
                    max_solve_time_seconds=240,
                    use_clustering=True,
                    min_cluster_size=25,
                    max_cluster_size=80,
                    epsilon=0.01,
                    unassigned_penalty=30000,
                    distance_weight=0.6,
                    vehicle_fixed_cost=2000,
                    distance_api={'api_priority': ['haversine']}
                )
            )
        ]
    
    def select_strategy(self, order_count: int, vehicle_count: int) -> Tuple[ORToolsVRPConfig, Dict]:
        """
        주문 수와 차량 수에 따라 최적 전략 선택
        
        Returns:
            (최적화 설정, 권장사항)
        """
        # 적합한 시나리오 찾기
        scenario = None
        for s in self.scenarios:
            if s.order_range[0] <= order_count < s.order_range[1]:
                scenario = s
                break
        
        if scenario is None:
            # 초대규모 기본값
            scenario = self.scenarios[-1]
        
        # 차량 수 검증
        recommendations = {}
        
        # 주문/차량 비율 계산
        if vehicle_count > 0:
            order_per_vehicle = order_count / vehicle_count
        else:
            order_per_vehicle = float('inf')
        
        # 권장사항 생성
        if vehicle_count < scenario.min_vehicles:
            recommendations['차량부족'] = (
                f"현재 {vehicle_count}대 → 권장 {scenario.min_vehicles}대 이상"
            )
        
        if order_per_vehicle > 30:
            recommendations['과부하'] = (
                f"차량당 {order_per_vehicle:.1f}개 주문 (권장: 20개 이하)"
            )
            recommendations['해결방안'] = "차량 추가 또는 주문 분할 처리 필요"
        
        # 대용량 특별 처리
        if order_count > 150:
            recommendations['대용량처리'] = (
                "권역별 또는 시간대별 분할 처리 권장"
            )
            
            # 동적으로 설정 조정
            if order_per_vehicle > 25:
                # 미배정 페널티 추가 감소
                scenario.config.unassigned_penalty = 20000
                scenario.config.vehicle_fixed_cost = 1500
                recommendations['설정조정'] = (
                    "차량 부족으로 페널티 자동 조정됨"
                )
        
        # 성능 예측
        if order_count < 100:
            expected_time = "1분 이내"
            expected_quality = "0.95 이상"
        elif order_count < 200:
            expected_time = "3-5분"
            expected_quality = "0.85 이상"
        else:
            expected_time = "10분 이상"
            expected_quality = "차량 수에 따라 변동"
        
        recommendations['예상성능'] = {
            '처리시간': expected_time,
            '품질점수': expected_quality,
            '시나리오': scenario.name
        }
        
        return scenario.config, recommendations
    
    def analyze_failure(self, order_count: int, vehicle_count: int, 
                        assigned_count: int, execution_time: float) -> Dict:
        """
        실패 원인 분석 및 개선안 제시
        """
        analysis = {}
        
        # 배정률 계산
        assignment_rate = assigned_count / order_count if order_count > 0 else 0
        
        if assignment_rate < 0.8:
            analysis['주요원인'] = "차량 용량 부족"
            
            # 필요 차량 수 계산
            required_vehicles = max(
                int(order_count / 20),  # 차량당 20개 기준
                int(vehicle_count * (1 / assignment_rate))  # 현재 배정률 기반
            )
            
            analysis['해결방안'] = {
                '즉시조치': f"차량을 {required_vehicles}대로 증가",
                '대안1': "주문을 2-3개 그룹으로 분할 처리",
                '대안2': "우선순위가 높은 주문만 선별 처리"
            }
        
        if execution_time > 600:  # 10분 초과
            analysis['성능문제'] = "처리 시간 과다"
            analysis['최적화방안'] = {
                '클러스터링': "epsilon을 0.01로 증가 (클러스터 크기 확대)",
                '솔버시간': "max_solve_time_seconds를 120초로 단축",
                '분할처리': f"{order_count // 2}개씩 2회 분할 처리"
            }
        
        return analysis


def main():
    """사용 예시"""
    strategy = AdaptiveVRPStrategy()
    
    # 테스트 시나리오들
    test_cases = [
        (50, 3, "소규모 센터"),
        (100, 5, "중규모 센터"),
        (200, 5, "대규모 센터 (차량 부족)"),
        (200, 15, "대규모 센터 (차량 충분)"),
        (500, 10, "초대규모 센터")
    ]
    
    print("=== 적응형 VRP 전략 분석 ===\n")
    
    for orders, vehicles, description in test_cases:
        print(f"📍 {description}: {orders}개 주문, {vehicles}대 차량")
        print("-" * 50)
        
        config, recommendations = strategy.select_strategy(orders, vehicles)
        
        print(f"선택된 전략: {recommendations['예상성능']['시나리오']}")
        print(f"VRP 솔버 시간: {config.max_solve_time_seconds}초")
        print(f"클러스터링: {'활성' if config.use_clustering else '비활성'}")
        
        if config.use_clustering:
            print(f"  - 최소 크기: {config.min_cluster_size}")
            print(f"  - 최대 크기: {config.max_cluster_size}")
        
        print(f"미배정 페널티: {config.unassigned_penalty:,}")
        print(f"차량 고정비용: {config.vehicle_fixed_cost:,}")
        
        print("\n📊 권장사항:")
        for key, value in recommendations.items():
            if key != '예상성능':
                print(f"  • {key}: {value}")
        
        print(f"\n⏱️ 예상 성능:")
        print(f"  • 처리 시간: {recommendations['예상성능']['처리시간']}")
        print(f"  • 품질 점수: {recommendations['예상성능']['품질점수']}")
        
        print("\n")
    
    # 실패 분석 예시
    print("=== 실패 원인 분석 예시 ===\n")
    
    failure_analysis = strategy.analyze_failure(
        order_count=200,
        vehicle_count=5,
        assigned_count=107,
        execution_time=1142
    )
    
    print("200개 주문, 5대 차량, 107개만 배정된 경우:")
    for key, value in failure_analysis.items():
        print(f"\n{key}:")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"  • {k}: {v}")
        else:
            print(f"  {value}")


if __name__ == "__main__":
    main()