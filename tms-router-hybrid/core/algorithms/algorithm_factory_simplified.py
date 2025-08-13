"""
간소화된 알고리즘 팩토리 - OR-Tools VRP 전용
"""
from typing import Dict, List, Optional
import logging

from ..models import Order, Vehicle, Region
from .base_algorithm import BaseAlgorithm
from .ortools_vrp_algorithm import ORToolsVRPAlgorithm, ORToolsVRPConfig


class SimplifiedAlgorithmFactory:
    """OR-Tools VRP 전용 간소화된 알고리즘 팩토리"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def create_algorithm(self, orders: List[Order], vehicles: List[Vehicle],
                        regions: List[Region], conditions: Dict = None) -> BaseAlgorithm:
        """OR-Tools VRP 알고리즘 생성 (적응형 설정)"""
        
        conditions = conditions or {}
        order_count = len(orders)
        vehicle_count = len(vehicles)
        
        self.logger.info(f"OR-Tools VRP 알고리즘 생성 - 주문: {order_count}개, 차량: {vehicle_count}대")
        
        # 적응형 설정 생성
        config = self._create_adaptive_config(order_count, vehicle_count, conditions)
        
        # OR-Tools VRP 알고리즘 생성
        algorithm = ORToolsVRPAlgorithm(config)
        
        return algorithm
    
    def _create_adaptive_config(self, order_count: int, vehicle_count: int, 
                               conditions: Dict) -> ORToolsVRPConfig:
        """주문 규모에 따른 적응형 설정 생성 & 최단 거리 구하기 haversine"""
        
        # 차량당 주문 수 계산
        orders_per_vehicle = order_count / max(vehicle_count, 1)
        
        # 규모별 설정
        if order_count <= 50:
            # 소규모: 빠른 처리
            config = ORToolsVRPConfig(
                max_solve_time_seconds=60,
                use_clustering=False,
                unassigned_penalty=100000,
                distance_weight=1.0,
                vehicle_fixed_cost=5000,
                distance_api={'api_priority': ['haversine']}
            )
            self.logger.info("소규모 설정 적용 (≤50개 주문)")
            
        elif order_count <= 100:
            # 중규모: 균형 잡힌 처리
            config = ORToolsVRPConfig(
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
            self.logger.info("중규모 설정 적용 (51-100개 주문)")
            
        elif order_count <= 200:
            # 대규모: 품질 중심
            config = ORToolsVRPConfig(
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
            self.logger.info("대규모 설정 적용 (101-200개 주문)")
            
        else:
            # 초대규모: 처리량 중심
            config = ORToolsVRPConfig(
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
            self.logger.info("초대규모 설정 적용 (200개+ 주문)")
        
        # 차량 부족 시 추가 조정
        if orders_per_vehicle > 25:
            config.unassigned_penalty = min(config.unassigned_penalty, 20000)
            config.vehicle_fixed_cost = min(config.vehicle_fixed_cost, 1500)
            self.logger.warning(f"차량 부족 감지 (차량당 {orders_per_vehicle:.1f}개) - 페널티 자동 조정")
        
        # 시간 제한 조건 반영
        time_limit = conditions.get('time_limit_seconds')
        if time_limit and time_limit < config.max_solve_time_seconds:
            config.max_solve_time_seconds = max(30, time_limit - 30)  # 30초 여유
            self.logger.info(f"시간 제한으로 VRP 솔버 시간 조정: {config.max_solve_time_seconds}초")
        
        return config