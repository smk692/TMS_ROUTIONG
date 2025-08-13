"""
OR-Tools VRP 기반 최적화 알고리즘
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .base_algorithm import BaseAlgorithm, AlgorithmResult, AlgorithmConfig
from .vrp_solver import VRPSolver
from .adapters.result_adapter import ResultAdapter
from ..models import Order, Vehicle, VehicleAssignment


@dataclass
class ORToolsVRPConfig(AlgorithmConfig):
    """OR-Tools VRP 알고리즘 설정"""
    
    # VRP 솔버 설정
    max_solve_time_seconds: int = 180  # 3분으로 증가 (대용량 처리 대응)
    use_clustering: bool = True
    
    # HDBSCAN 클러스터링 설정
    min_cluster_size: int = 8
    max_cluster_size: int = 35
    epsilon: float = 0.005  # ~500m
    
    # 제약조건 설정
    max_work_hours: int = 8
    max_distance_km: int = 120
    break_interval_hours: int = 4
    break_duration_minutes: int = 15
    
    # 목적함수 가중치
    unassigned_penalty: int = 100000
    distance_weight: float = 1.0
    vehicle_fixed_cost: int = 5000
    time_balance_penalty: int = 50
    
    # 거리 계산 API 설정
    distance_api: Dict = None
    
    def __post_init__(self):
        if self.distance_api is None:
            self.distance_api = {
                'api_priority': ['haversine'],  # 빠른 처리를 위해 Haversine만 사용
                'distance_cache_ttl': 24 * 3600,
                'max_locations_per_request': 200,
                'request_delay': 0.1
            }


class ORToolsVRPAlgorithm(BaseAlgorithm):
    """OR-Tools VRP 기반 최적화 알고리즘"""
    
    def __init__(self, config: ORToolsVRPConfig = None):
        super().__init__(config or ORToolsVRPConfig())
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        vrp_config = {
            'max_solve_time_seconds': self.config.max_solve_time_seconds,
            'use_clustering': self.config.use_clustering,
            'clustering': {
                'min_cluster_size': self.config.min_cluster_size,
                'max_cluster_size': self.config.max_cluster_size,
                'epsilon': self.config.epsilon,
            },
            'max_work_hours': self.config.max_work_hours,
            'max_distance_km': self.config.max_distance_km,
            'break_interval_hours': self.config.break_interval_hours,
            'break_duration_minutes': self.config.break_duration_minutes,
            'unassigned_penalty': self.config.unassigned_penalty,
            'distance_weight': self.config.distance_weight,
            'vehicle_fixed_cost': self.config.vehicle_fixed_cost,
            'time_balance_penalty': self.config.time_balance_penalty,
            'distance_api': self.config.distance_api,
        }
        
        self.vrp_solver = VRPSolver(vrp_config)
        self.result_adapter = ResultAdapter()
        
    @property
    def algorithm_name(self) -> str:
        return "OR-Tools VRP"
    
    @property
    def algorithm_type(self) -> str:
        return "ortools_vrp"
    
    @property
    def complexity_score(self) -> float:
        """알고리즘 복잡도 점수 (0.0-1.0)"""
        return 0.95  # 매우 고복잡도
    
    @property
    def recommended_order_range(self) -> tuple:
        """권장 주문 수 범위"""
        return (20, 1000)  # 20개 이상에서 효과적
    
    async def optimize_async(self, orders: List[Order], vehicles: List[Vehicle], 
                           regions: List = None, conditions: Dict = None) -> AlgorithmResult:
        """비동기 최적화 실행"""
        
        start_time = time.time()
        self.logger.info(f"OR-Tools VRP 최적화 시작: {len(orders)}개 주문, {len(vehicles)}대 차량")
        
        try:
            # 1. 입력 데이터 검증
            if not orders:
                raise ValueError("최적화할 주문이 없습니다")
            if not vehicles:
                raise ValueError("사용 가능한 차량이 없습니다")
            
            # 2. VRP 솔빙
            vrp_solution = await self.vrp_solver.solve(orders, vehicles, regions, conditions)
            
            # 3. 결과 변환
            assignments = self.result_adapter.convert_vrp_solution_to_assignments(
                vrp_solution, vehicles
            )
            
            # 4. 결과 요약
            summary = self.result_adapter.generate_assignment_summary(assignments, vrp_solution)
            
            # 5. 검증
            warnings = self.result_adapter.validate_assignments(assignments, vehicles)
            
            execution_time = time.time() - start_time
            
            # 6. 품질 점수 계산
            quality_score = self._calculate_quality_score(summary, len(orders), execution_time)
            
            # 미배정 주문 ID 리스트 생성
            assigned_order_ids = []
            for assignment in assignments:
                assigned_order_ids.extend(assignment.assigned_orders)
            
            unassigned_order_ids = [order.id for order in orders if order.id not in assigned_order_ids]
            
            result = AlgorithmResult(
                assignments=assignments,
                unassigned_orders=unassigned_order_ids,
                execution_time_seconds=execution_time,
                quality_score=quality_score,
                algorithm_name=self.algorithm_name,
                iteration_count=0,
                convergence_info={
                    'vrp_objective_value': summary.get('vrp_objective_value', 0),
                    'vrp_is_optimal': summary.get('vrp_is_optimal', False),
                    'average_capacity_utilization': summary.get('average_capacity_utilization', 0.0),
                    'assignment_rate': summary.get('assignment_rate', 0.0),
                    'warnings': warnings,
                    'clustering_used': self.config.use_clustering,
                    'solve_time_limit': self.config.max_solve_time_seconds,
                    'total_orders': len(orders),
                    'assigned_orders': summary['total_orders'],
                    'total_vehicles': len(vehicles),
                    'used_vehicles': summary['total_vehicles'],
                    'total_distance': summary['total_distance'],
                    'total_time': summary['total_time']
                }
            )
            
            self.logger.info(f"OR-Tools VRP 최적화 완료: "
                           f"{len(assignments)}대 배차, "
                           f"{summary['total_orders']}개 배정, "
                           f"{summary['unassigned_orders']}개 미배정, "
                           f"품질점수 {quality_score:.3f}, "
                           f"{execution_time:.1f}초")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"OR-Tools VRP 최적화 오류: {str(e)}")
            
            # 오류 시 빈 결과 반환
            return AlgorithmResult(
                assignments=[],
                unassigned_orders=[order.id for order in orders],
                execution_time_seconds=execution_time,
                quality_score=0.0,
                algorithm_name=self.algorithm_name,
                iteration_count=0,
                convergence_info={'error': True, 'error_message': str(e)}
            )
    
    def optimize(self, orders: List[Order], vehicles: List[Vehicle], 
                regions: List = None, conditions: Dict = None) -> AlgorithmResult:
        """동기 최적화 실행 (비동기 버전을 래핑)"""
        
        # 이벤트 루프 처리 - 새로운 루프에서 실행
        try:
            return asyncio.run(self.optimize_async(orders, vehicles, regions, conditions))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                # 이미 실행 중인 루프가 있으면 새 스레드에서 실행
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self.optimize_async(orders, vehicles, regions, conditions)
                    )
                    return future.result()
            else:
                raise
    
    def _calculate_quality_score(self, summary: Dict, total_orders: int, execution_time: float) -> float:
        """품질 점수 계산"""
        
        try:
            # 기본 점수 요소들
            assignment_rate = summary.get('assignment_rate', 0.0)
            avg_capacity_util = summary.get('average_capacity_utilization', 0.0)
            is_optimal = summary.get('vrp_is_optimal', False)
            
            # 1. 배정률 점수 (40%)
            assignment_score = assignment_rate * 0.4
            
            # 2. 용량 활용도 점수 (30%) - 70-90%가 이상적
            if 0.7 <= avg_capacity_util <= 0.9:
                capacity_score = 0.3
            elif avg_capacity_util > 0.9:
                capacity_score = 0.3 * (1.0 - (avg_capacity_util - 0.9) * 2)  # 초과시 감점
            else:
                capacity_score = 0.3 * (avg_capacity_util / 0.7)  # 부족시 비례 감점
            
            # 3. 최적성 점수 (20%)
            optimality_score = 0.2 if is_optimal else 0.15
            
            # 4. 처리 시간 점수 (10%)
            time_score = 0.1
            if execution_time > 180:  # 3분 초과시 감점
                time_score *= max(0.5, 180 / execution_time)
            
            quality_score = assignment_score + capacity_score + optimality_score + time_score
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            self.logger.warning(f"품질 점수 계산 오류: {str(e)}")
            return 0.5  # 기본값
    
    def can_handle(self, orders: List[Order], vehicles: List[Vehicle], 
                   conditions: Dict = None) -> bool:
        """알고리즘이 해당 조건을 처리할 수 있는지 확인"""
        
        # 기본 조건 확인
        if not orders or not vehicles:
            return False
        
        # 주문 수 범위 확인
        min_orders, max_orders = self.recommended_order_range
        if len(orders) < min_orders or len(orders) > max_orders:
            return False
        
        # 차량-주문 비율 확인 (차량당 최소 3개 주문)
        if len(orders) < len(vehicles) * 3:
            return False
        
        return True
    
    def estimate_execution_time(self, orders: List[Order], vehicles: List[Vehicle]) -> float:
        """예상 실행 시간 추정 (초)"""
        
        # 기본 시간 + 주문 수와 차량 수에 따른 증가
        base_time = 10.0
        order_factor = len(orders) * 0.1
        vehicle_factor = len(vehicles) * 0.5
        clustering_overhead = 5.0 if self.config.use_clustering else 0.0
        
        estimated_time = base_time + order_factor + vehicle_factor + clustering_overhead
        
        # 최대 솔빙 시간으로 제한
        return min(estimated_time, self.config.max_solve_time_seconds + 20)
    
    def get_configuration_summary(self) -> Dict:
        """알고리즘 설정 요약"""
        
        return {
            'algorithm_name': self.algorithm_name,
            'algorithm_type': self.algorithm_type,
            'complexity_score': self.complexity_score,
            'recommended_order_range': self.recommended_order_range,
            'max_solve_time_seconds': self.config.max_solve_time_seconds,
            'use_clustering': self.config.use_clustering,
            'clustering_config': {
                'min_cluster_size': self.config.min_cluster_size,
                'max_cluster_size': self.config.max_cluster_size,
                'epsilon': self.config.epsilon,
            },
            'constraints': {
                'max_work_hours': self.config.max_work_hours,
                'max_distance_km': self.config.max_distance_km,
                'break_interval_hours': self.config.break_interval_hours,
            },
            'objective_weights': {
                'unassigned_penalty': self.config.unassigned_penalty,
                'distance_weight': self.config.distance_weight,
                'vehicle_fixed_cost': self.config.vehicle_fixed_cost,
                'time_balance_penalty': self.config.time_balance_penalty,
            }
        }
    
    def get_algorithm_name(self) -> str:
        """알고리즘 이름 반환"""
        return self.algorithm_name
    
    def _solve_implementation(self, orders: List[Order], vehicles: List[Vehicle],
                            vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """알고리즘 구체적 구현 - BaseAlgorithm의 추상 메서드 구현"""
        
        # vehicle_capacities를 조건에 추가하여 optimize 메서드 호출
        conditions = {'vehicle_capacities': vehicle_capacities}
        
        return self.optimize(orders, vehicles, regions=[], conditions=conditions)