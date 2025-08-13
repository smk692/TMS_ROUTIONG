"""
배차 최적화 알고리즘 기본 인터페이스
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
import logging

from ..models import Order, Vehicle, VehicleAssignment
from ..utils.time_calculator import get_time_calculator


@dataclass
class AlgorithmResult:
    """알고리즘 실행 결과"""
    assignments: List[VehicleAssignment]
    unassigned_orders: List[str]
    execution_time_seconds: float
    quality_score: float  # 0.0-1.0
    algorithm_name: str
    iteration_count: int = 0
    convergence_info: Optional[Dict] = None


@dataclass
class AlgorithmConfig:
    """알고리즘 설정"""
    time_limit_seconds: int = 300
    quality_threshold: float = 0.8
    max_iterations: int = 1000
    early_stopping_enabled: bool = True
    verbose: bool = False


class BaseAlgorithm(ABC):
    """배차 최적화 알고리즘 기본 클래스"""
    
    def __init__(self, config: AlgorithmConfig = None):
        self.config = config or AlgorithmConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._start_time = None
        self._best_solution = None
        self._iteration_count = 0
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """알고리즘 이름 반환"""
        pass
    
    @abstractmethod
    def _solve_implementation(self, orders: List[Order], vehicles: List[Vehicle],
                            vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """알고리즘 구체적 구현 - 상속 클래스에서 구현"""
        pass
    
    def solve(self, orders: List[Order], vehicles: List[Vehicle],
             vehicle_capacities: Dict[str, int]) -> AlgorithmResult:
        """배차 최적화 실행 (공통 처리 로직 포함)"""
        
        self.logger.info(f"{self.get_algorithm_name()} 실행 시작 - 주문: {len(orders)}개, 차량: {len(vehicles)}대")
        
        # 입력 유효성 검증
        self._validate_inputs(orders, vehicles, vehicle_capacities)
        
        # 시간 측정 시작
        self._start_time = time.time()
        
        try:
            # 알고리즘 실행
            result = self._solve_implementation(orders, vehicles, vehicle_capacities)
            
            # 결과 후처리
            result = self._post_process_result(result, orders, vehicles)
            
            self.logger.info(f"{self.get_algorithm_name()} 완료 - "
                           f"실행시간: {result.execution_time_seconds:.1f}초, "
                           f"품질점수: {result.quality_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"{self.get_algorithm_name()} 실행 오류: {str(e)}")
            raise
    
    def _validate_inputs(self, orders: List[Order], vehicles: List[Vehicle],
                        vehicle_capacities: Dict[str, int]):
        """입력 데이터 유효성 검증"""
        if not orders:
            raise ValueError("주문 목록이 비어있습니다")
        
        if not vehicles:
            raise ValueError("차량 목록이 비어있습니다")
        
        if not vehicle_capacities:
            raise ValueError("차량 용량 정보가 비어있습니다")
        
        # 차량 ID와 용량 정보 일치 확인
        vehicle_ids = {v.id for v in vehicles}
        capacity_ids = set(vehicle_capacities.keys())
        
        if not vehicle_ids.issubset(capacity_ids):
            missing_ids = vehicle_ids - capacity_ids
            raise ValueError(f"용량 정보가 없는 차량: {missing_ids}")
    
    def _post_process_result(self, result: AlgorithmResult, orders: List[Order],
                           vehicles: List[Vehicle]) -> AlgorithmResult:
        """결과 후처리 및 품질 계산"""
        
        # 실행 시간 계산
        if self._start_time:
            result.execution_time_seconds = time.time() - self._start_time
        
        # 품질 점수 계산
        result.quality_score = self._calculate_quality_score(result, orders, vehicles)
        
        # 알고리즘 이름 설정
        result.algorithm_name = self.get_algorithm_name()
        
        return result
    
    def _calculate_quality_score(self, result: AlgorithmResult, orders: List[Order],
                               vehicles: List[Vehicle]) -> float:
        """품질 점수 계산 (0.0-1.0)"""
        
        if not result.assignments:
            return 0.0
        
        total_orders = len(orders)
        assigned_orders = sum(len(assignment.assigned_orders) for assignment in result.assignments)
        
        # 기본 점수: 배정률
        assignment_rate = assigned_orders / total_orders if total_orders > 0 else 0
        
        # 거리 효율성 점수
        distance_efficiency = self._calculate_distance_efficiency(result.assignments)
        
        # 용량 활용도 점수
        capacity_utilization = self._calculate_capacity_utilization(result.assignments)
        
        # 시간 효율성 점수
        time_efficiency = self._calculate_time_efficiency(result.assignments, orders, vehicles)
        
        # 가중 평균으로 최종 점수 계산
        quality_score = (
            assignment_rate * 0.4 +           # 배정률 40%
            distance_efficiency * 0.25 +      # 거리 효율성 25%
            capacity_utilization * 0.15 +     # 용량 활용도 15%
            time_efficiency * 0.2             # 시간 효율성 20%
        )
        
        return min(1.0, max(0.0, quality_score))
    
    def _calculate_distance_efficiency(self, assignments: List[VehicleAssignment]) -> float:
        """거리 효율성 계산"""
        if not assignments:
            return 0.0
        
        total_efficiency = 0.0
        for assignment in assignments:
            order_count = len(assignment.assigned_orders)
            if order_count > 0 and assignment.estimated_distance_km > 0:
                # 주문 당 평균 거리 (낮을수록 좋음)
                avg_distance_per_order = assignment.estimated_distance_km / order_count
                # 효율성 = 1 / (평균 거리 + 1) -> 0~1 범위로 정규화
                efficiency = 1 / (avg_distance_per_order + 1)
                total_efficiency += efficiency
        
        return total_efficiency / len(assignments) if assignments else 0.0
    
    def _calculate_capacity_utilization(self, assignments: List[VehicleAssignment]) -> float:
        """용량 활용도 계산"""
        if not assignments:
            return 0.0
        
        total_utilization = sum(assignment.capacity_utilization for assignment in assignments)
        return total_utilization / len(assignments)
    
    def _calculate_time_efficiency(self, assignments: List[VehicleAssignment], 
                                 orders: List[Order], vehicles: List[Vehicle]) -> float:
        """시간 효율성 계산"""
        if not assignments:
            return 0.0
        
        time_calculator = get_time_calculator()
        total_efficiency = 0.0
        efficiency_count = 0
        
        # 차량별 주문 그룹화
        vehicle_dict = {v.id: v for v in vehicles}
        order_dict = {o.id: o for o in orders}
        
        for assignment in assignments:
            if assignment.vehicle_id not in vehicle_dict:
                continue
            
            vehicle = vehicle_dict[assignment.vehicle_id]
            assigned_order_objects = []
            
            # 배정된 주문 ID를 Order 객체로 변환
            for order_id in assignment.assigned_orders:
                if order_id in order_dict:
                    assigned_order_objects.append(order_dict[order_id])
            
            if not assigned_order_objects:
                continue
            
            # 실제 최적 시간 vs 예상 시간 비교
            optimal_time = time_calculator.estimate_optimal_time_for_orders(assigned_order_objects)
            estimated_time = assignment.estimated_time_minutes
            
            if optimal_time > 0:
                efficiency = time_calculator.calculate_time_efficiency(estimated_time, optimal_time)
                total_efficiency += efficiency
                efficiency_count += 1
        
        return total_efficiency / efficiency_count if efficiency_count > 0 else 0.0
    
    def _is_time_limit_exceeded(self) -> bool:
        """시간 제한 초과 확인"""
        if not self._start_time:
            return False
        
        elapsed = time.time() - self._start_time
        return elapsed >= self.config.time_limit_seconds
    
    def _should_stop_early(self, current_quality: float, iteration: int) -> bool:
        """조기 종료 조건 확인"""
        if not self.config.early_stopping_enabled:
            return False
        
        # 품질 임계값 달성
        if current_quality >= self.config.quality_threshold:
            self.logger.info(f"품질 임계값 달성으로 조기 종료: {current_quality:.3f}")
            return True
        
        # 시간 제한 초과
        if self._is_time_limit_exceeded():
            self.logger.info(f"시간 제한 초과로 조기 종료: {iteration}회 반복")
            return True
        
        return False
    
    def _create_empty_result(self) -> AlgorithmResult:
        """빈 결과 객체 생성"""
        return AlgorithmResult(
            assignments=[],
            unassigned_orders=[],
            execution_time_seconds=0.0,
            quality_score=0.0,
            algorithm_name=self.get_algorithm_name(),
            iteration_count=0
        )
    
    def _log_progress(self, iteration: int, quality: float, message: str = ""):
        """진행 상황 로깅"""
        if self.config.verbose and iteration % 10 == 0:  # 10회마다 로그
            elapsed = time.time() - self._start_time if self._start_time else 0
            self.logger.debug(f"반복 {iteration}: 품질 {quality:.3f}, 경과시간 {elapsed:.1f}초 {message}")


class AlgorithmError(Exception):
    """알고리즘 실행 관련 예외"""
    pass