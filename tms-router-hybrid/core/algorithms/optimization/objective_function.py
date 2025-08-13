"""
VRP 목적함수 정의 및 관리
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .vrp_model import VRPModel


class ObjectiveType(Enum):
    """목적함수 타입"""
    MINIMIZE_DISTANCE = "minimize_distance"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_VEHICLES = "minimize_vehicles"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_UTILIZATION = "maximize_utilization"
    BALANCE_WORKLOAD = "balance_workload"


@dataclass
class ObjectiveComponent:
    """목적함수 구성 요소"""
    objective_type: ObjectiveType
    weight: float
    description: str
    calculator: Optional[Callable] = None


@dataclass
class ObjectiveResult:
    """목적함수 결과"""
    total_value: float
    components: Dict[ObjectiveType, float]
    is_minimization: bool = True
    
    def get_component_value(self, objective_type: ObjectiveType) -> float:
        """특정 목적함수 구성요소 값 조회"""
        return self.components.get(objective_type, 0.0)
    
    def get_weighted_total(self, weights: Dict[ObjectiveType, float]) -> float:
        """가중치 적용된 총합 계산"""
        weighted_sum = 0.0
        
        for obj_type, value in self.components.items():
            weight = weights.get(obj_type, 1.0)
            weighted_sum += value * weight
        
        return weighted_sum


class BaseObjective(ABC):
    """목적함수 기본 클래스"""
    
    def __init__(self, objective_type: ObjectiveType, weight: float = 1.0):
        self.objective_type = objective_type
        self.weight = weight
        self.logger = logging.getLogger(f"{__name__}.{objective_type.value}")
    
    @abstractmethod
    def calculate(self, routes: List[List[int]], model: VRPModel) -> float:
        """목적함수 값 계산"""
        pass
    
    def get_description(self) -> str:
        """목적함수 설명"""
        return f"{self.objective_type.value} (weight: {self.weight})"


class MinimizeDistanceObjective(BaseObjective):
    """총 이동거리 최소화"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(ObjectiveType.MINIMIZE_DISTANCE, weight)
    
    def calculate(self, routes: List[List[int]], model: VRPModel) -> float:
        """총 이동거리 계산"""
        
        if model.distance_matrix is None:
            return 0.0
        
        total_distance = 0.0
        
        for route in routes:
            if len(route) < 2:
                continue
            
            # 경로의 총 거리 계산
            route_distance = 0.0
            for i in range(len(route) - 1):
                from_idx = route[i]
                to_idx = route[i + 1]
                route_distance += model.distance_matrix[from_idx][to_idx]
            
            total_distance += route_distance
        
        return total_distance


class MinimizeTimeObjective(BaseObjective):
    """총 소요시간 최소화"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(ObjectiveType.MINIMIZE_TIME, weight)
    
    def calculate(self, routes: List[List[int]], model: VRPModel) -> float:
        """총 소요시간 계산"""
        
        if model.time_matrix is None:
            return 0.0
        
        total_time = 0.0
        
        for route in routes:
            if len(route) < 2:
                continue
            
            # 경로의 총 시간 계산 (이동시간 + 서비스시간)
            route_time = 0.0
            for i in range(len(route) - 1):
                from_idx = route[i]
                to_idx = route[i + 1]
                
                # 이동시간
                route_time += model.time_matrix[from_idx][to_idx]
                
                # 서비스시간 (도착지가 고객인 경우)
                if to_idx not in model.get_depot_indices():
                    location = model.locations[to_idx]
                    route_time += location.service_time
            
            total_time += route_time
        
        return total_time


class MinimizeVehiclesObjective(BaseObjective):
    """사용 차량 수 최소화"""
    
    def __init__(self, weight: float = 5000.0):
        super().__init__(ObjectiveType.MINIMIZE_VEHICLES, weight)
    
    def calculate(self, routes: List[List[int]], model: VRPModel) -> float:
        """사용된 차량 수 계산"""
        
        # 비어있지 않은 경로의 개수
        used_vehicles = sum(1 for route in routes if len(route) > 2)  # depot -> ... -> depot
        
        return float(used_vehicles) * self.weight


class MinimizeCostObjective(BaseObjective):
    """총 비용 최소화"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(ObjectiveType.MINIMIZE_COST, weight)
    
    def calculate(self, routes: List[List[int]], model: VRPModel) -> float:
        """총 운송비용 계산"""
        
        if model.distance_matrix is None:
            return 0.0
        
        total_cost = 0.0
        
        for vehicle_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            
            # 해당 차량 정보 조회
            vehicle = model.vehicles[vehicle_idx] if vehicle_idx < len(model.vehicles) else model.vehicles[0]
            
            # 고정비용
            total_cost += vehicle.fixed_cost
            
            # 변동비용 (거리 기반)
            route_distance = 0.0
            for i in range(len(route) - 1):
                from_idx = route[i]
                to_idx = route[i + 1]
                route_distance += model.distance_matrix[from_idx][to_idx]
            
            total_cost += route_distance * vehicle.cost_per_km
        
        return total_cost


class MaximizeUtilizationObjective(BaseObjective):
    """차량 용량 활용률 최대화"""
    
    def __init__(self, weight: float = 1000.0):
        super().__init__(ObjectiveType.MAXIMIZE_UTILIZATION, weight)
    
    def calculate(self, routes: List[List[int]], model: VRPModel) -> float:
        """용량 활용률 계산 (최대화를 위해 음수 반환)"""
        
        total_utilization = 0.0
        used_vehicles = 0
        
        for vehicle_idx, route in enumerate(routes):
            if len(route) <= 2:  # 빈 경로
                continue
            
            used_vehicles += 1
            
            # 해당 차량 정보 조회
            vehicle = model.vehicles[vehicle_idx] if vehicle_idx < len(model.vehicles) else model.vehicles[0]
            
            # 경로의 총 수요량 계산
            route_demand = 0
            for location_idx in route:
                if location_idx not in model.get_depot_indices():  # depot이 아닌 경우
                    route_demand += model.locations[location_idx].demand
            
            # 활용률 계산
            if vehicle.capacity > 0:
                utilization = route_demand / vehicle.capacity
                total_utilization += utilization
        
        # 평균 활용률
        avg_utilization = total_utilization / used_vehicles if used_vehicles > 0 else 0.0
        
        # 최대화 목적이므로 음수 반환 (미활용률을 최소화)
        return (1.0 - avg_utilization) * self.weight


class BalanceWorkloadObjective(BaseObjective):
    """작업량 균형 최적화"""
    
    def __init__(self, weight: float = 500.0):
        super().__init__(ObjectiveType.BALANCE_WORKLOAD, weight)
    
    def calculate(self, routes: List[List[int]], model: VRPModel) -> float:
        """작업량 불균형 정도 계산"""
        
        if model.time_matrix is None:
            return 0.0
        
        workloads = []
        
        for route in routes:
            if len(route) <= 2:  # 빈 경로
                workloads.append(0.0)
                continue
            
            # 경로의 총 작업시간 계산
            route_time = 0.0
            for i in range(len(route) - 1):
                from_idx = route[i]
                to_idx = route[i + 1]
                
                # 이동시간
                route_time += model.time_matrix[from_idx][to_idx]
                
                # 서비스시간
                if to_idx not in model.get_depot_indices():
                    route_time += model.locations[to_idx].service_time
            
            workloads.append(route_time)
        
        if not workloads or len(workloads) <= 1:
            return 0.0
        
        # 작업량 표준편차 계산 (불균형 정도)
        workloads = [w for w in workloads if w > 0]  # 빈 경로 제외
        
        if len(workloads) <= 1:
            return 0.0
        
        mean_workload = sum(workloads) / len(workloads)
        variance = sum((w - mean_workload) ** 2 for w in workloads) / len(workloads)
        std_deviation = variance ** 0.5
        
        return std_deviation * self.weight


class ObjectiveFunction:
    """복합 목적함수 관리자"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 목적함수 구성요소들
        self.objectives: List[BaseObjective] = []
        
        # 기본 가중치
        self.default_weights = {
            ObjectiveType.MINIMIZE_DISTANCE: 1.0,
            ObjectiveType.MINIMIZE_TIME: 0.5,
            ObjectiveType.MINIMIZE_VEHICLES: 5000.0,
            ObjectiveType.MINIMIZE_COST: 1.0,
            ObjectiveType.MAXIMIZE_UTILIZATION: 1000.0,
            ObjectiveType.BALANCE_WORKLOAD: 500.0
        }
        
        # 목적함수 팩토리
        self._objective_factories = {
            ObjectiveType.MINIMIZE_DISTANCE: MinimizeDistanceObjective,
            ObjectiveType.MINIMIZE_TIME: MinimizeTimeObjective,
            ObjectiveType.MINIMIZE_VEHICLES: MinimizeVehiclesObjective,
            ObjectiveType.MINIMIZE_COST: MinimizeCostObjective,
            ObjectiveType.MAXIMIZE_UTILIZATION: MaximizeUtilizationObjective,
            ObjectiveType.BALANCE_WORKLOAD: BalanceWorkloadObjective
        }
    
    def add_objective(self, objective_type: ObjectiveType, weight: float = None) -> None:
        """목적함수 구성요소 추가"""
        
        if weight is None:
            weight = self.default_weights.get(objective_type, 1.0)
        
        objective_class = self._objective_factories.get(objective_type)
        
        if objective_class:
            objective = objective_class(weight)
            self.objectives.append(objective)
            self.logger.info(f"목적함수 추가: {objective.get_description()}")
        else:
            self.logger.warning(f"지원하지 않는 목적함수 타입: {objective_type}")
    
    def add_default_objectives(self) -> None:
        """기본 목적함수들 추가"""
        
        # 거리 최소화 (주 목적)
        self.add_objective(ObjectiveType.MINIMIZE_DISTANCE, 1.0)
        
        # 차량 수 최소화 (고정비용)
        self.add_objective(ObjectiveType.MINIMIZE_VEHICLES, 5000.0)
        
        # 용량 활용률 최대화
        self.add_objective(ObjectiveType.MAXIMIZE_UTILIZATION, 1000.0)
        
        self.logger.info(f"기본 목적함수 {len(self.objectives)}개 추가 완료")
    
    def calculate_total_objective(self, routes: List[List[int]], model: VRPModel) -> ObjectiveResult:
        """전체 목적함수 값 계산"""
        
        if not self.objectives:
            self.add_default_objectives()
        
        components = {}
        total_value = 0.0
        
        for objective in self.objectives:
            component_value = objective.calculate(routes, model)
            components[objective.objective_type] = component_value
            total_value += component_value
            
            self.logger.debug(f"{objective.objective_type.value}: {component_value:.2f}")
        
        result = ObjectiveResult(
            total_value=total_value,
            components=components,
            is_minimization=True
        )
        
        return result
    
    def evaluate_solution_quality(self, routes: List[List[int]], model: VRPModel) -> Dict[str, Any]:
        """솔루션 품질 평가"""
        
        objective_result = self.calculate_total_objective(routes, model)
        
        # 기본 통계 계산
        total_orders = len(model.get_customer_indices())
        assigned_orders = sum(len([idx for idx in route if idx not in model.get_depot_indices()]) 
                            for route in routes)
        unassigned_orders = total_orders - assigned_orders
        
        used_vehicles = sum(1 for route in routes if len(route) > 2)
        total_vehicles = len(model.vehicles)
        
        # 품질 점수 계산 (0.0 ~ 1.0)
        assignment_rate = assigned_orders / total_orders if total_orders > 0 else 0.0
        efficiency_score = 1.0 - (used_vehicles / total_vehicles) if total_vehicles > 0 else 0.0
        
        # 전체 품질 점수 (할당률 80% + 효율성 20%)
        quality_score = assignment_rate * 0.8 + efficiency_score * 0.2
        
        return {
            'objective_value': objective_result.total_value,
            'quality_score': quality_score,
            'assignment_rate': assignment_rate,
            'efficiency_score': efficiency_score,
            'total_orders': total_orders,
            'assigned_orders': assigned_orders,
            'unassigned_orders': unassigned_orders,
            'used_vehicles': used_vehicles,
            'total_vehicles': total_vehicles,
            'objective_components': objective_result.components
        }
    
    def get_objective_summary(self) -> Dict[str, Any]:
        """목적함수 요약 정보"""
        
        objective_info = {}
        for objective in self.objectives:
            objective_info[objective.objective_type.value] = {
                'weight': objective.weight,
                'description': objective.get_description()
            }
        
        return {
            'total_objectives': len(self.objectives),
            'objectives': objective_info
        }
    
    def reset_objectives(self) -> None:
        """목적함수 초기화"""
        self.objectives.clear()
        self.logger.info("목적함수 초기화 완료")