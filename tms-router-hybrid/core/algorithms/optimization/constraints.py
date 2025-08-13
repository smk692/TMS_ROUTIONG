"""
VRP 제약조건 관리자
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .vrp_model import VRPModel, VRPConstraint, VRPConstraintType


@dataclass
class ConstraintViolation:
    """제약조건 위반 정보"""
    constraint_type: VRPConstraintType
    vehicle_id: str
    location_ids: List[str]
    violation_amount: float
    penalty_cost: float
    description: str


class BaseConstraint(ABC):
    """제약조건 기본 클래스"""
    
    def __init__(self, constraint_type: VRPConstraintType, parameters: Dict[str, Any]):
        self.constraint_type = constraint_type
        self.parameters = parameters
        self.logger = logging.getLogger(f"{__name__}.{constraint_type.value}")
    
    @abstractmethod
    def validate_route(self, route: List[int], model: VRPModel) -> Tuple[bool, Optional[ConstraintViolation]]:
        """경로가 제약조건을 만족하는지 검증"""
        pass
    
    @abstractmethod
    def calculate_penalty(self, violation_amount: float) -> float:
        """위반량에 대한 페널티 계산"""
        pass


class CapacityConstraint(BaseConstraint):
    """용량 제약조건"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(VRPConstraintType.CAPACITY, parameters)
        self.enforce_capacity = parameters.get('enforce_capacity', True)
        self.penalty_per_unit = parameters.get('penalty_per_unit', 1000.0)
    
    def validate_route(self, route: List[int], model: VRPModel) -> Tuple[bool, Optional[ConstraintViolation]]:
        """용량 제약조건 검증"""
        
        if not self.enforce_capacity:
            return True, None
        
        # 경로의 총 수요량 계산
        total_demand = sum(model.locations[location_idx].demand for location_idx in route)
        
        # 차량 용량 조회 (첫 번째 차량으로 가정)
        vehicle_capacity = model.vehicles[0].capacity if model.vehicles else 0
        
        if total_demand > vehicle_capacity:
            violation_amount = total_demand - vehicle_capacity
            penalty = self.calculate_penalty(violation_amount)
            
            violation = ConstraintViolation(
                constraint_type=self.constraint_type,
                vehicle_id="vehicle_0",
                location_ids=[model.locations[idx].id for idx in route],
                violation_amount=violation_amount,
                penalty_cost=penalty,
                description=f"용량 초과: {total_demand} > {vehicle_capacity}"
            )
            
            return False, violation
        
        return True, None
    
    def calculate_penalty(self, violation_amount: float) -> float:
        """용량 초과에 대한 페널티"""
        return violation_amount * self.penalty_per_unit


class DistanceConstraint(BaseConstraint):
    """거리 제약조건"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(VRPConstraintType.DISTANCE, parameters)
        self.max_distance_km = parameters.get('max_distance_km', 120.0)
        self.penalty_per_km = parameters.get('penalty_per_km', 100.0)
    
    def validate_route(self, route: List[int], model: VRPModel) -> Tuple[bool, Optional[ConstraintViolation]]:
        """거리 제약조건 검증"""
        
        if model.distance_matrix is None or len(route) < 2:
            return True, None
        
        # 경로의 총 거리 계산
        total_distance = 0.0
        for i in range(len(route) - 1):
            from_idx = route[i]
            to_idx = route[i + 1]
            total_distance += model.distance_matrix[from_idx][to_idx]
        
        if total_distance > self.max_distance_km:
            violation_amount = total_distance - self.max_distance_km
            penalty = self.calculate_penalty(violation_amount)
            
            violation = ConstraintViolation(
                constraint_type=self.constraint_type,
                vehicle_id="vehicle_0",
                location_ids=[model.locations[idx].id for idx in route],
                violation_amount=violation_amount,
                penalty_cost=penalty,
                description=f"거리 초과: {total_distance:.1f}km > {self.max_distance_km}km"
            )
            
            return False, violation
        
        return True, None
    
    def calculate_penalty(self, violation_amount: float) -> float:
        """거리 초과에 대한 페널티"""
        return violation_amount * self.penalty_per_km


class TimeWindowConstraint(BaseConstraint):
    """시간 창 제약조건"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(VRPConstraintType.TIME_WINDOW, parameters)
        self.enforce_time_windows = parameters.get('enforce_time_windows', True)
        self.penalty_per_minute = parameters.get('penalty_per_minute', 10.0)
    
    def validate_route(self, route: List[int], model: VRPModel) -> Tuple[bool, Optional[ConstraintViolation]]:
        """시간 창 제약조건 검증"""
        
        if not self.enforce_time_windows or model.time_matrix is None:
            return True, None
        
        current_time = 0
        violations = []
        
        for i, location_idx in enumerate(route[1:], 1):  # 첫 번째는 depot이므로 skip
            # 이전 위치에서 현재 위치로 이동 시간
            if i > 0:
                prev_idx = route[i-1]
                travel_time = model.time_matrix[prev_idx][location_idx]
                current_time += travel_time
            
            location = model.locations[location_idx]
            
            # 시간 창 검증
            if current_time < location.time_window_start:
                # 너무 일찍 도착 - 대기 시간
                wait_time = location.time_window_start - current_time
                current_time = location.time_window_start
            elif current_time > location.time_window_end:
                # 너무 늦게 도착 - 위반
                violation_amount = current_time - location.time_window_end
                penalty = self.calculate_penalty(violation_amount)
                
                violation = ConstraintViolation(
                    constraint_type=self.constraint_type,
                    vehicle_id="vehicle_0",
                    location_ids=[location.id],
                    violation_amount=violation_amount,
                    penalty_cost=penalty,
                    description=f"시간 창 위반: {current_time} > {location.time_window_end}"
                )
                violations.append(violation)
            
            # 서비스 시간 추가
            current_time += location.service_time
        
        if violations:
            # 첫 번째 위반만 반환 (단순화)
            return False, violations[0]
        
        return True, None
    
    def calculate_penalty(self, violation_amount: float) -> float:
        """시간 창 위반에 대한 페널티"""
        return violation_amount * self.penalty_per_minute


class SkillConstraint(BaseConstraint):
    """스킬 제약조건 (특정 차량만 특정 주문 처리 가능)"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(VRPConstraintType.SKILL, parameters)
        self.required_skills = parameters.get('required_skills', {})  # location_id: [skills]
        self.penalty_fixed = parameters.get('penalty_fixed', 10000.0)
    
    def validate_route(self, route: List[int], model: VRPModel) -> Tuple[bool, Optional[ConstraintViolation]]:
        """스킬 제약조건 검증"""
        
        if not self.required_skills:
            return True, None
        
        # 차량 스킬 조회 (첫 번째 차량으로 가정)
        vehicle_skills = model.vehicles[0].skills if model.vehicles else []
        
        for location_idx in route:
            location = model.locations[location_idx]
            required_skills = self.required_skills.get(location.id, [])
            
            # 필요한 스킬이 있는지 확인
            if required_skills:
                missing_skills = [skill for skill in required_skills if skill not in vehicle_skills]
                
                if missing_skills:
                    violation = ConstraintViolation(
                        constraint_type=self.constraint_type,
                        vehicle_id="vehicle_0",
                        location_ids=[location.id],
                        violation_amount=len(missing_skills),
                        penalty_cost=self.calculate_penalty(len(missing_skills)),
                        description=f"스킬 부족: {missing_skills}"
                    )
                    
                    return False, violation
        
        return True, None
    
    def calculate_penalty(self, violation_amount: float) -> float:
        """스킬 부족에 대한 페널티"""
        return self.penalty_fixed * violation_amount


class BreakConstraint(BaseConstraint):
    """휴식 시간 제약조건"""
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__(VRPConstraintType.BREAK, parameters)
        self.max_work_time = parameters.get('max_work_time_minutes', 240)  # 4시간
        self.break_duration = parameters.get('break_duration_minutes', 15)   # 15분
        self.penalty_per_minute = parameters.get('penalty_per_minute', 50.0)
    
    def validate_route(self, route: List[int], model: VRPModel) -> Tuple[bool, Optional[ConstraintViolation]]:
        """휴식 시간 제약조건 검증"""
        
        if model.time_matrix is None or len(route) < 3:  # depot -> location -> depot 최소
            return True, None
        
        total_work_time = 0
        
        # 경로의 총 작업시간 계산
        for i in range(len(route) - 1):
            from_idx = route[i]
            to_idx = route[i + 1]
            
            # 이동 시간
            travel_time = model.time_matrix[from_idx][to_idx]
            total_work_time += travel_time
            
            # 서비스 시간 (depot이 아닌 경우)
            depot_count = len(model.get_depot_indices())
            if int(to_idx) >= int(depot_count):  # depot이 아닌 경우 (수정된 조건)
                service_time = model.locations[to_idx].service_time
                total_work_time += service_time
        
        # 휴식 없이 최대 작업시간 초과 시 위반
        if total_work_time > self.max_work_time:
            violation_amount = total_work_time - self.max_work_time
            penalty = self.calculate_penalty(violation_amount)
            
            violation = ConstraintViolation(
                constraint_type=self.constraint_type,
                vehicle_id="vehicle_0",
                location_ids=[model.locations[idx].id for idx in route],
                violation_amount=violation_amount,
                penalty_cost=penalty,
                description=f"휴식시간 필요: {total_work_time}분 > {self.max_work_time}분"
            )
            
            return False, violation
        
        return True, None
    
    def calculate_penalty(self, violation_amount: float) -> float:
        """휴식 시간 위반에 대한 페널티"""
        return violation_amount * self.penalty_per_minute


class ConstraintManager:
    """제약조건 관리자"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 제약조건 등록
        self._constraint_factories = {
            VRPConstraintType.CAPACITY: CapacityConstraint,
            VRPConstraintType.DISTANCE: DistanceConstraint,
            VRPConstraintType.TIME_WINDOW: TimeWindowConstraint,
            VRPConstraintType.SKILL: SkillConstraint,
            VRPConstraintType.BREAK: BreakConstraint
        }
        
        self.constraints: List[BaseConstraint] = []
        self.violations: List[ConstraintViolation] = []
    
    def add_constraint_from_vrp_constraint(self, vrp_constraint: VRPConstraint) -> None:
        """VRPConstraint로부터 제약조건 추가"""
        
        constraint_class = self._constraint_factories.get(vrp_constraint.constraint_type)
        
        if constraint_class:
            constraint = constraint_class(vrp_constraint.parameters)
            self.constraints.append(constraint)
            self.logger.info(f"제약조건 추가: {vrp_constraint.constraint_type.value}")
        else:
            self.logger.warning(f"지원하지 않는 제약조건 타입: {vrp_constraint.constraint_type}")
    
    def add_default_constraints(self) -> None:
        """기본 제약조건들 추가"""
        
        # 용량 제약조건
        capacity_constraint = CapacityConstraint({
            'enforce_capacity': True,
            'penalty_per_unit': 1000.0
        })
        self.constraints.append(capacity_constraint)
        
        # 거리 제약조건
        distance_constraint = DistanceConstraint({
            'max_distance_km': 120.0,
            'penalty_per_km': 100.0
        })
        self.constraints.append(distance_constraint)
        
        # 휴식 시간 제약조건
        break_constraint = BreakConstraint({
            'max_work_time_minutes': 240,
            'break_duration_minutes': 15,
            'penalty_per_minute': 50.0
        })
        self.constraints.append(break_constraint)
        
        self.logger.info(f"기본 제약조건 {len(self.constraints)}개 추가 완료")
    
    def validate_route(self, route: List[int], model: VRPModel) -> Tuple[bool, List[ConstraintViolation]]:
        """경로에 대한 모든 제약조건 검증"""
        
        violations = []
        is_valid = True
        
        for constraint in self.constraints:
            valid, violation = constraint.validate_route(route, model)
            
            if not valid and violation:
                is_valid = False
                violations.append(violation)
                self.logger.debug(f"제약조건 위반: {violation.description}")
        
        return is_valid, violations
    
    def calculate_total_penalty(self, violations: List[ConstraintViolation]) -> float:
        """위반들에 대한 총 페널티 계산"""
        
        total_penalty = sum(violation.penalty_cost for violation in violations)
        return total_penalty
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """제약조건 요약 정보"""
        
        constraint_counts = {}
        for constraint in self.constraints:
            constraint_type = constraint.constraint_type.value
            constraint_counts[constraint_type] = constraint_counts.get(constraint_type, 0) + 1
        
        return {
            'total_constraints': len(self.constraints),
            'constraint_types': constraint_counts,
            'total_violations': len(self.violations)
        }
    
    def reset_violations(self) -> None:
        """위반 기록 초기화"""
        self.violations.clear()
    
    def get_violations_by_type(self, constraint_type: VRPConstraintType) -> List[ConstraintViolation]:
        """타입별 위반 조회"""
        return [v for v in self.violations if v.constraint_type == constraint_type]