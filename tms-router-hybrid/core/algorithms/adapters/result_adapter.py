"""
VRP 솔루션을 기존 VehicleAssignment 형태로 변환하는 어댑터
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from ...models import VehicleAssignment, Vehicle
from ..vrp_solver import VRPSolution, VRPRoute


class ResultAdapter:
    """VRP 솔루션을 VehicleAssignment로 변환"""
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
    
    def convert_vrp_solution_to_assignments(self, vrp_solution: VRPSolution, 
                                          vehicles: List[Vehicle]) -> List[VehicleAssignment]:
        """VRP 솔루션을 VehicleAssignment 리스트로 변환"""
        
        if not vrp_solution.routes:
            self.logger.warning("VRP 솔루션에 경로가 없음")
            return []
        
        assignments = []
        vehicle_map = {v.id: v for v in vehicles}
        
        for route in vrp_solution.routes:
            try:
                # 해당 차량 정보 찾기
                vehicle = vehicle_map.get(route.vehicle_id)
                if not vehicle:
                    self.logger.warning(f"차량 {route.vehicle_id}를 찾을 수 없음")
                    continue
                
                # VehicleAssignment 생성
                assignment = VehicleAssignment(
                    vehicle_id=route.vehicle_id,
                    driver_name=vehicle.driver_name,
                    vehicle_type=vehicle.vehicle_type.value,
                    region_name=f"권역_{vehicle.region_id}",
                    assigned_orders=route.order_sequence,
                    estimated_distance_km=route.total_distance,
                    estimated_time_minutes=route.total_time,
                    capacity_utilization=route.capacity_usage
                )
                
                assignments.append(assignment)
                
                self.logger.debug(f"변환 완료 - 차량 {route.vehicle_id}: "
                                f"{len(route.order_sequence)}개 주문, "
                                f"{route.total_distance:.1f}km, "
                                f"{route.total_time}분")
                
            except Exception as e:
                self.logger.error(f"VRP 경로 변환 오류 (차량 {route.vehicle_id}): {str(e)}")
                continue
        
        self.logger.info(f"VRP 솔루션 변환 완료: {len(assignments)}개 배차 생성")
        
        return assignments
    
    def generate_assignment_summary(self, assignments: List[VehicleAssignment], 
                                  vrp_solution: VRPSolution) -> Dict:
        """배차 결과 요약 생성"""
        
        if not assignments:
            return {
                'total_vehicles': 0,
                'total_orders': 0,
                'total_distance': 0.0,
                'total_time': 0,
                'average_capacity_utilization': 0.0,
                'unassigned_orders': len(vrp_solution.unassigned_orders),
                'assignment_rate': 0.0
            }
        
        total_orders = sum(len(a.assigned_orders) for a in assignments)
        total_distance = sum(a.estimated_distance_km for a in assignments)
        total_time = sum(a.estimated_time_minutes for a in assignments)
        
        avg_capacity_utilization = sum(a.capacity_utilization for a in assignments) / len(assignments)
        
        total_orders_attempted = total_orders + len(vrp_solution.unassigned_orders)
        assignment_rate = total_orders / total_orders_attempted if total_orders_attempted > 0 else 0.0
        
        return {
            'total_vehicles': len(assignments),
            'total_orders': total_orders,
            'total_distance': total_distance,
            'total_time': total_time,
            'average_capacity_utilization': avg_capacity_utilization,
            'unassigned_orders': len(vrp_solution.unassigned_orders),
            'assignment_rate': assignment_rate,
            'vrp_objective_value': vrp_solution.objective_value,
            'vrp_is_optimal': vrp_solution.is_optimal
        }
    
    def validate_assignments(self, assignments: List[VehicleAssignment], 
                           vehicles: List[Vehicle]) -> List[str]:
        """배차 결과 유효성 검증"""
        
        warnings = []
        vehicle_map = {v.id: v for v in vehicles}
        
        for assignment in assignments:
            vehicle = vehicle_map.get(assignment.vehicle_id)
            if not vehicle:
                warnings.append(f"차량 {assignment.vehicle_id}를 찾을 수 없음")
                continue
            
            # 1. 용량 초과 검증
            if assignment.capacity_utilization > 1.1:  # 10% 초과 허용
                warnings.append(f"차량 {assignment.vehicle_id}: 용량 초과 "
                              f"({assignment.capacity_utilization:.1%})")
            
            # 2. 시간 제한 검증
            max_work_minutes = 8 * 60  # 8시간
            if assignment.estimated_time_minutes > max_work_minutes * 1.2:  # 20% 초과 허용
                warnings.append(f"차량 {assignment.vehicle_id}: 작업시간 초과 "
                              f"({assignment.estimated_time_minutes}분 > {max_work_minutes}분)")
            
            # 3. 거리 제한 검증
            max_distance = 120.0  # 120km
            if assignment.estimated_distance_km > max_distance * 1.2:  # 20% 초과 허용
                warnings.append(f"차량 {assignment.vehicle_id}: 이동거리 초과 "
                              f"({assignment.estimated_distance_km:.1f}km > {max_distance}km)")
            
            # 4. 최소 효율성 검증
            if assignment.capacity_utilization < 0.3:  # 30% 미만
                warnings.append(f"차량 {assignment.vehicle_id}: 용량 활용도 낮음 "
                              f"({assignment.capacity_utilization:.1%})")
        
        if warnings:
            self.logger.warning(f"배차 검증 경고 {len(warnings)}건: {warnings}")
        else:
            self.logger.info("배차 결과 검증 통과")
        
        return warnings