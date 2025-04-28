from typing import List, Dict, Tuple
from datetime import datetime
from ..models.delivery_point import DeliveryPoint
from ..models.vehicle import Vehicle
from ..models.route import Route, RoutePoint
from ..services.area_divider import AreaDivider
from ..services.time_optimizer import TimeOptimizer
from ..services.vehicle_optimizer import VehicleOptimizer
from ..services.constraint_handler import ConstraintHandler

class TMSOptimizer:
    """TMS 경로 최적화 핵심 클래스"""
    
    def __init__(self):
        self.area_divider = AreaDivider()
        self.time_optimizer = TimeOptimizer()
        self.vehicle_optimizer = VehicleOptimizer()
        self.constraint_handler = ConstraintHandler()

    def optimize(
        self,
        delivery_points: List[DeliveryPoint],
        vehicles: List[Vehicle],
        optimization_params: dict = None
    ) -> List[Route]:
        """
        전체 배송 경로 최적화 수행
        
        Args:
            delivery_points: 배송지점 목록
            vehicles: 사용 가능한 차량 목록
            optimization_params: 최적화 파라미터
        
        Returns:
            최적화된 경로 목록
        """
        if optimization_params is None:
            optimization_params = self._get_default_params()

        # 1. 지역 분할
        areas = self.area_divider.divide(
            delivery_points,
            num_areas=len(vehicles),
            method=optimization_params.get('area_division_method', 'kmeans')
        )

        # 2. 차량 할당
        vehicle_assignments = self.vehicle_optimizer.assign_vehicles(
            areas=areas,
            vehicles=vehicles,
            assignment_strategy=optimization_params.get('vehicle_assignment_strategy', 'balanced')
        )

        # 3. 각 구역별 경로 최적화
        routes = []
        for area, vehicle in vehicle_assignments:
            # 시간 최적화
            optimized_sequence = self.time_optimizer.optimize(
                points=area,
                vehicle=vehicle,
                start_time=vehicle.start_time,
                end_time=vehicle.end_time
            )

            # 경로 생성
            route = self._create_route(
                vehicle=vehicle,
                points=optimized_sequence,
                optimization_params=optimization_params
            )

            if route and route.is_valid:
                routes.append(route)

        # 4. 제약조건 검증 및 처리
        valid_routes = self.constraint_handler.validate_and_adjust(
            routes=routes,
            constraints=optimization_params.get('constraints', {})
        )

        return valid_routes

    def _get_default_params(self) -> dict:
        """기본 최적화 파라미터 반환"""
        return {
            'area_division_method': 'kmeans',
            'vehicle_assignment_strategy': 'balanced',
            'constraints': {
                'max_working_hours': 8,
                'max_points_per_vehicle': 50,
                'min_points_per_vehicle': 10,
                'allow_overtime': False,
                'consider_traffic': True
            }
        }

    def _create_route(
        self,
        vehicle: Vehicle,
        points: List[DeliveryPoint],
        optimization_params: dict
    ) -> Route:
        """최적화된 경로 생성"""
        route_points = []
        current_time = vehicle.start_time
        current_distance = 0
        current_load = {'volume': 0, 'weight': 0}

        for point in points:
            # 실제 구현에서는 여기에 더 복잡한 로직 필요
            arrival_time = current_time
            service_time = point.service_time
            departure_time = arrival_time + timedelta(minutes=service_time)

            route_point = RoutePoint(
                point=point,
                arrival_time=arrival_time,
                departure_time=departure_time,
                cumulative_distance=current_distance,
                cumulative_load={
                    'volume': current_load['volume'] + point.volume,
                    'weight': current_load['weight'] + point.weight
                }
            )

            route_points.append(route_point)
            current_time = departure_time
            current_load = route_point.cumulative_load

        return Route(
            id=f"R{vehicle.id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            vehicle=vehicle,
            points=route_points,
            total_distance=current_distance,
            total_time=int((current_time - vehicle.start_time).total_seconds() / 60),
            total_load=current_load,
            start_time=vehicle.start_time,
            end_time=current_time,
            status='PLANNED'
        )