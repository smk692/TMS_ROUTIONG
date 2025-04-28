from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from ..models.route import Route
from ..models.delivery_point import DeliveryPoint
from ..utils.geo_utils import calculate_distance

class ConstraintHandler:
    """경로의 제약조건을 처리하고 검증하는 클래스"""
    
    def __init__(self):
        self.hard_constraints = {
            'vehicle_capacity': self._check_vehicle_capacity,
            'time_windows': self._check_time_windows,
            'vehicle_type_compatibility': self._check_vehicle_compatibility,
            'working_hours': self._check_working_hours,
            'route_overlap': self._check_route_overlap
        }
        
        self.soft_constraints = {
            'balanced_workload': self._check_workload_balance,
            'route_efficiency': self._check_route_efficiency,
            'driver_preferences': self._check_driver_preferences
        }

    def validate_and_adjust(
        self,
        routes: List[Route],
        constraints: Dict[str, Any]
    ) -> List[Route]:
        """
        경로들의 제약조건을 검증하고 필요한 경우 조정

        Args:
            routes: 검증할 경로 리스트
            constraints: 적용할 제약조건 설정

        Returns:
            조정된 경로 리스트
        """
        adjusted_routes = routes.copy()
        violations = []

        # 하드 제약조건 검사
        for route in routes:
            route_violations = self._check_hard_constraints(route, constraints)
            if route_violations:
                violations.append((route, route_violations))

        # 위반사항이 있는 경우 경로 조정
        if violations:
            adjusted_routes = self._adjust_routes(routes, violations, constraints)

        # 소프트 제약조건 최적화
        adjusted_routes = self._optimize_soft_constraints(adjusted_routes, constraints)

        return adjusted_routes

    def _check_hard_constraints(
        self,
        route: Route,
        constraints: Dict[str, Any]
    ) -> List[str]:
        """하드 제약조건 검사"""
        violations = []
        
        for constraint_name, constraint_value in constraints.items():
            if constraint_name in self.hard_constraints:
                checker = self.hard_constraints[constraint_name]
                if not checker(route, constraint_value):
                    violations.append(constraint_name)
        
        return violations

    def _check_vehicle_capacity(
        self,
        route: Route,
        constraint_value: Dict[str, float]
    ) -> bool:
        """차량 용량 제약조건 검사"""
        total_volume = 0
        total_weight = 0
        
        for point in route.points:
            total_volume += point.point.volume
            total_weight += point.point.weight
            
            if (total_volume > route.vehicle.capacity.volume or
                total_weight > route.vehicle.capacity.weight):
                return False
        
        return True

    def _check_time_windows(
        self,
        route: Route,
        constraint_value: Dict[str, Any]
    ) -> bool:
        """시간 제약조건 검사"""
        current_time = route.start_time
        
        for i, point in enumerate(route.points):
            # 도착 시간이 허용 범위를 벗어나는지 검사
            if point.point.time_window and all(point.point.time_window):
                if not (point.point.time_window[0] <= point.arrival_time <= point.point.time_window[1]):
                    return False
            
            # 서비스 시간과 이동 시간이 전체 근무 시간을 초과하는지 검사
            if i < len(route.points) - 1:
                next_point = route.points[i + 1]
                travel_time = (next_point.arrival_time - point.departure_time).total_seconds() / 60
                
                if travel_time > constraint_value.get('max_travel_time', float('inf')):
                    return False
        
        return True

    def _check_vehicle_compatibility(
        self,
        route: Route,
        constraint_value: Dict[str, List[str]]
    ) -> bool:
        """차량 유형 호환성 검사"""
        vehicle_type = route.vehicle.type
        
        for point in route.points:
            required_features = point.point.special_requirements
            if required_features:
                if not all(feature in route.vehicle.features for feature in required_features):
                    return False
        
        return True

    def _check_working_hours(
        self,
        route: Route,
        constraint_value: Dict[str, int]
    ) -> bool:
        """근무 시간 제약조건 검사"""
        max_working_minutes = constraint_value.get('max_working_minutes', 480)  # 기본 8시간
        total_minutes = (route.end_time - route.start_time).total_seconds() / 60
        
        return total_minutes <= max_working_minutes

    def _check_route_overlap(
        self,
        routes: List[Route],
        constraint_value: Dict[str, Any]
    ) -> bool:
        """경로 중복 검사"""
        for i, route1 in enumerate(routes):
            for j, route2 in enumerate(routes[i+1:], i+1):
                overlapping_points = self._find_overlapping_points(route1, route2)
                if overlapping_points:
                    return False
        return True

    def _check_workload_balance(
        self,
        routes: List[Route],
        constraint_value: Dict[str, float]
    ) -> float:
        """작업량 균형 점수 계산"""
        if not routes:
            return 1.0
            
        workloads = [len(route.points) for route in routes]
        avg_workload = sum(workloads) / len(workloads)
        max_deviation = max(abs(w - avg_workload) for w in workloads)
        
        return 1.0 - (max_deviation / avg_workload)

    def _check_route_efficiency(
        self,
        route: Route,
        constraint_value: Dict[str, float]
    ) -> float:
        """경로 효율성 점수 계산"""
        total_distance = route.total_distance
        num_points = len(route.points)
        
        if num_points <= 1:
            return 1.0
            
        # 거리당 배송지점 수로 효율성 계산
        efficiency = num_points / total_distance if total_distance > 0 else 0
        target_efficiency = constraint_value.get('target_efficiency', 0.1)
        
        return min(efficiency / target_efficiency, 1.0)

    def _check_driver_preferences(
        self,
        route: Route,
        constraint_value: Dict[str, Any]
    ) -> float:
        """운전자 선호도 점수 계산"""
        # 실제 구현에서는 운전자 선호도 데이터 필요
        return 1.0

    def _adjust_routes(
        self,
        routes: List[Route],
        violations: List[Tuple[Route, List[str]]],
        constraints: Dict[str, Any]
    ) -> List[Route]:
        """제약조건 위반을 해결하기 위한 경로 조정"""
        adjusted_routes = routes.copy()
        
        for route, violation_types in violations:
            for violation_type in violation_types:
                if violation_type == 'vehicle_capacity':
                    adjusted_routes = self._adjust_capacity_violation(
                        adjusted_routes,
                        route,
                        constraints
                    )
                elif violation_type == 'time_windows':
                    adjusted_routes = self._adjust_time_violation(
                        adjusted_routes,
                        route,
                        constraints
                    )
                elif violation_type == 'route_overlap':
                    adjusted_routes = self._adjust_overlap_violation(
                        adjusted_routes,
                        route,
                        constraints
                    )
        
        return adjusted_routes

    def _adjust_capacity_violation(
        self,
        routes: List[Route],
        violated_route: Route,
        constraints: Dict[str, Any]
    ) -> List[Route]:
        """용량 제약 위반 조정"""
        # 가장 가까운 다른 경로로 초과 포인트 이동
        excess_points = []
        current_volume = 0
        current_weight = 0
        
        for point in violated_route.points:
            if (current_volume + point.point.volume > violated_route.vehicle.capacity.volume or
                current_weight + point.point.weight > violated_route.vehicle.capacity.weight):
                excess_points.append(point)
            else:
                current_volume += point.point.volume
                current_weight += point.point.weight
        
        if excess_points:
            # 다른 경로들 중 수용 가능한 경로 찾기
            for point in excess_points:
                best_route = None
                min_distance = float('inf')
                
                for route in routes:
                    if route != violated_route and route.can_add_point(point.point):
                        distance = calculate_distance(
                            (point.point.latitude, point.point.longitude),
                            (route.points[-1].point.latitude, route.points[-1].point.longitude)
                        )
                        if distance < min_distance:
                            min_distance = distance
                            best_route = route
                
                if best_route:
                    best_route.add_point(point.point)
                    violated_route.points.remove(point)
        
        return routes

    def _adjust_time_violation(
        self,
        routes: List[Route],
        violated_route: Route,
        constraints: Dict[str, Any]
    ) -> List[Route]:
        """시간 제약 위반 조정"""
        # 시간 창을 위반하는 포인트들을 다른 경로로 재할당
        violated_points = []
        valid_points = []
        
        current_time = violated_route.start_time
        for point in violated_route.points:
            if point.point.time_window and all(point.point.time_window):
                if current_time <= point.point.time_window[1]:
                    valid_points.append(point)
                    current_time = max(
                        current_time + timedelta(minutes=point.point.service_time),
                        point.point.time_window[0]
                    )
                else:
                    violated_points.append(point)
            else:
                valid_points.append(point)
                current_time += timedelta(minutes=point.point.service_time)
        
        # 위반된 포인트들을 다른 경로에 재할당
        for point in violated_points:
            best_route = None
            best_insertion_time = None
            
            for route in routes:
                if route != violated_route:
                    insertion_time = route.find_best_insertion_time(point.point)
                    if insertion_time:
                        best_route = route
                        best_insertion_time = insertion_time
                        break
            
            if best_route:
                best_route.add_point(point.point)
            else:
                # 새로운 경로 생성 필요
                pass
        
        violated_route.points = valid_points
        return routes

    def _adjust_overlap_violation(
        self,
        routes: List[Route],
        violated_route: Route,
        constraints: Dict[str, Any]
    ) -> List[Route]:
        """경로 중복 위반 조정"""
        # 중복되는 구역 식별 및 재할당
        for route in routes:
            if route != violated_route:
                overlapping_points = self._find_overlapping_points(violated_route, route)
                if overlapping_points:
                    # 중복 지점들을 거리 기반으로 재할당
                    for point in overlapping_points:
                        dist1 = self._calculate_route_center_distance(violated_route, point)
                        dist2 = self._calculate_route_center_distance(route, point)
                        
                        if dist1 < dist2:
                            route.points.remove(point)
                        else:
                            violated_route.points.remove(point)
        
        return routes

    def _optimize_soft_constraints(
        self,
        routes: List[Route],
        constraints: Dict[str, Any]
    ) -> List[Route]:
        """소프트 제약조건 최적화"""
        improved = True
        while improved:
            improved = False
            
            # 작업량 균형 최적화
            workload_score = self._check_workload_balance(routes, constraints)
            if workload_score < constraints.get('min_workload_balance', 0.8):
                routes = self._balance_workload(routes)
                improved = True
            
            # 경로 효율성 최적화
            for route in routes:
                efficiency_score = self._check_route_efficiency(route, constraints)
                if efficiency_score < constraints.get('min_route_efficiency', 0.7):
                    route = self._improve_route_efficiency(route)
                    improved = True
        
        return routes

    def _find_overlapping_points(
        self,
        route1: Route,
        route2: Route
    ) -> List[DeliveryPoint]:
        """두 경로 간의 중복되는 지점 찾기"""
        overlapping = []
        for point1 in route1.points:
            for point2 in route2.points:
                if calculate_distance(
                    (point1.point.latitude, point1.point.longitude),
                    (point2.point.latitude, point2.point.longitude)
                ) < 0.1:  # 100m 이내를 중복으로 간주
                    overlapping.append(point1.point)
        return overlapping

    def _calculate_route_center_distance(
        self,
        route: Route,
        point: DeliveryPoint
    ) -> float:
        """경로의 중심점과 특정 지점 간의 거리 계산"""
        if not route.points:
            return float('inf')
            
        center_lat = sum(p.point.latitude for p in route.points) / len(route.points)
        center_lon = sum(p.point.longitude for p in route.points) / len(route.points)
        
        return calculate_distance(
            (center_lat, center_lon),
            (point.latitude, point.longitude)
        )

    def _balance_workload(self, routes: List[Route]) -> List[Route]:
        """작업량 균형 조정"""
        if not routes:
            return routes
            
        # 작업량이 가장 많은/적은 경로 찾기
        max_route = max(routes, key=lambda r: len(r.points))
        min_route = min(routes, key=lambda r: len(r.points))
        
        while len(max_route.points) - len(min_route.points) > 1:
            # 가장 적합한 포인트를 이동
            best_point = None
            min_cost = float('inf')
            
            for point in max_route.points[1:]:  # 첫 포인트(depot)는 제외
                # 이동 비용 계산
                cost = self._calculate_transfer_cost(point.point, max_route, min_route)
                if cost < min_cost:
                    min_cost = cost
                    best_point = point
            
            if best_point:
                max_route.points.remove(best_point)
                min_route.add_point(best_point.point)
            else:
                break
        
        return routes

    def _improve_route_efficiency(self, route: Route) -> Route:
        """경로 효율성 개선"""
        if len(route.points) <= 2:
            return route
            
        # 2-opt 알고리즘을 사용한 경로 최적화
        improved = True
        while improved:
            improved = False
            best_distance = route.total_distance
            
            for i in range(1, len(route.points) - 1):
                for j in range(i + 1, len(route.points)):
                    new_route = self._two_opt_swap(route, i, j)
                    if new_route.total_distance < best_distance:
                        route = new_route
                        best_distance = new_route.total_distance
                        improved = True
                        break
                if improved:
                    break
        
        return route

    def _calculate_transfer_cost(
        self,
        point: DeliveryPoint,
        from_route: Route,
        to_route: Route
    ) -> float:
        """포인트 이동 비용 계산"""
        # 거리 기반 비용
        from_center = (
            sum(p.point.latitude for p in from_route.points) / len(from_route.points),
            sum(p.point.longitude for p in from_route.points) / len(from_route.points)
        )
        to_center = (
            sum(p.point.latitude for p in to_route.points) / len(to_route.points),
            sum(p.point.longitude for p in to_route.points) / len(to_route.points)
        )
        
        from_dist = calculate_distance(
            (point.latitude, point.longitude),
            from_center
        )
        to_dist = calculate_distance(
            (point.latitude, point.longitude),
            to_center
        )
        
        # 작업량 균형 비용
        workload_diff = len(from_route.points) - len(to_route.points)
        
        # 종합 비용 계산
        return to_dist - from_dist + workload_diff * 0.5

    def _two_opt_swap(self, route: Route, i: int, j: int) -> Route:
        """2-opt 스왑을 통한 경로 개선"""
        new_points = route.points[:i]
        new_points.extend(reversed(route.points[i:j + 1]))
        new_points.extend(route.points[j + 1:])
        
        new_route = Route(
            id=route.id,
            vehicle=route.vehicle,
            points=new_points,
            total_distance=self._calculate_route_distance(new_points),
            total_time=route.total_time,
            total_load=route.total_load,
            start_time=route.start_time,
            end_time=route.end_time,
            status=route.status
        )
        
        return new_route

    def _calculate_route_distance(self, points: List[DeliveryPoint]) -> float:
        """경로의 총 거리 계산"""
        total_distance = 0
        for i in range(len(points) - 1):
            total_distance += calculate_distance(
                (points[i].point.latitude, points[i].point.longitude),
                (points[i + 1].point.latitude, points[i + 1].point.longitude)
            )
        return total_distance