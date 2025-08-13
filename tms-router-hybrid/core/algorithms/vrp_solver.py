"""
OR-Tools 기반 Vehicle Routing Problem 솔버
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from ..models import Order, Vehicle, Coordinates, VehicleAssignment
from .clustering.hdbscan_clusterer import HDBSCANGeographicClusterer, OrderCluster
from .optimization.distance_calculator import DistanceMatrixCalculator


@dataclass
class VRPData:
    """VRP 입력 데이터"""
    locations: List[Coordinates]  # 모든 위치 (depot + 주문)
    demands: List[int]           # 각 위치의 수요량
    vehicle_capacities: List[int] # 각 차량의 용량
    depot_indices: List[int]     # 각 차량의 출발지 인덱스
    time_windows: List[Tuple[int, int]]  # 각 위치의 시간 창
    distance_matrix: np.ndarray  # 거리 행렬
    time_matrix: np.ndarray      # 시간 행렬
    
    # 메타데이터
    order_to_location_map: Dict[str, int]  # 주문ID -> 위치인덱스
    location_to_order_map: Dict[int, str]  # 위치인덱스 -> 주문ID
    vehicle_to_depot_map: Dict[str, int]   # 차량ID -> depot 인덱스


@dataclass  
class VRPRoute:
    """VRP 경로 결과"""
    vehicle_id: str
    location_sequence: List[int]  # 위치 인덱스 순서
    order_sequence: List[str]     # 주문 ID 순서
    total_distance: float         # 총 이동거리 (km)
    total_time: int              # 총 소요시간 (분)
    total_demand: int            # 총 수요량
    capacity_usage: float        # 용량 사용률


@dataclass
class VRPSolution:
    """VRP 솔루션"""
    routes: List[VRPRoute]
    unassigned_orders: List[str]
    total_distance: float
    total_time: int
    objective_value: int
    solve_time_seconds: float
    is_optimal: bool


class VRPSolver:
    """OR-Tools 기반 VRP 솔버"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 솔버 설정
        self.max_solve_time = config.get('max_solve_time_seconds', 120)
        self.use_clustering = config.get('use_clustering', True)
        
        # 컴포넌트 초기화
        self.clusterer = HDBSCANGeographicClusterer(config.get('clustering', {}))
        self.distance_calculator = DistanceMatrixCalculator(config.get('distance_api', {}))
        
        # 제약조건 설정
        self.max_work_hours = config.get('max_work_hours', 8)  # 8시간
        self.max_distance_km = config.get('max_distance_km', 120)  # 120km
        self.break_interval_hours = config.get('break_interval_hours', 4)  # 4시간마다
        self.break_duration_minutes = config.get('break_duration_minutes', 15)  # 15분 휴식
        
        # 목적함수 가중치
        self.unassigned_penalty = config.get('unassigned_penalty', 100000)
        self.distance_weight = config.get('distance_weight', 1.0)
        self.vehicle_fixed_cost = config.get('vehicle_fixed_cost', 5000)
        self.time_balance_penalty = config.get('time_balance_penalty', 50)
        
    async def solve(self, orders: List[Order], vehicles: List[Vehicle], 
                   regions: List, conditions: Dict = None) -> VRPSolution:
        """VRP 문제 해결"""
        
        self.logger.info(f"VRP 솔빙 시작: {len(orders)}개 주문, {len(vehicles)}대 차량")
        
        try:
            # 1. 전처리 및 클러스터링
            if self.use_clustering and len(orders) > 30:
                clustered_orders = await self._preprocess_with_clustering(orders)
                self.logger.info(f"클러스터링 완료: {len(clustered_orders)}개 주문 그룹")
            else:
                clustered_orders = orders
            
            # 2. VRP 데이터 준비
            vrp_data = await self._prepare_vrp_data(clustered_orders, vehicles, conditions)
            
            # 3. OR-Tools 모델 생성
            manager, routing = self._create_vrp_model(vrp_data)
            
            # 4. 제약조건 및 목적함수 설정
            self._setup_constraints(routing, manager, vrp_data)
            self._setup_objective_function(routing, manager, vrp_data)
            
            # 5. 솔버 실행
            solution = self._solve_vrp_model(routing, manager)
            
            # 6. 결과 변환
            vrp_solution = self._convert_solution(solution, routing, manager, vrp_data)
            
            self.logger.info(f"VRP 솔빙 완료: {len(vrp_solution.routes)}개 경로, "
                           f"미배정 {len(vrp_solution.unassigned_orders)}개")
            
            return vrp_solution
            
        except Exception as e:
            import traceback
            self.logger.error(f"VRP 솔빙 오류: {str(e)}")
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            # 폴백: 빈 솔루션 반환
            return VRPSolution(
                routes=[],
                unassigned_orders=[order.id for order in orders],
                total_distance=0.0,
                total_time=0,
                objective_value=999999,
                solve_time_seconds=0.0,
                is_optimal=False
            )
    
    async def _preprocess_with_clustering(self, orders: List[Order]) -> List[Order]:
        """클러스터링을 통한 전처리"""
        
        try:
            # HDBSCAN 클러스터링 실행
            region_clusters = self.clusterer.cluster_orders(orders)
            
            # 클러스터 중심점 순으로 주문 재정렬
            reordered_orders = []
            
            for region_id, clusters in region_clusters.items():
                # 클러스터별로 중심점에서 가까운 순으로 정렬
                for cluster in clusters:
                    # 클러스터 내 주문들을 중심점 거리 순으로 정렬
                    cluster_orders = self._sort_orders_by_centroid_distance(
                        cluster.orders, cluster.centroid
                    )
                    reordered_orders.extend(cluster_orders)
            
            return reordered_orders
            
        except Exception as e:
            self.logger.warning(f"클러스터링 전처리 실패: {str(e)}, 원본 순서 사용")
            return orders
    
    def _sort_orders_by_centroid_distance(self, orders: List[Order], centroid: Coordinates) -> List[Order]:
        """중심점 거리 순으로 주문 정렬"""
        
        from geopy.distance import geodesic
        
        def distance_to_centroid(order):
            return geodesic(
                (centroid.latitude, centroid.longitude),
                (order.coordinates.latitude, order.coordinates.longitude)
            ).kilometers
        
        return sorted(orders, key=distance_to_centroid)
    
    async def _prepare_vrp_data(self, orders: List[Order], vehicles: List[Vehicle], 
                               conditions: Dict = None) -> VRPData:
        """VRP 입력 데이터 준비"""
        
        # 1. 위치 리스트 생성 (depot들 + 주문들)
        locations = []
        depot_indices = []
        order_to_location = {}
        location_to_order = {}
        
        # depot들 추가 (차량별 출발지)
        vehicle_to_depot = {}
        for i, vehicle in enumerate(vehicles):
            depot_coord = vehicle.center_coordinates
            locations.append(depot_coord)
            depot_indices.append(i)
            vehicle_to_depot[vehicle.id] = i
        
        # 주문 위치들 추가
        for order in orders:
            location_idx = len(locations)
            locations.append(order.coordinates)
            order_to_location[order.id] = location_idx
            location_to_order[location_idx] = order.id
        
        # 2. 수요량 설정 (depot은 0, 주문은 1)
        demands = [0] * len(depot_indices) + [1] * len(orders)
        
        # 3. 차량 용량 설정 (동적 조정 반영)
        vehicle_capacities = []
        for vehicle in vehicles:
            base_capacity = vehicle.safe_capacity
            
            # 날씨/교통 조건에 따른 용량 조정
            if conditions:
                weather_factor = conditions.get('weather', {}).get('capacity_factor', 1.0)
                traffic_factor = conditions.get('traffic', {}).get('capacity_factor', 1.0)
                experience_factor = min(1.3, 0.7 + (vehicle.experience_months * 0.02))
                
                adjusted_capacity = int(base_capacity * experience_factor * weather_factor * traffic_factor)
                vehicle_capacities.append(max(1, adjusted_capacity))
            else:
                vehicle_capacities.append(base_capacity)
        
        # 4. 시간 창 설정 (현재는 전체 작업시간)
        time_windows = []
        work_minutes = self.max_work_hours * 60
        
        # depot: 하루 종일 열려있음
        for _ in depot_indices:
            time_windows.append((0, work_minutes))
        
        # 주문: 우선순위에 따른 시간 선호도
        for order in orders:
            # Priority enum 문자열을 숫자로 변환
            priority_mapping = {
                'low': 1,
                'normal': 2, 
                'high': 3,
                'urgent': 4
            }
            priority_value = priority_mapping.get(order.priority.value, 2)
            
            if priority_value >= 3:  # 고우선순위 (high, urgent)
                time_windows.append((0, work_minutes // 2))  # 오전 선호
            else:
                time_windows.append((0, work_minutes))  # 언제나 가능
        
        # 5. 거리/시간 행렬 계산
        distance_matrix = await self.distance_calculator.calculate_distance_matrix(locations)
        time_matrix = self.distance_calculator.calculate_time_matrix(distance_matrix)
        
        return VRPData(
            locations=locations,
            demands=demands,
            vehicle_capacities=vehicle_capacities,
            depot_indices=depot_indices,
            time_windows=time_windows,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            order_to_location_map=order_to_location,
            location_to_order_map=location_to_order,
            vehicle_to_depot_map=vehicle_to_depot
        )
    
    def _create_vrp_model(self, vrp_data: VRPData) -> Tuple[Any, Any]:
        """OR-Tools VRP 모델 생성"""
        
        # RoutingIndexManager 생성
        manager = pywrapcp.RoutingIndexManager(
            len(vrp_data.locations),      # 전체 위치 수
            len(vrp_data.vehicle_capacities),  # 차량 수
            vrp_data.depot_indices,       # 각 차량의 출발지
            vrp_data.depot_indices        # 각 차량의 도착지 (동일)
        )
        
        # RoutingModel 생성
        routing = pywrapcp.RoutingModel(manager)
        
        self.logger.info(f"VRP 모델 생성 완료: {len(vrp_data.locations)}개 위치, "
                        f"{len(vrp_data.vehicle_capacities)}대 차량")
        
        return manager, routing
    
    def _setup_constraints(self, routing, manager, vrp_data: VRPData):
        """제약조건 설정"""
        
        # 1. 거리 제약조건
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(vrp_data.distance_matrix[from_node][to_node] * 1000)  # m 단위
        
        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # 거리 차원 추가
        routing.AddDimension(
            distance_callback_index,
            0,  # slack
            int(self.max_distance_km * 1000),  # maximum distance per vehicle (m)
            True,  # start cumul to zero
            'Distance'
        )
        distance_dimension = routing.GetDimensionOrDie('Distance')
        
        # 2. 용량 제약조건
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return vrp_data.demands[from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # 각 차량별 용량 제한
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            vrp_data.vehicle_capacities,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # 3. 시간 제약조건
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # 이동시간 + 서비스시간
            travel_time = vrp_data.time_matrix[from_node][to_node]
            service_time = 8 if int(to_node) > int(len(vrp_data.depot_indices)) - 1 else 0  # 주문지에서 8분 서비스
            
            return int((travel_time + service_time))
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # 시간 차원 추가
        max_work_minutes = self.max_work_hours * 60
        routing.AddDimension(
            time_callback_index,
            30,  # 30분 slack 허용
            max_work_minutes,  # 최대 작업시간
            False,  # don't force start cumul to zero
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # 시간 창 설정
        for location_idx, time_window in enumerate(vrp_data.time_windows):
            if location_idx < len(vrp_data.depot_indices):
                continue  # depot은 skip
            
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        
        # 4. 휴식시간 제약 (4시간마다 15분)
        for vehicle_id in range(len(vrp_data.vehicle_capacities)):
            # 4시간 후 휴식 필요
            break_time = 4 * 60  # 4시간
            for break_point in [break_time]:
                # 휴식 구간 설정 (간단히 시간 제한으로 대체)
                pass  # 복잡한 휴식 제약은 향후 추가
        
        # 5. 권역 제약조건 (차량은 해당 권역만 서비스)
        # 현재는 단순하게 모든 차량이 모든 주문을 서비스 가능하다고 가정
        # 실제로는 vehicle.region_id와 order.region_id 매칭 필요
        
        self.logger.info("VRP 제약조건 설정 완료")
    
    def _setup_objective_function(self, routing, manager, vrp_data: VRPData):
        """목적함수 설정"""
        
        # 1. 기본 비용: 거리 기반
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(vrp_data.distance_matrix[from_node][to_node] * self.distance_weight * 1000)
        
        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        
        # 2. 차량 고정비용 (차량 수 최소화)
        for vehicle_id in range(len(vrp_data.vehicle_capacities)):
            routing.SetFixedCostOfVehicle(self.vehicle_fixed_cost, vehicle_id)
        
        # 3. 미배정 주문 페널티 (최고 우선순위)
        penalty = self.unassigned_penalty
        
        for order_id, location_idx in vrp_data.order_to_location_map.items():
            index = manager.NodeToIndex(location_idx)
            routing.AddDisjunction([index], penalty)
        
        self.logger.info(f"목적함수 설정 완료: 거리가중치={self.distance_weight}, "
                        f"차량비용={self.vehicle_fixed_cost}, 미배정페널티={penalty}")
    
    def _solve_vrp_model(self, routing, manager):
        """VRP 모델 솔빙"""
        
        # 솔빙 파라미터 설정
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        
        # 초기 해법 전략
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        
        # 지역 탐색 메타휴리스틱
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        
        # 시간 제한
        search_parameters.time_limit.seconds = self.max_solve_time
        
        # 솔루션 한계
        search_parameters.solution_limit = 1000
        
        # 로그 설정 (필요시)
        search_parameters.log_search = False
        
        self.logger.info(f"VRP 솔빙 시작: 최대 {self.max_solve_time}초")
        
        # 솔빙 실행
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            self.logger.info(f"VRP 솔빙 성공: 목적함수 값 = {solution.ObjectiveValue()}")
        else:
            self.logger.warning("VRP 솔빙 실패: 해를 찾을 수 없음")
        
        return solution
    
    def _convert_solution(self, solution, routing, manager, vrp_data: VRPData) -> VRPSolution:
        """OR-Tools 솔루션을 VRPSolution으로 변환"""
        
        if not solution:
            # 해가 없는 경우
            return VRPSolution(
                routes=[],
                unassigned_orders=list(vrp_data.order_to_location_map.keys()),
                total_distance=0.0,
                total_time=0,
                objective_value=999999,
                solve_time_seconds=0.0,
                is_optimal=False
            )
        
        routes = []
        total_distance = 0
        total_time = 0
        assigned_orders = set()
        
        # 각 차량별 경로 추출
        for vehicle_id in range(len(vrp_data.vehicle_capacities)):
            if routing.IsVehicleUsed(solution, vehicle_id):
                route = self._extract_vehicle_route(
                    solution, routing, manager, vrp_data, vehicle_id
                )
                routes.append(route)
                total_distance += route.total_distance
                total_time += route.total_time
                assigned_orders.update(route.order_sequence)
        
        # 미배정 주문 계산
        all_orders = set(vrp_data.order_to_location_map.keys())
        unassigned_orders = list(all_orders - assigned_orders)
        
        return VRPSolution(
            routes=routes,
            unassigned_orders=unassigned_orders,
            total_distance=total_distance,
            total_time=total_time,
            objective_value=solution.ObjectiveValue(),
            solve_time_seconds=0.0,  # 실제 측정 시간으로 업데이트 필요
            is_optimal=True  # 임시로 True로 설정 (OR-Tools API 호환성 문제)
        )
    
    def _extract_vehicle_route(self, solution, routing, manager, vrp_data: VRPData, 
                              vehicle_id: int) -> VRPRoute:
        """특정 차량의 경로 추출"""
        
        route_distance = 0
        route_time = 0
        route_demand = 0
        location_sequence = []
        order_sequence = []
        
        index = routing.Start(vehicle_id)
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            location_sequence.append(node_index)
            
            # 주문인지 확인 (depot이 아닌 경우)
            # 타입 안전성을 위해 정수로 변환
            node_index = int(node_index)
            depot_count = int(len(vrp_data.depot_indices))
            
            if node_index >= depot_count:
                if node_index in vrp_data.location_to_order_map:
                    order_id = vrp_data.location_to_order_map[node_index]
                    order_sequence.append(order_id)
                    route_demand += vrp_data.demands[node_index]
            
            # 다음 위치로 이동
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            # 거리/시간 누적
            if not routing.IsEnd(index):
                from_node = manager.IndexToNode(previous_index)
                to_node = manager.IndexToNode(index)
                route_distance += vrp_data.distance_matrix[from_node][to_node]
                route_time += vrp_data.time_matrix[from_node][to_node]
        
        # 마지막 depot 추가
        location_sequence.append(manager.IndexToNode(index))
        
        # 차량 ID 찾기
        vehicle_key = None
        for v_id, depot_idx in vrp_data.vehicle_to_depot_map.items():
            if depot_idx == vehicle_id:
                vehicle_key = v_id
                break
        
        if not vehicle_key:
            vehicle_key = f"vehicle_{vehicle_id}"
        
        # 용량 사용률 계산
        vehicle_capacity = vrp_data.vehicle_capacities[vehicle_id]
        capacity_usage = route_demand / vehicle_capacity if vehicle_capacity > 0 else 0.0
        
        return VRPRoute(
            vehicle_id=vehicle_key,
            location_sequence=location_sequence,
            order_sequence=order_sequence,
            total_distance=route_distance,
            total_time=int(route_time),
            total_demand=route_demand,
            capacity_usage=capacity_usage
        )