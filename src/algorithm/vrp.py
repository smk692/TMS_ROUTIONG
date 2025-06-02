# src/algorithm/vrp.py
from typing import List, Optional
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from src.model.vehicle import Vehicle
from src.model.delivery_point import DeliveryPoint
from src.model.route import Route, RoutePoint
from src.core.logger import setup_logger
from src.core.distance_matrix import DistanceMatrix
from datetime import timedelta

logger = setup_logger('vrp')
distance_matrix = DistanceMatrix()

class VRPSolver:
    """VRP 문제 해결을 위한 클래스"""
    def __init__(self, vehicles: List[Vehicle], delivery_points: List[DeliveryPoint]):
        self.vehicles = vehicles
        self.delivery_points = delivery_points
        self.distance_matrix = distance_matrix.compute_matrix(delivery_points)
        self.manager = None
        self.routing = None

    def validate_input(self) -> bool:
        """입력 데이터 검증"""
        if not self.vehicles:
            logger.error("차량 데이터가 없습니다.")
            return False
        if not self.delivery_points:
            logger.error("배송지점 데이터가 없습니다.")
            return False
            
        # 기본 유효성 검사
        total_volume = sum(p.volume for p in self.delivery_points)
        total_weight = sum(p.weight for p in self.delivery_points)
        total_vehicle_volume = sum(v.capacity.volume for v in self.vehicles)
        total_vehicle_weight = sum(v.capacity.weight for v in self.vehicles)
        
        # 품목 부피 정보가 있는 경우에만 부피 제약 확인
        if any(p.volume > 0 for p in self.delivery_points):
            if total_volume > total_vehicle_volume:
                logger.error(f"총 화물 부피({total_volume})가 차량 용량({total_vehicle_volume})을 초과합니다.")
                return False
                
        # 품목 무게 정보가 있는 경우에만 무게 제약 확인
        if any(p.weight > 0 for p in self.delivery_points):
            if total_weight > total_vehicle_weight:
                logger.error(f"총 화물 무게({total_weight})가 차량 용량({total_vehicle_weight})을 초과합니다.")
                return False
            
        return True
    
    def setup_model(self):
        """OR-Tools 모델 설정"""
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.delivery_points),
            len(self.vehicles),
            0  # depot
        )
        self.routing = pywrapcp.RoutingModel(self.manager)
        
        # 거리 콜백 등록
        def distance_callback(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return int(self.distance_matrix[from_node][to_node] * 1e6)
            
        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # 용량 제약 추가
        def demand_callback(from_index):
            point = self.delivery_points[self.manager.IndexToNode(from_index)]
            return int(point.volume * 1000)
            
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            [int(v.capacity.volume * 1000) for v in self.vehicles],
            True,
            "Capacity"
        )
        
        # 시간 제약 추가
        time_callback_index = self.routing.RegisterTransitCallback(
            lambda from_index, to_index: int(distance_callback(from_index, to_index) / 1e6 * 60)  # 분 단위
        )
        self.routing.AddDimension(
            time_callback_index,
            30,  # 허용 대기 시간
            480,  # 최대 근무 시간 (8시간)
            False,
            "Time"
        )
    
    def solve(self) -> Optional[List[Route]]:
        """VRP 해결"""
        try:
            logger.info("VRP 해결 시작")
            
            if not self.validate_input():
                return None
                
            self.setup_model()
            
            # 해 찾기 파라미터 최적화
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            
            # 첫 번째 해결책 전략 개선
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
            )
            
            # 메타휴리스틱 알고리즘 개선
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
            )
            
            # 시간 제한 및 기타 파라미터 조정
            search_parameters.time_limit.seconds = 15  # 시간 제한 감소
            search_parameters.solution_limit = 100    # 해의 개수 제한
            search_parameters.log_search = True       # 검색 로깅 활성화
            
            # 병렬 처리 설정
            search_parameters.num_search_workers = 4  # 작업자 수 설정
            
            solution = self.routing.SolveWithParameters(search_parameters)
            
            if not solution:
                logger.error("해를 찾을 수 없습니다.")
                return None
                
            # 결과 변환
            routes = self._convert_solution_to_routes(solution)
            
            logger.info(f"VRP 해결 완료: {len(routes)}개 경로 생성")
            return routes
            
        except Exception as e:
            logger.error(f"VRP 해결 중 오류 발생: {str(e)}")
            raise
    
    def _convert_solution_to_routes(self, solution) -> List[Route]:
        """OR-Tools 해를 Route 객체로 변환"""
        routes = []
        for vehicle_id in range(len(self.vehicles)):
            vehicle = self.vehicles[vehicle_id]
            route_points = []
            
            index = self.routing.Start(vehicle_id)
            if self.routing.IsEnd(index):
                continue
                
            current_time = vehicle.start_time
            cumulative_distance = 0.0
            
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                point = self.delivery_points[node_index]
                
                route_point = RoutePoint(
                    point=point,
                    arrival_time=current_time,
                    departure_time=current_time + timedelta(minutes=point.service_time),
                    cumulative_distance=cumulative_distance,
                    cumulative_load={
                        'volume': point.volume,
                        'weight': point.weight
                    }
                )
                route_points.append(route_point)
                
                current_time += timedelta(minutes=point.service_time)
                next_index = solution.Value(self.routing.NextVar(index))
                
                if not self.routing.IsEnd(next_index):
                    next_node = self.manager.IndexToNode(next_index)
                    dist = self.distance_matrix[node_index][next_node]
                    cumulative_distance += dist
                    current_time += timedelta(minutes=int(dist * 60))  # km/h 기준
                
                index = next_index
            
            if route_points:
                route = Route(
                    id=f"R{vehicle_id}",
                    vehicle=vehicle,
                    points=route_points,
                    total_distance=cumulative_distance,
                    total_time=int((current_time - vehicle.start_time).total_seconds() / 60),
                    total_load={
                        'volume': sum(p.point.volume for p in route_points),
                        'weight': sum(p.point.weight for p in route_points)
                    },
                    start_time=vehicle.start_time,
                    end_time=current_time,
                    status='PLANNED'
                )
                routes.append(route)
        
        return routes

def solve_vrp(vehicles: List[Vehicle], delivery_points: List[DeliveryPoint]) -> Optional[List[Route]]:
    """VRP 해결을 위한 편의 함수"""
    solver = VRPSolver(vehicles, delivery_points)
    return solver.solve()