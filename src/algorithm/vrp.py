import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def solve_vrp(vehicles, orders):
    """
    Solve a VRP given vehicles and orders.
    vehicles: list of dicts with 'capacity'
    orders: list of dicts with 'latitude', 'longitude', 'demand'
    """
    if not vehicles or not orders:
        print("🚫 차량 또는 주문 데이터가 없습니다.")
        return
    locations = [[o["latitude"], o["longitude"]] for o in orders]
    demands = [o["demand"] for o in orders]
    depot = 0
    distance_matrix = compute_euclidean_distance_matrix(locations)

    num_vehicles = len(vehicles)
    capacities = [v["capacity"] for v in vehicles]

    manager = pywrapcp.RoutingIndexManager(len(locations), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]
    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb,
        0,
        capacities,
        True,
        "Capacity"
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    search_params.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_params)

    if solution:
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_load = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                route_load += demands[node]
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            print(f"🚚 차량 {vehicle_id}: 경로={route}, 적재량={route_load}")
    else:
        print("❌ VRP 해를 찾지 못했습니다.")




def compute_euclidean_distance_matrix(locations):
    size = len(locations)
    matrix = {}
    for from_idx in range(size):
        matrix[from_idx] = {}
        for to_idx in range(size):
            if from_idx == to_idx:
                matrix[from_idx][to_idx] = 0
            else:
                dx = locations[from_idx][0] - locations[to_idx][0]
                dy = locations[from_idx][1] - locations[to_idx][1]
                matrix[from_idx][to_idx] = int(math.hypot(dx, dy) * 1e6)
    return matrix


def solve_cvrp(cluster, vehicle_capacity=50):
    locations = [[p["latitude"], p["longitude"]] for p in cluster]
    demands = [1] * len(cluster)
    depot = 0
    distance_matrix = compute_euclidean_distance_matrix(locations)

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [vehicle_capacity],
        True,
        "Capacity"
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        print(f"🚚 최적 경로: {route}")
    else:
        print("❌ 경로를 찾을 수 없습니다.")