import math
import numpy as np
from sklearn.cluster import KMeans
from .vrp import solve_cvrp


def required_trucks(total_items, capacity_per_truck=50):
    return math.ceil(total_items / capacity_per_truck)


def cluster_points(points, max_per_vehicle=50):
    total_items = len(points)
    num_vehicles = required_trucks(total_items, max_per_vehicle)
    coords = np.array([[p["latitude"], p["longitude"]] for p in points])
    kmeans = KMeans(n_clusters=num_vehicles, random_state=42)
    labels = kmeans.fit_predict(coords)
    clustered_points = [[] for _ in range(num_vehicles)]
    for point, label in zip(points, labels):
        clustered_points[label].append(point)
    return clustered_points, num_vehicles


def run_optimization(clustered_points, max_per_vehicle=50):
    for i, cluster in enumerate(clustered_points):
        print(f"\n🔹 클러스터 {i+1} (지점 수: {len(cluster)}개)")
        solve_cvrp(cluster, vehicle_capacity=max_per_vehicle)