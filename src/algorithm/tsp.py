from itertools import permutations
from src.model.distance import distance

def solve_tsp(points):
    """
    Brute-force TSP solver for a list of points (dicts with 'latitude' and 'longitude').
    Returns the best order of points and the total distance.
    """
    best_order = None
    min_cost = float('inf')
    
    for perm in permutations(points):
        cost = 0
        for i in range(len(perm) - 1):
            cost += distance(
                perm[i]['latitude'], perm[i]['longitude'],
                perm[i+1]['latitude'], perm[i+1]['longitude'],
            )
        if cost < min_cost:
            min_cost = cost
            best_order = perm

    return list(best_order), min_cost

def two_opt(path, dist_func):
    """
    2-opt heuristic for improving an initial TSP route.
    path: list of points (dicts with 'latitude' and 'longitude')
    dist_func: function that takes lat1, lng1, lat2, lng2 and returns distance
    """
    best = path
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                # Compute old and new costs
                old_cost = sum(
                    dist_func(best[k]['latitude'], best[k]['longitude'],
                              best[k+1]['latitude'], best[k+1]['longitude'])
                    for k in range(len(best)-1)
                )
                new_cost = sum(
                    dist_func(new_route[k]['latitude'], new_route[k]['longitude'],
                              new_route[k+1]['latitude'], new_route[k+1]['longitude'])
                    for k in range(len(new_route)-1)
                )
                if new_cost < old_cost:
                    best = new_route
                    improved = True
        path = best
    return best
