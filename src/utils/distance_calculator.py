from math import radians, sin, cos, sqrt, atan2
import numpy as np
from typing import List, Tuple, Union
from geopy.distance import geodesic

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 지점 간의 Haversine 거리를 계산합니다.
    
    Args:
        lat1: 첫 번째 지점의 위도
        lon1: 첫 번째 지점의 경도
        lat2: 두 번째 지점의 위도
        lon2: 두 번째 지점의 경도
        
    Returns:
        float: 두 지점 간의 거리 (km)
    """
    R = 6371  # 지구의 반경 (km)

    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = radians(lat2), radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance 

def calculate_distance_geodesic(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    geopy를 사용한 정확한 거리 계산 (fallback 포함)
    
    Args:
        lat1, lon1: 첫 번째 지점의 좌표
        lat2, lon2: 두 번째 지점의 좌표
        
    Returns:
        float: 거리 (km)
    """
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except:
        # geopy 실패 시 하버사인 공식 사용
        return calculate_distance(lat1, lon1, lat2, lon2)

def calculate_distances_batch(coords1: List[Tuple[float, float]], 
                            coords2: List[Tuple[float, float]]) -> np.ndarray:
    """
    배치 거리 계산 (성능 최적화)
    
    Args:
        coords1: 첫 번째 좌표 리스트 [(lat, lon), ...]
        coords2: 두 번째 좌표 리스트 [(lat, lon), ...]
        
    Returns:
        np.ndarray: 거리 행렬 (km)
    """
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    
    # 라디안 변환
    coords1_rad = np.radians(coords1)
    coords2_rad = np.radians(coords2)
    
    # 브로드캐스팅을 위한 차원 확장
    lat1 = coords1_rad[:, 0:1]  # (n, 1)
    lon1 = coords1_rad[:, 1:2]  # (n, 1)
    lat2 = coords2_rad[:, 0].T   # (1, m)
    lon2 = coords2_rad[:, 1].T   # (1, m)
    
    # 하버사인 공식 벡터화
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return 6371 * c  # 지구 반지름 곱하기

def find_nearest_point(target_lat: float, target_lon: float, 
                      candidates: List[Tuple[float, float]]) -> Tuple[int, float]:
    """
    가장 가까운 지점 찾기 (최적화된 버전)
    
    Args:
        target_lat, target_lon: 대상 지점 좌표
        candidates: 후보 지점들 [(lat, lon), ...]
        
    Returns:
        Tuple[int, float]: (가장 가까운 지점의 인덱스, 거리)
    """
    if not candidates:
        return -1, float('inf')
    
    # 벡터화된 거리 계산
    target_coords = [(target_lat, target_lon)]
    distances = calculate_distances_batch(target_coords, candidates)[0]
    
    min_idx = np.argmin(distances)
    return int(min_idx), float(distances[min_idx])

def assign_points_to_nearest_centers(points: List[Tuple[float, float]], 
                                   centers: List[Tuple[float, float]]) -> List[int]:
    """
    각 포인트를 가장 가까운 센터에 할당 (벡터화)
    
    Args:
        points: 할당할 포인트들 [(lat, lon), ...]
        centers: 센터 포인트들 [(lat, lon), ...]
        
    Returns:
        List[int]: 각 포인트가 할당된 센터의 인덱스
    """
    if not points or not centers:
        return []
    
    # 모든 포인트와 모든 센터 간의 거리 계산
    distances = calculate_distances_batch(points, centers)
    
    # 각 포인트에 대해 가장 가까운 센터 찾기
    assignments = np.argmin(distances, axis=1)
    
    return assignments.tolist()

# 하위 호환성을 위한 별칭
distance = calculate_distance
haversine_distance = calculate_distance 