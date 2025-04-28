from src.model.delivery_point import DeliveryPoint
from math import radians, sin, cos, sqrt, atan2

def calculate_distance(point1: DeliveryPoint, point2: DeliveryPoint) -> float:
    """
    두 지점 간의 Haversine 거리를 계산합니다.
    """
    R = 6371  # 지구의 반경 (km)

    lat1, lon1 = radians(point1.latitude), radians(point1.longitude)
    lat2, lon2 = radians(point2.latitude), radians(point2.longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance 