"""
좌표 접근을 위한 헬퍼 함수들
"""

def get_order_latitude(order):
    """주문 객체에서 안전하게 위도를 가져오기"""
    if hasattr(order, 'coordinates') and order.coordinates:
        return order.coordinates.latitude
    elif hasattr(order, 'latitude'):
        return order.latitude
    else:
        raise AttributeError(f"Order 객체에서 위도를 찾을 수 없습니다: {type(order)}")

def get_order_longitude(order):
    """주문 객체에서 안전하게 경도를 가져오기"""
    if hasattr(order, 'coordinates') and order.coordinates:
        return order.coordinates.longitude
    elif hasattr(order, 'longitude'):
        return order.longitude
    else:
        raise AttributeError(f"Order 객체에서 경도를 찾을 수 없습니다: {type(order)}")

def get_order_coordinates(order):
    """주문 객체에서 좌표 튜플을 안전하게 가져오기"""
    return [get_order_latitude(order), get_order_longitude(order)]

def get_order_id(order):
    """주문 객체에서 안전하게 ID를 가져오기"""
    if hasattr(order, 'order_id'):
        return order.order_id
    elif hasattr(order, 'id'):
        return order.id
    else:
        return 'N/A'
