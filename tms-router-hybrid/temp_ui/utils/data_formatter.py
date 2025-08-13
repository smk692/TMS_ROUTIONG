"""
데이터 포맷팅 유틸리티
"""
import pandas as pd
import json
from typing import List, Dict, Any
from datetime import datetime
from decimal import Decimal


def format_time(seconds: float) -> str:
    """시간 포맷팅"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}분"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}시간"


def format_percentage(value: float) -> str:
    """퍼센트 포맷팅"""
    return f"{value * 100:.1f}%"


def format_distance(km: float) -> str:
    """거리 포맷팅"""
    if km < 1:
        return f"{km * 1000:.0f}m"
    return f"{km:.1f}km"


def create_dispatch_dataframe(vehicle_assignments: List) -> pd.DataFrame:
    """배차 결과를 DataFrame으로 변환"""
    data = []
    for assignment in vehicle_assignments:
        data.append({
            '차량 ID': assignment.vehicle_id,
            '기사명': assignment.driver_name,
            '차량 유형': assignment.vehicle_type,
            '권역': assignment.region_name,
            '배정 주문 수': len(assignment.assigned_orders),
            '예상 거리': format_distance(assignment.estimated_distance_km),
            '예상 시간': f"{assignment.estimated_time_minutes}분",
            '용량 활용도': format_percentage(assignment.capacity_utilization)
        })
    
    return pd.DataFrame(data)


def create_orders_dataframe(orders: List) -> pd.DataFrame:
    """주문 목록을 DataFrame으로 변환"""
    data = []
    for order in orders:
        data.append({
            '주문 ID': order.order_id,
            '센터': order.center_id,
            '권역': order.region_id,
            '주소': order.address,
            '우선순위': order.priority,
            '상태': order.status,
            '생성일시': order.created_at.strftime('%Y-%m-%d %H:%M') if order.created_at else ''
        })
    
    return pd.DataFrame(data)


def create_history_dataframe(history: List[Dict]) -> pd.DataFrame:
    """배차 이력을 DataFrame으로 변환"""
    data = []
    for item in history:
        data.append({
            '배치 ID': item['batch_id'],
            '센터': item['center_id'],
            '상태': item['status'],
            '총 주문': item['total_orders'],
            '배정 주문': item['assigned_orders'],
            '차량 수': f"{item['used_vehicles']}/{item['total_vehicles']}",
            '실행 시간': format_time(item['execution_time']),
            '실행 일시': item['created_at']
        })
    
    return pd.DataFrame(data)


class DecimalEncoder(json.JSONEncoder):
    """Decimal 타입을 JSON 직렬화 가능한 형태로 변환하는 인코더"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


def get_status_color(status: str) -> str:
    """상태별 색상 반환"""
    status_colors = {
        'success': '#28a745',
        'partial_success': '#ffc107',
        'failed': '#dc3545',
        'pending': '#6c757d',
        'assigned': '#17a2b8',
        'completed': '#28a745',
        'cancelled': '#dc3545'
    }
    return status_colors.get(status.lower(), '#6c757d')


def get_priority_emoji(priority: str) -> str:
    """우선순위별 이모지 반환"""
    priority_emojis = {
        'urgent': '🔴',
        'high': '🟠',
        'normal': '🟢',
        'low': '🔵'
    }
    return priority_emojis.get(priority.lower(), '⚪')