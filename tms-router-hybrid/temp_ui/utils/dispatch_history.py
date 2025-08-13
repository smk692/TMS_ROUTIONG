"""
배차 이력 조회 유틸리티
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.database.connection import db_session
from core.database.models import (
    DispatchBatch, VehicleAssignment, OrderAssignment, 
    Order, Center
)
from core.models.map_display_result import MapDisplayResult, VehicleAssignmentResult
from core.models.coordinates import Coordinates
from core.models.order import Order as OrderModel
from sqlalchemy.orm import joinedload
from sqlalchemy import desc, and_


class DispatchHistoryManager:
    """배차 이력 관리 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_recent_dispatch_batches(self, 
                                    center_id: Optional[str] = None,
                                    limit: int = 10,
                                    days_back: int = 30) -> List[Dict[str, Any]]:
        """최근 배차 배치 목록 조회"""
        try:
            with db_session() as session:
                query = session.query(DispatchBatch)
                
                # 센터 필터
                if center_id:
                    query = query.filter(DispatchBatch.center_id == center_id)
                
                # 날짜 필터 (최근 N일)
                since_date = datetime.now() - timedelta(days=days_back)
                query = query.filter(DispatchBatch.created_at >= since_date)
                
                # 정렬 및 제한
                batches = query.order_by(desc(DispatchBatch.created_at)).limit(limit).all()
                
                return [
                    {
                        'batch_id': batch.batch_id,
                        'center_id': batch.center_id,
                        'status': batch.status,
                        'total_orders': batch.total_orders,
                        'assigned_orders': batch.assigned_orders,
                        'used_vehicles': batch.used_vehicles,
                        'total_vehicles': batch.total_vehicles,
                        'algorithm_used': batch.algorithm_used,
                        'execution_time': float(batch.execution_time_seconds),
                        'created_at': batch.created_at,
                        'completed_at': batch.completed_at
                    } for batch in batches
                ]
                
        except Exception as e:
            self.logger.error(f"배차 배치 목록 조회 실패: {str(e)}")
            return []
    
    def get_dispatch_result_by_batch_id(self, batch_id: str) -> Optional[MapDisplayResult]:
        """배치 ID로 배차 결과 조회"""
        try:
            with db_session() as session:
                # 배치 정보 조회
                batch = session.query(DispatchBatch).filter(
                    DispatchBatch.batch_id == batch_id
                ).first()
                
                if not batch:
                    self.logger.warning(f"배치를 찾을 수 없습니다: {batch_id}")
                    return None
                
                # 센터 정보 조회
                center = session.query(Center).filter(
                    Center.id == batch.center_id
                ).first()
                
                # 차량 배정 정보 조회
                vehicle_assignments = session.query(VehicleAssignment).filter(
                    VehicleAssignment.batch_id == batch_id
                ).all()
                
                # 주문 배정 정보 조회
                order_assignments = session.query(OrderAssignment).options(
                    joinedload(OrderAssignment.order),
                    joinedload(OrderAssignment.vehicle)
                ).filter(
                    OrderAssignment.batch_id == batch_id
                ).order_by(
                    OrderAssignment.vehicle_id, 
                    OrderAssignment.assignment_order
                ).all()
                
                # 미배정 주문 조회
                assigned_order_ids = [oa.order_id for oa in order_assignments]
                unassigned_orders = session.query(Order).filter(
                    and_(
                        Order.center_id == batch.center_id,
                        Order.status == 'pending',
                        ~Order.id.in_(assigned_order_ids) if assigned_order_ids else True
                    )
                ).all() if assigned_order_ids else []
                
                # MapDisplayResult 객체 생성
                return self._build_dispatch_result(
                    batch, center, vehicle_assignments, order_assignments, unassigned_orders
                )
                
        except Exception as e:
            self.logger.error(f"배차 결과 조회 실패 (batch_id: {batch_id}): {str(e)}")
            return None
    
    def _build_dispatch_result(self, 
                               batch: DispatchBatch,
                               center: Center,
                               vehicle_assignments: List[VehicleAssignment],
                               order_assignments: List[OrderAssignment],
                               unassigned_orders: List[Order]) -> MapDisplayResult:
        """데이터베이스 데이터로부터 MapDisplayResult 객체 생성"""
        
        # 센터 좌표
        center_coords = Coordinates(
            latitude=float(center.latitude),
            longitude=float(center.longitude)
        ) if center else None
        
        # 차량별 주문 그룹화
        vehicle_order_map = {}
        for oa in order_assignments:
            if oa.vehicle_id not in vehicle_order_map:
                vehicle_order_map[oa.vehicle_id] = []
            vehicle_order_map[oa.vehicle_id].append(oa)
        
        # 차량 배정 결과 생성
        vehicle_assignment_results = []
        for va in vehicle_assignments:
            orders_for_vehicle = vehicle_order_map.get(va.vehicle_id, [])
            
            # 주문 객체 변환
            assigned_orders = []
            route_coordinates = []
            if center_coords:
                route_coordinates.append([center_coords.latitude, center_coords.longitude])
            
            for oa in sorted(orders_for_vehicle, key=lambda x: x.assignment_order):
                order = oa.order
                order_coords = Coordinates(
                    latitude=float(order.latitude),
                    longitude=float(order.longitude)
                )
                order_model = OrderModel(
                    id=order.id,
                    center_id=order.center_id,
                    region_id=order.region_id,
                    coordinates=order_coords,
                    address=order.address,
                    priority=order.priority,
                    status=order.status,
                    estimated_delivery_time=oa.estimated_delivery_minutes
                )
                assigned_orders.append(order_model)
                route_coordinates.append([float(order.latitude), float(order.longitude)])
            
            # 색상 할당 (차량 인덱스 기반)
            colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue']
            color_index = len(vehicle_assignment_results) % len(colors)
            
            vehicle_result = VehicleAssignmentResult(
                vehicle_id=va.vehicle_id,
                driver_name=va.driver_name,
                vehicle_type=va.vehicle_type,
                region_name=va.region_name,
                assigned_orders=assigned_orders,
                route_coordinates=route_coordinates,
                estimated_distance_km=float(va.estimated_distance_km),
                estimated_time_minutes=va.estimated_time_minutes,
                capacity_utilization=float(va.capacity_utilization),
                color=colors[color_index]
            )
            vehicle_assignment_results.append(vehicle_result)
        
        # 미배정 주문 변환
        unassigned_order_models = []
        for order in unassigned_orders:
            order_coords = Coordinates(
                latitude=float(order.latitude),
                longitude=float(order.longitude)
            )
            order_model = OrderModel(
                id=order.id,
                center_id=order.center_id,
                region_id=order.region_id,
                coordinates=order_coords,
                address=order.address,
                priority=order.priority,
                status=order.status
            )
            unassigned_order_models.append(order_model)
        
        # 총 거리 및 시간 계산 (안전한 처리)
        total_distance = sum(float(va.estimated_distance_km or 0) for va in vehicle_assignments)
        total_time = sum(int(va.estimated_time_minutes or 0) for va in vehicle_assignments)
        
        # MapDisplayResult 생성
        # 센터 정보를 별도로 처리
        center_info = None
        if center:
            center_coords = Coordinates(
                latitude=float(center.latitude),
                longitude=float(center.longitude)
            )
            # center 정보를 담은 객체 생성 (Coordinates를 확장하기 위해 임시 클래스 사용)
            class CenterInfo:
                def __init__(self, coords, name, center_id, address):
                    self.latitude = coords.latitude
                    self.longitude = coords.longitude
                    self.name = name
                    self.center_id = center_id
                    self.address = address
            
            center_info = CenterInfo(center_coords, center.name, center.id, center.address)
        
        return MapDisplayResult(
            center=center_info,
            vehicle_assignments=vehicle_assignment_results,
            unassigned_orders=unassigned_order_models,
            total_orders=batch.total_orders,
            assigned_orders=batch.assigned_orders,
            total_vehicles=batch.total_vehicles,
            used_vehicles=batch.used_vehicles,
            total_distance=float(total_distance),
            total_time=int(total_time),
            algorithm_used=batch.algorithm_used or "Unknown",
            execution_time=float(batch.execution_time_seconds),
            batch_id=batch.batch_id,
            created_at=batch.created_at
        )
    
    def get_centers_list(self) -> List[Dict[str, Any]]:
        """센터 목록 조회"""
        try:
            with db_session() as session:
                centers = session.query(Center).filter(Center.is_active == True).all()
                return [
                    {
                        'id': center.id,
                        'name': center.name,
                        'address': center.address
                    } for center in centers
                ]
        except Exception as e:
            self.logger.error(f"센터 목록 조회 실패: {str(e)}")
            return []
