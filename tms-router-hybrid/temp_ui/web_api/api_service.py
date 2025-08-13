"""
TMS API 서비스 - Streamlit과 기존 시스템 연결
"""
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from core.services import DispatchOrchestrator, DataCollector
from core.database.connection import get_session
from core.database.models import Center, Region, Vehicle, Order as DBOrder, DispatchBatch
from core.models import DispatchStatus
from .data_models import (
    WebDispatchResult, WebVehicleAssignment, WebOrder, WebCenter
)
from core.config import get_settings


class TmsApiService:
    """TMS 시스템을 웹에서 사용하기 위한 API 래핑"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        
        # 서비스 초기화
        config = {
            'database_url': self.settings.database_url,
            'weather_api_key': self.settings.external_api.openweather_api_key,
            'traffic_api_key': self.settings.external_api.here_api_key
        }
        
        self.orchestrator = DispatchOrchestrator(config)
        self.data_collector = DataCollector(config)
        
        # 색상 팔레트 (차량별 구분용)
        self.colors = [
            'blue', 'green', 'purple', 'orange', 'darkred',
            'lightred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple'
        ]
    
    def execute_dispatch(self, center_id: str, algorithm: str = 'auto') -> WebDispatchResult:
        """동기 배차 실행"""
        try:
            # 기존 시스템의 배차 실행
            result = self.orchestrator.execute_dispatch(center_id=center_id)
            
            # 웹 모델로 변환
            web_result = self._convert_to_web_result(result, center_id)
            return web_result
            
        except Exception as e:
            self.logger.error(f"배차 실행 오류: {str(e)}")
            return self._create_error_result(str(e), center_id)
    
    async def execute_dispatch_async(self, center_id: str, algorithm: str = 'auto') -> WebDispatchResult:
        """비동기 배차 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_dispatch, center_id, algorithm)
    
    def get_centers_list(self) -> List[Dict]:
        """물류센터 목록 조회"""
        session = get_session()
        try:
            centers = session.query(Center).filter(Center.is_active == True).all()
            return [
                {
                    'center_id': c.id,
                    'name': c.name,
                    'address': c.address,
                    'latitude': float(c.latitude),
                    'longitude': float(c.longitude)
                }
                for c in centers
            ]
        finally:
            session.close()
    
    def get_regions_by_center(self, center_id: str) -> List[Dict]:
        """센터별 권역 목록 조회"""
        session = get_session()
        try:
            regions = session.query(Region).filter(
                Region.center_id == center_id,
                Region.is_active == True
            ).all()
            return [
                {
                    'region_id': r.id,
                    'name': r.name,
                    'center_latitude': float(r.center_latitude),
                    'center_longitude': float(r.center_longitude)
                }
                for r in regions
            ]
        finally:
            session.close()
    
    def get_dispatch_history(self, limit: int = 50) -> List[Dict]:
        """배차 이력 조회"""
        session = get_session()
        try:
            batches = session.query(DispatchBatch).order_by(
                DispatchBatch.created_at.desc()
            ).limit(limit).all()
            
            history = []
            for batch in batches:
                # enum 값을 문자열로 변환
                status_value = batch.status.value if hasattr(batch.status, 'value') else str(batch.status)
                
                # Decimal 값을 float로 변환
                execution_time = float(batch.execution_time_seconds) if batch.execution_time_seconds else 0.0
                
                history.append({
                    'batch_id': batch.batch_id,
                    'center_id': batch.center_id,
                    'status': status_value,
                    'total_orders': batch.total_orders or 0,
                    'assigned_orders': batch.assigned_orders or 0,
                    'total_vehicles': batch.total_vehicles or 0,
                    'used_vehicles': batch.used_vehicles or 0,
                    'created_at': batch.created_at.isoformat() if batch.created_at else None,
                    'execution_time': execution_time
                })
            
            return history
        finally:
            session.close()
    
    def get_order_details(self, order_id: str) -> Optional[WebOrder]:
        """주문 상세 정보 조회"""
        session = get_session()
        try:
            order = session.query(DBOrder).filter(DBOrder.id == order_id).first()
            if order:
                # enum 값들을 문자열로 변환
                priority_value = order.priority.value if hasattr(order.priority, 'value') else str(order.priority)
                status_value = order.status.value if hasattr(order.status, 'value') else str(order.status)
                
                return WebOrder(
                    order_id=order.id,
                    center_id=order.center_id,
                    region_id=order.region_id,
                    address=order.address,
                    latitude=float(order.latitude),
                    longitude=float(order.longitude),
                    priority=priority_value,
                    status=status_value,
                    created_at=order.created_at
                )
            return None
        finally:
            session.close()
    
    def get_center_statistics(self, center_id: Optional[str] = None) -> Dict:
        """센터별 통계 정보"""
        session = get_session()
        try:
            # 전체 또는 특정 센터 통계
            query = session.query(DBOrder)
            if center_id:
                query = query.filter(DBOrder.center_id == center_id)
            
            pending_orders = query.filter(DBOrder.status == 'pending').count()
            assigned_orders = query.filter(DBOrder.status == 'assigned').count()
            completed_orders = query.filter(DBOrder.status == 'completed').count()
            
            # 차량 통계
            vehicle_query = session.query(Vehicle)
            if center_id:
                vehicle_query = vehicle_query.filter(Vehicle.center_id == center_id)
            
            total_vehicles = vehicle_query.count()
            active_vehicles = vehicle_query.filter(Vehicle.status == 'ACTIVE').count()
            
            return {
                'pending_orders': pending_orders,
                'assigned_orders': assigned_orders,
                'completed_orders': completed_orders,
                'total_orders': pending_orders + assigned_orders + completed_orders,
                'total_vehicles': total_vehicles,
                'active_vehicles': active_vehicles,
                'inactive_vehicles': total_vehicles - active_vehicles
            }
        finally:
            session.close()
    
    def _convert_to_web_result(self, dispatch_result, center_id: str) -> WebDispatchResult:
        """기존 배차 결과를 웹 모델로 변환"""
        # 센터 정보 조회
        session = get_session()
        try:
            center = session.query(Center).filter(Center.id == center_id).first()
            web_center = WebCenter(
                center_id=center.id,
                name=center.name,
                address=center.address,
                latitude=float(center.latitude),
                longitude=float(center.longitude),
                is_active=center.is_active
            ) if center else None
            
            # 차량 배정 정보 변환
            web_assignments = []
            for i, assignment in enumerate(dispatch_result.vehicle_assignments):
                # 배정된 주문들의 상세 정보 조회
                assigned_orders = []
                for order_id in assignment.assigned_orders:
                    order = self.get_order_details(order_id)
                    if order:
                        assigned_orders.append(order)
                
                # 경로 좌표 생성 (센터 -> 주문들 -> 센터)
                route_coords = []
                if web_center:
                    route_coords.append((web_center.latitude, web_center.longitude))
                for order in assigned_orders:
                    route_coords.append((order.latitude, order.longitude))
                if web_center and len(route_coords) > 1:
                    route_coords.append((web_center.latitude, web_center.longitude))
                
                web_assignment = WebVehicleAssignment(
                    vehicle_id=assignment.vehicle_id,
                    driver_name=assignment.driver_name,
                    vehicle_type=assignment.vehicle_type,
                    region_name=assignment.region_name,
                    assigned_orders=assigned_orders,
                    route_coordinates=route_coords,
                    estimated_distance_km=assignment.estimated_distance_km,
                    estimated_time_minutes=assignment.estimated_time_minutes,
                    capacity_utilization=assignment.capacity_utilization,
                    color=self.colors[i % len(self.colors)]
                )
                web_assignments.append(web_assignment)
            
            # 미배정 주문 정보
            unassigned_orders = []
            for order_id in dispatch_result.unassigned_orders:
                order = self.get_order_details(order_id)
                if order:
                    unassigned_orders.append(order)
            
            # 웹 결과 생성
            metrics = dispatch_result.metrics if dispatch_result.metrics else None
            
            # enum 값을 문자열로 변환
            status_value = dispatch_result.status.value if hasattr(dispatch_result.status, 'value') else str(dispatch_result.status)
            
            return WebDispatchResult(
                batch_id=dispatch_result.batch_id,
                timestamp=dispatch_result.timestamp,
                status=status_value,
                center=web_center,
                vehicle_assignments=web_assignments,
                unassigned_orders=unassigned_orders,
                total_orders=metrics.total_orders if metrics else 0,
                assigned_orders=metrics.assigned_orders if metrics else 0,
                total_vehicles=metrics.total_vehicles if metrics else 0,
                used_vehicles=metrics.used_vehicles if metrics else 0,
                total_distance=metrics.total_estimated_distance if metrics else 0,
                total_time=metrics.total_estimated_time if metrics else 0,
                execution_time=dispatch_result.execution_time_seconds,
                algorithm_used=metrics.algorithm_used if metrics else "",
                quality_score=metrics.quality_score if metrics else 0,
                error_message=dispatch_result.error_message,
                warnings=dispatch_result.warnings
            )
        finally:
            session.close()
    
    def _create_error_result(self, error_message: str, center_id: str) -> WebDispatchResult:
        """오류 결과 생성"""
        session = get_session()
        try:
            center = session.query(Center).filter(Center.id == center_id).first()
            web_center = WebCenter(
                center_id=center.id,
                name=center.name,
                address=center.address,
                latitude=float(center.latitude),
                longitude=float(center.longitude),
                is_active=center.is_active
            ) if center else None
            
            return WebDispatchResult(
                batch_id=f"ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                status='failed',
                center=web_center,
                error_message=error_message
            )
        finally:
            session.close()