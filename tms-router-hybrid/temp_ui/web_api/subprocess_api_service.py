"""
TMS API 서비스 - Subprocess 기반 Core CLI 호출
"""
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
import sys

# temp_ui 경로를 추가하여 core_executor 사용 가능하게 함
sys.path.append(str(Path(__file__).parent.parent))

from core_executor import CoreDispatchExecutor
from .data_models import (
    WebDispatchResult, WebVehicleAssignment, WebOrder, WebCenter
)

# 프로젝트 루트 경로 추가 (데이터베이스 직접 접근용)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from core.database.connection import get_session
    from core.database.models import Center, Region, Vehicle, Order as DBOrder, DispatchBatch
    from core.models import DispatchStatus
    DATABASE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"데이터베이스 모듈 import 실패: {e}")
    DATABASE_AVAILABLE = False


class SubprocessTmsApiService:
    """Subprocess 기반 TMS API 서비스 - Core와 완전히 분리된 UI"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core 실행기 초기화
        self.executor = CoreDispatchExecutor()
        
        # 색상 팔레트 (차량별 구분용)
        self.colors = [
            'blue', 'green', 'purple', 'orange', 'darkred',
            'lightred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple'
        ]
        
        # 데이터베이스 사용 가능 여부 확인
        self.db_available = DATABASE_AVAILABLE
        if not self.db_available:
            self.logger.warning("데이터베이스 직접 접근 불가 - 일부 기능 제한")
    
    def execute_dispatch(self, center_id: str, algorithm: str = 'auto') -> WebDispatchResult:
        """배차 실행 - Core CLI를 subprocess로 호출"""
        try:
            # Core CLI를 subprocess로 실행
            success, result = self.executor.execute_dispatch(
                center_id=center_id,
                algorithm=algorithm,
                dry_run=False
            )
            
            if success:
                # 성공한 경우 WebDispatchResult로 변환
                return self._convert_core_result_to_web(result, center_id)
            else:
                # 실패한 경우 오류 결과 생성
                return self._create_error_result(
                    error_message=result.get('error_message', 'Unknown error'),
                    center_id=center_id
                )
                
        except Exception as e:
            self.logger.error(f"배차 실행 오류: {str(e)}")
            return self._create_error_result(str(e), center_id)
    
    async def execute_dispatch_async(self, center_id: str, algorithm: str = 'auto') -> WebDispatchResult:
        """비동기 배차 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_dispatch, center_id, algorithm)
    
    def get_centers_list(self) -> List[Dict]:
        """물류센터 목록 조회"""
        if not self.db_available:
            # 데이터베이스 접근 불가시 더미 데이터 반환
            return [
                {
                    'center_id': 'CENTER_GANGNAM',
                    'name': '강남 물류센터',
                    'address': '서울시 강남구',
                    'latitude': 37.5665,
                    'longitude': 126.9780
                }
            ]
        
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
    
    def get_dispatch_history(self, limit: int = 50) -> List[Dict]:
        """배차 이력 조회"""
        if not self.db_available:
            return []
        
        session = get_session()
        try:
            batches = session.query(DispatchBatch).order_by(
                DispatchBatch.created_at.desc()
            ).limit(limit).all()
            
            history = []
            for batch in batches:
                # enum 값을 문자열로 변환
                status_value = batch.status if isinstance(batch.status, str) else str(batch.status)
                
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
    
    def get_center_statistics(self, center_id: Optional[str] = None) -> Dict:
        """센터별 통계 정보"""
        if not self.db_available:
            return {
                'pending_orders': 0,
                'assigned_orders': 0,
                'completed_orders': 0,
                'total_orders': 0,
                'total_vehicles': 0,
                'active_vehicles': 0,
                'inactive_vehicles': 0
            }
        
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
    
    def _convert_core_result_to_web(self, core_result: Dict, center_id: str) -> WebDispatchResult:
        """Core CLI 결과를 웹 모델로 변환"""
        
        # 센터 정보 조회 (데이터베이스 사용 가능한 경우)
        web_center = None
        if self.db_available:
            session = get_session()
            try:
                center = session.query(Center).filter(Center.id == center_id).first()
                if center:
                    web_center = WebCenter(
                        center_id=center.id,
                        name=center.name,
                        address=center.address,
                        latitude=float(center.latitude),
                        longitude=float(center.longitude),
                        is_active=center.is_active
                    )
            finally:
                session.close()
        
        # 기본 센터 정보 (데이터베이스 접근 실패시)
        if not web_center:
            web_center = WebCenter(
                center_id=center_id,
                name=f"센터 {center_id}",
                address="정보 없음",
                latitude=37.5665,
                longitude=126.9780,
                is_active=True
            )
        
        # 차량 배정 정보 변환
        web_assignments = []
        vehicle_assignments = core_result.get('vehicle_assignments', [])
        
        for i, assignment in enumerate(vehicle_assignments):
            # 주문 정보를 WebOrder 형태로 변환
            assigned_orders = []
            for order_id in assignment.get('assigned_orders', []):
                # 간단한 주문 정보 생성 (실제 구현에서는 DB에서 조회)
                order = WebOrder(
                    order_id=order_id,
                    center_id=center_id,
                    region_id=assignment.get('region_name', ''),
                    address=f"주문 {order_id} 주소",
                    latitude=37.5665 + (i * 0.001),  # 더미 좌표
                    longitude=126.9780 + (i * 0.001),
                    priority='NORMAL',
                    status='pending',
                    created_at=datetime.now()
                )
                assigned_orders.append(order)
            
            # 경로 좌표 생성
            route_coords = [(web_center.latitude, web_center.longitude)]
            for order in assigned_orders:
                route_coords.append((order.latitude, order.longitude))
            if len(route_coords) > 1:
                route_coords.append((web_center.latitude, web_center.longitude))
            
            web_assignment = WebVehicleAssignment(
                vehicle_id=assignment.get('vehicle_id', ''),
                driver_name=assignment.get('driver_name', ''),
                vehicle_type=assignment.get('vehicle_type', ''),
                region_name=assignment.get('region_name', ''),
                assigned_orders=assigned_orders,
                route_coordinates=route_coords,
                estimated_distance_km=assignment.get('estimated_distance_km', 0),
                estimated_time_minutes=assignment.get('estimated_time_minutes', 0),
                capacity_utilization=assignment.get('capacity_utilization', 0),
                color=self.colors[i % len(self.colors)]
            )
            web_assignments.append(web_assignment)
        
        # 메트릭스 정보
        metrics = core_result.get('metrics', {})
        
        return WebDispatchResult(
            batch_id=core_result.get('batch_id', ''),
            timestamp=datetime.now(),
            status=core_result.get('status', 'unknown'),
            center=web_center,
            vehicle_assignments=web_assignments,
            unassigned_orders=[],  # TODO: 미배정 주문 처리
            total_orders=metrics.get('total_orders', 0),
            assigned_orders=metrics.get('assigned_orders', 0),
            total_vehicles=metrics.get('total_vehicles', 0),
            used_vehicles=metrics.get('used_vehicles', 0),
            total_distance=metrics.get('total_estimated_distance', 0),
            total_time=metrics.get('total_estimated_time', 0),
            execution_time=core_result.get('execution_time_seconds', 0),
            algorithm_used=metrics.get('algorithm_used', ''),
            quality_score=metrics.get('quality_score', 0),
            error_message=core_result.get('error_message'),
            warnings=core_result.get('warnings', [])
        )
    
    def _create_error_result(self, error_message: str, center_id: str) -> WebDispatchResult:
        """오류 결과 생성"""
        web_center = WebCenter(
            center_id=center_id,
            name=f"센터 {center_id}",
            address="정보 없음",
            latitude=37.5665,
            longitude=126.9780,
            is_active=True
        )
        
        return WebDispatchResult(
            batch_id=f"ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            status='failed',
            center=web_center,
            error_message=error_message
        )


# 테스트 코드
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    
    service = SubprocessTmsApiService()
    
    print("=== 센터 목록 조회 ===")
    centers = service.get_centers_list()
    print(json.dumps(centers, indent=2, ensure_ascii=False))
    
    print("\n=== 배차 실행 테스트 ===")
    result = service.execute_dispatch('CENTER_GANGNAM')
    print(f"상태: {result.status}")
    print(f"배치 ID: {result.batch_id}")
    print(f"오류 메시지: {result.error_message}")