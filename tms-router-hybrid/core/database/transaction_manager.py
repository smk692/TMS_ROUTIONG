"""
데이터베이스 트랜잭션 관리자
"""
import logging
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .connection import db_session, get_session
from .models import (
    DispatchBatch, VehicleAssignment, OrderAssignment, TransactionLog,
    Order, Vehicle, TransactionOperationEnum
)
from ..models import VehicleAssignment as DomainVehicleAssignment


class TransactionManager:
    """배차 트랜잭션 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def dispatch_transaction(self, batch_id: str, center_id: str):
        """배차 트랜잭션 컨텍스트 매니저"""
        session = get_session()
        batch_created = False
        
        try:
            # 트랜잭션 시작
            session.begin()
            
            # 배치 레코드 생성
            batch = DispatchBatch(
                batch_id=batch_id,
                center_id=center_id,
                status="processing"
            )
            session.add(batch)
            
            # 트랜잭션 로그
            log = TransactionLog(
                batch_id=batch_id,
                operation_type=TransactionOperationEnum.START
            )
            session.add(log)
            
            session.flush()  # ID 생성을 위한 flush
            batch_created = True
            
            self.logger.info(f"배차 트랜잭션 시작: {batch_id}")
            
            # 트랜잭션 컨텍스트 반환
            transaction_context = DispatchTransactionContext(
                session=session,
                batch_id=batch_id,
                batch=batch,
                transaction_manager=self
            )
            
            yield transaction_context
            
            # 성공 시 커밋
            session.commit()
            self.logger.info(f"배차 트랜잭션 커밋 완료: {batch_id}")
            
        except Exception as e:
            # 오류 시 롤백
            if session.in_transaction():
                session.rollback()
                
            if batch_created:
                self._handle_rollback(batch_id, str(e))
                
            self.logger.error(f"배차 트랜잭션 롤백: {batch_id}, 오류: {str(e)}")
            raise
            
        finally:
            session.close()
    
    def _handle_rollback(self, batch_id: str, error_message: str):
        """롤백 처리 - Python 기반 구현"""
        try:
            with db_session() as session:
                # 먼저 배치가 존재하는지 확인
                batch_exists = session.execute(
                    text("SELECT COUNT(*) FROM dispatch_batches WHERE batch_id = :batch_id"),
                    {"batch_id": batch_id}
                ).scalar() > 0
                
                if batch_exists:
                    # 주문의 할당 정보 초기화
                    session.execute(
                        text("""
                            UPDATE orders SET 
                                assigned_vehicle_id = NULL,
                                status = 'pending',
                                assigned_at = NULL,
                                estimated_delivery_time_minutes = NULL
                            WHERE id IN (
                                SELECT order_id FROM order_assignments WHERE batch_id = :batch_id
                            )
                        """),
                        {"batch_id": batch_id}
                    )
                    
                    # 배차 배치 상태를 rollback으로 변경
                    session.execute(
                        text("""
                            UPDATE dispatch_batches SET 
                                status = 'rollback',
                                error_message = :error_message,
                                completed_at = NOW()
                            WHERE batch_id = :batch_id
                        """),
                        {"batch_id": batch_id, "error_message": error_message}
                    )
                    
                    # 롤백 로그 기록 (배치가 존재할 때만)
                    log = TransactionLog(
                        batch_id=batch_id,
                        operation_type=TransactionOperationEnum.ROLLBACK,
                        error_message=error_message
                    )
                    session.add(log)
                    
                    session.commit()
                    self.logger.info(f"롤백 처리 완료: {batch_id}")
                else:
                    # 배치가 존재하지 않는 경우, 로그만 기록
                    self.logger.warning(f"롤백 처리 스킵 - 배치 없음: {batch_id}, 오류: {error_message}")
                
        except Exception as rollback_error:
            self.logger.error(f"롤백 처리 실패: {batch_id}, 오류: {str(rollback_error)}")


class DispatchTransactionContext:
    """배차 트랜잭션 컨텍스트"""
    
    def __init__(self, session, batch_id: str, batch: DispatchBatch, transaction_manager: TransactionManager):
        self.session = session
        self.batch_id = batch_id
        self.batch = batch
        self.transaction_manager = transaction_manager
        self.logger = logging.getLogger(__name__)
        
        # 통계 추적
        self.assigned_orders = 0
        self.used_vehicles = 0
    
    def assign_orders_to_vehicle(self, 
                               vehicle_assignments: List[DomainVehicleAssignment]) -> bool:
        """차량별 주문 배정 원자적 처리"""
        try:
            for assignment in vehicle_assignments:
                # 1. 차량 배정 레코드 생성
                vehicle_assignment = VehicleAssignment(
                    batch_id=self.batch_id,
                    vehicle_id=assignment.vehicle_id,
                    driver_name=assignment.driver_name,
                    vehicle_type=assignment.vehicle_type,
                    region_name=assignment.region_name,
                    total_orders=len(assignment.assigned_orders),
                    estimated_distance_km=assignment.estimated_distance_km,
                    estimated_time_minutes=assignment.estimated_time_minutes,
                    capacity_utilization=assignment.capacity_utilization
                )
                self.session.add(vehicle_assignment)
                
                # 2. 주문별 배정 처리
                for i, order_id in enumerate(assignment.assigned_orders):
                    # 주문 상태 업데이트
                    result = self.session.execute(
                        text("""
                            UPDATE orders 
                            SET status = 'assigned', 
                                assigned_vehicle_id = :vehicle_id,
                                assigned_at = NOW()
                            WHERE id = :order_id AND status = 'pending'
                        """),
                        {"vehicle_id": assignment.vehicle_id, "order_id": order_id}
                    )
                    
                    if result.rowcount == 0:
                        raise ValueError(f"주문 {order_id}를 배정할 수 없습니다")
                    
                    # 주문 배정 이력 생성
                    order_assignment = OrderAssignment(
                        batch_id=self.batch_id,
                        order_id=order_id,
                        vehicle_id=assignment.vehicle_id,
                        assignment_order=i + 1,
                        estimated_delivery_minutes=assignment.estimated_time_minutes
                    )
                    self.session.add(order_assignment)
                    
                    self.assigned_orders += 1
                
                # 3. 차량 상태 업데이트
                self.session.execute(
                    text("""
                        UPDATE vehicles 
                        SET status = 'in_delivery', updated_at = NOW()
                        WHERE id = :vehicle_id
                    """),
                    {"vehicle_id": assignment.vehicle_id}
                )
                
                self.used_vehicles += 1
                
                # 4. 트랜잭션 로그
                log = TransactionLog(
                    batch_id=self.batch_id,
                    operation_type=TransactionOperationEnum.ORDER_ASSIGN,
                    table_name="orders",
                    record_id=assignment.vehicle_id,
                    new_data={
                        "vehicle_id": assignment.vehicle_id,
                        "order_count": len(assignment.assigned_orders),
                        "order_ids": assignment.assigned_orders
                    }
                )
                self.session.add(log)
            
            # 중간 플러시
            self.session.flush()
            
            self.logger.info(f"주문 배정 완료: {self.used_vehicles}대 차량, {self.assigned_orders}개 주문")
            return True
            
        except Exception as e:
            self.logger.error(f"주문 배정 실패: {str(e)}")
            raise
    
    def complete_dispatch(self, 
                         algorithm_used: str, 
                         execution_time: float,
                         weather_conditions: Optional[Dict] = None,
                         traffic_conditions: Optional[Dict] = None) -> bool:
        """배차 완료 처리"""
        try:
            # 상태 결정
            if self.used_vehicles > 0 and self.assigned_orders > 0:
                status = "success"
            elif self.used_vehicles > 0:
                status = "partial_success"
            else:
                status = "failed"
            
            # 배치 상태 업데이트
            self.batch.status = status
            self.batch.assigned_orders = self.assigned_orders
            self.batch.used_vehicles = self.used_vehicles
            self.batch.algorithm_used = algorithm_used
            self.batch.execution_time_seconds = execution_time
            self.batch.weather_conditions = weather_conditions
            self.batch.traffic_conditions = traffic_conditions
            self.batch.completed_at = datetime.now()
            
            # 커밋 로그
            log = TransactionLog(
                batch_id=self.batch_id,
                operation_type=TransactionOperationEnum.COMMIT,
                new_data={
                    "status": status,  # status는 이미 문자열
                    "assigned_orders": self.assigned_orders,
                    "used_vehicles": self.used_vehicles,
                    "algorithm": algorithm_used
                }
            )
            self.session.add(log)
            
            # 플러시
            self.session.flush()
            
            self.logger.info(f"배차 완료: 상태={status}, 차량={self.used_vehicles}, 주문={self.assigned_orders}")
            return True
            
        except Exception as e:
            self.logger.error(f"배차 완료 처리 실패: {str(e)}")
            raise
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """배치 통계 반환"""
        return {
            "batch_id": self.batch_id,
            "assigned_orders": self.assigned_orders,
            "used_vehicles": self.used_vehicles,
            "status": self.batch.status if self.batch.status else "unknown"
        }


# 전역 트랜잭션 매니저
_transaction_manager: Optional[TransactionManager] = None


def get_transaction_manager() -> TransactionManager:
    """트랜잭션 매니저 싱글톤 반환"""
    global _transaction_manager
    if _transaction_manager is None:
        _transaction_manager = TransactionManager()
    return _transaction_manager