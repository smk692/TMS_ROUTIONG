"""
데이터 수집 서비스
"""
from typing import List, Dict, Optional
from datetime import datetime
import logging

from ..models import Order, Vehicle, Region, Coordinates, OrderStatus, VehicleStatus, VehicleType, Priority
from ..database.connection import db_session
from ..database.models import (
    Order as DBOrder, Vehicle as DBVehicle, Region as DBRegion, Center as DBCenter,
    VehicleStatusEnum, OrderStatusEnum, VehicleTypeEnum, OrderPriorityEnum
)
from sqlalchemy import and_, or_


class DataCollector:
    """TMS 데이터베이스에서 배차 데이터 수집"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def get_pending_orders(self, center_id: str = None, region_id: str = None) -> List[Order]:
        """대기 중인 주문 조회 - 데이터베이스에서 직접 조회"""
        try:
            return self._load_orders_from_database(center_id, region_id)
        except Exception as e:
            self.logger.error(f"주문 데이터 로드 오류: {str(e)}")
            return []
    
    def _load_orders_from_database(self, center_id: str = None, region_id: str = None) -> List[Order]:
        """데이터베이스에서 주문 조회"""
        try:
            with db_session() as session:
                query = session.query(DBOrder).filter(
                    DBOrder.status == 'pending'
                )
                
                # 센터 필터
                if center_id:
                    query = query.filter(DBOrder.center_id == center_id)
                
                # 권역 필터
                if region_id:
                    query = query.filter(DBOrder.region_id == region_id)
                
                # 우선순위 순, 생성 시간 순 정렬
                query = query.order_by(
                    DBOrder.priority.desc(),
                    DBOrder.created_at.asc()
                )
                
                db_orders = query.all()
                
                # 도메인 모델로 변환
                orders = []
                for db_order in db_orders:
                    order = Order(
                        id=db_order.id,
                        center_id=db_order.center_id,
                        region_id=db_order.region_id,
                        coordinates=Coordinates(
                            latitude=float(db_order.latitude),
                            longitude=float(db_order.longitude)
                        ),
                        address=db_order.address,
                        priority=Priority(db_order.priority),
                        status=OrderStatus(db_order.status),
                        created_at=db_order.created_at,
                        assigned_vehicle_id=db_order.assigned_vehicle_id,
                        estimated_delivery_time=db_order.estimated_delivery_time_minutes
                    )
                    orders.append(order)
                
                self.logger.info(f"주문 {len(orders)}개 조회 완료: center_id={center_id}, region_id={region_id}")
                return orders
                
        except Exception as e:
            self.logger.error(f"데이터베이스 주문 조회 오류: {str(e)}")
            raise
    
    def get_available_vehicles(self, center_id: str = None) -> List[Vehicle]:
        """사용 가능한 차량 조회 - 데이터베이스에서 직접 조회"""
        try:
            return self._load_vehicles_from_database(center_id)
        except Exception as e:
            self.logger.error(f"차량 데이터 로드 오류: {str(e)}")
            return []
    
    def _load_vehicles_from_database(self, center_id: str = None) -> List[Vehicle]:
        """데이터베이스에서 차량 조회"""
        try:
            with db_session() as session:
                query = session.query(DBVehicle, DBRegion).join(
                    DBRegion, DBVehicle.region_id == DBRegion.id
                ).filter(
                    and_(
                        DBVehicle.status == VehicleStatusEnum.ACTIVE,
                        DBVehicle.auto_dispatch == True,
                        or_(
                            DBVehicle.vehicle_type == VehicleTypeEnum.TOP_CAR,
                            DBVehicle.vehicle_type == VehicleTypeEnum.CARGO
                        )
                    )
                )
                
                # 센터 필터
                if center_id:
                    query = query.filter(DBVehicle.center_id == center_id)
                
                results = query.all()
                
                # 도메인 모델로 변환
                vehicles = []
                for db_vehicle, db_region in results:
                    vehicle = Vehicle(
                        id=db_vehicle.id,
                        driver_name=db_vehicle.driver_name,
                        vehicle_type=VehicleType(db_vehicle.vehicle_type.value),
                        region_id=db_vehicle.region_id,
                        center_coordinates=Coordinates(
                            latitude=float(db_region.center_latitude),
                            longitude=float(db_region.center_longitude)
                        ),
                        experience_months=db_vehicle.experience_months,
                        max_capacity=db_vehicle.max_capacity,
                        safe_capacity=db_vehicle.safe_capacity,
                        status=VehicleStatus(db_vehicle.status.value),
                        auto_dispatch=db_vehicle.auto_dispatch
                    )
                    vehicles.append(vehicle)
                
                self.logger.info(f"차량 {len(vehicles)}대 조회 완료: center_id={center_id}")
                return vehicles
                
        except Exception as e:
            self.logger.error(f"데이터베이스 차량 조회 오류: {str(e)}")
            raise
    
    def get_regions(self, center_id: str = None) -> List[Region]:
        """권역 정보 조회 - 데이터베이스에서 직접 조회"""
        try:
            return self._load_regions_from_database(center_id)
        except Exception as e:
            self.logger.error(f"권역 데이터 로드 오류: {str(e)}")
            return []
    
    def _load_regions_from_database(self, center_id: str = None) -> List[Region]:
        """데이터베이스에서 권역 조회"""
        try:
            with db_session() as session:
                query = session.query(DBRegion).filter(
                    DBRegion.is_active == True
                )
                
                # 센터 필터
                if center_id:
                    query = query.filter(DBRegion.center_id == center_id)
                
                db_regions = query.all()
                
                # 도메인 모델로 변환
                regions = []
                for db_region in db_regions:
                    region = Region(
                        id=db_region.id,
                        name=db_region.name,
                        center_id=db_region.center_id,
                        center_coordinates=Coordinates(
                            latitude=float(db_region.center_latitude),
                            longitude=float(db_region.center_longitude)
                        ),
                        difficulty_score=float(db_region.difficulty_score),
                        max_delivery_distance_km=float(db_region.max_delivery_distance_km)
                    )
                    regions.append(region)
                
                self.logger.info(f"권역 {len(regions)}개 조회 완료: center_id={center_id}")
                return regions
                
        except Exception as e:
            self.logger.error(f"데이터베이스 권역 조회 오류: {str(e)}")
            raise
    
    def get_excluded_vehicles(self, center_id: str = None) -> List[Vehicle]:
        """수동 배차 대상 차량 조회 - 데이터베이스에서 직접 조회"""
        try:
            return self._load_excluded_vehicles_from_database(center_id)
        except Exception as e:
            self.logger.error(f"수동 배차 차량 데이터 로드 오류: {str(e)}")
            return []
    
    def _load_excluded_vehicles_from_database(self, center_id: str = None) -> List[Vehicle]:
        """데이터베이스에서 수동 배차 차량 조회"""
        try:
            with db_session() as session:
                query = session.query(DBVehicle, DBRegion).join(
                    DBRegion, DBVehicle.region_id == DBRegion.id
                ).filter(
                    and_(
                        DBVehicle.status == VehicleStatusEnum.ACTIVE,
                        or_(
                            DBVehicle.auto_dispatch == False,
                            DBVehicle.vehicle_type == VehicleTypeEnum.OTHER
                        )
                    )
                )
                
                # 센터 필터
                if center_id:
                    query = query.filter(DBVehicle.center_id == center_id)
                
                results = query.all()
                
                # 도메인 모델로 변환
                vehicles = []
                for db_vehicle, db_region in results:
                    vehicle = Vehicle(
                        id=db_vehicle.id,
                        driver_name=db_vehicle.driver_name,
                        vehicle_type=VehicleType(db_vehicle.vehicle_type.value),
                        region_id=db_vehicle.region_id,
                        center_coordinates=Coordinates(
                            latitude=float(db_region.center_latitude),
                            longitude=float(db_region.center_longitude)
                        ),
                        experience_months=db_vehicle.experience_months,
                        max_capacity=db_vehicle.max_capacity,
                        safe_capacity=db_vehicle.safe_capacity,
                        status=VehicleStatus(db_vehicle.status.value),
                        auto_dispatch=db_vehicle.auto_dispatch
                    )
                    vehicles.append(vehicle)
                
                self.logger.info(f"수동 배차 차량 {len(vehicles)}대 조회 완료: center_id={center_id}")
                return vehicles
                
        except Exception as e:
            self.logger.error(f"데이터베이스 수동 배차 차량 조회 오류: {str(e)}")
            raise
    
    def validate_data_consistency(self, orders: List[Order], vehicles: List[Vehicle], 
                                regions: List[Region]) -> bool:
        """데이터 일관성 검증"""
        errors = []
        
        # 주문의 권역 ID가 유효한지 확인
        region_ids = {r.id for r in regions}
        for order in orders:
            if order.region_id not in region_ids:
                errors.append(f"주문 {order.id}의 권역 {order.region_id}이 존재하지 않음")
        
        # 차량의 권역 ID가 유효한지 확인
        for vehicle in vehicles:
            if vehicle.region_id not in region_ids:
                errors.append(f"차량 {vehicle.id}의 권역 {vehicle.region_id}이 존재하지 않음")
        
        if errors:
            for error in errors:
                self.logger.error(error)
            return False
        
        self.logger.info("데이터 일관성 검증 통과")
        return True