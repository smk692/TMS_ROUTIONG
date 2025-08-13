"""
SQLAlchemy 데이터베이스 모델
"""
from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    String, Integer, Numeric, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.mysql import ENUM


class Base(DeclarativeBase):
    """베이스 모델"""
    pass


class VehicleTypeEnum(PyEnum):
    """차량 유형"""
    TOP_CAR = "TOP_CAR"
    CARGO = "CARGO"
    OTHER = "OTHER"


class VehicleStatusEnum(PyEnum):
    """차량 상태"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MAINTENANCE = "MAINTENANCE"
    IN_DELIVERY = "IN_DELIVERY"


class OrderStatusEnum(PyEnum):
    """주문 상태"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class OrderPriorityEnum(PyEnum):
    """주문 우선순위"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class DispatchStatusEnum(PyEnum):
    """배차 상태"""
    PROCESSING = "processing"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    ROLLBACK = "rollback"



class TransactionOperationEnum(PyEnum):
    """트랜잭션 작업 유형"""
    START = "start"
    ORDER_ASSIGN = "order_assign"
    VEHICLE_UPDATE = "vehicle_update"
    COMMIT = "commit"
    ROLLBACK = "rollback"


class Center(Base):
    """물류센터 모델"""
    __tablename__ = "centers"
    
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    address: Mapped[str] = mapped_column(String(255), nullable=False)
    latitude: Mapped[Decimal] = mapped_column(Numeric(10, 8), nullable=False)
    longitude: Mapped[Decimal] = mapped_column(Numeric(11, 8), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 관계
    regions: Mapped[List["Region"]] = relationship("Region", back_populates="center", cascade="all, delete-orphan")
    vehicles: Mapped[List["Vehicle"]] = relationship("Vehicle", back_populates="center", cascade="all, delete-orphan")
    orders: Mapped[List["Order"]] = relationship("Order", back_populates="center", cascade="all, delete-orphan")
    dispatch_batches: Mapped[List["DispatchBatch"]] = relationship("DispatchBatch", back_populates="center")
    
    # 인덱스
    __table_args__ = (
        Index('idx_centers_active', 'is_active'),
        Index('idx_centers_coordinates', 'latitude', 'longitude'),
    )


class Region(Base):
    """권역 모델"""
    __tablename__ = "regions"
    
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    center_id: Mapped[str] = mapped_column(String(50), ForeignKey("centers.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    center_latitude: Mapped[Decimal] = mapped_column(Numeric(10, 8), nullable=False)
    center_longitude: Mapped[Decimal] = mapped_column(Numeric(11, 8), nullable=False)
    difficulty_score: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=1.00)
    max_delivery_distance_km: Mapped[Decimal] = mapped_column(Numeric(5, 2), default=20.00)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 관계
    center: Mapped["Center"] = relationship("Center", back_populates="regions")
    vehicles: Mapped[List["Vehicle"]] = relationship("Vehicle", back_populates="region", cascade="all, delete-orphan")
    orders: Mapped[List["Order"]] = relationship("Order", back_populates="region", cascade="all, delete-orphan")
    
    # 인덱스
    __table_args__ = (
        Index('idx_regions_center', 'center_id'),
        Index('idx_regions_active', 'is_active'),
        Index('idx_regions_coordinates', 'center_latitude', 'center_longitude'),
    )


class Vehicle(Base):
    """차량 모델"""
    __tablename__ = "vehicles"
    
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    center_id: Mapped[str] = mapped_column(String(50), ForeignKey("centers.id", ondelete="CASCADE"))
    region_id: Mapped[str] = mapped_column(String(50), ForeignKey("regions.id", ondelete="CASCADE"))
    driver_name: Mapped[str] = mapped_column(String(100), nullable=False)
    vehicle_type: Mapped[VehicleTypeEnum] = mapped_column(ENUM(VehicleTypeEnum), default=VehicleTypeEnum.TOP_CAR)
    experience_months: Mapped[int] = mapped_column(Integer, default=0)
    max_capacity: Mapped[int] = mapped_column(Integer, default=40)
    safe_capacity: Mapped[int] = mapped_column(Integer, default=35)
    status: Mapped[VehicleStatusEnum] = mapped_column(ENUM(VehicleStatusEnum), default=VehicleStatusEnum.ACTIVE)
    auto_dispatch: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 관계
    center: Mapped["Center"] = relationship("Center", back_populates="vehicles")
    region: Mapped["Region"] = relationship("Region", back_populates="vehicles")
    assigned_orders: Mapped[List["Order"]] = relationship("Order", back_populates="assigned_vehicle")
    vehicle_assignments: Mapped[List["VehicleAssignment"]] = relationship("VehicleAssignment", back_populates="vehicle")
    order_assignments: Mapped[List["OrderAssignment"]] = relationship("OrderAssignment", back_populates="vehicle")
    
    # 인덱스
    __table_args__ = (
        Index('idx_vehicles_center', 'center_id'),
        Index('idx_vehicles_region', 'region_id'),
        Index('idx_vehicles_status', 'status'),
        Index('idx_vehicles_auto_dispatch', 'auto_dispatch'),
        Index('idx_vehicles_driver', 'driver_name'),
        Index('idx_vehicles_center_region_status', 'center_id', 'region_id', 'status'),
    )


class Order(Base):
    """주문 모델"""
    __tablename__ = "orders"
    
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    center_id: Mapped[str] = mapped_column(String(50), ForeignKey("centers.id", ondelete="CASCADE"))
    region_id: Mapped[str] = mapped_column(String(50), ForeignKey("regions.id", ondelete="CASCADE"))
    address: Mapped[str] = mapped_column(String(255), nullable=False)
    latitude: Mapped[Decimal] = mapped_column(Numeric(10, 8), nullable=False)
    longitude: Mapped[Decimal] = mapped_column(Numeric(11, 8), nullable=False)
    priority: Mapped[str] = mapped_column(String(10), default='normal')
    status: Mapped[str] = mapped_column(String(20), default='pending')
    assigned_vehicle_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("vehicles.id", ondelete="SET NULL"))
    estimated_delivery_time_minutes: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    assigned_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # 관계
    center: Mapped["Center"] = relationship("Center", back_populates="orders")
    region: Mapped["Region"] = relationship("Region", back_populates="orders")
    assigned_vehicle: Mapped[Optional["Vehicle"]] = relationship("Vehicle", back_populates="assigned_orders")
    order_assignments: Mapped[List["OrderAssignment"]] = relationship("OrderAssignment", back_populates="order")
    
    # 인덱스
    __table_args__ = (
        Index('idx_orders_center', 'center_id'),
        Index('idx_orders_region', 'region_id'),
        Index('idx_orders_status', 'status'),
        Index('idx_orders_assigned_vehicle', 'assigned_vehicle_id'),
        Index('idx_orders_priority', 'priority'),
        Index('idx_orders_coordinates', 'latitude', 'longitude'),
        Index('idx_orders_created_at', 'created_at'),
        Index('idx_orders_center_region_status', 'center_id', 'region_id', 'status'),
    )


class DispatchBatch(Base):
    """배차 배치 모델"""
    __tablename__ = "dispatch_batches"
    
    batch_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    center_id: Mapped[str] = mapped_column(String(50), ForeignKey("centers.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(20), default="processing")
    total_orders: Mapped[int] = mapped_column(Integer, default=0)
    assigned_orders: Mapped[int] = mapped_column(Integer, default=0)
    total_vehicles: Mapped[int] = mapped_column(Integer, default=0)
    used_vehicles: Mapped[int] = mapped_column(Integer, default=0)
    algorithm_used: Mapped[Optional[str]] = mapped_column(String(100))
    execution_time_seconds: Mapped[Decimal] = mapped_column(Numeric(8, 3), default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    weather_conditions: Mapped[Optional[dict]] = mapped_column(JSON)
    traffic_conditions: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # 관계
    center: Mapped["Center"] = relationship("Center", back_populates="dispatch_batches")
    vehicle_assignments: Mapped[List["VehicleAssignment"]] = relationship("VehicleAssignment", back_populates="batch", cascade="all, delete-orphan")
    order_assignments: Mapped[List["OrderAssignment"]] = relationship("OrderAssignment", back_populates="batch", cascade="all, delete-orphan")
    transaction_logs: Mapped[List["TransactionLog"]] = relationship("TransactionLog", back_populates="batch")
    
    # 인덱스
    __table_args__ = (
        Index('idx_dispatch_batches_center', 'center_id'),
        Index('idx_dispatch_batches_status', 'status'),
        Index('idx_dispatch_batches_created', 'created_at'),
    )


class VehicleAssignment(Base):
    """차량별 배정 결과 모델"""
    __tablename__ = "vehicle_assignments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(100), ForeignKey("dispatch_batches.batch_id", ondelete="CASCADE"))
    vehicle_id: Mapped[str] = mapped_column(String(50), ForeignKey("vehicles.id", ondelete="CASCADE"))
    driver_name: Mapped[str] = mapped_column(String(100), nullable=False)
    vehicle_type: Mapped[str] = mapped_column(String(20), nullable=False)
    region_name: Mapped[str] = mapped_column(String(100), nullable=False)
    total_orders: Mapped[int] = mapped_column(Integer, default=0)
    estimated_distance_km: Mapped[Decimal] = mapped_column(Numeric(8, 2), default=0)
    estimated_time_minutes: Mapped[int] = mapped_column(Integer, default=0)
    capacity_utilization: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    
    # 관계
    batch: Mapped["DispatchBatch"] = relationship("DispatchBatch", back_populates="vehicle_assignments")
    vehicle: Mapped["Vehicle"] = relationship("Vehicle", back_populates="vehicle_assignments")
    
    # 인덱스
    __table_args__ = (
        Index('idx_vehicle_assignments_batch', 'batch_id'),
        Index('idx_vehicle_assignments_vehicle', 'vehicle_id'),
        Index('idx_vehicle_assignments_batch_vehicle', 'batch_id', 'vehicle_id'),
    )


class OrderAssignment(Base):
    """주문별 배정 이력 모델"""
    __tablename__ = "order_assignments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(100), ForeignKey("dispatch_batches.batch_id", ondelete="CASCADE"))
    order_id: Mapped[str] = mapped_column(String(50), ForeignKey("orders.id", ondelete="CASCADE"))
    vehicle_id: Mapped[str] = mapped_column(String(50), ForeignKey("vehicles.id", ondelete="CASCADE"))
    assignment_order: Mapped[int] = mapped_column(Integer, default=0)
    estimated_delivery_minutes: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    
    # 관계
    batch: Mapped["DispatchBatch"] = relationship("DispatchBatch", back_populates="order_assignments")
    order: Mapped["Order"] = relationship("Order", back_populates="order_assignments")
    vehicle: Mapped["Vehicle"] = relationship("Vehicle", back_populates="order_assignments")
    
    # 인덱스
    __table_args__ = (
        Index('idx_order_assignments_batch', 'batch_id'),
        Index('idx_order_assignments_order', 'order_id'),
        Index('idx_order_assignments_vehicle', 'vehicle_id'),
        Index('idx_order_assignments_batch_vehicle', 'batch_id', 'vehicle_id'),
        UniqueConstraint('batch_id', 'order_id', name='uk_order_assignments_batch_order'),
    )




class SystemSetting(Base):
    """시스템 설정 모델"""
    __tablename__ = "system_settings"
    
    setting_key: Mapped[str] = mapped_column(String(100), primary_key=True)
    setting_value: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class TransactionLog(Base):
    """트랜잭션 로그 모델"""
    __tablename__ = "transaction_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(100), ForeignKey("dispatch_batches.batch_id"), nullable=False)
    operation_type: Mapped[TransactionOperationEnum] = mapped_column(ENUM(TransactionOperationEnum), nullable=False)
    table_name: Mapped[Optional[str]] = mapped_column(String(100))
    record_id: Mapped[Optional[str]] = mapped_column(String(100))
    old_data: Mapped[Optional[dict]] = mapped_column(JSON)
    new_data: Mapped[Optional[dict]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    
    # 관계
    batch: Mapped["DispatchBatch"] = relationship("DispatchBatch", back_populates="transaction_logs")
    
    # 인덱스
    __table_args__ = (
        Index('idx_transaction_logs_batch', 'batch_id'),
        Index('idx_transaction_logs_operation', 'operation_type'),
        Index('idx_transaction_logs_created', 'created_at'),
    )