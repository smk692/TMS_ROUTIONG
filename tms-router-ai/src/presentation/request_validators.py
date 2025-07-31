"""
API 요청 검증기

외부 API 요청의 형식과 내용을 검증합니다.
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid

from src.use_cases.optimize_route_use_case import TmsRequest
from src.use_cases.process_feedback_use_case import FeedbackRequest
from src.shared.exceptions import ValidationError


class LocationModel(BaseModel):
    """위치 모델"""
    lat: float = Field(..., ge=-90, le=90, description="위도")
    lng: float = Field(..., ge=-180, le=180, description="경도")


class TimeWindowModel(BaseModel):
    """시간창 모델"""
    start: str = Field(..., description="시작 시간 (ISO 형식)")
    end: str = Field(..., description="종료 시간 (ISO 형식)")
    
    @validator('start', 'end')
    def validate_datetime_format(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")


class VehicleModel(BaseModel):
    """차량 모델"""
    vehicle_id: str = Field(..., min_length=1, description="차량 ID")
    capacity_tons: float = Field(..., gt=0, description="용량 (톤)")
    current_location: LocationModel = Field(..., description="현재 위치")
    driver_id: str = Field(None, description="운전자 ID")
    fuel_efficiency_kmpl: float = Field(10.0, gt=0, description="연비 (km/L)")
    hourly_cost: float = Field(20000.0, ge=0, description="시간당 비용")
    fuel_cost_per_liter: float = Field(1500.0, ge=0, description="연료 단가")
    special_capabilities: List[str] = Field(default_factory=list, description="특수 능력")


class OrderModel(BaseModel):
    """주문 모델"""
    order_id: str = Field(..., min_length=1, description="주문 ID")
    pickup_location: LocationModel = Field(..., description="픽업 위치")
    delivery_location: LocationModel = Field(..., description="배송 위치")
    weight_tons: float = Field(..., ge=0, description="중량 (톤)")
    volume_cbm: float = Field(0.0, ge=0, description="부피 (㎥)")
    priority: str = Field("MEDIUM", description="우선순위")
    time_window: TimeWindowModel = Field(None, description="배송 시간창")
    special_requirements: List[str] = Field(default_factory=list, description="특수 요구사항")
    customer_id: str = Field(None, description="고객 ID")
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ['LOW', 'MEDIUM', 'HIGH', 'URGENT']
        if v not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")
        return v


class TmsRequestModel(BaseModel):
    """TMS 배차 요청 모델"""
    vehicles: List[VehicleModel] = Field(..., min_items=1, max_items=50, description="차량 리스트")
    orders: List[OrderModel] = Field(..., min_items=1, max_items=200, description="주문 리스트")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="제약 조건")
    scenario_type: str = Field(None, description="시나리오 타입")
    conversation_id: str = Field(None, description="대화 ID")
    
    @validator('scenario_type')
    def validate_scenario_type(cls, v):
        if v is None:
            return v
        valid_scenarios = ['vrp', 'tsp', 'load_consolidation', 'emergency_dispatch', 'realtime_adjustment']
        if v not in valid_scenarios:
            raise ValueError(f"Scenario type must be one of: {valid_scenarios}")
        return v


class FeedbackRequestModel(BaseModel):
    """피드백 요청 모델"""
    conversation_id: str = Field(..., min_length=1, description="대화 ID")
    feedback_type: str = Field(..., description="피드백 타입")
    feedback_content: str = Field(..., min_length=1, max_length=5000, description="피드백 내용")
    route_specific: Dict[str, Any] = Field(None, description="특정 경로 관련 피드백")
    rating: int = Field(None, ge=1, le=5, description="평점 (1-5)")
    
    @validator('feedback_type')
    def validate_feedback_type(cls, v):
        valid_types = ['positive', 'negative', 'suggestion', 'correction', 'neutral']
        if v not in valid_types:
            raise ValueError(f"Feedback type must be one of: {valid_types}")
        return v


def validate_tms_request(request_data: Dict[str, Any]) -> TmsRequest:
    """
    TMS 배차 요청 검증
    
    Args:
        request_data: 요청 데이터
        
    Returns:
        검증된 TMS 요청 객체
        
    Raises:
        ValidationError: 검증 실패시
    """
    try:
        # Pydantic 모델로 검증
        validated_model = TmsRequestModel(**request_data)
        
        # Use Case 요청 객체로 변환
        tms_request = TmsRequest(
            request_id=str(uuid.uuid4()),
            vehicles=[vehicle.dict() for vehicle in validated_model.vehicles],
            orders=[order.dict() for order in validated_model.orders],
            constraints=validated_model.constraints,
            conversation_id=validated_model.conversation_id
        )
        
        # 추가 비즈니스 검증
        _validate_business_rules(tms_request)
        
        return tms_request
        
    except Exception as e:
        raise ValidationError("request_validation", f"Invalid request data: {e}")


def validate_feedback_request(request_data: Dict[str, Any]) -> FeedbackRequest:
    """
    피드백 요청 검증
    
    Args:
        request_data: 요청 데이터
        
    Returns:
        검증된 피드백 요청 객체
        
    Raises:
        ValidationError: 검증 실패시
    """
    try:
        # Pydantic 모델로 검증
        validated_model = FeedbackRequestModel(**request_data)
        
        # Use Case 요청 객체로 변환
        feedback_request = FeedbackRequest(
            request_id=str(uuid.uuid4()),
            conversation_id=validated_model.conversation_id,
            feedback_type=validated_model.feedback_type,
            feedback_content=validated_model.feedback_content,
            route_specific=validated_model.route_specific,
            rating=validated_model.rating
        )
        
        return feedback_request
        
    except Exception as e:
        raise ValidationError("feedback_validation", f"Invalid feedback data: {e}")


def _validate_business_rules(request: TmsRequest):
    """비즈니스 규칙 검증"""
    # 총 용량 vs 총 중량 기본 검증
    total_capacity = sum(vehicle['capacity_tons'] for vehicle in request.vehicles)
    total_weight = sum(order['weight_tons'] for order in request.orders)
    
    if total_weight > total_capacity:
        raise ValidationError("capacity_exceeded", 
            f"Total weight ({total_weight:.1f}t) exceeds total capacity ({total_capacity:.1f}t)")
    
    # 중복 ID 검증
    vehicle_ids = [vehicle['vehicle_id'] for vehicle in request.vehicles]
    if len(vehicle_ids) != len(set(vehicle_ids)):
        raise ValidationError("duplicate_vehicle_ids", "Duplicate vehicle IDs found")
    
    order_ids = [order['order_id'] for order in request.orders]
    if len(order_ids) != len(set(order_ids)):
        raise ValidationError("duplicate_order_ids", "Duplicate order IDs found")
    
    # 위치 검증 (픽업과 배송 위치가 같으면 안됨)
    for order in request.orders:
        pickup = order['pickup_location']
        delivery = order['delivery_location']
        
        if (abs(pickup['lat'] - delivery['lat']) < 0.001 and 
            abs(pickup['lng'] - delivery['lng']) < 0.001):
            raise ValidationError("same_pickup_delivery", 
                f"Order {order['order_id']} has same pickup and delivery location")


def validate_request_id(request_id: str) -> str:
    """요청 ID 검증"""
    if not request_id or not isinstance(request_id, str):
        raise ValidationError("request_id", "Invalid request ID")
    
    try:
        # UUID 형식인지 확인
        uuid.UUID(request_id)
        return request_id
    except ValueError:
        # UUID가 아니면 그대로 사용 (임의 문자열 허용)
        if len(request_id) > 100:
            raise ValidationError("request_id", "Request ID too long (max 100 characters)")
        return request_id


def validate_conversation_id(conversation_id: str) -> str:
    """대화 ID 검증"""
    if not conversation_id or not isinstance(conversation_id, str):
        raise ValidationError("conversation_id", "Invalid conversation ID")
    
    if len(conversation_id) > 50:
        raise ValidationError("conversation_id", "Conversation ID too long (max 50 characters)")
    
    return conversation_id 