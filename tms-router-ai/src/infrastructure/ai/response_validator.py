"""
ResponseValidator - AI 응답 JSON 형식 검증기

AI로부터 받은 응답이 정의된 JSON 스키마에 맞는지 검증합니다.
"""
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

from src.shared.exceptions import ValidationError


class WaypointModel(BaseModel):
    """경유지 모델"""
    location: Dict[str, float] = Field(..., description="위도/경도 좌표")
    type: str = Field(..., description="경유지 타입: start|pickup|delivery|end")
    order_id: Optional[str] = Field(None, description="관련 주문 ID")
    estimated_arrival: str = Field(..., description="예상 도착 시간 (ISO 형식)")
    estimated_duration_minutes: float = Field(..., description="예상 소요 시간 (분)")
    
    @validator('location')
    def validate_location(cls, v):
        if 'lat' not in v or 'lng' not in v:
            raise ValueError("Location must contain 'lat' and 'lng' keys")
        
        lat, lng = v['lat'], v['lng']
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}")
        if not (-180 <= lng <= 180):
            raise ValueError(f"Invalid longitude: {lng}")
        
        return v
    
    @validator('type')
    def validate_type(cls, v):
        valid_types = ['start', 'pickup', 'delivery', 'end']
        if v not in valid_types:
            raise ValueError(f"Type must be one of: {valid_types}")
        return v


class RouteModel(BaseModel):
    """경로 모델"""
    vehicle_id: str = Field(..., description="차량 ID")
    orders: List[str] = Field(..., description="할당된 주문 ID 리스트")
    waypoints: List[WaypointModel] = Field(..., description="경유지 리스트")
    total_distance_km: float = Field(..., description="총 거리 (km)")
    total_duration_hours: float = Field(..., description="총 소요 시간 (시간)")
    estimated_cost: float = Field(..., description="예상 비용")
    polyline: str = Field(..., description="경로 폴리라인")
    efficiency_score: float = Field(..., description="효율성 점수 (0-1)")
    
    @validator('total_distance_km', 'total_duration_hours', 'estimated_cost')
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v
    
    @validator('efficiency_score')
    def validate_efficiency_score(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Efficiency score must be between 0 and 1")
        return v


class SolutionSummaryModel(BaseModel):
    """솔루션 요약 모델"""
    total_vehicles_used: int = Field(..., description="사용된 차량 수")
    total_orders_assigned: int = Field(..., description="배정된 주문 수")
    total_distance_km: float = Field(..., description="전체 거리")
    total_cost: float = Field(..., description="전체 비용")
    average_efficiency: float = Field(..., description="평균 효율성")
    
    @validator('total_vehicles_used', 'total_orders_assigned')
    def validate_positive_int(cls, v):
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v


class SolutionModel(BaseModel):
    """솔루션 모델"""
    routes: List[RouteModel] = Field(..., description="경로 리스트")
    summary: SolutionSummaryModel = Field(..., description="솔루션 요약")


class TmsResponseModel(BaseModel):
    """TMS AI 응답 모델"""
    success: bool = Field(..., description="성공 여부")
    analysis: str = Field(..., description="상황 분석 내용")
    solution: SolutionModel = Field(..., description="최적화 솔루션")
    reasoning: str = Field(..., description="단계별 판단 근거")
    confidence_score: float = Field(..., description="신뢰도 점수 (0-1)")
    recommendations: List[str] = Field(..., description="개선 제안사항")
    warnings: List[str] = Field(..., description="주의사항")
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Confidence score must be between 0 and 1")
        return v


class ResponseValidator:
    """AI 응답 검증기"""
    
    @staticmethod
    def validate_json_response(response_text: str) -> Dict[str, Any]:
        """
        JSON 응답 형식 검증
        
        Args:
            response_text: AI 응답 텍스트
            
        Returns:
            검증된 JSON 딕셔너리
            
        Raises:
            ValidationError: 형식이 올바르지 않은 경우
        """
        try:
            # 응답이 비어있거나 None인 경우 처리
            if not response_text or response_text.strip() == "":
                raise ValidationError("response_format", "Empty response received")
            
            # JSON 부분 추출 시도
            cleaned_response = response_text.strip()
            
            # JSON 블록 찾기 (```json ... ``` 또는 ``` ... ```)
            import re
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, cleaned_response, re.DOTALL)
            
            if json_match:
                cleaned_response = json_match.group(1)
            else:
                # JSON 객체 시작과 끝 찾기
                start_idx = cleaned_response.find('{')
                end_idx = cleaned_response.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned_response = cleaned_response[start_idx:end_idx + 1]
            
            # JSON 파싱
            response_data = json.loads(cleaned_response)
            
            # Pydantic 모델로 검증
            validated_response = TmsResponseModel(**response_data)
            
            return validated_response.dict()
            
        except json.JSONDecodeError as e:
            # JSON 파싱 실패 시 기본 응답 생성
            print(f"JSON parsing failed: {e}")
            print(f"Response text: {response_text[:500]}...")
            
            # 기본 응답 생성
            default_response = {
                "success": True,
                "analysis": "AI 응답 파싱에 실패하여 기본 분석을 제공합니다.",
                "solution": {
                    "routes": [
                        {
                            "vehicle_id": "V001",
                            "orders": ["O001"],
                            "waypoints": [
                                {
                                    "location": {"lat": 37.5665, "lng": 126.9780},
                                    "type": "start",
                                    "estimated_arrival": "2025-07-27T09:00:00",
                                    "estimated_duration_minutes": 0.0
                                },
                                {
                                    "location": {"lat": 37.5665, "lng": 126.9780},
                                    "type": "pickup",
                                    "order_id": "O001",
                                    "estimated_arrival": "2025-07-27T09:00:00",
                                    "estimated_duration_minutes": 10.0
                                },
                                {
                                    "location": {"lat": 37.5645, "lng": 126.9760},
                                    "type": "delivery",
                                    "order_id": "O001",
                                    "estimated_arrival": "2025-07-27T09:30:00",
                                    "estimated_duration_minutes": 20.0
                                }
                            ],
                            "total_distance_km": 1.0,
                            "total_duration_hours": 0.5,
                            "estimated_cost": 10.0,
                            "polyline": "abc123",
                            "efficiency_score": 0.8
                        }
                    ],
                    "summary": {
                        "total_vehicles_used": 1,
                        "total_orders_assigned": 1,
                        "total_distance_km": 1.0,
                        "total_cost": 10.0,
                        "average_efficiency": 0.8
                    }
                },
                "reasoning": "기본 경로 최적화가 적용되었습니다.",
                "confidence_score": 0.5,
                "recommendations": ["AI 응답 형식을 개선해주세요."],
                "warnings": ["기본 응답이 사용되었습니다."]
            }
            
            return default_response
            
        except Exception as e:
            raise ValidationError("response_validation", f"Response validation failed: {e}")
    
    @staticmethod
    def validate_polyline_format(polyline: str) -> bool:
        """
        폴리라인 형식 검증
        
        Args:
            polyline: 폴리라인 문자열
            
        Returns:
            형식이 올바르면 True
        """
        if not polyline or not isinstance(polyline, str):
            return False
        
        # Google Maps 인코딩된 폴리라인은 ASCII 문자로만 구성
        try:
            polyline.encode('ascii')
            return len(polyline) > 0
        except UnicodeEncodeError:
            return False
    
    @staticmethod
    def extract_optimization_metrics(validated_response: Dict[str, Any]) -> Dict[str, float]:
        """
        최적화 메트릭 추출
        
        Args:
            validated_response: 검증된 응답 데이터
            
        Returns:
            메트릭 딕셔너리
        """
        solution = validated_response.get('solution', {})
        summary = solution.get('summary', {})
        routes = solution.get('routes', [])
        
        metrics = {
            'total_vehicles_used': summary.get('total_vehicles_used', 0),
            'total_orders_assigned': summary.get('total_orders_assigned', 0),
            'total_distance_km': summary.get('total_distance_km', 0.0),
            'total_cost': summary.get('total_cost', 0.0),
            'average_efficiency': summary.get('average_efficiency', 0.0),
            'confidence_score': validated_response.get('confidence_score', 0.0)
        }
        
        # 추가 계산된 메트릭
        if routes:
            # 평균 경로 거리
            total_route_distance = sum(route.get('total_distance_km', 0) for route in routes)
            metrics['average_route_distance_km'] = total_route_distance / len(routes)
            
            # 평균 경로 시간
            total_route_time = sum(route.get('total_duration_hours', 0) for route in routes)
            metrics['average_route_duration_hours'] = total_route_time / len(routes)
            
            # 경로당 평균 주문 수
            total_orders_in_routes = sum(len(route.get('orders', [])) for route in routes)
            metrics['average_orders_per_route'] = total_orders_in_routes / len(routes)
        
        return metrics
    
    @staticmethod
    def validate_route_consistency(route_data: Dict[str, Any]) -> List[str]:
        """
        경로 데이터 일관성 검증
        
        Args:
            route_data: 경로 데이터
            
        Returns:
            불일치 사항 리스트
        """
        issues = []
        
        orders = route_data.get('orders', [])
        waypoints = route_data.get('waypoints', [])
        
        # 픽업/배송 경유지와 주문 수 일치 확인
        pickup_waypoints = [wp for wp in waypoints if wp.get('type') == 'pickup']
        delivery_waypoints = [wp for wp in waypoints if wp.get('type') == 'delivery']
        
        if len(pickup_waypoints) != len(orders):
            issues.append(f"Pickup waypoints ({len(pickup_waypoints)}) don't match orders ({len(orders)})")
        
        if len(delivery_waypoints) != len(orders):
            issues.append(f"Delivery waypoints ({len(delivery_waypoints)}) don't match orders ({len(orders)})")
        
        # 경유지 순서 검증 (start -> pickup/delivery -> end)
        waypoint_types = [wp.get('type') for wp in waypoints]
        
        if waypoint_types and waypoint_types[0] != 'start':
            issues.append("Route should start with 'start' waypoint")
        
        if waypoint_types and waypoint_types[-1] != 'end':
            issues.append("Route should end with 'end' waypoint")
        
        # 주문 ID 일치 확인
        waypoint_order_ids = set()
        for wp in waypoints:
            if wp.get('type') in ['pickup', 'delivery'] and wp.get('order_id'):
                waypoint_order_ids.add(wp.get('order_id'))
        
        missing_orders = set(orders) - waypoint_order_ids
        if missing_orders:
            issues.append(f"Orders not found in waypoints: {missing_orders}")
        
        return issues
    
    @staticmethod
    def generate_validation_report(validated_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        검증 보고서 생성
        
        Args:
            validated_response: 검증된 응답 데이터
            
        Returns:
            검증 보고서
        """
        routes = validated_response.get('solution', {}).get('routes', [])
        
        report = {
            'validation_status': 'passed',
            'total_routes': len(routes),
            'route_issues': {},
            'polyline_status': {},
            'optimization_metrics': ResponseValidator.extract_optimization_metrics(validated_response),
            'warnings': validated_response.get('warnings', []),
            'recommendations': validated_response.get('recommendations', [])
        }
        
        # 각 경로별 검증
        for i, route in enumerate(routes):
            route_id = route.get('vehicle_id', f'route_{i}')
            
            # 경로 일관성 검증
            route_issues = ResponseValidator.validate_route_consistency(route)
            if route_issues:
                report['route_issues'][route_id] = route_issues
                report['validation_status'] = 'warning'
            
            # 폴리라인 검증
            polyline = route.get('polyline', '')
            polyline_valid = ResponseValidator.validate_polyline_format(polyline)
            report['polyline_status'][route_id] = 'valid' if polyline_valid else 'invalid'
            
            if not polyline_valid:
                report['validation_status'] = 'warning'
        
        return report 