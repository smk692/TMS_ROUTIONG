"""
API 응답 포맷터

API 응답을 일관된 형식으로 포맷팅합니다.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback

from src.domain.value_objects.optimization_result import OptimizationResult
from src.use_cases.process_feedback_use_case import FeedbackResult
from src.shared.exceptions import TmsError, ValidationError, AIServiceError


def format_success_response(data: Any, message: str = "Success", 
                          request_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    성공 응답 포맷팅
    
    Args:
        data: 응답 데이터
        message: 성공 메시지
        request_id: 요청 ID
        metadata: 추가 메타데이터
        
    Returns:
        포맷된 응답
    """
    response = {
        "status": "success",
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": _serialize_data(data)
    }
    
    if request_id:
        response["request_id"] = request_id
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def format_error_response(error: Exception, request_id: Optional[str] = None,
                         include_trace: bool = False) -> Dict[str, Any]:
    """
    에러 응답 포맷팅
    
    Args:
        error: 에러 객체
        request_id: 요청 ID
        include_trace: 스택 트레이스 포함 여부
        
    Returns:
        포맷된 에러 응답
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    # HTTP 상태 코드 결정
    status_code = _get_http_status_code(error)
    
    response = {
        "status": "error",
        "error": {
            "type": error_type,
            "message": error_message,
            "code": _get_error_code(error)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    if request_id:
        response["request_id"] = request_id
    
    # 개발 환경에서만 스택 트레이스 포함
    if include_trace:
        response["error"]["trace"] = traceback.format_exc()
    
    # ValidationError의 경우 추가 정보 포함
    if isinstance(error, ValidationError):
        response["error"]["field"] = getattr(error, 'field', 'unknown')
        response["error"]["details"] = getattr(error, 'details', {})
    
    response["status_code"] = status_code
    
    return response


def format_optimization_result(result: OptimizationResult) -> Dict[str, Any]:
    """
    최적화 결과 포맷팅
    
    Args:
        result: 최적화 결과
        
    Returns:
        포맷된 결과
    """
    return {
        "request_id": result.request_id,
        "scenario_type": result.scenario_type,
        "solution": {
            "routes": result.routes,
            "summary": {
                "total_distance_km": result.total_distance_km,
                "total_duration_hours": result.total_duration_hours,
                "total_cost": result.total_cost,
                "total_vehicles_used": len(result.routes),
                "total_orders_assigned": sum(len(route.get('orders', [])) for route in result.routes),
                "average_efficiency": _calculate_average_efficiency(result.routes)
            }
        },
        "analysis": result.analysis_reasoning,
        "reasoning": result.optimization_reasoning,
        "recommendations": result.recommendations,
        "warnings": result.warnings,
        "confidence_score": result.confidence_score,
        "metadata": {
            **result.processing_metadata,
            "generated_at": datetime.now().isoformat()
        }
    }


def format_feedback_result(result: FeedbackResult) -> Dict[str, Any]:
    """
    피드백 결과 포맷팅
    
    Args:
        result: 피드백 결과
        
    Returns:
        포맷된 결과
    """
    response = {
        "feedback_id": result.feedback_id,
        "status": result.status,
        "message": result.message,
        "processed_at": datetime.now().isoformat()
    }
    
    if result.conversation_summary:
        response["conversation_summary"] = result.conversation_summary
    
    return response


def format_health_check() -> Dict[str, Any]:
    """
    헬스 체크 응답 포맷팅
    
    Returns:
        헬스 체크 응답
    """
    return {
        "status": "healthy",
        "service": "tms-router-ai",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "operational",
            "ai_service": "operational",
            "memory": "operational"
        }
    }


def _serialize_data(data: Any) -> Any:
    """데이터 직렬화"""
    if hasattr(data, 'to_dict'):
        return data.to_dict()
    elif hasattr(data, '__dict__'):
        return data.__dict__
    elif isinstance(data, (list, tuple)):
        return [_serialize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: _serialize_data(value) for key, value in data.items()}
    else:
        return data


def _get_http_status_code(error: Exception) -> int:
    """에러 타입에 따른 HTTP 상태 코드 반환"""
    if isinstance(error, ValidationError):
        return 400  # Bad Request
    elif isinstance(error, AIServiceError):
        return 500  # Internal Server Error
    elif isinstance(error, TmsError):
        return 422  # Unprocessable Entity
    else:
        return 500  # Internal Server Error


def _get_error_code(error: Exception) -> str:
    """에러 코드 생성"""
    error_codes = {
        ValidationError: "VALIDATION_ERROR",
        AIServiceError: "AI_SERVICE_ERROR",
        TmsError: "TMS_ERROR",
    }
    
    return error_codes.get(type(error), "INTERNAL_ERROR")


def _calculate_average_efficiency(routes: List[Dict[str, Any]]) -> float:
    """경로들의 평균 효율성 계산"""
    if not routes:
        return 0.0
    
    efficiency_scores = []
    for route in routes:
        efficiency = route.get('efficiency_score', 0.0)
        if isinstance(efficiency, (int, float)):
            efficiency_scores.append(efficiency)
    
    if not efficiency_scores:
        return 0.0
    
    return sum(efficiency_scores) / len(efficiency_scores)


def format_api_documentation() -> Dict[str, Any]:
    """
    API 문서 포맷팅
    
    Returns:
        API 문서
    """
    return {
        "name": "TMS Router AI API",
        "version": "1.0.0",
        "description": "AI-powered Transportation Management System routing API",
        "endpoints": {
            "POST /optimize-route": {
                "description": "Optimize vehicle routes using AI",
                "parameters": {
                    "vehicles": "List of vehicles with capabilities",
                    "orders": "List of delivery orders",
                    "constraints": "Optimization constraints",
                    "scenario_type": "Optional scenario type (vrp, tsp, etc.)"
                },
                "response": "Optimized routes with polylines"
            },
            "POST /feedback": {
                "description": "Submit feedback on route optimization",
                "parameters": {
                    "conversation_id": "Conversation identifier",
                    "feedback_type": "Type of feedback (positive, negative, etc.)",
                    "feedback_content": "Feedback text",
                    "rating": "Optional rating (1-5)"
                },
                "response": "Feedback processing result"
            },
            "GET /health": {
                "description": "Health check endpoint",
                "response": "Service health status"
            }
        },
        "examples": {
            "optimize_request": {
                "vehicles": [
                    {
                        "vehicle_id": "truck_01",
                        "capacity_tons": 5.0,
                        "current_location": {"lat": 37.5665, "lng": 126.9780}
                    }
                ],
                "orders": [
                    {
                        "order_id": "order_001",
                        "pickup_location": {"lat": 37.5665, "lng": 126.9780},
                        "delivery_location": {"lat": 37.5759, "lng": 126.9768},
                        "weight_tons": 1.5
                    }
                ]
            }
        }
    } 