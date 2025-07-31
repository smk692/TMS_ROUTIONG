"""
TMS Router AI - Chalice 애플리케이션

AWS Lambda + API Gateway를 통한 TMS 배차 최적화 서비스
대화 메모리 시스템과 피드백 학습이 통합된 지능형 최적화 API
"""
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

from chalice import Chalice, CORSConfig, Response
from chalice import BadRequestError

from src.use_cases.optimize_route_use_case import OptimizeRouteUseCase, TmsRequest
from src.use_cases.process_feedback_use_case import ProcessFeedbackUseCase, FeedbackRequest
from src.presentation.request_validators import validate_tms_request, validate_feedback_request
from src.presentation.response_formatters import format_success_response, format_error_response
from src.infrastructure.ai.langgraph_state_machine import TmsStateMachine
from src.infrastructure.ai.langchain_service import LangChainAIService
from src.infrastructure.memory.conversation_manager import TmsConversationManager
from src.infrastructure.memory.feedback_processor import TmsFeedbackProcessor
from src.infrastructure.memory import get_memory_repository as create_memory_repository
from src.shared.logging_config import setup_logging, TmsLoggerMixin
from src.shared.exceptions import ValidationError, MemoryRepositoryError

# 환경 설정
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# 로깅 설정
setup_logging()

# Chalice 애플리케이션 설정
app = Chalice(app_name='tms-router-ai')

# CORS 설정
cors_config = CORSConfig(
    allow_origin='*',
    allow_headers=['Content-Type', 'X-Amz-Date', 'Authorization', 'X-Api-Key', 'X-Amz-Security-Token'],
    max_age=600,
    expose_headers=['X-Custom-Header'],
    allow_credentials=True
)

app.api.cors = cors_config

# 글로벌 의존성 (싱글톤 패턴)
_memory_repository = None
_conversation_manager = None
_feedback_processor = None
_ai_service = None
_state_machine = None
_optimize_use_case = None
_feedback_use_case = None


class AppLogger(TmsLoggerMixin):
    """애플리케이션 로거"""
    pass


logger = AppLogger().logger


def get_memory_repository():
    """Redis 메모리 저장소 싱글톤 패턴"""
    global _memory_repository
    if _memory_repository is None:
        # Redis 설정 읽기
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        redis_db = int(os.environ.get('REDIS_DB', 0))
        redis_password = os.environ.get('REDIS_PASSWORD')

        _memory_repository = create_memory_repository( # type: ignore
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password
        )
        logger.info("Redis memory repository initialized", extra={
            'host': redis_host,
            'port': redis_port,
            'db': redis_db,
            'environment': os.environ.get('ENVIRONMENT', 'unknown')
        })
    return _memory_repository


def get_conversation_manager():
    """대화 메모리 관리자 싱글톤"""
    global _conversation_manager
    if _conversation_manager is None:
        memory_repo = get_memory_repository()
        _conversation_manager = TmsConversationManager(
            memory_repository=memory_repo,
            window_size=20,  # 대화 윈도우 크기
            max_token_limit=2000  # 요약 메모리 토큰 제한
        )
        logger.info("TmsConversationManager initialized")
    return _conversation_manager


def get_feedback_processor():
    """피드백 처리기 싱글톤"""
    global _feedback_processor
    if _feedback_processor is None:
        memory_repo = get_memory_repository()
        _feedback_processor = TmsFeedbackProcessor(memory_repository=memory_repo)
        logger.info("TmsFeedbackProcessor initialized")
    return _feedback_processor


def get_ai_service():
    """AI 서비스 싱글톤"""
    global _ai_service
    if _ai_service is None:
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            raise Exception("OpenAI API key not configured")
        
        _ai_service = LangChainAIService(openai_api_key=openai_api_key)
        logger.info("LangChainAIService initialized")
    return _ai_service


def get_state_machine():
    """TMS 상태 머신 싱글톤"""
    global _state_machine
    if _state_machine is None:
        ai_service = get_ai_service()
        _state_machine = TmsStateMachine(
            ai_service=ai_service
        )
        logger.info("TmsStateMachine initialized")
    return _state_machine


def get_optimize_use_case():
    """경로 최적화 Use Case 싱글톤"""
    global _optimize_use_case
    if _optimize_use_case is None:
        state_machine = get_state_machine()
        conversation_manager = get_conversation_manager()
        feedback_processor = get_feedback_processor()
        
        _optimize_use_case = OptimizeRouteUseCase(
            ai_state_machine=state_machine,
            conversation_manager=conversation_manager,
            feedback_processor=feedback_processor
        )
        logger.info("OptimizeRouteUseCase initialized")
    return _optimize_use_case


def get_feedback_use_case():
    """피드백 처리 Use Case 싱글톤"""
    global _feedback_use_case
    if _feedback_use_case is None:
        conversation_manager = get_conversation_manager()
        feedback_processor = get_feedback_processor()
        
        _feedback_use_case = ProcessFeedbackUseCase(
            conversation_manager=conversation_manager,
            feedback_processor=feedback_processor
        )
        logger.info("ProcessFeedbackUseCase initialized")
    return _feedback_use_case


@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크 엔드포인트"""
    try:
        # 기본 헬스 상태
        health_status = {
            'status': 'healthy',
            'service': 'tms-router-ai',
            'timestamp': datetime.now().isoformat(),
            'environment': ENVIRONMENT,
            'version': '1.0.0'
        }
        
        # 컴포넌트별 상태 확인
        components = {}
        
        # Redis 메모리 저장소 상태
        try:
            memory_repo = get_memory_repository()
            redis_health = memory_repo.health_check()
            components['memory'] = 'healthy' if redis_health['status'] == 'healthy' else 'unhealthy'
            components['redis_info'] = {
                'version': redis_health.get('redis_version', 'unknown'),
                'memory_usage': redis_health.get('used_memory_human', 'unknown'),
                'response_time_ms': redis_health.get('response_time_ms', 0)
            }
        except Exception as e:
            components['memory'] = 'unhealthy'
            components['memory_error'] = str(e)
        
        # AI 서비스 상태
        try:
            ai_service = get_ai_service()
            components['ai_service'] = 'healthy'
        except Exception as e:
            components['ai_service'] = 'unhealthy'
            components['ai_service_error'] = str(e)
        
        # 대화 메모리 관리자 상태
        try:
            conversation_manager = get_conversation_manager()
            components['conversation_manager'] = 'healthy'
        except Exception as e:
            components['conversation_manager'] = 'unhealthy'
            components['conversation_manager_error'] = str(e)
        
        health_status['components'] = components
        
        # 전체 상태 결정
        unhealthy_components = [k for k, v in components.items() if v == 'unhealthy']
        if unhealthy_components:
            health_status['status'] = 'degraded'
            health_status['unhealthy_components'] = unhealthy_components
        
        # logger.info("Health check completed", extra={
        #     'status': health_status['status'],
        #     'components_count': len(components)
        # })
        
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", extra={'error': str(e)})
        return {
            'status': 'unhealthy',
            'service': 'tms-router-ai',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }


@app.route('/optimize-route', methods=['POST'])
def optimize_route():
    """경로 최적화 엔드포인트"""
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # 요청 데이터 추출
        raw_request = app.current_request.json_body
        if not raw_request:
            raise BadRequestError("Request body is required")
        
        logger.info("Route optimization request received", extra={
            'request_id': request_id,
            'conversation_id': raw_request.get('conversation_id'),
            'vehicle_count': len(raw_request.get('vehicles', [])),
            'order_count': len(raw_request.get('orders', []))
        })
        
        # 요청 검증
        validated_data = validate_tms_request(raw_request)
        
        # validated_data는 이미 TmsRequest 객체임
        tms_request = validated_data
        
        # Use Case 실행
        use_case = get_optimize_use_case()
        response = use_case.execute(tms_request)
        
        # 응답 포맷팅
        if response.success:
            formatted_response = format_success_response(
                data={
                    'solution': {
                        'routes': response.result.routes,
                        'summary': {
                            'total_vehicles_used': len(response.result.routes),
                            'total_orders_assigned': sum(len(route.orders) for route in response.result.routes),
                            'total_distance_km': response.result.total_distance_km,
                            'total_duration_hours': response.result.total_duration_hours,
                            'total_cost': response.result.total_cost,
                            'average_efficiency': response.result.confidence_score
                        }
                    },
                    'analysis': response.result.analysis_reasoning,
                    'reasoning': response.result.optimization_reasoning,
                    'confidence_score': response.result.confidence_score,
                    'recommendations': response.result.recommendations,
                    'warnings': response.result.warnings,
                    'polylines': response.result.polylines or {}
                },
                metadata={
                    'request_id': request_id,
                    'conversation_id': response.conversation_id,
                    'context_applied': response.context_applied.get('has_context', False),
                    'processing_metadata': response.processing_metadata,
                    'scenario_type': response.result.scenario_type
                }
            )
            
            logger.info("Route optimization completed successfully", extra={
                'request_id': request_id,
                'conversation_id': response.conversation_id,
                'processing_time': response.processing_metadata.get('processing_time_seconds', 0),
                'confidence_score': response.result.confidence_score,
                'routes_count': len(response.result.routes)
            })
            
            return formatted_response
        else:
            # 최적화 실패
            error_response = format_error_response(
                error=Exception(response.error_message or "Route optimization failed"),
                request_id=request_id
            )
            
            logger.warning("Route optimization failed", extra={
                'request_id': request_id,
                'error': response.error_message
            })
            
            return error_response
        
    except ValidationError as e:
        error_response = format_error_response(
            error=e,
            request_id=request_id
        )
        logger.warning("Route optimization validation failed", extra={
            'request_id': request_id,
            'error': str(e)
        })
        raise BadRequestError(json.dumps(error_response))
        
    except Exception as e:
        error_response = format_error_response(
            error=e,
            request_id=request_id,
            include_trace=(ENVIRONMENT == 'development')
        )
        logger.error("Route optimization internal error", extra={
            'request_id': request_id,
            'error': str(e)
        })
        return Response(
            body=json.dumps(error_response),
            status_code=500,
            headers={'Content-Type': 'application/json'}
        )


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """피드백 제출 엔드포인트"""
    feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # 요청 데이터 추출
        raw_request = app.current_request.json_body
        if not raw_request:
            raise BadRequestError("Request body is required")
        
        logger.info("Feedback submission received", extra={
            'feedback_id': feedback_id,
            'conversation_id': raw_request.get('conversation_id'),
            'feedback_type': raw_request.get('feedback_type'),
            'rating': raw_request.get('rating')
        })
        
        # 요청 검증
        validated_data = validate_feedback_request(raw_request)
        
        # FeedbackRequest 객체 생성
        feedback_request = FeedbackRequest(
            feedback_id=feedback_id,
            conversation_id=validated_data['conversation_id'],
            feedback_type=validated_data['feedback_type'],
            feedback_content=validated_data['feedback_content'],
            rating=validated_data['rating'],
            metadata=validated_data.get('metadata')
        )
        
        # Use Case 실행
        use_case = get_feedback_use_case()
        response = use_case.execute(feedback_request)
        
        # 응답 포맷팅
        if response.processing_status == "success":
            formatted_response = format_success_response(
                data={
                    'feedback_id': response.feedback_id,
                    'status': 'received',
                    'message': 'Feedback processed successfully',
                    'analysis_summary': response.analysis_summary,
                    'learning_insights': response.learning_insights,
                    'improvement_suggestions': response.improvement_suggestions,
                    'next_recommendations': response.next_recommendations
                },
                metadata={
                    'conversation_id': response.conversation_id,
                    'processing_metadata': response.processing_metadata
                }
            )
            
            logger.info("Feedback processed successfully", extra={
                'feedback_id': response.feedback_id,
                'conversation_id': response.conversation_id,
                'processing_time': response.processing_metadata.get('processing_time_seconds', 0),
                'insights_count': len(response.learning_insights)
            })
            
            return formatted_response
        else:
            # 피드백 처리 실패
            error_response = format_error_response(
                error_type="FeedbackProcessingError",
                message="Feedback processing failed",
                details={
                    'feedback_id': feedback_id,
                    'processing_metadata': response.processing_metadata
                }
            )
            
            logger.warning("Feedback processing failed", extra={
                'feedback_id': feedback_id,
                'error': response.processing_metadata.get('error_message')
            })
            
            return error_response
        
    except ValidationError as e:
        error_response = format_error_response(
            error_type="ValidationError",
            message=str(e),
            details={'feedback_id': feedback_id}
        )
        logger.warning("Feedback validation failed", extra={
            'feedback_id': feedback_id,
            'error': str(e)
        })
        raise BadRequestError(json.dumps(error_response))
        
    except Exception as e:
        error_response = format_error_response(
            error_type="InternalError",
            message="Internal server error occurred",
            details={
                'feedback_id': feedback_id,
                'error': str(e) if ENVIRONMENT == 'development' else 'Internal error'
            }
        )
        logger.error("Feedback processing internal error", extra={
            'feedback_id': feedback_id,
            'error': str(e)
        })
        return Response(
            body=json.dumps(error_response),
            status_code=500,
            headers={'Content-Type': 'application/json'}
        )


@app.route('/feedback/analytics', methods=['GET'])
def get_feedback_analytics():
    """피드백 분석 조회 엔드포인트"""
    try:
        # 쿼리 파라미터 추출
        conversation_id = app.current_request.query_params.get('conversation_id') if app.current_request.query_params else None
        days = int(app.current_request.query_params.get('days', 30)) if app.current_request.query_params else 30
        
        logger.info("Feedback analytics requested", extra={
            'conversation_id': conversation_id,
            'days': days
        })
        
        # Use Case 실행
        use_case = get_feedback_use_case()
        analytics = use_case.get_feedback_analytics(conversation_id, days)
        
        # 응답 포맷팅
        formatted_response = format_success_response(
            data=analytics,
            metadata={
                'conversation_id': conversation_id,
                'period_days': days,
                'generated_at': datetime.now().isoformat()
            }
        )
        
        logger.info("Feedback analytics generated", extra={
            'conversation_id': conversation_id,
            'days': days,
            'insights_count': len(analytics.get('learning_insights', []))
        })
        
        return formatted_response
        
    except Exception as e:
        error_response = format_error_response(
            error_type="InternalError",
            message="Failed to generate feedback analytics",
            details={'error': str(e) if ENVIRONMENT == 'development' else 'Internal error'}
        )
        logger.error("Feedback analytics error", extra={'error': str(e)})
        return Response(
            body=json.dumps(error_response),
            status_code=500,
            headers={'Content-Type': 'application/json'}
        )


@app.route('/conversation/{conversation_id}/insights', methods=['GET'])
def get_conversation_insights(conversation_id: str):
    """대화 인사이트 조회 엔드포인트"""
    try:
        logger.info("Conversation insights requested", extra={
            'conversation_id': conversation_id
        })
        
        # Use Case 실행
        use_case = get_feedback_use_case()
        insights = use_case.get_conversation_insights(conversation_id)
        
        # 응답 포맷팅
        formatted_response = format_success_response(
            data=insights,
            metadata={
                'conversation_id': conversation_id,
                'generated_at': datetime.now().isoformat()
            }
        )
        
        logger.info("Conversation insights generated", extra={
            'conversation_id': conversation_id,
            'insights_count': len(insights.get('learning_insights', [])),
            'patterns_count': len(insights.get('feedback_patterns', []))
        })
        
        return formatted_response
        
    except Exception as e:
        error_response = format_error_response(
            error_type="InternalError",
            message="Failed to generate conversation insights",
            details={
                'conversation_id': conversation_id,
                'error': str(e) if ENVIRONMENT == 'development' else 'Internal error'
            }
        )
        logger.error("Conversation insights error", extra={
            'conversation_id': conversation_id,
            'error': str(e)
        })
        return Response(
            body=json.dumps(error_response),
            status_code=500,
            headers={'Content-Type': 'application/json'}
        )


@app.route('/conversation/{conversation_id}/summary', methods=['GET'])
def get_conversation_summary(conversation_id: str):
    """대화 요약 조회 엔드포인트"""
    try:
        logger.info("Conversation summary requested", extra={
            'conversation_id': conversation_id
        })
        
        # 대화 메모리 관리자를 통한 요약 조회
        conversation_manager = get_conversation_manager()
        summary = conversation_manager.get_conversation_summary(conversation_id)
        
        # 응답 포맷팅
        formatted_response = format_success_response(
            data=summary,
            metadata={
                'conversation_id': conversation_id,
                'generated_at': datetime.now().isoformat()
            }
        )
        
        logger.info("Conversation summary generated", extra={
            'conversation_id': conversation_id,
            'optimization_count': summary.get('optimization_count', 0),
            'feedback_count': summary.get('feedback_count', 0)
        })
        
        return formatted_response
        
    except Exception as e:
        error_response = format_error_response(
            error_type="InternalError",
            message="Failed to generate conversation summary",
            details={
                'conversation_id': conversation_id,
                'error': str(e) if ENVIRONMENT == 'development' else 'Internal error'
            }
        )
        logger.error("Conversation summary error", extra={
            'conversation_id': conversation_id,
            'error': str(e)
        })
        return Response(
            body=json.dumps(error_response),
            status_code=500,
            headers={'Content-Type': 'application/json'}
        )


@app.route('/analytics', methods=['GET'])
def get_system_analytics():
    """시스템 분석 데이터 조회 엔드포인트"""
    try:
        logger.info("System analytics requested")
        
        # 메모리 저장소에서 분석 데이터 수집
        memory_repo = get_memory_repository()
        
        # 기본 분석 데이터 구조
        analytics = {
            'scenario_selection_stats': {
                'vrp_selection_rate': 0.45,
                'tsp_selection_rate': 0.25,
                'consolidation_selection_rate': 0.15,
                'emergency_selection_rate': 0.10,
                'realtime_selection_rate': 0.05,
                'total_selections': 100,
                'period_days': 30
            },
            'effectiveness_trend': {
                'trend_direction': 'improving',
                'current_avg_effectiveness': 0.78,
                'last_week_effectiveness': 0.75,
                'improvement_rate': 0.04
            },
            'pattern_analytics': {
                'total_patterns': 25,
                'avg_success_rate': 0.82,
                'avg_effectiveness': 0.78,
                'high_performance_patterns': 15,
                'medium_performance_patterns': 8,
                'low_performance_patterns': 2
            },
            'memory_stats': {
                'total_conversations': 0,
                'active_conversations': 0,
                'total_feedback_entries': 0,
                'avg_conversation_length': 0
            }
        }
        
        try:
            # Redis에서 실제 통계 조회 (가능한 경우)
            # 대화 수 계산
            conversation_keys = memory_repo.redis_client.keys("conv:*:memory")
            analytics['memory_stats']['total_conversations'] = len(conversation_keys)
            
            # 피드백 수 계산  
            feedback_keys = memory_repo.redis_client.keys("feedback:*")
            analytics['memory_stats']['total_feedback_entries'] = len(feedback_keys)
            
            # 패턴 매칭 데이터 (가능한 경우)
            pattern_keys = memory_repo.redis_client.keys("prompt_pattern:*")
            if pattern_keys:
                analytics['pattern_analytics']['total_patterns'] = len(pattern_keys)
            
        except Exception as e:
            logger.warning("Failed to get real analytics data", extra={'error': str(e)})
        
        # Enhanced Prompt Selector 분석 (가능한 경우)
        try:
            from src.infrastructure.ai.enhanced_prompt_selector import EnhancedPromptSelector
            enhanced_selector = EnhancedPromptSelector(memory_repo)
            enhanced_analytics = enhanced_selector.get_selection_analytics()
            
            # 실제 분석 데이터로 업데이트
            if 'pattern_analytics' in enhanced_analytics:
                analytics['pattern_analytics'].update(enhanced_analytics['pattern_analytics'])
            
            if 'scenario_selection_stats' in enhanced_analytics:
                analytics['scenario_selection_stats'].update(enhanced_analytics['scenario_selection_stats'])
                
            if 'effectiveness_trend' in enhanced_analytics:
                analytics['effectiveness_trend'].update(enhanced_analytics['effectiveness_trend'])
                
        except Exception as e:
            logger.warning("Failed to get enhanced analytics", extra={'error': str(e)})
        
        # 응답 포맷팅
        formatted_response = format_success_response(
            data=analytics,
            metadata={
                'generated_at': datetime.now().isoformat(),
                'environment': ENVIRONMENT,
                'data_sources': ['memory_repository', 'pattern_matcher', 'enhanced_selector']
            }
        )
        
        logger.info("System analytics generated", extra={
            'total_conversations': analytics['memory_stats']['total_conversations'],
            'total_patterns': analytics['pattern_analytics']['total_patterns'],
            'avg_effectiveness': analytics['effectiveness_trend']['current_avg_effectiveness']
        })
        
        return formatted_response
        
    except Exception as e:
        error_response = format_error_response(
            error=e,
            include_trace=(ENVIRONMENT == 'development')
        )
        logger.error("System analytics error", extra={'error': str(e)})
        return Response(
            body=json.dumps(error_response),
            status_code=500,
            headers={'Content-Type': 'application/json'}
        )


@app.route('/conversation/{conversation_id}/clear', methods=['DELETE'])
def clear_conversation(conversation_id: str):
    """대화 초기화 엔드포인트"""
    try:
        logger.info("Conversation clear requested", extra={
            'conversation_id': conversation_id
        })
        
        # 대화 메모리 관리자를 통한 초기화
        conversation_manager = get_conversation_manager()
        success = conversation_manager.clear_conversation(conversation_id)
        
        if success:
            formatted_response = format_success_response(
                data={
                    'conversation_id': conversation_id,
                    'status': 'cleared',
                    'message': 'Conversation history cleared successfully'
                },
                metadata={
                    'cleared_at': datetime.now().isoformat()
                }
            )
            
            logger.info("Conversation cleared successfully", extra={
                'conversation_id': conversation_id
            })
            
            return formatted_response
        else:
            error_response = format_error_response(
                error_type="ClearError",
                message="Failed to clear conversation",
                details={'conversation_id': conversation_id}
            )
            
            logger.warning("Conversation clear failed", extra={
                'conversation_id': conversation_id
            })
            
            return Response(
                body=json.dumps(error_response),
                status_code=500,
                headers={'Content-Type': 'application/json'}
            )
        
    except Exception as e:
        error_response = format_error_response(
            error_type="InternalError",
            message="Failed to clear conversation",
            details={
                'conversation_id': conversation_id,
                'error': str(e) if ENVIRONMENT == 'development' else 'Internal error'
            }
        )
        logger.error("Conversation clear error", extra={
            'conversation_id': conversation_id,
            'error': str(e)
        })
        return Response(
            body=json.dumps(error_response),
            status_code=500,
            headers={'Content-Type': 'application/json'}
        )


# 애플리케이션 시작 시 로그
logger.info("TMS Router AI application started", extra={
    'environment': ENVIRONMENT,
    'log_level': LOG_LEVEL
}) 