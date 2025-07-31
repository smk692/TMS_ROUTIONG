"""
TMS 경로 최적화 Use Case

Clean Architecture의 Use Case 레이어에서 TMS 배차 최적화 비즈니스 로직을 처리합니다.
대화 메모리 시스템과 통합하여 컨텍스트 기반 최적화를 제공합니다.
"""
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.shared.logging_config import TmsLoggerMixin
from src.shared.exceptions import ValidationError, OptimizationError
from src.shared.constants import ScenarioType
from src.domain.entities.vehicle import Vehicle
from src.domain.entities.delivery_order import DeliveryOrder
from src.domain.entities.route import Route
from src.domain.value_objects.optimization_result import OptimizationResult
from src.infrastructure.ai.langgraph_state_machine import TmsStateMachine
from src.infrastructure.memory.conversation_manager import TmsConversationManager
from src.infrastructure.memory.feedback_processor import TmsFeedbackProcessor


@dataclass
class TmsRequest:
    """TMS 최적화 요청"""
    request_id: str
    conversation_id: Optional[str]
    vehicles: List[Dict[str, Any]]
    orders: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    preferences: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationResponse:
    """최적화 응답"""
    request_id: str
    conversation_id: Optional[str]
    success: bool
    result: Optional[OptimizationResult]
    context_applied: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    error_message: Optional[str] = None


class OptimizeRouteUseCase(TmsLoggerMixin):
    """경로 최적화 Use Case"""
    
    def __init__(self, 
                 ai_state_machine: TmsStateMachine,
                 conversation_manager: TmsConversationManager,
                 feedback_processor: TmsFeedbackProcessor):
        """
        Args:
            ai_state_machine: TMS AI 상태 머신
            conversation_manager: 대화 메모리 관리자
            feedback_processor: 피드백 처리기
        """
        super().__init__()  # TmsLoggerMixin 초기화
        self.ai_state_machine = ai_state_machine
        self.conversation_manager = conversation_manager
        self.feedback_processor = feedback_processor
        
        self.logger.info("OptimizeRouteUseCase initialized")
    
    def execute(self, request: TmsRequest) -> OptimizationResponse:
        """경로 최적화 실행"""
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting route optimization", extra={
                'request_id': request.request_id,
                'conversation_id': request.conversation_id,
                'vehicle_count': len(request.vehicles),
                'order_count': len(request.orders)
            })
            
            # 1. 요청 검증
            self._validate_request(request)
            
            # 2. 대화 컨텍스트 조회 및 적용
            context_info = self._get_conversation_context(request)
            
            # 3. 도메인 객체 변환
            vehicles, orders = self._convert_to_domain_objects(request)
            
            # 4. 컨텍스트 기반 최적화 실행
            optimization_result = self._execute_optimization_with_context(
                request, vehicles, orders, context_info
            )
            
            # 5. 결과 검증 및 후처리
            validated_result = self._validate_and_enhance_result(
                optimization_result, context_info
            )
            
            # 6. 대화 메모리 업데이트
            self._update_conversation_memory(request, validated_result, context_info)
            
            # 7. 응답 생성
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = OptimizationResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                success=True,
                result=validated_result,
                context_applied=context_info,
                processing_metadata={
                    'processing_time_seconds': processing_time,
                    'context_used': bool(request.conversation_id),
                    'optimization_type': validated_result.scenario_type,
                    'confidence_score': validated_result.confidence_score,
                    'polyline_status': 'included' if validated_result.polylines else 'not_available'
                }
            )
            
            self.logger.info("Route optimization completed successfully", extra={
                'request_id': request.request_id,
                'processing_time': processing_time,
                'confidence_score': validated_result.confidence_score,
                'routes_generated': len(validated_result.routes)
            })
            
            return response
            
        except ValidationError as e:
            self.logger.error("Request validation failed", extra={
                'request_id': request.request_id,
                'error': str(e)
            })
            return self._create_error_response(request, f"Validation error: {e}")
            
        except OptimizationError as e:
            self.logger.error("Optimization failed", extra={
                'request_id': request.request_id,
                'error': str(e)
            })
            return self._create_error_response(request, f"Optimization error: {e}")
            
        except Exception as e:
            self.logger.error("Unexpected error during optimization", extra={
                'request_id': request.request_id,
                'error': str(e)
            })
            return self._create_error_response(request, f"Internal error: {e}")
    
    def _validate_request(self, request: TmsRequest) -> None:
        """요청 검증"""
        if not request.vehicles:
            raise ValidationError("At least one vehicle is required")
        
        if not request.orders:
            raise ValidationError("At least one order is required")
        
        # 차량 용량 vs 주문 중량 검증
        total_vehicle_capacity = sum(v.get('capacity_tons', 0) for v in request.vehicles)
        total_order_weight = sum(o.get('weight_tons', 0) for o in request.orders)
        
        if total_order_weight > total_vehicle_capacity:
            raise ValidationError(
                f"Total order weight ({total_order_weight:.1f}t) exceeds "
                f"total vehicle capacity ({total_vehicle_capacity:.1f}t)"
            )
        
        # 좌표 유효성 검증
        for vehicle in request.vehicles:
            location = vehicle.get('current_location', {})
            if not self._is_valid_coordinate(location):
                raise ValidationError(f"Invalid vehicle location: {location}")
        
        for order in request.orders:
            pickup = order.get('pickup_location', {})
            delivery = order.get('delivery_location', {})
            if not self._is_valid_coordinate(pickup):
                raise ValidationError(f"Invalid pickup location: {pickup}")
            if not self._is_valid_coordinate(delivery):
                raise ValidationError(f"Invalid delivery location: {delivery}")
    
    def _get_conversation_context(self, request: TmsRequest) -> Dict[str, Any]:
        """대화 컨텍스트 조회"""
        if not request.conversation_id:
            return {
                'has_context': False,
                'user_preferences': {},
                'recent_messages': [],
                'learned_patterns': {},
                'preference_weights': {'distance': 0.33, 'time': 0.33, 'cost': 0.34},
                'context_hints': []
            }
        
        try:
            # 사용자 메시지 추가 (요청이 있는 경우)
            if request.metadata and request.metadata.get('user_message'):
                self.conversation_manager.add_user_message(
                    request.conversation_id,
                    request.metadata['user_message'],
                    {
                        'request_id': request.request_id,
                        'vehicle_count': len(request.vehicles),
                        'order_count': len(request.orders)
                    }
                )
            
            # 최적화 컨텍스트 조회
            context = self.conversation_manager.get_context_for_optimization(request.conversation_id)
            context['has_context'] = True
            
            self.logger.debug("Retrieved conversation context", extra={
                'conversation_id': request.conversation_id,
                'preference_weights': context.get('preference_weights', {}),
                'feedback_count': len(context.get('feedback_summary', {}).get('total_count', 0))
            })
            
            return context
            
        except Exception as e:
            self.logger.warning("Failed to retrieve conversation context", extra={
                'conversation_id': request.conversation_id,
                'error': str(e)
            })
            # 컨텍스트 조회 실패 시 기본값 반환
            return {
                'has_context': False,
                'user_preferences': {},
                'recent_messages': [],
                'learned_patterns': {},
                'preference_weights': {'distance': 0.33, 'time': 0.33, 'cost': 0.34},
                'context_hints': [],
                'context_error': str(e)
            }
    
    def _convert_to_domain_objects(self, request: TmsRequest) -> tuple[List[Vehicle], List[DeliveryOrder]]:
        """요청 데이터를 도메인 객체로 변환"""
        vehicles = []
        for vehicle_data in request.vehicles:
            vehicle = Vehicle.from_dict(vehicle_data)
            vehicles.append(vehicle)
        
        orders = []
        for order_data in request.orders:
            order = DeliveryOrder.from_dict(order_data)
            orders.append(order)
        
        return vehicles, orders
    
    def _execute_optimization_with_context(self, 
                                         request: TmsRequest,
                                         vehicles: List[Vehicle],
                                         orders: List[DeliveryOrder],
                                         context_info: Dict[str, Any]) -> OptimizationResult:
        """컨텍스트 기반 최적화 실행"""
        
        # 컨텍스트 기반 제약조건 조정
        enhanced_constraints = self._enhance_constraints_with_context(
            request.constraints, context_info
        )
        
        # AI 상태 머신에 컨텍스트 정보 전달
        ai_context = {
            'conversation_context': context_info,
            'user_preferences': context_info.get('user_preferences', {}),
            'preference_weights': context_info.get('preference_weights', {}),
            'learned_patterns': context_info.get('learned_patterns', {}),
            'context_hints': context_info.get('context_hints', []),
            'recent_feedback_summary': context_info.get('feedback_summary', {})
        }
        
        # TMS 상태 머신으로 최적화 실행
        try:
            # 요청 파라미터 구성
            request_parameters = {
                'vehicles': [v.to_dict() for v in vehicles],
                'orders': [o.to_dict() for o in orders],
                'constraints': enhanced_constraints,
                'ai_context': ai_context
            }
            
            result = self.ai_state_machine.process_tms_request(
                request_parameters=request_parameters,
                request_id=request.request_id
            )
            
            if not result:
                raise OptimizationError("AI optimization failed to generate valid result")
            
            # AI 상태 머신 결과에서 solution 추출
            solution = result.get('solution', {})
            if not solution:
                self.logger.error("No solution found in AI state machine result", extra={
                    'request_id': request.request_id,
                    'result_keys': list(result.keys()) if result else [],
                    'result': str(result)[:500] if result else "None"
                })
                raise OptimizationError("No routes generated")
            
            # solution에서 경로 정보 추출 (AI 응답 구조에 따라)
            routes = []
            
            # 에러 발생 지점에 상세 디버깅 정보 출력
            import json
            
            # solution 객체의 상세 분석
            debug_info = {
                'solution_type': type(solution).__name__,
                'solution_keys': list(solution.keys()) if isinstance(solution, dict) else [],
                'solution_str': str(solution)[:1000],
                'has_routes': 'routes' in solution if isinstance(solution, dict) else False,
                'has_solution': 'solution' in solution if isinstance(solution, dict) else False,
                'routes_direct': solution.get('routes', []) if isinstance(solution, dict) else [],
                'inner_solution': solution.get('solution', {}) if isinstance(solution, dict) else {},
                'inner_routes': solution.get('solution', {}).get('routes', []) if isinstance(solution, dict) and 'solution' in solution else []
            }
            
            self.logger.error("DEBUG: Solution analysis at error point", extra={
                'request_id': request.request_id,
                'debug_info': json.dumps(debug_info, indent=2, ensure_ascii=False)
            })
            
            if isinstance(solution, dict):
                # 직접 routes 키가 있는 경우
                routes = solution.get('routes', [])
                if not routes and 'solution' in solution:
                    # solution 안에 또 다른 solution이 있는 경우
                    inner_solution = solution.get('solution', {})
                    routes = inner_solution.get('routes', [])
                
                # 디버깅을 위한 로그 추가
                self.logger.info("Solution structure analysis", extra={
                    'request_id': request.request_id,
                    'solution_keys': list(solution.keys()) if solution else [],
                    'has_routes': 'routes' in solution,
                    'has_inner_solution': 'solution' in solution,
                    'routes_found': len(routes),
                    'solution_type': type(solution).__name__,
                    'solution_str': str(solution)[:200]
                })
            
            if not routes:
                self.logger.error("No routes found in solution", extra={
                    'request_id': request.request_id,
                    'solution_keys': list(solution.keys()) if solution else [],
                    'solution': str(solution)[:1000] if solution else "None",
                    'solution_type': type(solution).__name__,
                    'has_solution_key': 'solution' in solution if isinstance(solution, dict) else False,
                    'inner_solution_keys': list(solution.get('solution', {}).keys()) if isinstance(solution, dict) and 'solution' in solution else []
                })
                raise OptimizationError("No routes generated")
            
            # 결과를 OptimizationResult로 변환
            optimization_result = OptimizationResult(
                request_id=request.request_id,
                scenario_type=ScenarioType(result.get('scenario_type', 'vrp')),
                routes=routes,
                confidence_score=result.get('metadata', {}).get('confidence_score', 0.0),
                ai_reasoning=solution.get('reasoning', 'AI optimization completed'),
                total_distance_km=solution.get('total_distance_km', 0.0),
                total_estimated_duration_hours=solution.get('total_duration_hours', 0.0),
                optimization_metrics=result.get('metadata', {})
            )
            
            return optimization_result
            
        except Exception as e:
            self.logger.error("AI optimization execution failed", extra={
                'request_id': request.request_id,
                'error': str(e)
            })
            raise OptimizationError(f"Optimization execution failed: {e}")
    
    def _enhance_constraints_with_context(self, 
                                        base_constraints: Dict[str, Any],
                                        context_info: Dict[str, Any]) -> Dict[str, Any]:
        """컨텍스트 기반 제약조건 강화"""
        enhanced = base_constraints.copy()
        
        # 학습된 선호도 가중치 적용
        preference_weights = context_info.get('preference_weights', {})
        if preference_weights:
            enhanced['optimization_weights'] = preference_weights
        
        # 사용자 선호도 적용
        user_prefs = context_info.get('user_preferences', {})
        if user_prefs.get('optimization_priority'):
            enhanced['priority'] = user_prefs['optimization_priority']
        
        # 컨텍스트 힌트 적용
        context_hints = context_info.get('context_hints', [])
        if context_hints:
            enhanced['context_hints'] = context_hints
        
        # 성공 패턴 기반 조정
        learned_patterns = context_info.get('learned_patterns', {})
        successful_scenarios = learned_patterns.get('successful_scenarios', [])
        if len(successful_scenarios) >= 3:
            # 최근 성공 패턴의 공통 특성 추출
            common_topics = self._extract_common_topics(successful_scenarios[-3:])
            if common_topics:
                enhanced['preferred_optimization_focus'] = common_topics
        
        self.logger.debug("Enhanced constraints with context", extra={
            'has_weights': 'optimization_weights' in enhanced,
            'has_priority': 'priority' in enhanced,
            'hints_count': len(enhanced.get('context_hints', []))
        })
        
        return enhanced
    
    def _validate_and_enhance_result(self, 
                                   result: OptimizationResult, 
                                   context_info: Dict[str, Any]) -> OptimizationResult:
        """결과 검증 및 강화"""
        if not result.routes:
            raise OptimizationError("No routes generated")
        
        # 컨텍스트 기반 결과 평가
        context_evaluation = self._evaluate_result_against_context(result, context_info)
        
        # 결과에 컨텍스트 평가 추가
        if hasattr(result, 'metadata'):
            result.metadata.update(context_evaluation)
        else:
            result.metadata = context_evaluation
        
        return result
    
    def _update_conversation_memory(self, 
                                  request: TmsRequest,
                                  result: OptimizationResult,
                                  context_info: Dict[str, Any]) -> None:
        """대화 메모리 업데이트"""
        if not request.conversation_id:
            return
        
        try:
            # AI 응답 메시지 생성
            ai_response_content = self._generate_ai_response_message(result, context_info)
            
            # 최적화 결과 메타데이터
            optimization_metadata = {
                'scenario_type': result.scenario_type,
                'confidence_score': result.confidence_score,
                'routes': [
                    {
                        'vehicle_id': route.vehicle_id,
                        'order_count': len(route.orders),
                        'total_distance_km': route.total_distance_km,
                        'total_duration_hours': route.total_duration_hours,
                        'estimated_cost': route.estimated_cost
                    } for route in result.routes
                ],
                'total_distance_km': result.total_distance_km,
                'total_cost': result.total_cost,
                'polylines_included': bool(result.polylines)
            }
            
            # AI 메시지 추가
            self.conversation_manager.add_ai_message(
                request.conversation_id,
                ai_response_content,
                optimization_metadata,
                {
                    'request_id': request.request_id,
                    'context_applied': context_info.get('has_context', False),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            self.logger.debug("Updated conversation memory", extra={
                'conversation_id': request.conversation_id,
                'confidence_score': result.confidence_score
            })
            
        except Exception as e:
            self.logger.warning("Failed to update conversation memory", extra={
                'conversation_id': request.conversation_id,
                'error': str(e)
            })
    
    def _generate_ai_response_message(self, 
                                    result: OptimizationResult,
                                    context_info: Dict[str, Any]) -> str:
        """AI 응답 메시지 생성"""
        message_parts = []
        
        # 기본 결과 정보
        message_parts.append(f"최적화 완료: {len(result.routes)}개 경로 생성")
        message_parts.append(f"총 거리: {result.total_distance_km:.1f}km")
        message_parts.append(f"예상 비용: {result.total_cost:,}원")
        message_parts.append(f"신뢰도: {result.confidence_score:.2f}")
        
        # 컨텍스트 적용 정보
        if context_info.get('has_context'):
            preference_weights = context_info.get('preference_weights', {})
            max_weight = max(preference_weights.keys(), key=lambda x: preference_weights[x])
            message_parts.append(f"최적화 우선순위: {max_weight}")
            
            hints = context_info.get('context_hints', [])
            if hints:
                message_parts.append(f"학습된 선호도 반영: {hints[0]}")
        
        # 경로별 상세 정보
        for i, route in enumerate(result.routes, 1):
            message_parts.append(
                f"경로 {i}: {route.vehicle_id} - "
                f"{len(route.orders)}개 주문, "
                f"{route.total_distance_km:.1f}km"
            )
        
        return " | ".join(message_parts)
    
    def _create_error_response(self, request: TmsRequest, error_message: str) -> OptimizationResponse:
        """에러 응답 생성"""
        return OptimizationResponse(
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            success=False,
            result=None,
            context_applied={},
            processing_metadata={
                'error': True,
                'error_message': error_message
            },
            error_message=error_message
        )
    
    # Helper methods
    
    def _is_valid_coordinate(self, location: Dict[str, Any]) -> bool:
        """좌표 유효성 검증"""
        try:
            lat = float(location.get('lat', 0))
            lng = float(location.get('lng', 0))
            return -90 <= lat <= 90 and -180 <= lng <= 180
        except (ValueError, TypeError):
            return False
    
    def _extract_common_topics(self, scenarios: List[Dict[str, Any]]) -> List[str]:
        """성공 시나리오에서 공통 토픽 추출"""
        topic_counts = {}
        for scenario in scenarios:
            for topic in scenario.get('topics', []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # 2회 이상 등장한 토픽들을 공통 토픽으로 간주
        common_topics = [topic for topic, count in topic_counts.items() if count >= 2]
        return common_topics
    
    def _evaluate_result_against_context(self, 
                                       result: OptimizationResult,
                                       context_info: Dict[str, Any]) -> Dict[str, Any]:
        """컨텍스트 대비 결과 평가"""
        evaluation = {
            'context_alignment_score': 0.0,
            'preference_match': {},
            'pattern_consistency': 'unknown'
        }
        
        if not context_info.get('has_context'):
            return evaluation
        
        # 선호도 가중치와 결과 일치도 평가
        preference_weights = context_info.get('preference_weights', {})
        if preference_weights:
            # 단순화된 평가 (실제로는 더 복잡한 로직 필요)
            max_preference = max(preference_weights.keys(), key=lambda x: preference_weights[x])
            evaluation['preference_match']['primary_preference'] = max_preference
            evaluation['context_alignment_score'] = 0.8  # 임시 점수
        
        # 성공 패턴 일치도 평가
        successful_scenarios = context_info.get('learned_patterns', {}).get('successful_scenarios', [])
        if successful_scenarios:
            evaluation['pattern_consistency'] = 'high' if result.confidence_score >= 0.8 else 'medium'
        
        return evaluation 