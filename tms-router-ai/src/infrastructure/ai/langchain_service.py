"""
LangChainAIService - LangChain 기반 AI 서비스

GPT 모델과의 상호작용을 담당하는 핵심 서비스입니다.
프롬프트 실행, 응답 파싱, 에러 처리를 포함합니다.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.infrastructure.ai.prompt_templates import TmsPromptTemplates
from src.infrastructure.ai.prompt_selector import PromptSelector, PromptSelectionResult
from src.infrastructure.ai.response_validator import ResponseValidator
from src.infrastructure.ai.polyline_generator import get_polyline_generator
from src.shared.exceptions import AIServiceError, ValidationError
from src.shared.constants import AiConstants
from src.shared.logging_config import TmsLoggerMixin


@dataclass
class AIResponse:
    """AI 응답 결과"""
    success: bool
    response_data: Dict[str, Any]
    raw_response: str
    prompt_used: str
    scenario_type: str
    confidence_score: float
    token_usage: Dict[str, int]
    processing_time_ms: int
    validation_report: Dict[str, Any]


class LangChainAIService(TmsLoggerMixin):
    """LangChain 기반 AI 서비스"""
    
    def __init__(self, 
                 model_name: str = "gpt-4-1106-preview",
                 temperature: float = 0.1,
                 max_tokens: int = 4000,
                 openai_api_key: Optional[str] = None):
        """
        AI 서비스 초기화
        
        Args:
            model_name: OpenAI 모델명
            temperature: 창의성 조절 (0.0-1.0)
            max_tokens: 최대 토큰 수
            openai_api_key: OpenAI API 키
        """
        super().__init__()
        
        # OpenAI API 키 설정
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise AIServiceError("OpenAI API key is required")
        
        # ChatOpenAI 모델 초기화
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=self.api_key,
            request_timeout=AiConstants.REQUEST_TIMEOUT_SECONDS,
            max_retries=AiConstants.MAX_RETRIES
        )
        
        # 종속성 주입
        self.prompt_selector = PromptSelector()
        self.prompt_templates = TmsPromptTemplates()
        self.response_validator = ResponseValidator()
        self.polyline_generator = get_polyline_generator()
        
        self.logger.info("LangChain AI Service initialized", extra={
            'model_name': model_name,
            'temperature': temperature,
            'max_tokens': max_tokens
        })
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def process_tms_request(self, request_parameters: Dict[str, Any], 
                          previous_feedback: Optional[str] = None) -> AIResponse:
        """
        TMS 배차 요청 처리
        
        Args:
            request_parameters: TMS 요청 파라미터
            previous_feedback: 이전 피드백 (선택사항)
            
        Returns:
            AI 응답 결과
        """
        start_time = datetime.now()
        
        try:
            # 1. 프롬프트 선택
            prompt_selection = self.prompt_selector.select_optimal_prompt(request_parameters)
            self.logger.info("Prompt selected", extra={
                'scenario_type': prompt_selection.scenario_type.value,
                'confidence_score': prompt_selection.confidence_score,
                'selection_reasoning': prompt_selection.selection_reasoning
            })
            
            # 2. 프롬프트 파라미터 준비
            prompt_params = self._prepare_prompt_parameters(
                request_parameters, previous_feedback
            )
            
            # 3. 프롬프트 포맷팅
            formatted_prompt = prompt_selection.prompt_template.format(**prompt_params)
            
            # 4. GPT 호출
            with get_openai_callback() as cb:
                messages = [
                    SystemMessage(content="당신은 전문적인 TMS 배차 시스템입니다. 정확하고 실용적인 솔루션을 제공해주세요."),
                    HumanMessage(content=formatted_prompt)
                ]
                
                response = self.llm(messages)
                raw_response = response.content
                
                # 토큰 사용량 기록
                token_usage = {
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_tokens': cb.total_tokens,
                    'total_cost': cb.total_cost
                }
            
            # 5. 응답 검증 및 파싱
            validated_response = self.response_validator.validate_json_response(raw_response)
            validation_report = self.response_validator.generate_validation_report(validated_response)
            
            # 6. 폴리라인 생성 및 보완
            enhanced_response = self._enhance_response_with_polylines(validated_response)
            
            # 7. 처리 시간 계산
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # 8. 최종 응답 구성
            ai_response = AIResponse(
                success=True,
                response_data=enhanced_response,
                raw_response=raw_response,
                prompt_used=formatted_prompt,
                scenario_type=prompt_selection.scenario_type.value,
                confidence_score=prompt_selection.confidence_score,
                token_usage=token_usage,
                processing_time_ms=processing_time,
                validation_report=validation_report
            )
            
            self.logger.info("TMS request processed successfully", extra={
                'scenario_type': prompt_selection.scenario_type.value,
                'processing_time_ms': processing_time,
                'token_usage': token_usage,
                'validation_status': validation_report.get('validation_status')
            })
            
            return ai_response
            
        except ValidationError as e:
            self.logger.error("Response validation failed", extra={'error': str(e)})
            raise AIServiceError(f"Invalid AI response format: {e}")
        
        except Exception as e:
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            self.logger.error("TMS request processing failed", extra={
                'error': str(e),
                'processing_time_ms': processing_time
            })
            raise AIServiceError(f"AI request processing failed: {e}")
    
    def _prepare_prompt_parameters(self, request_params: Dict[str, Any], 
                                 previous_feedback: Optional[str]) -> Dict[str, Any]:
        """프롬프트 파라미터 준비"""
        return {
            'vehicles': json.dumps(request_params.get('vehicles', []), ensure_ascii=False),
            'orders': json.dumps(request_params.get('orders', []), ensure_ascii=False),
            'constraints': json.dumps(request_params.get('constraints', {}), ensure_ascii=False),
            'previous_feedback': previous_feedback or "이전 피드백 없음",
            'current_situation': json.dumps(request_params.get('current_situation', {}), ensure_ascii=False),
            'active_routes': json.dumps(request_params.get('active_routes', []), ensure_ascii=False),
            'change_reason': request_params.get('change_reason', ''),
            'realtime_constraints': json.dumps(request_params.get('realtime_constraints', {}), ensure_ascii=False),
            'emergency_order': json.dumps(request_params.get('emergency_order', {}), ensure_ascii=False),
            'available_vehicles': json.dumps(request_params.get('available_vehicles', []), ensure_ascii=False),
            'existing_routes': json.dumps(request_params.get('existing_routes', []), ensure_ascii=False),
            'urgency_level': request_params.get('urgency_level', 'NORMAL'),
            'vehicle': json.dumps(request_params.get('vehicle', {}), ensure_ascii=False),
            'start_location': json.dumps(request_params.get('start_location', {}), ensure_ascii=False),
            'consolidation_rules': json.dumps(request_params.get('consolidation_rules', {}), ensure_ascii=False),
            'vehicle_capacities': json.dumps(request_params.get('vehicle_capacities', []), ensure_ascii=False),
            'cargo_details': json.dumps(request_params.get('cargo_details', []), ensure_ascii=False),
            'loading_constraints': json.dumps(request_params.get('loading_constraints', {}), ensure_ascii=False),
            'delivery_conditions': json.dumps(request_params.get('delivery_conditions', {}), ensure_ascii=False),
            'planning_period': json.dumps(request_params.get('planning_period', {}), ensure_ascii=False),
            'daily_orders': json.dumps(request_params.get('daily_orders', []), ensure_ascii=False),
            'vehicle_availability': json.dumps(request_params.get('vehicle_availability', {}), ensure_ascii=False),
            'seasonal_factors': json.dumps(request_params.get('seasonal_factors', {}), ensure_ascii=False),
            'cost_structure': json.dumps(request_params.get('cost_structure', {}), ensure_ascii=False),
            'vehicle_costs': json.dumps(request_params.get('vehicle_costs', []), ensure_ascii=False),
            'fuel_prices': json.dumps(request_params.get('fuel_prices', {}), ensure_ascii=False),
            'labor_costs': json.dumps(request_params.get('labor_costs', {}), ensure_ascii=False)
        }
    
    def _enhance_response_with_polylines(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        응답에 실제 폴리라인 생성 및 보완
        
        Args:
            response_data: 검증된 AI 응답 데이터
            
        Returns:
            폴리라인이 보완된 응답 데이터
        """
        try:
            solution = response_data.get('solution', {})
            routes = solution.get('routes', [])
            
            for route in routes:
                waypoints = route.get('waypoints', [])
                
                if waypoints:
                    # OSRM API로 실제 도로 기반 폴리라인 생성 (상태 정보 포함)
                    route_result = self.polyline_generator.get_route_with_status(waypoints)
                    
                    # 폴리라인 상태에 따른 처리
                    if route_result['status'] == 'success':
                        # 성공: 실제 도로 기반 폴리라인 사용
                        route['polyline'] = route_result['polyline']
                        route['total_distance_km'] = route_result['distance_km']
                        route['total_duration_hours'] = route_result['duration_hours']
                        route['_polyline_source'] = 'osrm_real_roads'
                        route['_polyline_status'] = 'success'
                        
                        # AI 추정치와 실제 계산치 비교
                        ai_distance = route.get('total_distance_km', 0)
                        ai_duration = route.get('total_duration_hours', 0)
                        route['_distance_variance'] = abs(route_result['distance_km'] - ai_distance) if ai_distance > 0 else 0
                        route['_duration_variance'] = abs(route_result['duration_hours'] - ai_duration) if ai_duration > 0 else 0
                        
                    else:
                        # 실패: AI 응답 그대로 유지하되 상태 정보 추가
                        route['_polyline_source'] = 'ai_estimate_only'
                        route['_polyline_status'] = route_result['status']
                        route['_polyline_error'] = route_result.get('error', 'No real road route available')
                        
                        # 빈 폴리라인으로 설정 (프론트엔드에서 처리)
                        route['polyline'] = ""
                        
                        # 경고 추가
                        if 'warnings' not in response_data:
                            response_data['warnings'] = []
                        response_data['warnings'].append(
                            f"실제 도로 경로를 계산할 수 없습니다. 상태: {route_result['status']}"
                        )
            
            # 전체 요약 정보도 업데이트
            if routes:
                summary = solution.get('summary', {})
                summary['total_distance_km'] = sum(r.get('total_distance_km', 0) for r in routes)
                summary['total_duration_hours'] = sum(r.get('total_duration_hours', 0) for r in routes)
                
                # 평균 효율성 재계산
                efficiency_scores = [r.get('efficiency_score', 0) for r in routes]
                if efficiency_scores:
                    summary['average_efficiency'] = sum(efficiency_scores) / len(efficiency_scores)
            
            return response_data
            
        except Exception as e:
            self.logger.warning("Failed to enhance response with polylines", extra={'error': str(e)})
            # 폴리라인 생성 실패해도 원본 응답은 유지
            return response_data
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_name': self.llm.model_name,
            'temperature': self.llm.temperature,
            'max_tokens': self.llm.max_tokens,
            'api_provider': 'OpenAI'
        }
    
    def estimate_token_cost(self, prompt_text: str) -> Dict[str, Any]:
        """토큰 사용량 및 비용 추정"""
        # 대략적인 토큰 수 계산 (1 토큰 ≈ 4자)
        estimated_prompt_tokens = len(prompt_text) // 4
        estimated_completion_tokens = 1000  # 평균 응답 길이
        
        # GPT-4 가격 (2024년 기준, 실제 가격은 변동 가능)
        prompt_cost_per_1k = 0.03  # $0.03 per 1K tokens
        completion_cost_per_1k = 0.06  # $0.06 per 1K tokens
        
        estimated_cost = (
            (estimated_prompt_tokens / 1000) * prompt_cost_per_1k +
            (estimated_completion_tokens / 1000) * completion_cost_per_1k
        )
        
        return {
            'estimated_prompt_tokens': estimated_prompt_tokens,
            'estimated_completion_tokens': estimated_completion_tokens,
            'estimated_total_tokens': estimated_prompt_tokens + estimated_completion_tokens,
            'estimated_cost_usd': round(estimated_cost, 4)
        } 