"""
LangGraphStateMachine - LangGraph 기반 상태 관리

복잡한 TMS 배차 프로세스를 상태 기반으로 관리하고,
다단계 추론과 피드백 루프를 구현합니다.
"""
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from datetime import datetime
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.infrastructure.ai.langchain_service import LangChainAIService, AIResponse
from src.infrastructure.ai.prompt_selector import PromptSelector
from src.shared.exceptions import AIServiceError
from src.shared.constants import ScenarioType
from src.shared.logging_config import TmsLoggerMixin


class TmsState(TypedDict):
    """TMS 상태 정의"""
    request_id: str
    original_request: Dict[str, Any]
    scenario_type: Optional[ScenarioType]
    current_step: str
    conversation_history: List[Dict[str, Any]]
    preliminary_analysis: Optional[Dict[str, Any]]
    optimization_result: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    final_response: Optional[Dict[str, Any]]
    feedback_received: Optional[str]
    iteration_count: int
    error_count: int
    metadata: Dict[str, Any]


@dataclass
class StateTransition:
    """상태 전환 정의"""
    from_state: str
    to_state: str
    condition: str
    action: str


class TmsStateMachine(TmsLoggerMixin):
    """TMS 상태 기반 처리 머신"""
    
    def __init__(self, ai_service: LangChainAIService):
        """
        상태 머신 초기화
        
        Args:
            ai_service: LangChain AI 서비스
        """
        super().__init__()
        self.ai_service = ai_service
        self.prompt_selector = PromptSelector()
        self.graph = self._build_state_graph()
    
    def _build_state_graph(self) -> StateGraph:
        """상태 그래프 구성"""
        workflow = StateGraph(TmsState)
        
        # 노드 추가
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("optimize_routes", self._optimize_routes)
        workflow.add_node("validate_solution", self._validate_solution)
        workflow.add_node("enhance_response", self._enhance_response)
        workflow.add_node("process_feedback", self._process_feedback)
        workflow.add_node("error_recovery", self._error_recovery)
        
        # 엣지 추가 (상태 전환 규칙)
        workflow.set_entry_point("analyze_request")
        
        # 분석 → 최적화
        workflow.add_edge("analyze_request", "optimize_routes")
        
        # 최적화 → 검증
        workflow.add_edge("optimize_routes", "validate_solution")
        
        # 검증 → 향상 또는 피드백 처리
        workflow.add_conditional_edges(
            "validate_solution",
            self._should_enhance_or_feedback,
            {
                "enhance": "enhance_response",
                "feedback": "process_feedback",
                "retry": "optimize_routes",
                "error": "error_recovery"
            }
        )
        
        # 향상 → 완료
        workflow.add_edge("enhance_response", END)
        
        # 피드백 → 최적화 (재시도)
        workflow.add_edge("process_feedback", "optimize_routes")
        
        # 에러 복구 → 최적화 또는 완료
        workflow.add_conditional_edges(
            "error_recovery",
            self._should_retry_or_end,
            {
                "retry": "optimize_routes",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def process_tms_request(self, request_parameters: Dict[str, Any], 
                          request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        TMS 요청을 상태 기반으로 처리
        
        Args:
            request_parameters: TMS 요청 파라미터
            request_id: 요청 ID
            
        Returns:
            처리 결과
        """
        # 초기 상태 설정
        initial_state: TmsState = {
            "request_id": request_id or f"req_{datetime.now().isoformat()}",
            "original_request": request_parameters,
            "scenario_type": None,
            "current_step": "analyze_request",
            "conversation_history": [],
            "preliminary_analysis": None,
            "optimization_result": None,
            "validation_result": None,
            "final_response": None,
            "feedback_received": None,
            "iteration_count": 0,
            "error_count": 0,
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "processing_steps": []
            }
        }
        
        try:
            # 상태 그래프 실행
            final_state = self.graph.invoke(initial_state)
            
            self.logger.info("TMS request processed through state machine", extra={
                'request_id': final_state['request_id'],
                'iteration_count': final_state['iteration_count'],
                'final_step': final_state['current_step'],
                'processing_steps': final_state['metadata']['processing_steps']
            })
            
            return final_state['final_response'] or {}
            
        except Exception as e:
            self.logger.error("State machine processing failed", extra={
                'request_id': initial_state['request_id'],
                'error': str(e)
            })
            raise AIServiceError(f"State machine processing failed: {e}")
    
    def _analyze_request(self, state: TmsState) -> TmsState:
        """요청 분석 단계"""
        try:
            self.logger.info("Analyzing TMS request", extra={
                'request_id': state['request_id']
            })
            
            # 시나리오 타입 감지
            prompt_selection = self.prompt_selector.select_optimal_prompt(
                state['original_request']
            )
            
            state['scenario_type'] = prompt_selection.scenario_type
            state['preliminary_analysis'] = {
                'scenario_type': prompt_selection.scenario_type.value,
                'confidence_score': prompt_selection.confidence_score,
                'selection_reasoning': prompt_selection.selection_reasoning,
                'alternative_scenarios': [s.value for s in prompt_selection.alternative_scenarios]
            }
            
            state['current_step'] = "optimize_routes"
            state['metadata']['processing_steps'].append({
                'step': 'analyze_request',
                'timestamp': datetime.now().isoformat(),
                'result': 'success'
            })
            
            return state
            
        except Exception as e:
            state['error_count'] += 1
            state['current_step'] = "error_recovery"
            state['metadata']['processing_steps'].append({
                'step': 'analyze_request',
                'timestamp': datetime.now().isoformat(),
                'result': 'error',
                'error': str(e)
            })
            return state
    
    def _optimize_routes(self, state: TmsState) -> TmsState:
        """경로 최적화 단계"""
        try:
            self.logger.info("Optimizing routes", extra={
                'request_id': state['request_id'],
                'scenario_type': state['scenario_type'].value if state['scenario_type'] else 'unknown'
            })
            
            # AI 서비스로 최적화 실행
            ai_response = self.ai_service.process_tms_request(
                state['original_request'],
                state.get('feedback_received')
            )
            
            state['optimization_result'] = {
                'ai_response': ai_response.response_data,
                'raw_response': ai_response.raw_response,
                'scenario_type': ai_response.scenario_type,
                'confidence_score': ai_response.confidence_score,
                'token_usage': ai_response.token_usage,
                'processing_time_ms': ai_response.processing_time_ms
            }
            
            state['iteration_count'] += 1
            state['current_step'] = "validate_solution"
            state['metadata']['processing_steps'].append({
                'step': 'optimize_routes',
                'timestamp': datetime.now().isoformat(),
                'result': 'success',
                'iteration': state['iteration_count']
            })
            
            return state
            
        except Exception as e:
            state['error_count'] += 1
            state['current_step'] = "error_recovery"
            state['metadata']['processing_steps'].append({
                'step': 'optimize_routes',
                'timestamp': datetime.now().isoformat(),
                'result': 'error',
                'error': str(e),
                'iteration': state['iteration_count']
            })
            return state
    
    def _validate_solution(self, state: TmsState) -> TmsState:
        """솔루션 검증 단계"""
        try:
            self.logger.info("Validating solution", extra={
                'request_id': state['request_id']
            })
            
            optimization_result = state['optimization_result']
            if not optimization_result:
                raise AIServiceError("No optimization result to validate")
            
            ai_response_data = optimization_result['ai_response']
            
            # 응답 검증
            from src.infrastructure.ai.response_validator import ResponseValidator
            validator = ResponseValidator()
            validation_report = validator.generate_validation_report(ai_response_data)
            
            state['validation_result'] = validation_report
            
            # 검증 결과에 따라 다음 단계 결정
            validation_status = validation_report.get('validation_status', 'failed')
            
            if validation_status == 'passed':
                state['current_step'] = "enhance_response"
            elif validation_status == 'warning' and state['iteration_count'] < 3:
                state['current_step'] = "optimize_routes"  # 재시도
            else:
                state['current_step'] = "enhance_response"  # 경고 있어도 진행
            
            state['metadata']['processing_steps'].append({
                'step': 'validate_solution',
                'timestamp': datetime.now().isoformat(),
                'result': validation_status,
                'validation_report': validation_report
            })
            
            return state
            
        except Exception as e:
            state['error_count'] += 1
            state['current_step'] = "error_recovery"
            state['metadata']['processing_steps'].append({
                'step': 'validate_solution',
                'timestamp': datetime.now().isoformat(),
                'result': 'error',
                'error': str(e)
            })
            return state
    
    def _enhance_response(self, state: TmsState) -> TmsState:
        """응답 향상 단계"""
        try:
            self.logger.info("Enhancing response", extra={
                'request_id': state['request_id']
            })
            
            optimization_result = state['optimization_result']
            validation_result = state['validation_result']
            
            # 최종 응답 구성
            final_response = {
                'request_id': state['request_id'],
                'scenario_type': state['scenario_type'].value if state['scenario_type'] else 'unknown',
                'solution': optimization_result['ai_response'],
                'metadata': {
                    'processing_time_ms': optimization_result['processing_time_ms'],
                    'confidence_score': optimization_result['confidence_score'],
                    'iteration_count': state['iteration_count'],
                    'validation_status': validation_result.get('validation_status'),
                    'token_usage': optimization_result['token_usage'],
                    'processing_steps': state['metadata']['processing_steps']
                }
            }
            
            state['final_response'] = final_response
            state['current_step'] = "completed"
            state['metadata']['processing_steps'].append({
                'step': 'enhance_response',
                'timestamp': datetime.now().isoformat(),
                'result': 'success'
            })
            
            return state
            
        except Exception as e:
            state['error_count'] += 1
            state['current_step'] = "error_recovery"
            state['metadata']['processing_steps'].append({
                'step': 'enhance_response',
                'timestamp': datetime.now().isoformat(),
                'result': 'error',
                'error': str(e)
            })
            return state
    
    def _process_feedback(self, state: TmsState) -> TmsState:
        """피드백 처리 단계"""
        try:
            self.logger.info("Processing feedback", extra={
                'request_id': state['request_id']
            })
            
            # 피드백을 기반으로 요청 파라미터 수정
            feedback = state.get('feedback_received', '')
            
            # 대화 이력에 피드백 추가
            state['conversation_history'].append({
                'type': 'feedback',
                'content': feedback,
                'timestamp': datetime.now().isoformat()
            })
            
            state['current_step'] = "optimize_routes"
            state['metadata']['processing_steps'].append({
                'step': 'process_feedback',
                'timestamp': datetime.now().isoformat(),
                'result': 'success',
                'feedback_length': len(feedback)
            })
            
            return state
            
        except Exception as e:
            state['error_count'] += 1
            state['current_step'] = "error_recovery"
            state['metadata']['processing_steps'].append({
                'step': 'process_feedback',
                'timestamp': datetime.now().isoformat(),
                'result': 'error',
                'error': str(e)
            })
            return state
    
    def _error_recovery(self, state: TmsState) -> TmsState:
        """에러 복구 단계"""
        self.logger.warning("Entering error recovery", extra={
            'request_id': state['request_id'],
            'error_count': state['error_count'],
            'iteration_count': state['iteration_count']
        })
        
        # 최대 재시도 횟수 체크
        if state['error_count'] >= 3:
            # 기본 응답 생성
            state['final_response'] = {
                'request_id': state['request_id'],
                'success': False,
                'error': 'Maximum retry attempts exceeded',
                'partial_result': state.get('optimization_result'),
                'metadata': state['metadata']
            }
            state['current_step'] = "failed"
        else:
            # 재시도
            state['current_step'] = "optimize_routes"
        
        state['metadata']['processing_steps'].append({
            'step': 'error_recovery',
            'timestamp': datetime.now().isoformat(),
            'error_count': state['error_count'],
            'next_step': state['current_step']
        })
        
        return state
    
    def _should_enhance_or_feedback(self, state: TmsState) -> str:
        """검증 후 다음 단계 결정"""
        if state['current_step'] == "error_recovery":
            return "error"
        elif state.get('feedback_received'):
            return "feedback"
        elif state['current_step'] == "optimize_routes" and state['iteration_count'] < 3:
            return "retry"
        else:
            return "enhance"
    
    def _should_retry_or_end(self, state: TmsState) -> str:
        """에러 복구 후 다음 단계 결정"""
        if state['error_count'] >= 3:
            return "end"
        else:
            return "retry"
    
    def add_feedback_to_conversation(self, request_id: str, feedback: str) -> bool:
        """대화에 피드백 추가"""
        try:
            # 실제 구현에서는 메모리 저장소에서 상태를 찾아서 업데이트
            # 여기서는 기본 구조만 제공
            self.logger.info("Feedback added to conversation", extra={
                'request_id': request_id,
                'feedback_length': len(feedback)
            })
            return True
        except Exception as e:
            self.logger.error("Failed to add feedback", extra={
                'request_id': request_id,
                'error': str(e)
            })
            return False 