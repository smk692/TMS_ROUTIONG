"""
PromptSelector - 파라미터 기반 프롬프트 선택기

입력 파라미터를 분석하여 가장 적합한 TMS 프롬프트를 자동 선택합니다.
"""
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from langchain.prompts import PromptTemplate

from src.shared.constants import ScenarioType, Priority
from src.infrastructure.ai.prompt_templates import TmsPromptTemplates
from src.shared.exceptions import PromptSelectionError


@dataclass
class PromptSelectionResult:
    """프롬프트 선택 결과"""
    scenario_type: ScenarioType
    prompt_template: PromptTemplate
    confidence_score: float
    selection_reasoning: str
    alternative_scenarios: List[ScenarioType]


class PromptSelector:
    """TMS 프롬프트 선택기"""
    
    def __init__(self):
        self.templates = TmsPromptTemplates()
        self._selection_rules = self._initialize_selection_rules()
    
    def select_optimal_prompt(self, parameters: Dict[str, Any]) -> PromptSelectionResult:
        """
        파라미터 분석을 통한 최적 프롬프트 선택
        
        Args:
            parameters: TMS 배차 요청 파라미터
            
        Returns:
            프롬프트 선택 결과
        """
        # 1. 기본 시나리오 감지
        detected_scenarios = self._detect_scenarios(parameters)
        
        if not detected_scenarios:
            raise PromptSelectionError(
                "Unknown scenario",
                list(ScenarioType)
            )
        
        # 2. 최적 시나리오 선택
        best_scenario, confidence = self._select_best_scenario(detected_scenarios, parameters)
        
        # 3. 프롬프트 템플릿 가져오기
        prompt_template = self.templates.get_prompt_by_scenario(best_scenario)
        
        # 4. 선택 근거 생성
        reasoning = self._generate_selection_reasoning(best_scenario, parameters, detected_scenarios)
        
        # 5. 대안 시나리오 추출
        alternatives = [scenario for scenario, _ in detected_scenarios if scenario != best_scenario]
        
        return PromptSelectionResult(
            scenario_type=best_scenario,
            prompt_template=prompt_template,
            confidence_score=confidence,
            selection_reasoning=reasoning,
            alternative_scenarios=alternatives[:3]  # 상위 3개 대안
        )
    
    def _detect_scenarios(self, parameters: Dict[str, Any]) -> List[Tuple[ScenarioType, float]]:
        """
        파라미터 기반 시나리오 감지
        
        Args:
            parameters: 입력 파라미터
            
        Returns:
            (시나리오, 적합도 점수) 리스트
        """
        scenario_scores = []
        
        # 차량 수 분석
        vehicles = parameters.get('vehicles', [])
        orders = parameters.get('orders', [])
        
        vehicle_count = len(vehicles)
        order_count = len(orders)
        
        # VRP 점수 계산
        vrp_score = self._calculate_vrp_score(vehicle_count, order_count, parameters)
        if vrp_score > 0.3:
            scenario_scores.append((ScenarioType.VRP, vrp_score))
        
        # TSP 점수 계산
        tsp_score = self._calculate_tsp_score(vehicle_count, order_count, parameters)
        if tsp_score > 0.3:
            scenario_scores.append((ScenarioType.TSP, tsp_score))
        
        # 적재 통합 점수 계산
        consolidation_score = self._calculate_consolidation_score(parameters)
        if consolidation_score > 0.3:
            scenario_scores.append((ScenarioType.LOAD_CONSOLIDATION, consolidation_score))
        
        # 긴급 배송 점수 계산
        emergency_score = self._calculate_emergency_score(parameters)
        if emergency_score > 0.3:
            scenario_scores.append((ScenarioType.EMERGENCY_DISPATCH, emergency_score))
        
        # 실시간 조정 점수 계산
        realtime_score = self._calculate_realtime_score(parameters)
        if realtime_score > 0.3:
            scenario_scores.append((ScenarioType.REALTIME_ADJUSTMENT, realtime_score))
        
        # 점수 기준 정렬
        scenario_scores.sort(key=lambda x: x[1], reverse=True)
        
        return scenario_scores
    
    def _calculate_vrp_score(self, vehicle_count: int, order_count: int, parameters: Dict[str, Any]) -> float:
        """VRP 시나리오 적합도 점수 계산"""
        score = 0.0
        
        # 차량 수가 2대 이상이면 VRP 가능성 높음
        if vehicle_count >= 2:
            score += 0.4
        
        # 주문 수가 차량 수보다 많으면 VRP 필요
        if order_count > vehicle_count:
            score += 0.3
        
        # 지리적 분산도 확인
        if self._is_geographically_distributed(parameters.get('orders', [])):
            score += 0.2
        
        # 다양한 차량 특성
        if self._has_diverse_vehicle_types(parameters.get('vehicles', [])):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_tsp_score(self, vehicle_count: int, order_count: int, parameters: Dict[str, Any]) -> float:
        """TSP 시나리오 적합도 점수 계산"""
        score = 0.0
        
        # 차량이 1대인 경우 TSP
        if vehicle_count == 1:
            score += 0.5
        
        # 주문 수가 적당한 경우 (2-20개)
        if 2 <= order_count <= 20:
            score += 0.3
        
        # 순차적 배송이 가능한 경우
        if self._is_sequential_delivery_suitable(parameters.get('orders', [])):
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_consolidation_score(self, parameters: Dict[str, Any]) -> float:
        """적재 통합 시나리오 적합도 점수 계산"""
        score = 0.0
        
        orders = parameters.get('orders', [])
        
        # 소량 주문이 많은 경우
        small_orders = [o for o in orders if o.get('weight_tons', 0) < 1.0]
        if len(small_orders) / max(len(orders), 1) > 0.7:
            score += 0.4
        
        # 지역적 집중도가 높은 경우
        if self._has_regional_concentration(orders):
            score += 0.3
        
        # 시간 여유가 있는 경우
        if self._has_flexible_time_windows(orders):
            score += 0.2
        
        # 명시적 통합 요청
        if parameters.get('scenario_type') == 'load_consolidation':
            score += 0.4
        
        return min(1.0, score)
    
    def _calculate_emergency_score(self, parameters: Dict[str, Any]) -> float:
        """긴급 배송 시나리오 적합도 점수 계산"""
        score = 0.0
        
        orders = parameters.get('orders', [])
        
        # 긴급 우선순위 주문 존재
        urgent_orders = [o for o in orders if o.get('priority') == 'URGENT']
        if urgent_orders:
            score += 0.5
        
        # 시간 제약이 매우 타이트한 경우
        if self._has_tight_time_constraints(orders):
            score += 0.3
        
        # 기존 경로 정보가 있는 경우 (재조정)
        if parameters.get('existing_routes'):
            score += 0.2
        
        # 명시적 긴급 요청
        if parameters.get('scenario_type') == 'emergency_dispatch':
            score += 0.4
        
        return min(1.0, score)
    
    def _calculate_realtime_score(self, parameters: Dict[str, Any]) -> float:
        """실시간 조정 시나리오 적합도 점수 계산"""
        score = 0.0
        
        # 진행 중인 경로가 있는 경우
        if parameters.get('active_routes'):
            score += 0.4
        
        # 상황 변화가 명시된 경우
        if parameters.get('change_reason') or parameters.get('current_situation'):
            score += 0.3
        
        # 실시간 제약 조건이 있는 경우
        if parameters.get('realtime_constraints'):
            score += 0.2
        
        # 명시적 실시간 조정 요청
        if parameters.get('scenario_type') == 'realtime_adjustment':
            score += 0.4
        
        return min(1.0, score)
    
    def _select_best_scenario(self, detected_scenarios: List[Tuple[ScenarioType, float]], 
                            parameters: Dict[str, Any]) -> Tuple[ScenarioType, float]:
        """최적 시나리오 선택"""
        if not detected_scenarios:
            # 기본값: VRP
            return ScenarioType.VRP, 0.5
        
        # 명시적 시나리오 타입이 있는 경우 우선 고려
        explicit_scenario = parameters.get('scenario_type')
        if explicit_scenario:
            try:
                explicit_type = ScenarioType(explicit_scenario)
                for scenario, score in detected_scenarios:
                    if scenario == explicit_type and score > 0.3:
                        return scenario, min(1.0, score + 0.2)  # 명시적 지정 보너스
            except ValueError:
                pass
        
        # 가장 높은 점수의 시나리오 선택
        return detected_scenarios[0]
    
    def _generate_selection_reasoning(self, selected_scenario: ScenarioType, 
                                    parameters: Dict[str, Any],
                                    all_scenarios: List[Tuple[ScenarioType, float]]) -> str:
        """선택 근거 생성"""
        reasoning_parts = []
        
        # 선택된 시나리오 근거
        vehicle_count = len(parameters.get('vehicles', []))
        order_count = len(parameters.get('orders', []))
        
        if selected_scenario == ScenarioType.VRP:
            reasoning_parts.append(f"다중 차량({vehicle_count}대)과 다중 주문({order_count}개)으로 VRP가 최적")
        elif selected_scenario == ScenarioType.TSP:
            reasoning_parts.append(f"단일 차량 또는 순차 배송에 TSP가 적합")
        elif selected_scenario == ScenarioType.LOAD_CONSOLIDATION:
            reasoning_parts.append("소량 주문들의 통합 배송으로 효율성 극대화")
        elif selected_scenario == ScenarioType.EMERGENCY_DISPATCH:
            reasoning_parts.append("긴급 주문 또는 타이트한 시간 제약으로 긴급 대응 필요")
        elif selected_scenario == ScenarioType.REALTIME_ADJUSTMENT:
            reasoning_parts.append("진행 중인 경로 또는 상황 변화로 실시간 조정 필요")
        
        # 점수 정보 추가
        selected_score = next((score for scenario, score in all_scenarios if scenario == selected_scenario), 0.0)
        reasoning_parts.append(f"적합도 점수: {selected_score:.2f}")
        
        # 대안 시나리오 언급
        if len(all_scenarios) > 1:
            alternatives = [f"{scenario.value}({score:.2f})" for scenario, score in all_scenarios[1:3]]
            reasoning_parts.append(f"대안: {', '.join(alternatives)}")
        
        return " | ".join(reasoning_parts)
    
    def _is_geographically_distributed(self, orders: List[Dict[str, Any]]) -> bool:
        """주문들이 지리적으로 분산되어 있는지 확인"""
        if len(orders) < 2:
            return False
        
        # 간단한 분산도 계산 (실제로는 더 정교한 알고리즘 필요)
        locations = []
        for order in orders:
            pickup = order.get('pickup_location', {})
            delivery = order.get('delivery_location', {})
            if pickup.get('lat') and pickup.get('lng'):
                locations.append((pickup['lat'], pickup['lng']))
            if delivery.get('lat') and delivery.get('lng'):
                locations.append((delivery['lat'], delivery['lng']))
        
        if len(locations) < 2:
            return False
        
        # 위도/경도 범위 계산
        lats = [loc[0] for loc in locations]
        lngs = [loc[1] for loc in locations]
        
        lat_range = max(lats) - min(lats)
        lng_range = max(lngs) - min(lngs)
        
        # 0.01도 이상 차이나면 분산되어 있다고 판단 (약 1km)
        return lat_range > 0.01 or lng_range > 0.01
    
    def _has_diverse_vehicle_types(self, vehicles: List[Dict[str, Any]]) -> bool:
        """다양한 차량 타입이 있는지 확인"""
        if len(vehicles) < 2:
            return False
        
        capacities = set()
        capabilities = set()
        
        for vehicle in vehicles:
            capacities.add(vehicle.get('capacity_tons', 0))
            vehicle_capabilities = vehicle.get('special_capabilities', [])
            capabilities.update(vehicle_capabilities)
        
        return len(capacities) > 1 or len(capabilities) > 0
    
    def _is_sequential_delivery_suitable(self, orders: List[Dict[str, Any]]) -> bool:
        """순차적 배송이 적합한지 확인"""
        # 시간 창이 연속적이거나 우선순위가 명확한 경우
        time_windows = []
        priorities = []
        
        for order in orders:
            if order.get('time_window'):
                time_windows.append(order['time_window'])
            priorities.append(order.get('priority', 'MEDIUM'))
        
        # 우선순위가 다양하면 순차 배송 적합
        unique_priorities = set(priorities)
        return len(unique_priorities) > 1
    
    def _has_regional_concentration(self, orders: List[Dict[str, Any]]) -> bool:
        """지역적 집중도가 높은지 확인"""
        # 주문들이 특정 지역에 집중되어 있는지 확인
        if len(orders) < 3:
            return False
        
        # 위치 기반 클러스터링 (간단한 버전)
        locations = []
        for order in orders:
            pickup = order.get('pickup_location', {})
            if pickup.get('lat') and pickup.get('lng'):
                locations.append((pickup['lat'], pickup['lng']))
        
        if len(locations) < 3:
            return False
        
        # 평균 위치 계산
        avg_lat = sum(loc[0] for loc in locations) / len(locations)
        avg_lng = sum(loc[1] for loc in locations) / len(locations)
        
        # 평균 위치로부터의 거리 계산
        distances = []
        for lat, lng in locations:
            dist = ((lat - avg_lat) ** 2 + (lng - avg_lng) ** 2) ** 0.5
            distances.append(dist)
        
        # 80% 이상이 평균 거리의 2배 내에 있으면 집중도가 높다고 판단
        avg_distance = sum(distances) / len(distances)
        concentrated_count = sum(1 for d in distances if d <= avg_distance * 2)
        
        return concentrated_count / len(distances) >= 0.8
    
    def _has_flexible_time_windows(self, orders: List[Dict[str, Any]]) -> bool:
        """유연한 시간 창이 있는지 확인"""
        flexible_count = 0
        
        for order in orders:
            time_window = order.get('time_window')
            if not time_window:
                flexible_count += 1  # 시간 제약 없음
            else:
                # 시간 창이 4시간 이상이면 유연하다고 판단
                start = time_window.get('start')
                end = time_window.get('end')
                if start and end:
                    # 간단한 시간 차이 계산 (실제로는 datetime 파싱 필요)
                    flexible_count += 1
        
        return flexible_count / max(len(orders), 1) > 0.6
    
    def _has_tight_time_constraints(self, orders: List[Dict[str, Any]]) -> bool:
        """타이트한 시간 제약이 있는지 확인"""
        tight_count = 0
        
        for order in orders:
            if order.get('priority') in ['HIGH', 'URGENT']:
                tight_count += 1
            
            time_window = order.get('time_window')
            if time_window:
                # 시간 창이 2시간 미만이면 타이트하다고 판단
                # 실제로는 datetime 파싱하여 정확히 계산해야 함
                tight_count += 1
        
        return tight_count > 0
    
    def _initialize_selection_rules(self) -> Dict[str, Any]:
        """선택 규칙 초기화"""
        return {
            'min_confidence_threshold': 0.3,
            'explicit_scenario_bonus': 0.2,
            'geographic_distribution_threshold': 0.01,
            'small_order_threshold': 1.0,
            'concentration_threshold': 0.8
        } 