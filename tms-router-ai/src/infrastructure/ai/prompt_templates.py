"""
TMS AI 프롬프트 템플릿 라이브러리

각 TMS 시나리오별로 최적화된 프롬프트를 정의합니다.
모든 프롬프트는 [TMS 전문가] 역할로 작성되고 고정된 JSON 형식으로 출력됩니다.
"""
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from src.shared.constants import ScenarioType


class TmsPromptTemplates:
    """TMS 배차 프롬프트 템플릿 컬렉션"""
    
    # 공통 JSON 출력 형식 정의
    JSON_OUTPUT_FORMAT = """
반드시 다음과 같은 JSON 형식으로만 응답해주세요:

{{
  "success": true,
  "analysis": "상황 분석 내용",
  "solution": {{
    "routes": [
      {{
        "vehicle_id": "차량 ID",
        "orders": ["주문 ID 리스트"],
        "waypoints": [
          {{
            "location": {{"lat": 위도, "lng": 경도}},
            "type": "start|pickup|delivery|end",
            "order_id": "주문 ID (해당시)",
            "estimated_arrival": "ISO 시간",
            "estimated_duration_minutes": 소요시간
          }}
        ],
        "total_distance_km": 총거리,
        "total_duration_hours": 총시간,
        "estimated_cost": 예상비용,
        "polyline": "경로 폴리라인 (구글맵 호환)",
        "efficiency_score": 효율성점수
      }}
    ],
    "summary": {{
      "total_vehicles_used": 사용차량수,
      "total_orders_assigned": 배정주문수,
      "total_distance_km": 전체거리,
      "total_cost": 전체비용,
      "average_efficiency": 평균효율성
    }}
  }},
  "reasoning": "단계별 판단 근거",
  "confidence_score": 0.0에서1.0사이값,
  "recommendations": ["개선 제안사항"],
  "warnings": ["주의사항 또는 제약조건"]
}}
"""
    
    @classmethod
    def get_vrp_prompt(cls) -> PromptTemplate:
        """VRP (Vehicle Routing Problem) 프롬프트"""
        template = f"""
당신은 20년 경력의 TMS 전문가입니다. 다중 차량 경로 최적화(VRP) 문제를 해결해주세요.

## 입력 정보
**차량 정보**: {{vehicles}}
**배송 주문**: {{orders}}
**제약 조건**: {{constraints}}
**이전 피드백**: {{previous_feedback}}

## 분석 단계
1. **수요 분석**: 주문 분포, 중량, 우선순위 파악
2. **차량 배정**: 용량, 특수능력, 위치 기반 최적 매칭
3. **경로 최적화**: 거리, 시간, 비용 최소화
4. **제약 검증**: 근무시간, 시간창, 용량 한계 확인
5. **효율성 평가**: 차량 활용률, 연료 효율성 계산

## 최적화 목표
- 총 운송 거리 최소화
- 차량 활용률 극대화  
- 배송 시간 준수
- 연료비 절약
- 우선순위 주문 우선 처리

{cls.JSON_OUTPUT_FORMAT}

폴리라인은 각 경로별로 Google Maps 호환 형식으로 생성해주세요.
"""
        
        return PromptTemplate(
            input_variables=["vehicles", "orders", "constraints", "previous_feedback"],
            template=template
        )
    
    @classmethod
    def get_tsp_prompt(cls) -> PromptTemplate:
        """TSP (Traveling Salesman Problem) 프롬프트"""
        template = f"""
당신은 20년 경력의 TMS 전문가입니다. 단일 차량 최적 경로(TSP) 문제를 해결해주세요.

## 입력 정보
**차량**: {{vehicle}}
**배송 주문**: {{orders}}
**시작 위치**: {{start_location}}
**제약 조건**: {{constraints}}
**이전 피드백**: {{previous_feedback}}

## 분석 단계
1. **주문 우선순위 정렬**: 긴급도, 시간창, 고객 중요도
2. **경로 순서 최적화**: 최단 거리 알고리즘 적용
3. **시간 계획**: 픽업/배송 시간, 교통 상황 고려
4. **용량 관리**: 적재 순서, 하역 효율성
5. **연료 효율성**: 최적 속도, 공회전 최소화

## 최적화 목표
- 총 이동 거리 최소화
- 배송 시간 최적화
- 연료 소모량 최소화
- 고객 만족도 극대화

{cls.JSON_OUTPUT_FORMAT}

단일 차량이므로 routes 배열에는 1개 요소만 포함됩니다.
"""
        
        return PromptTemplate(
            input_variables=["vehicle", "orders", "start_location", "constraints", "previous_feedback"],
            template=template
        )
    
    @classmethod
    def get_load_consolidation_prompt(cls) -> PromptTemplate:
        """적재 통합 최적화 프롬프트"""
        template = f"""
당신은 20년 경력의 TMS 전문가입니다. 소량 다수 주문의 적재 통합 최적화를 수행해주세요.

## 입력 정보
**차량들**: {{vehicles}}
**소량 주문들**: {{orders}}
**통합 규칙**: {{consolidation_rules}}
**제약 조건**: {{constraints}}
**이전 피드백**: {{previous_feedback}}

## 분석 단계
1. **주문 클러스터링**: 지역별, 시간대별 그룹화
2. **통합 가능성 분석**: 화물 호환성, 시간창 겹침
3. **차량 할당**: 최소 차량으로 최대 효율
4. **적재 순서 계획**: 하역 편의성, 손상 방지
5. **비용 절감 계산**: 통합 전후 비교 분석

## 최적화 목표
- 사용 차량 수 최소화
- 적재율 극대화 (80% 이상)
- 배송 지연 방지
- 물류비 절감
- 소량 주문 효율적 처리

{cls.JSON_OUTPUT_FORMAT}

통합 효과가 높은 주문들을 우선적으로 그룹화해주세요.
"""
        
        return PromptTemplate(
            input_variables=["vehicles", "orders", "consolidation_rules", "constraints", "previous_feedback"],
            template=template
        )
    
    @classmethod
    def get_emergency_dispatch_prompt(cls) -> PromptTemplate:
        """긴급 배송 프롬프트"""
        template = f"""
당신은 20년 경력의 TMS 전문가입니다. 긴급 배송 요청에 대한 신속한 배차를 수행해주세요.

## 입력 정보
**긴급 주문**: {{emergency_order}}
**가용 차량**: {{available_vehicles}}
**기존 경로**: {{existing_routes}}
**긴급도**: {{urgency_level}}
**이전 피드백**: {{previous_feedback}}

## 분석 단계
1. **긴급도 평가**: 시간 제약, 고객 중요도, 비즈니스 영향
2. **가용 자원 확인**: 즉시 투입 가능한 차량/기사
3. **기존 계획 영향 분석**: 재배정 필요성, 지연 리스크
4. **최적 삽입점 탐색**: 최소 영향으로 긴급 배송 추가
5. **비상 대응 방안**: 예비 차량, 외주 활용

## 최적화 목표
- 긴급 배송 시간 최소화
- 기존 배송 영향 최소화
- 추가 비용 최소화
- 고객 만족도 유지
- 전체 시스템 안정성

{cls.JSON_OUTPUT_FORMAT}

긴급 주문은 최우선으로 처리하되, 기존 약속된 배송은 최대한 지켜주세요.
"""
        
        return PromptTemplate(
            input_variables=["emergency_order", "available_vehicles", "existing_routes", "urgency_level", "previous_feedback"],
            template=template
        )
    
    @classmethod
    def get_realtime_adjustment_prompt(cls) -> PromptTemplate:
        """실시간 경로 조정 프롬프트"""
        template = f"""
당신은 20년 경력의 TMS 전문가입니다. 실시간 상황 변화에 따른 경로 재조정을 수행해주세요.

## 입력 정보
**현재 상황**: {{current_situation}}
**진행 중인 경로**: {{active_routes}}
**변경 사유**: {{change_reason}}
**실시간 제약**: {{realtime_constraints}}
**이전 피드백**: {{previous_feedback}}

## 분석 단계
1. **상황 영향 분석**: 교통, 차량 고장, 주문 변경 등
2. **조정 범위 결정**: 전체 재최적화 vs 부분 수정
3. **대안 경로 탐색**: 우회로, 시간대 변경, 차량 교체
4. **비용 영향 계산**: 추가 연료, 시간, 인건비
5. **고객 소통 방안**: 지연 통보, 대안 제시

## 최적화 목표
- 변경 영향 최소화
- 기존 약속 최대한 유지
- 추가 비용 절약
- 고객 만족도 보호
- 신속한 대응

{cls.JSON_OUTPUT_FORMAT}

실시간 조정이므로 현실적이고 즉시 실행 가능한 방안을 제시해주세요.
"""
        
        return PromptTemplate(
            input_variables=["current_situation", "active_routes", "change_reason", "realtime_constraints", "previous_feedback"],
            template=template
        )
    
    @classmethod
    def get_capacity_optimization_prompt(cls) -> PromptTemplate:
        """용량 최적화 프롬프트"""
        template = f"""
당신은 20년 경력의 TMS 전문가입니다. 차량 용량 활용률 극대화를 위한 배차를 수행해주세요.

## 입력 정보
**차량 용량**: {{vehicle_capacities}}
**화물 정보**: {{cargo_details}}
**적재 제약**: {{loading_constraints}}
**배송 조건**: {{delivery_conditions}}
**이전 피드백**: {{previous_feedback}}

## 분석 단계
1. **용량 분석**: 중량, 부피, 형태별 제약
2. **적재 계획**: 3D 패킹, 무게 분산, 하역 순서
3. **호환성 검토**: 화물 간 충돌, 온도 조건
4. **안전성 확보**: 적재 안정성, 운전 안전
5. **효율성 측정**: 용량 활용률, 공간 효율

## 최적화 목표
- 용량 활용률 90% 이상
- 안전한 적재 배치
- 하역 작업 효율성
- 화물 손상 방지
- 운송비 절감

{cls.JSON_OUTPUT_FORMAT}

적재 계획은 3차원 공간 활용과 안전성을 모두 고려해주세요.
"""
        
        return PromptTemplate(
            input_variables=["vehicle_capacities", "cargo_details", "loading_constraints", "delivery_conditions", "previous_feedback"],
            template=template
        )
    
    @classmethod
    def get_multi_day_planning_prompt(cls) -> PromptTemplate:
        """다일 계획 프롬프트"""
        template = f"""
당신은 20년 경력의 TMS 전문가입니다. 여러 날에 걸친 배송 계획을 수립해주세요.

## 입력 정보
**계획 기간**: {{planning_period}}
**일별 주문량**: {{daily_orders}}
**차량 가용성**: {{vehicle_availability}}
**계절적 요인**: {{seasonal_factors}}
**이전 피드백**: {{previous_feedback}}

## 분석 단계
1. **수요 예측**: 일별 변동성, 계절성 반영
2. **자원 계획**: 차량/기사 일정, 유지보수
3. **부하 분산**: 피크 시간 분산, 여유 시간 활용
4. **리스크 관리**: 날씨, 교통, 차량 고장 대비
5. **최적화 연계**: 일간 계획의 연속성

## 최적화 목표
- 일별 효율성 극대화
- 자원 활용 최적화
- 비용 평준화
- 서비스 품질 일관성
- 장기적 지속가능성

{cls.JSON_OUTPUT_FORMAT}

여러 날의 연관성을 고려한 전략적 계획을 수립해주세요.
"""
        
        return PromptTemplate(
            input_variables=["planning_period", "daily_orders", "vehicle_availability", "seasonal_factors", "previous_feedback"],
            template=template
        )
    
    @classmethod
    def get_cost_optimization_prompt(cls) -> PromptTemplate:
        """비용 최적화 프롬프트"""
        template = f"""
당신은 20년 경력의 TMS 전문가입니다. 물류 비용 최소화에 중점을 둔 배차를 수행해주세요.

## 입력 정보
**비용 구조**: {{cost_structure}}
**차량별 비용**: {{vehicle_costs}}
**연료 가격**: {{fuel_prices}}
**인건비**: {{labor_costs}}
**이전 피드백**: {{previous_feedback}}

## 분석 단계
1. **비용 분석**: 고정비, 변동비, 기회비용
2. **효율성 계산**: 톤-km당 비용, 시간당 비용
3. **최적화 전략**: 공차 최소화, 왕복 화물
4. **대안 평가**: 외주 vs 직영, 차량 크기
5. **ROI 분석**: 투자 대비 절감 효과

## 최적화 목표
- 총 운송비 최소화
- 연료비 절약
- 인건비 효율화
- 차량 가동률 최적화
- 수익성 개선

{cls.JSON_OUTPUT_FORMAT}

비용 절감과 서비스 품질의 균형을 맞춘 방안을 제시해주세요.
"""
        
        return PromptTemplate(
            input_variables=["cost_structure", "vehicle_costs", "fuel_prices", "labor_costs", "previous_feedback"],
            template=template
        )
    
    @classmethod
    def get_prompt_by_scenario(cls, scenario_type: ScenarioType) -> PromptTemplate:
        """시나리오 타입에 따른 프롬프트 반환"""
        prompt_map = {
            ScenarioType.VRP: cls.get_vrp_prompt(),
            ScenarioType.TSP: cls.get_tsp_prompt(),
            ScenarioType.LOAD_CONSOLIDATION: cls.get_load_consolidation_prompt(),
            ScenarioType.EMERGENCY_DISPATCH: cls.get_emergency_dispatch_prompt(),
            ScenarioType.REALTIME_ADJUSTMENT: cls.get_realtime_adjustment_prompt()
        }
        
        if scenario_type not in prompt_map:
            raise ValueError(f"Unsupported scenario type: {scenario_type}")
        
        return prompt_map[scenario_type]
    
    @classmethod
    def get_all_scenario_types(cls) -> Dict[ScenarioType, str]:
        """지원하는 모든 시나리오 타입과 설명"""
        return {
            ScenarioType.VRP: "다중 차량 경로 최적화",
            ScenarioType.TSP: "단일 차량 최적 경로",
            ScenarioType.LOAD_CONSOLIDATION: "적재 통합 최적화",
            ScenarioType.EMERGENCY_DISPATCH: "긴급 배송 처리",
            ScenarioType.REALTIME_ADJUSTMENT: "실시간 경로 조정"
        } 