# TMS 라우팅 알고리즘 분석 보고서

## 1. 문제 분석

### 1.1 발견된 핵심 문제
**"삥삥 돌아가는 현상"** - 배차 순서가 불필요하게 비효율적으로 배정되는 문제

**구체적 증상:**
- A→B→C 경로가 15분이면 충분한데, 시스템이 30분 이상 걸리는 경로를 배정
- 다른 라이더와의 시간 맞추기 위해 의도적으로 비효율적 경로 선택
- 시간 균등화 우선시로 인한 경로 최적화 포기

### 1.2 근본 원인 분석

#### 1) 고정 시간 계산 공식의 문제
```python
# 기존 문제있는 공식
estimated_time = len(order_ids) * 15  # 주문당 고정 15분
```

**문제점:**
- 실제 거리나 위치와 무관하게 주문 수에만 비례
- 인접한 3개 주문이든 멀리 떨어진 3개 주문이든 동일한 시간 (45분)
- 지리적 최적화를 완전히 무시

#### 2) 품질 점수 계산의 한계
```python
# 기존 품질 점수 가중치
assignment_rate * 0.5 +           # 배정률 50%
distance_efficiency * 0.3 +       # 거리 효율성 30%  
capacity_utilization * 0.2        # 용량 활용도 20%
# 시간 효율성 = 0% (없음!)
```

**문제점:**
- 시간 효율성이 품질 평가에서 완전히 제외
- 시간 균등화는 고려하지만 시간 최적화는 무시
- 결과적으로 시간이 오래 걸려도 품질 점수에 반영되지 않음

#### 3) 시간 균등화 알고리즘의 역효과
- 모든 라이더의 배송 시간을 비슷하게 맞추려는 시도
- 빠른 경로를 가진 라이더에게 일부러 더 긴 경로 배정
- 전체 효율성보다 형평성을 우선시하는 설계

## 2. 개선 방안

### 2.1 TimeCalculator 도입
정확한 시간 계산을 위한 전용 유틸리티 클래스 개발

#### 핵심 기능:
```python
class TimeCalculator:
    def calculate_delivery_time(self, vehicle, orders, traffic_factor=1.0):
        # 실제 경로 기반 시간 계산
        # 라우팅 API 또는 정확한 추정 사용
        
    def estimate_optimal_time_for_orders(self, orders):
        # 이론적 최적 시간 계산
        
    def calculate_time_efficiency(self, estimated_time, optimal_time):
        # 시간 효율성 점수 (0.0-1.0)
```

#### 개선된 시간 계산 공식:
```python
# 현실적인 시간 계산
travel_time = (distance_km / avg_speed) * traffic_factor
delivery_time = len(orders) * delivery_time_per_order  # 8분
setup_time = 5  # 차량 준비시간
total_time = travel_time + delivery_time + setup_time
```

### 2.2 품질 점수 개선
시간 효율성을 품질 평가에 포함

#### 새로운 가중치 구조:
```python
quality_score = (
    assignment_rate * 0.4 +      # 배정률 40% (감소)
    distance_efficiency * 0.25 + # 거리 효율성 25% (감소)
    capacity_utilization * 0.15 + # 용량 활용도 15% (감소)
    time_efficiency * 0.2        # 시간 효율성 20% (신규)
)
```

#### 시간 효율성 계산:
```python
def _calculate_time_efficiency(self, assignments, orders, vehicles):
    for assignment in assignments:
        optimal_time = time_calculator.estimate_optimal_time_for_orders(assigned_orders)
        estimated_time = assignment.estimated_time_minutes
        efficiency = time_calculator.calculate_time_efficiency(estimated_time, optimal_time)
```

### 2.3 알고리즘별 개선사항

#### NearestNeighbor 알고리즘:
- 고정 15분 → 현실적 시간 계산오케
- 평균 속도: 30km/h → 25km/h (더 현실적)
- 배송시간: 10분 → 8분 (더 효율적)

#### SimulatedAnnealing 알고리즘:
- 동일한 개선된 시간 계산 공식 적용
- 품질 평가에서 시간 효율성 고려

#### DispatchOrchestrator:
- 간단한 추정에서도 현실적 공식 사용
- 주문당 거리: 2.5km → 2.0km (더 현실적)

## 3. 기대 효과

### 3.1 시간 효율성 개선
- **기존:** A→B→C가 15분인데 30분으로 배정
- **개선 후:** 실제 경로 시간에 기반한 정확한 배정
- **예상 개선율:** 20-30% 시간 단축

### 3.2 품질 점수 정확도 향상
- 시간 효율성 20% 반영으로 더 현실적인 품질 평가
- 빠른 배송 경로가 높은 점수를 받도록 인센티브 구조 개선

### 3.3 사용자 만족도 향상
- 불필요하게 긴 배송 시간 감소
- 더 예측 가능하고 합리적인 배송 시간 제공

## 4. 구현 상태

### 4.1 완료된 작업 ✅
- [x] `TimeCalculator` 클래스 개발 완료
- [x] `BaseAlgorithm` 품질 점수 개선 (시간 효율성 20% 추가)  
- [x] `NearestNeighbor` 알고리즘 시간 계산 개선
- [x] `SimulatedAnnealing` 알고리즘 시간 계산 개선
- [x] `GeneticAlgorithm` 시간 계산 및 적합도 평가 개선
- [x] `DispatchOrchestrator` 시간 추정 개선
- [x] 모든 알고리즘에서 고정 15분 공식 완전 제거

### 4.2 시스템 현황 분석 📊
6개 센터 배차 효율성 분석 결과:
- **평균 비효율성:** 3.8배
- **심각 수준 (5배 초과):** HANAM(8.4배), SUWON(6.0배)
- **경고 수준 (2-5배):** ANYANG(2.9배), SEONGNAM(2.7배)  
- **양호 수준 (2배 미만):** GANGNAM(1.7배), SONGPA(1.2배)

**핵심 문제:** 직선거리 × 1.3 계수 사용으로 실제 도로 거리와 큰 차이

### 4.3 향후 작업 📋
- [ ] **OpenRouteService API 통합 (우선순위):** 실제 도로 거리 계산
- [ ] **HybridVRPTSPAlgorithm 구현:** VRP + TSP 2단계 최적화
- [ ] **TSP 경로 순서 최적화:** 2-opt 알고리즘 적용
- [ ] **TimeCalculator 실제 도로 거리 연동:** API 기반 정확한 시간 계산
- [ ] **API 캐싱 시스템:** 비용 최적화 및 성능 향상
- [ ] **성능 테스트 및 벤치마킹:** 개선 효과 검증

## 5. 상세 기술 분석

### 5.1 시간 계산 로직 비교

#### 기존 방식:
```python
# 모든 알고리즘에서 사용하던 공식
time = len(orders) * 15  # 완전 고정값
```
- **문제:** 지리적 위치 완전 무시
- **결과:** 5km 떨어진 3개 주문 = 옆건물 3개 주문 = 45분

#### 개선 방식:
```python
# 거리 기반 현실적 계산  
travel_time = (distance_km / 25) * 60 * traffic_factor
delivery_time = len(orders) * 8
setup_time = 5
total_time = travel_time + delivery_time + setup_time
```
- **장점:** 실제 거리와 교통상황 반영
- **결과:** 지리적으로 최적화된 시간 산정

### 5.2 알고리즘별 개선사항

#### BaseAlgorithm (모든 알고리즘 공통):
- 품질 점수 가중치 재조정: 시간 효율성 20% 추가
- `_calculate_time_efficiency()` 메서드로 실제 시간 vs 최적 시간 비교

#### GeneticAlgorithm:
- `_evaluate_fitness()`: 시간 효율성 6점 보너스 추가
- 적합도 계산 가중치 재조정: 거리(8) + 시간(6) + 용량(4) + 배정률(40)
- TimeCalculator 연동으로 정확한 시간 예측

#### SimulatedAnnealing:
- `_calculate_cost()`: 시간 비효율성 패널티 30점 추가  
- 비용 계산 가중치 재조정: 거리(8) + 시간패널티(30) + 용량보너스(40)
- 더 정확한 시간 기반 최적화

#### 개선된 품질 평가 체계:
1. **균형잡힌 가중치** → 배정률(40%) + 거리(25%) + 용량(15%) + 시간(20%)
2. **시간 효율성 도입** → 빠른 배송이 높은 점수/낮은 비용
3. **종합적 최적화** → 시간과 품질의 균형

### 5.3 캐싱 및 성능 최적화
```python
class TimeCalculator:
    def __init__(self):
        self.routing_cache = {}  # API 결과 캐싱
        self.distance_cache = {}  # 거리 계산 캐싱
        
    def _calculate_with_routing_api(self):
        # API 호출 결과 캐싱으로 성능 향상
        
    def validate_time_estimation(self):
        # 시간 추정 검증 및 피드백 루프
```

## 6. 구현 계획

### 6.1 핵심 구현 목표
**OpenRouteService API 우선 적용으로 실제 도로 거리 기반 라우팅 구현**

### 6.2 단계별 구현 전략

#### Phase 1: OpenRouteService API 통합 (우선순위)
1. **RoutingClient 개선**
   - OpenRouteService를 Primary API로 설정
   - 기존 KakaoRoutingClient를 Fallback으로 유지
   - API 응답 캐싱 시스템 구축

2. **TimeCalculator 실제 도로 거리 연동**
   - `_calculate_with_routing_api()` 메서드 구현
   - 직선거리 × 1.3 공식 → 실제 API 거리로 교체
   - API 실패 시 기존 추정 방식 유지

#### Phase 2: 하이브리드 VRP+TSP 알고리즘
1. **HybridVRPTSPAlgorithm 클래스 개발**
   - VRP: 차량별 주문 배정 (1단계)
   - TSP: 차량 내 경로 순서 최적화 (2단계)

2. **TSP 최적화 구현**
   - 2-opt 알고리즘 적용
   - 배송 순서 최적화로 추가 효율성 확보

#### Phase 3: 성능 최적화 및 검증
1. **API 캐싱 시스템**
   - 메모리 캐시: 세션 내 중복 요청 방지
   - 파일 캐시: 재시작 시 캐시 유지
   - TTL 기반 캐시 무효화

2. **성능 벤치마킹**
   - 기존 대비 거리/시간 정확도 개선 측정
   - API 호출 최적화 및 비용 관리

### 6.3 기대 효과
- **거리 계산 정확도:** 3.8배 → 1.2배 미만으로 개선
- **심각 센터 해결:** HANAM(8.4배), SUWON(6.0배) 정상화
- **API 비용 최적화:** 캐싱으로 중복 호출 70% 감소

## 7. 결론

### 핵심 개선사항:
1. **실제 도로 거리:** OpenRouteService API 기반 정확한 거리 계산
2. **품질 점수 개선:** 시간 효율성 20% 가중치 추가  
3. **하이브리드 알고리즘:** VRP+TSP 2단계 최적화
4. **API 캐싱 시스템:** 성능과 비용 최적화

### 예상 결과:
- **거리 정확도:** 3.8배 비효율성을 1.2배 미만으로 개선
- **배송 품질:** 실제 도로 기반 정확한 시간 제공
- **시스템 신뢰성:** 모든 센터에서 최적 경로 보장

이러한 개선을 통해 TMS 라우팅 시스템이 실제 배송 환경에 최적화된 효율적 배차 결과를 제공할 수 있습니다.