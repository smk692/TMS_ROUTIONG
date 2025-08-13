# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고할 가이드라인을 제공합니다.

## 프로젝트 개요

TMS Router Hybrid 프로젝트 - 배송 경로 최적화에 중점을 둔 운송 관리 시스템(Transportation Management System)입니다. 이 디렉토리는 하이브리드 TMS 라우팅 시스템 구현을 위한 포괄적인 계획 문서와 함께 개발 작업 공간으로 사용됩니다.

이 프로젝트는 전통적인 알고리즘 접근법과 현대적인 최적화 기법을 결합하며, 클린 아키텍처 원칙과 UseCase 기반 디자인 패턴을 따르도록 설계되었습니다.

## 명령어

### 기본 개발 명령어
```bash
# 메인 Python 스크립트 실행 (현재는 템플릿)
python main.py

# 현재 템플릿 프로젝트이므로 추가 빌드/테스트 명령어는 구성되지 않음
# 프로젝트가 발전함에 따라 의존성과 빌드 도구가 추가될 예정
```

## 아키텍처 개요

프로젝트는 프로덕션 준비가 된 TMS 시스템 구현을 위한 상세한 문서화와 함께 포괄적인 계획 접근법을 따릅니다:

### 핵심 설계 원칙
- **UseCase 중심 아키텍처**: 비즈니스 로직을 독립적인 UseCase 클래스로 분리
- **전략 패턴 구현**: 플러그 가능한 알고리즘 및 조정 전략  
- **다층 캐싱**: API 최적화를 위한 메모리, 파일, 데이터베이스 캐싱
- **성능 최적화**: 모든 배차 작업은 품질을 유지하며 효율적으로 완료
- **실시간 적응성**: 날씨, 교통, 기사 경험에 기반한 동적 조정

### 주요 처리 구성 요소

1. **알고리즘 선택 엔진** (01-algorithm-selection-rules.md)
   - 복잡도 기반 알고리즘 선택 (최근접 이웃 → 유전자 알고리즘 → 시뮬레이티드 어닐링 → 대규모 근방 탐색)
   - 날씨 및 교통 상황 적응
   - 성능 기반 동적 알고리즘 전환

2. **동적 주문 조정 시스템** (02-dynamic-order-adjustment.md)
   - 기사 경험 수준 기반 용량 조정 (신입 70% → 전문가 130%)
   - 날씨 심각도 스케일링 (맑음 1.1배 → 폭풍 0.3배)
   - 교통 정체 요소 (원활 1.1배 → 심각한 정체 0.6배)

3. **경로 최적화 전략** (03-route-optimization-rules.md)
   - 다중 API 통합 (OpenRouteService → HERE Maps → 카카오맵 → Mapbox)
   - 할당량과 요청 특성에 기반한 지능형 API 선택
   - 3단계 캐싱 시스템 (메모리 → 파일 → 데이터베이스)

4. **프로세스 플로우 통합** (04-process-flow-integration.md)
   - 확장성을 위한 병렬 권역 처리
   - 오류 복구 및 폴백 전략
   - 적응형 최적화를 통한 실시간 성능 모니터링

### 제안된 프로젝트 구조 (05-improved-project-structure.md)

문서는 포괄적인 클린 아키텍처 구조를 설명합니다:

```
tms_dispatch_system/
├── usecases/              # 비즈니스 로직 레이어
├── strategies/            # 알고리즘 및 조정 전략
├── models/               # 도메인 엔티티와 값 객체
├── services/             # 도메인 서비스
├── repositories/         # 데이터 접근 레이어
├── algorithms/           # 최적화 알고리즘 구현
├── external/             # 외부 API 통합
└── infrastructure/       # 횡단 관심사
```

## 주요 기능 및 제약사항

### 성능 요구사항
- **처리 최적화**: 품질 중심의 효율적 배차 처리
- **처리 전략**: 데이터 수집 최적화 → 지능형 경로 계산 → 알고리즘 최적화 → 결과 검증
- **확장성**: 권역 기반 병렬 처리로 50-300개 이상의 주문 지원

### 알고리즘 복잡도 매트릭스
- **주문 ≤30개**: 최근접 이웃 (빠른 처리, 70-80% 품질)
- **주문 31-100개**: 유전자 알고리즘 (균형 처리, 85-90% 품질)  
- **주문 101-300개**: 시뮬레이티드 어닐링 (품질 중심, 88-93% 품질)
- **주문 300개 이상**: 대규모 근방 탐색 (최고 품질, 90-95% 품질)

### 동적 조정 요소
- **기사 경험 레벨**: 신입(70% 용량)부터 전문가(130% 용량)까지 5단계
- **날씨 심각도 점수**: 배송 용량에 영향을 미치는 1.0-5.0 스케일
- **교통 정체 영향**: 정체 수준에 기반한 실시간 조정
- **권역 난이도 점수**: 배송 시간에 영향을 미치는 지리적 복잡도

## 개발 패턴

### 현재 상태
이것은 계획 및 문서화 단계 프로젝트입니다. 실제 구현은 마크다운 파일의 상세한 사양을 따라야 합니다:

1. **UseCase 구현으로 시작**: `usecases/` 디렉토리의 핵심 비즈니스 로직부터 시작
2. **전략 패턴 구현**: 플러그 가능한 알고리즘 선택 및 조정 전략 생성  
3. **캐싱 인프라 구축**: API 최적화를 위한 3단계 캐싱 시스템 구현
4. **외부 API 통합 추가**: 폴백 전략과 함께 날씨, 교통, 라우팅 API 통합
5. **성능 최적화 시스템**: 효율적 처리를 위한 동적 최적화 구현

### 핵심 구현 가이드라인
- 품질과 성능의 최적 균형 추구
- 포괄적인 오류 처리 및 폴백 전략 구현
- 권역 기반 최적화를 위한 병렬 처리 사용
- 실시간 조건에 기반한 동적 조정 알고리즘 적용
- 비즈니스 로직과 인프라 관심사 간의 명확한 분리 유지

## 파일 구조

### 문서 파일
- `01-algorithm-selection-rules.md`: 상세한 알고리즘 선택 매트릭스 및 규칙
- `02-dynamic-order-adjustment.md`: 기사 및 조건 기반 조정 로직
- `03-route-optimization-rules.md`: API 관리 및 캐싱 전략  
- `04-process-flow-integration.md`: 종단간 프로세스 플로우 및 오류 처리
- `05-improved-project-structure.md`: UseCase 예제를 포함한 포괄적인 아키텍처 제안

### 현재 구현
- `main.py`: 기본 Python 템플릿 (한국어 주석, 간단한 구조)

## 다음 개발 단계

포괄적인 계획 문서를 기반으로:

1. **핵심 도메인 모델 구현**: Order, Vehicle, Route, DispatchResult 엔티티 생성
2. **UseCase 레이어 구축**: ExecuteDispatchUseCase 및 지원 유스케이스부터 시작
3. **알고리즘 팩토리 생성**: 플러그 가능한 알고리즘 선택 시스템 구현
4. **외부 API 통합**: OpenWeatherMap, HERE Maps Traffic, 라우팅 API 직접 호출
5. **캐싱 시스템 구현**: API 최적화를 위한 다층 캐싱
6. **CLI 인터페이스 구축**: 사용자 친화적 명령어 도구

## 필수 참고 링크
[00-execution-guide.md](docs/00-execution-guide.md)
[01-algorithm-selection-rules.md](docs/01-algorithm-selection-rules.md)
[02-dynamic-order-adjustment.md](docs/02-dynamic-order-adjustment.md)
[03-route-optimization-rules.md](docs/03-route-optimization-rules.md)
[04-process-flow-integration.md](docs/04-process-flow-integration.md)
[05-improved-project-structure.md](docs/05-improved-project-structure.md)

# terminal-setup
node: v18.17.0
npm: 9.6.7
git config --global user.email "myemail@example.com"
alias ll='ls -la'
export PATH="/usr/local/bin:$PATH"

## 실용적 날씨/교통 정보 통합 전략

### OpenWeatherMap API 활용
```bash
# 무료 API 키 설정
export OPENWEATHER_API_KEY="your_api_key_here"

# 권역별 날씨 데이터 수집
GET https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}
```

#### 날씨 데이터 활용 로직
```
날씨 영향 계수 적용:
- 맑음: 배송량 × 1.1 (10% 증가)
- 비: 배송량 × 0.8 (20% 감소)  
- 폭우: 배송량 × 0.6 (40% 감소)
- 눈/폭설: 배송량 × 0.5 (50% 감소)

차량별 날씨 영향:
- 오토바이: 비/눈 시 추가 30% 감소
- 승용차: 비/눈 시 추가 10% 감소
- 트럭: 강풍 시 추가 20% 감소
```

### HERE Maps Traffic API 활용
```bash
# HERE API 설정
export HERE_API_KEY="your_api_key_here"

# 실시간 교통 정보 수집
GET https://traffic.ls.hereapi.com/traffic/6.0/incidents.json?bbox={bbox}&apikey={API_KEY}
```

#### 교통 정보 활용 로직
```
교통 정체도별 조정:
- 원활 (0.0-0.2): 배송량 × 1.1 (10% 증가)
- 보통 (0.2-0.6): 배송량 × 1.0 (변화 없음)
- 정체 (0.6-0.8): 배송량 × 0.8 (20% 감소)
- 심각한 정체 (0.8-1.0): 배송량 × 0.6 (40% 감소)

실시간 조정:
- 교통사고 발생: 해당 권역 배송량 30% 감소
- 도로공사: 우회 경로 시간 50% 증가 적용
- 집회/행사: 해당 지역 배송 일시 중단
```

### 통합 조정 공식
```
최종_배송량 = 기본_용량 
              × 경험도_계수 
              × 날씨_계수 
              × 교통_계수 
              × 권역_난이도_계수

실시간 재조정 트리거:
- 날씨 심각도 1.0 이상 변화
- 교통 정체도 0.3 이상 변화  
- 새로운 교통사고/공사 발생
- 30분마다 자동 재평가
```

프로젝트는 클린 아키텍처 원칙과 실용적인 외부 데이터 통합을 통한 현실적인 배차 시스템 구현을 목표로 합니다.