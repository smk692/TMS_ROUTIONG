# 🚛 TMS 배송 경로 최적화 시스템 - 전체 구조 분석

## 📋 프로젝트 개요

**TMS (Transportation Management System) 배송 경로 최적화 시스템**은 고급 TSP 알고리즘과 실제 도로 경로를 활용한 엔터프라이즈급 배송 경로 최적화 시스템입니다.

### 🎯 핵심 기능
1. **TSP 알고리즘으로 최적 경로 JSON 생성**
2. **실제 도로 경로 폴리라인 추가 (OSRM API)**
3. **고급 인터랙티브 지도로 최종 시각화**

### 🔧 주요 특징
- **🧠 고급 TSP 알고리즘**: OR-Tools 기반 최적화
- **🗺️ 실제 도로 경로**: OSRM API 연동
- **📊 인터랙티브 시각화**: Folium 기반 고급 지도
- **⚙️ Config 기반 관리**: 중앙화된 설정 시스템
- **🚛 제약조건 처리**: 차량 용량, 시간, 거리 제약
- **📈 성능 최적화**: 클러스터링 + TSP 하이브리드

---

## 🏗️ 프로젝트 구조

```
TMS_ROUTING/
├── 📁 config/                          # 🔧 설정 관리 시스템
│   ├── tms_config.py                   # 중앙 설정 시스템 (364줄)
│   ├── example_configs.py              # 시나리오별 예시 설정 (144줄)
│   ├── db_config.py                    # 데이터베이스 설정 (8줄)
│   └── __init__.py                     # 패키지 초기화 (9줄)
│
├── 📁 src/                             # 📦 핵심 서비스 모듈
│   ├── 📁 services/                    # 비즈니스 로직 서비스
│   │   ├── route_extractor.py          # TSP 최적화 서비스 (998줄)
│   │   ├── integrated_route_service.py # 통합 최적화 서비스 (663줄)
│   │   ├── polyline_service.py         # 실제 경로 서비스 (384줄)
│   │   ├── constraint_handler.py       # 제약조건 처리 (550줄)
│   │   ├── vehicle_id_service.py       # 차량 ID 관리 (139줄)
│   │   └── depot_selector.py           # 물류센터 선택 (158줄)
│   │
│   ├── 📁 algorithm/                   # 최적화 알고리즘
│   │   ├── cluster.py                  # 클러스터링 알고리즘 (1414줄)
│   │   ├── cluster_optimizer.py        # 클러스터링 최적화 (448줄)
│   │   ├── voronoi_optimizer.py        # Voronoi 최적화 (559줄)
│   │   ├── tsp.py                      # TSP 알고리즘 (234줄)
│   │   ├── vrp.py                      # VRP 알고리즘 (206줄)
│   │   └── __init__.py
│   │
│   ├── 📁 visualization/               # 시각화 모듈
│   │   ├── route_visualizer.py         # 경로 시각화 (1208줄)
│   │   └── clustering_visualizer.py    # 클러스터링 시각화 (148줄)
│   │
│   ├── 📁 analysis/                    # 분석 도구
│   │   ├── route_analyzer.py           # 경로 분석 (456줄)
│   │   └── voronoi_analyzer.py         # Voronoi 분석 (309줄)
│   │
│   ├── 📁 db/                          # 데이터베이스 연동
│   │   ├── maria_client.py             # MariaDB 클라이언트 (42줄)
│   │   └── __init__.py
│   │
│   ├── 📁 core/                        # 핵심 모듈
│   ├── 📁 model/                       # 데이터 모델
│   ├── 📁 monitoring/                  # 모니터링
│   ├── 📁 utils/                       # 유틸리티
│   └── __init__.py
│
├── 📁 data/                            # 📊 데이터 저장소
│   ├── extracted_coordinates.json      # 최적화된 경로 데이터 (6.8MB)
│   └── route_data.json                 # 기본 경로 데이터 (2.4KB)
│
├── 📁 output/                          # 📈 결과물 저장소
├── 📁 logs/                            # 📝 로그 파일
│
├── 🚀 run_all.py                       # 메인 실행 스크립트 (571줄)
├── 📖 README.md                        # 프로젝트 문서 (379줄)
├── 📦 requirements.txt                 # 의존성 패키지 (19줄)
├── ⚙️ setup.py                         # 패키지 설정 (22줄)
├── 🚫 .gitignore                       # Git 제외 파일 (98줄)
│
├── 🗺️ current_route_map_new.html       # 생성된 지도 파일 (17MB)
└── 🧪 test_output.html                 # 테스트 결과 (6.1MB)
```

---

## 🔧 핵심 모듈 분석

### 1. 📋 Config 시스템 (`config/`)

#### `tms_config.py` - 중앙 설정 관리 시스템
- **역할**: 모든 시스템 설정을 중앙에서 관리
- **주요 기능**:
  - 점 표기법으로 설정값 접근 (`vehicles.count`)
  - 프리셋 시스템 (fast, quality, large_scale, test)
  - 설정 검증 및 요약 출력
  - 명령행 인수 기반 설정 업데이트

#### 주요 설정 카테고리 (23개 변수):
```python
# 🚗 차량 관련 설정 (6개)
vehicles: {
    count: 15,                    # 차량 수
    capacity: {volume: 5.0, weight: 1000.0},
    operating_hours: {start: 6, end: 14},
    cost_per_km: 500.0,
    average_speed: 30.0
}

# 📍 물류센터 & 배송 설정 (7개)
logistics: {
    depots: [10개 물류센터],      # 다중 물류센터
    delivery: {
        max_distance: 12.0,       # 배송 반경
        points_per_vehicle: 50,   # 차량당 배송지
        service_time: 3           # 서비스 시간
    }
}

# ⚖️ 제약조건 설정 (6개)
constraints: {
    max_working_hours: 10,
    max_points_per_vehicle: 50,
    allow_overtime: False
}

# 🧠 알고리즘 파라미터 (3개)
algorithms: {
    tsp: {max_iterations: 100},
    clustering: {strategy: "enhanced_kmeans"}
}

# 🎨 시각화 설정 (1개)
visualization: {colors: [15색 배열]}
```

### 2. 🚀 메인 실행기 (`run_all.py`)

#### 핵심 실행 모드:
1. **통합 최적화 모드** (`--integrated`)
   - 기존 2단계 → 1단계 통합
   - 50% 성능 향상
   - OSRM API 호출 최적화

2. **단계별 실행 모드**
   - Step 1: TSP 최적화 (`step1_route_extraction`)
   - Step 2: 폴리라인 추가 (`step2_polyline_addition`)
   - Step 3: 시각화 (`step3_visualization`)

3. **고급 최적화 옵션**
   - Voronoi 최적화: 99.2% 중복 해결
   - 클러스터링 최적화: 대규모 처리

#### 명령행 옵션:
```bash
# 기본 실행
python run_all.py

# 통합 최적화 (50% 빠름)
python run_all.py --integrated

# 고급 최적화
python run_all.py --voronoi-optimization
python run_all.py --cluster-optimization

# 프리셋 사용
python run_all.py --preset fast
python run_all.py --preset quality

# 개별 설정
python run_all.py --vehicles 20 --max-distance 25
```

### 3. 📦 서비스 모듈 (`src/services/`)

#### `route_extractor.py` - TSP 최적화 서비스 (998줄)
- **역할**: 핵심 경로 최적화 엔진
- **주요 기능**:
  - 데이터베이스에서 배송 데이터 로드
  - 거리 기반 필터링
  - 클러스터링 + TSP 하이브리드 최적화
  - 제약조건 처리 (용량, 시간, 거리)

#### `integrated_route_service.py` - 통합 최적화 서비스 (663줄)
- **역할**: 모든 단계를 통합한 고성능 서비스
- **특징**: 
  - 2단계 → 1단계 통합
  - 메모리 효율성 개선
  - API 호출 최적화

#### `polyline_service.py` - 실제 경로 서비스 (384줄)
- **역할**: OSRM API를 통한 실제 도로 경로 추가
- **최적화**: N번 API 호출 → 차량 수만큼 호출 (99% 시간 단축)

### 4. 🧠 알고리즘 모듈 (`src/algorithm/`)

#### `cluster.py` - 클러스터링 알고리즘 (1414줄)
- **역할**: 대규모 배송지를 효율적으로 그룹화
- **알고리즘**: Enhanced K-means, DBSCAN

#### `voronoi_optimizer.py` - Voronoi 최적화 (559줄)
- **역할**: 99.2% 경로 중복 해결
- **특징**: 전략적 시드 포인트 배치

#### `tsp.py` - TSP 알고리즘 (234줄)
- **역할**: 각 클러스터 내 최적 경로 계산
- **알고리즘**: OR-Tools 기반 최적화

### 5. 📊 시각화 모듈 (`src/visualization/`)

#### `route_visualizer.py` - 경로 시각화 (1208줄)
- **역할**: 고급 인터랙티브 지도 생성
- **특징**:
  - TC별 선택/해제 토글
  - 차량별 레이어 제어
  - 실제 도로 곡선 폴리라인
  - 반응형 컨트롤 패널
  - 3D 효과 마커

---

## 🔄 시스템 실행 흐름

### 1. 초기화 단계
```python
# Config 시스템 로드
from config.tms_config import tms_config

# 프리셋 적용 (선택사항)
apply_preset('fast')

# 명령행 인수 적용
tms_config.update_from_args(args)
```

### 2. 데이터 처리 단계
```python
# 1단계: TSP 최적화
extractor = RouteExtractorService(config=tms_config)
result = extractor.extract_routes()

# 2단계: 폴리라인 추가
polyline_service = PolylineService(config=tms_config)
success = polyline_service.add_polylines_to_data()

# 3단계: 시각화
visualizer = RouteVisualizerService(config=tms_config)
map_file = visualizer.visualize_routes()
```

### 3. 통합 최적화 모드
```python
# 모든 단계를 한 번에 처리
integrated_service = IntegratedRouteService(tms_config)
result = integrated_service.extract_routes_integrated()
```

---

## 📊 의존성 패키지

### 운영 필수 패키지
```
pymysql>=1.0.0          # MariaDB 연결
sqlalchemy>=1.4.0       # ORM
pandas>=1.5.0           # 데이터 처리
numpy>=1.21.0           # 수치 계산
scikit-learn>=1.1.0     # 머신러닝 (클러스터링)
ortools>=9.0            # 최적화 엔진
folium>=0.14.0          # 지도 시각화
geopy>=2.2.0            # 지리 계산
hdbscan>=0.8.0          # 고급 클러스터링
psutil>=5.8.0           # 시스템 모니터링
```

### 개발/테스트 패키지 (선택사항)
```
pytest>=7.0.0           # 테스트 프레임워크
matplotlib>=3.5.0       # 그래프 시각화
jupyter>=1.0.0          # 노트북 환경
```

---

## 🎯 주요 최적화 기법

### 1. Voronoi 기반 최적화
- **목표**: 99.2% 경로 중복 해결
- **방법**: 전략적 시드 포인트 배치
- **효과**: 중복 경로 제거, 효율성 향상

### 2. 클러스터링 기반 최적화
- **알고리즘**: DBSCAN + Balanced K-means
- **특징**: 지역 기반 자연스러운 클러스터링
- **효과**: 대규모 데이터 처리 최적화

### 3. 통합 최적화
- **혁신**: 기존 2단계 → 1단계 통합
- **성능**: OSRM API 호출 50% 감소
- **결과**: 처리 시간 50% 단축

### 4. API 최적화
- **기존**: N번 API 호출 (배송지 수만큼)
- **개선**: 차량 수만큼만 API 호출
- **효과**: 99% 시간 단축

---

## 🔧 설정 가능한 변수 (23개)

### 🚗 차량 관련 (6개)
| 변수 | 기본값 | 설명 |
|------|--------|------|
| 차량 수 | 15대 | 투입할 차량 대수 |
| 부피 용량 | 5.0m³ | 차량당 부피 제한 |
| 무게 용량 | 1000kg | 차량당 무게 제한 |
| 운영시간 | 6-14시 | 차량 운영 시간대 |
| km당 비용 | 500원 | 차량 운영 비용 |
| 평균 속도 | 30km/h | 차량 평균 속도 |

### 📍 물류센터 & 배송 (7개)
| 변수 | 기본값 | 설명 |
|------|--------|------|
| 물류센터 | 10개 센터 | 다중 물류센터 |
| 배송 반경 | 12km | 최대 배송 거리 |
| 차량당 배송지 | 50개 | 차량별 최대 배송지 |
| 서비스 시간 | 3분 | 배송지당 소요시간 |
| 기본 부피 | 0.0m³ | 배송지 기본 부피 |
| 기본 무게 | 0.0kg | 배송지 기본 무게 |
| 우선순위 | 3 | 배송 우선순위 |

### ⚖️ 제약조건 (6개)
| 변수 | 기본값 | 설명 |
|------|--------|------|
| 최대 근무시간 | 10시간 | 차량별 최대 근무 |
| 최대 배송지 | 50개 | 차량당 최대 배송지 |
| 최소 배송지 | 10개 | 차량당 최소 배송지 |
| 초과근무 허용 | False | 초과근무 가능 여부 |
| 교통상황 고려 | True | 실시간 교통 반영 |
| 목표 효율성 | 0.1 | 최적화 목표 수준 |

### 🧠 알고리즘 파라미터 (3개)
| 변수 | 기본값 | 설명 |
|------|--------|------|
| TSP 최대 반복 | 100회 | TSP 알고리즘 반복 |
| 개선 허용 횟수 | 20회 | 개선 없을 때 중단 |
| SA 온도 | 80.0 | Simulated Annealing |

### 🎨 시각화 설정 (1개)
| 변수 | 기본값 | 설명 |
|------|--------|------|
| 차량별 색상 | 15색 배열 | 지도 시각화 색상 |

---

## 🚀 사용 시나리오

### 1. 빠른 테스트
```bash
python run_all.py --preset test --test-mode
```

### 2. 고품질 최적화
```bash
python run_all.py --preset quality --voronoi-optimization
```

### 3. 대규모 처리
```bash
python run_all.py --preset large_scale --cluster-optimization
```

### 4. 통합 최적화 (권장)
```bash
python run_all.py --integrated --vehicles 20
```

---

## 📈 성능 지표

### 최적화 효과
- **Voronoi 최적화**: 99.2% 중복 해결
- **통합 최적화**: 50% 성능 향상
- **API 최적화**: 99% 시간 단축
- **클러스터링**: 대규모 데이터 처리 가능

### 처리 능력
- **배송지**: 수천 개 동시 처리
- **차량**: 최대 50대 동시 관리
- **물류센터**: 10개 센터 자동 선택
- **실시간**: 교통상황 반영

---

## 🔮 확장 가능성

### 1. 추가 최적화 알고리즘
- 유전 알고리즘 (Genetic Algorithm)
- 개미 군집 최적화 (Ant Colony Optimization)
- 강화학습 기반 최적화

### 2. 실시간 기능
- 실시간 교통 정보 연동
- 동적 경로 재계산
- 실시간 배송 상태 추적

### 3. 고급 제약조건
- 시간 윈도우 제약
- 차량별 특성 제약
- 고객 우선순위 제약

### 4. 클라우드 확장
- 마이크로서비스 아키텍처
- 컨테이너 기반 배포
- 자동 스케일링

---

## 📝 결론

TMS 배송 경로 최적화 시스템은 **Config 기반의 유연한 설정 관리**와 **고급 최적화 알고리즘**을 결합한 엔터프라이즈급 시스템입니다. 

**주요 강점**:
- 🔧 **유연성**: 23개 설정 변수로 다양한 시나리오 대응
- ⚡ **성능**: 통합 최적화로 50% 성능 향상
- 🎯 **정확성**: Voronoi 최적화로 99.2% 중복 해결
- 📊 **시각화**: 고급 인터랙티브 지도 제공
- 🔄 **확장성**: 모듈화된 아키텍처로 쉬운 확장

이 시스템은 소규모 사업체부터 대기업까지 다양한 규모의 배송 최적화 요구사항을 효과적으로 해결할 수 있습니다. 