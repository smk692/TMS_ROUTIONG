# 🚛 TMS 배송 경로 최적화 시스템 (Config 기반)

고급 TSP 알고리즘과 실제 도로 경로를 활용한 엔터프라이즈급 배송 경로 최적화 시스템입니다.

## 🔄 최근 업데이트 (2025-05-28)

### ✅ Route Visualizer 복구 완료
- **route_visualizer.py 완전 재생성**: 모든 CSS 스타일과 JavaScript 기능 통합
- **현대적인 UI 컨트롤**: TC별 정보 패널, 차량별 레이어 제어 패널
- **인터랙티브 기능**: TC 선택, 차량 필터링, 레이어 토글
- **3D 효과 마커**: 그라데이션과 호버 효과가 적용된 현대적 디자인
- **반응형 디자인**: 모바일 및 데스크톱 환경 지원

### 🎨 주요 개선사항
- **CSS 스타일**: 현대적인 카드 스타일, 그라데이션, 애니메이션
- **JavaScript 기능**: TC별 차량 매핑, 실시간 레이어 제어
- **마커 디자인**: 32px 크기, 3D 효과, 색상 구분
- **팝업 스타일**: 구조화된 테이블, 아이콘 사용, 색상 구분

### 🧪 테스트 완료
- **샘플 데이터 생성**: 3대 차량, 2개 TC, 9개 배송지
- **시각화 검증**: HTML 파일 생성 및 기능 확인
- **CSS/JS 통합**: 모든 스타일과 기능이 정상 작동

## 🎯 주요 특징

- **🧠 고급 TSP 알고리즘**: OR-Tools 기반 최적화
- **🗺️ 실제 도로 경로**: OSRM API 연동
- **📊 인터랙티브 시각화**: Folium 기반 고급 지도
- **⚙️ Config 기반 관리**: 중앙화된 설정 시스템
- **🚛 제약조건 처리**: 차량 용량, 시간, 거리 제약
- **📈 성능 최적화**: 클러스터링 + TSP 하이브리드

## 🚀 빠른 시작

### 기본 실행
```bash
# 전체 과정 실행 (기본 설정)
python run_all.py

# 프리셋 사용
python run_all.py --preset fast        # 빠른 처리
python run_all.py --preset quality     # 높은 품질
python run_all.py --preset test        # 테스트 모드

# 개별 설정
python run_all.py --vehicles 10 --max-distance 20
python run_all.py --capacity-volume 8.0 --capacity-weight 1500
```

### Config 기반 관리
```bash
# 현재 설정 확인
python run_all.py --show-config

# 프리셋 목록 보기
python run_all.py --list-presets

# 단계별 실행
python run_all.py --step1-only         # TSP 최적화만
python run_all.py --step2-only         # 폴리라인 추가만
python run_all.py --step3-only         # 시각화만
```

## ⚙️ Config 시스템

### 1. 중앙화된 설정 관리

모든 시스템 설정이 `config/tms_config.py`에서 중앙 관리됩니다:

```python
from config import tms_config

# 설정 조회
vehicle_count = tms_config.get('vehicles.count')
max_distance = tms_config.get('logistics.delivery.max_distance')

# 설정 변경
tms_config.set('vehicles.count', 20)
tms_config.set('logistics.delivery.max_distance', 25.0)
```

### 2. 프리셋 시스템

다양한 시나리오에 맞는 사전 정의된 설정:

| 프리셋 | 설명 | 특징 |
|--------|------|------|
| `fast` | 빠른 처리 | 낮은 품질, 빠른 속도 |
| `quality` | 높은 품질 | 느린 처리, 최적 결과 |
| `large_scale` | 대규모 처리 | 많은 차량, 넓은 반경 |
| `test` | 테스트 모드 | 소규모 데이터 |

### 3. 비즈니스 시나리오별 Config

```python
from config.example_configs import *

# 소규모 사업체
apply_small_business_config()      # 5대, 8km 반경

# 대기업
apply_enterprise_config()          # 50대, 30km 반경

# 당일 배송
apply_same_day_delivery_config()   # 20대, 14시간 운영

# 농촌 배송
apply_rural_delivery_config()      # 8대, 50km 반경

# 친환경 배송
apply_eco_friendly_config()        # 12대, 전기차

# 성수기
apply_peak_season_config()         # 30대, 18시간 운영
```

## 📋 변경 가능한 변수 포인트 (23개)

### 🚗 차량 관련 설정 (6개)
| 변수 | 기본값 | 설명 | Config 경로 |
|------|--------|------|-------------|
| 차량 수 | 15대 | 투입할 차량 대수 | `vehicles.count` |
| 부피 용량 | 5.0m³ | 차량당 부피 제한 | `vehicles.capacity.volume` |
| 무게 용량 | 1000kg | 차량당 무게 제한 | `vehicles.capacity.weight` |
| 운영시간 | 6-14시 | 차량 운영 시간대 | `vehicles.operating_hours` |
| km당 비용 | 500원 | 차량 운영 비용 | `vehicles.cost_per_km` |
| 평균 속도 | 30km/h | 차량 평균 속도 | `vehicles.average_speed` |

### 📍 물류센터 & 배송 설정 (7개)
| 변수 | 기본값 | 설명 | Config 경로 |
|------|--------|------|-------------|
| 물류센터 위치 | 수원 | 배송 출발지 좌표 | `logistics.depot` |
| 배송 반경 | 15km | 최대 배송 거리 | `logistics.delivery.max_distance` |
| 차량당 배송지 | 50개 | 차량별 최대 배송지 | `logistics.delivery.points_per_vehicle` |
| 서비스 시간 | 5분 | 배송지당 소요시간 | `logistics.delivery.service_time` |
| 기본 부피 | 0.1m³ | 배송지 기본 부피 | `logistics.delivery.default_volume` |
| 기본 무게 | 5kg | 배송지 기본 무게 | `logistics.delivery.default_weight` |
| 우선순위 | 3 | 배송 우선순위 | `logistics.delivery.default_priority` |

### ⚖️ 제약조건 설정 (6개)
| 변수 | 기본값 | 설명 | Config 경로 |
|------|--------|------|-------------|
| 최대 근무시간 | 8시간 | 차량별 최대 근무 | `constraints.max_working_hours` |
| 최대 배송지 | 50개 | 차량당 최대 배송지 | `constraints.max_points_per_vehicle` |
| 최소 배송지 | 10개 | 차량당 최소 배송지 | `constraints.min_points_per_vehicle` |
| 초과근무 허용 | False | 초과근무 가능 여부 | `constraints.allow_overtime` |
| 교통상황 고려 | True | 실시간 교통 반영 | `constraints.consider_traffic` |
| 목표 효율성 | 0.1 | 최적화 목표 수준 | `constraints.target_efficiency` |

### 🧠 알고리즘 파라미터 (3개)
| 변수 | 기본값 | 설명 | Config 경로 |
|------|--------|------|-------------|
| TSP 최대 반복 | 150회 | TSP 알고리즘 반복 | `algorithms.tsp.max_iterations` |
| 개선 허용 횟수 | 30회 | 개선 없을 때 중단 | `algorithms.tsp.no_improvement_limit` |
| SA 온도 | 100.0 | Simulated Annealing | `algorithms.tsp.initial_temperature` |

### 🎨 시각화 설정 (1개)
| 변수 | 기본값 | 설명 | Config 경로 |
|------|--------|------|-------------|
| 차량별 색상 | 15색 배열 | 지도 시각화 색상 | `visualization.colors` |

## 🏗️ 시스템 아키텍처

### 📁 프로젝트 구조
```
TMS_ROUTING/
├── config/                     # 🔧 설정 관리
│   ├── tms_config.py          # 중앙 설정 시스템
│   ├── example_configs.py     # 시나리오별 예시
│   └── __init__.py
├── src/                       # 📦 핵심 서비스
│   ├── services/              # 비즈니스 로직
│   │   ├── route_extractor.py # TSP 최적화 서비스
│   │   └── polyline_service.py # 실제 경로 서비스
│   ├── visualization/         # 시각화
│   │   └── route_visualizer.py
│   ├── models/               # 데이터 모델
│   ├── algorithms/           # 알고리즘
│   └── utils/               # 유틸리티
├── data/                    # 📊 데이터
├── output/                  # 📈 결과물
└── run_all.py              # 🚀 메인 실행기
```

## 🔄 실행 로직

### 1단계: TSP 최적화
```python
# Config 기반 서비스 초기화
extractor = RouteExtractorService(config=tms_config)

# 데이터 로드 및 클러스터링
delivery_points = extractor.load_delivery_data()
filtered_points = extractor.filter_points_by_distance()
routes = extractor.clustering_optimization()
```

### 2단계: 실제 경로 추가
```python
# Config 기반 폴리라인 서비스
polyline_service = PolylineService(config=tms_config)

# OSRM API로 실제 도로 경로 계산
enhanced_routes = polyline_service.add_polylines_to_routes()
```

### 3단계: 시각화
```python
# Config 기반 시각화 서비스
visualizer = RouteVisualizerService(config=tms_config)

# 인터랙티브 지도 생성
visualizer.create_interactive_map()
```

## 📊 성능 최적화

### 클러스터링 최적화
- **HDBSCAN 알고리즘**: 밀도 기반 클러스터링
- **제약조건 검증**: 용량, 시간, 거리 제약 실시간 체크
- **동적 조정**: 제약 위반 시 자동 재클러스터링

### TSP 최적화
- **OR-Tools**: Google 최적화 라이브러리
- **Simulated Annealing**: 지역 최적해 탈출
- **시간 제한**: 실용적 처리 시간 보장

### 병렬 처리
- **멀티스레딩**: OSRM API 호출 병렬화
- **비동기 처리**: 대용량 데이터 효율 처리
- **메모리 최적화**: 점진적 데이터 로딩

## 🛠️ 개발자 가이드

### Config 시스템 확장

새로운 설정 추가:
```python
# config/tms_config.py에서
"new_feature": {
    "enabled": True,
    "parameter": 100
}

# 코드에서 사용
if tms_config.get('new_feature.enabled'):
    value = tms_config.get('new_feature.parameter')
```

### 새로운 프리셋 추가

```python
# config/tms_config.py의 PRESETS에 추가
"custom_preset": {
    "description": "커스텀 설정",
    "config": {
        "vehicles.count": 25,
        "logistics.delivery.max_distance": 20.0
    }
}
```

### 서비스 확장

새로운 서비스 추가:
```python
class NewService:
    def __init__(self, config=None):
        self.config = config
        # config에서 설정 로드
        self.setting = config.get('new_service.setting') if config else default_value
```

## 📈 변수 변경 가이드

### 높은 영향도 (성능에 큰 영향)
- `vehicles.count`: 차량 수 ↑ → 처리 속도 ↑, 비용 ↑
- `logistics.delivery.max_distance`: 반경 ↑ → 배송지 ↑, 거리 ↑
- `algorithms.tsp.max_iterations`: 반복 ↑ → 품질 ↑, 시간 ↑

### 중간 영향도 (균형 조정)
- `vehicles.capacity.*`: 용량 ↑ → 효율성 ↑, 유연성 ↓
- `constraints.max_working_hours`: 시간 ↑ → 배송량 ↑, 비용 ↑
- `logistics.delivery.points_per_vehicle`: 배송지 ↑ → 효율성 ↑, 복잡도 ↑

### 낮은 영향도 (미세 조정)
- `vehicles.average_speed`: 속도 조정
- `logistics.delivery.service_time`: 서비스 시간
- `visualization.*`: 시각화 옵션

## 💡 실용적인 변경 팁

### 성능 향상
```bash
# 빠른 처리가 필요한 경우
python run_all.py --preset fast --vehicles 10

# 고품질 결과가 필요한 경우  
python run_all.py --preset quality --tsp-iterations 200
```

### 정확도 향상
```bash
# 더 정밀한 최적화
python run_all.py --tsp-iterations 300 --max-distance 12

# 제약조건 강화
python run_all.py --max-working-hours 6 --capacity-volume 4.0
```

### 대용량 처리
```bash
# 대규모 데이터 처리
python run_all.py --preset large_scale --vehicles 30

# 성수기 대응
python run_all.py --vehicles 25 --max-working-hours 12
```

## 📋 명령행 옵션

### 실행 모드
- `--step1-only`: TSP 최적화만 실행
- `--step2-only`: 폴리라인 추가만 실행
- `--step3-only`: 시각화만 실행
- `--skip-polylines`: 폴리라인 추가 건너뛰기
- `--test-mode`: 테스트 모드 (소규모 처리)

### 설정 옵션
- `--preset {fast,quality,large_scale,test}`: 프리셋 적용
- `--vehicles N`: 차량 수 지정
- `--capacity-volume N`: 차량 부피 용량 (m³)
- `--capacity-weight N`: 차량 무게 용량 (kg)
- `--max-distance N`: 배송 반경 (km)
- `--points-per-vehicle N`: 차량당 배송지 수
- `--max-working-hours N`: 최대 근무시간
- `--tsp-iterations N`: TSP 최대 반복 횟수

### 출력 옵션
- `--output-dir DIR`: 출력 디렉토리 지정
- `--visualization-dir DIR`: 시각화 출력 디렉토리
- `--data-file FILE`: 데이터 파일 경로

### 정보 옵션
- `--show-config`: 현재 설정 표시
- `--list-presets`: 사용 가능한 프리셋 목록
- `--help`: 도움말 표시

## 🔧 요구사항

### Python 패키지
```
pymysql>=1.0.2
sqlalchemy>=1.4.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
ortools>=9.4.0
folium>=0.14.0
geopy>=2.2.0
hdbscan>=0.8.29
psutil>=5.9.0
```

### 시스템 요구사항
- Python 3.8+
- MySQL 데이터베이스
- 인터넷 연결 (OSRM API)
- 메모리 4GB+ 권장

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

---

**TMS 배송 경로 최적화 시스템** - Config 기반 엔터프라이즈급 솔루션
