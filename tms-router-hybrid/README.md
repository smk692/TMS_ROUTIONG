# TMS Router Hybrid - 배차 최적화 시스템

> 전통적인 알고리즘과 현대적인 최적화 기법을 결합한 하이브리드 운송 관리 시스템(TMS)
> 
> **실제 한국 지역 데이터 기반** - 6개 물류센터와 90개 실제 주문 데이터  
> **클린 아키텍처 설계** - Core 엔진과 UI 레이어의 완전 분리된 독립 실행 구조

## 🚀 주요 특징

### 🧠 지능형 알고리즘 선택
- **≤30개**: 최근접 이웃 (30초, 70-80% 품질)
- **31-100개**: 유전자 알고리즘 (2-5분, 85-90% 품질)  
- **101-300개**: 시뮬레이티드 어닐링 (5-10분, 88-93% 품질)
- **300개+**: 대규모 근방 탐색 (최고 품질, 90-95%)

### 🌤️ 실시간 조건 분석
- **날씨**: OpenWeatherMap API 연동, 심각도 점수 계산
- **교통**: HERE Maps Traffic API 연동, 실시간 정체/사고 정보
- **동적 조정**: 기사 경험도(70-130%), 날씨(30-110%), 교통(60-110%)

### 🏗️ 아키텍처 특징
- **독립 실행**: Core 엔진(`core/`)과 UI 레이어(`temp_ui/`) 완전 분리
- **Subprocess 통신**: JSON 기반 안전한 프로세스 간 통신
- **다층 캐싱**: 메모리/디스크 캐시, TTL 자동 관리
- **Docker 통합**: MySQL, Redis, phpMyAdmin 완전 자동화

---

## 📋 빠른 시작 (5분)

### 1. 환경 준비
```bash
# 저장소 클론 및 Docker 시작
git clone <repository-url>
cd tms-router-hybrid
docker compose up -d

# 가상환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. 실행 방법 선택

#### CLI 배차 엔진 (Core)
```bash
# 1분 대기 후 Core 엔진 실행
sleep 60
python -m core.main dispatch -c CENTER_GANGNAM

# 결과: 콘솔 테이블 + phpMyAdmin (localhost:8080)
```

#### 웹 인터페이스 (UI)
```bash
# 웹 인터페이스 실행
cd temp_ui
python run_web.py

# 접속: http://localhost:8504
# 기능: 대시보드, 지도, 배차실행, 이력관리
```

---

## 🖥️ 사용법

### Core 엔진 명령어
```bash
# 기본 배차 실행
python -m core.main dispatch -c CENTER_GANGNAM

# 알고리즘 지정
python -m core.main dispatch -c CENTER_GANGNAM -a genetic

# 시뮬레이션 모드
python -m core.main dispatch -c CENTER_GANGNAM --dry-run

# 시스템 상태 확인
python -m core.main status

# 캐시 관리
python -m core.main clear-cache --cache-type weather
```

### 사용 가능한 센터 ID
- `CENTER_GANGNAM`, `CENTER_GANGBUK`, `CENTER_GANGDONG`
- `CENTER_GANGSEO`, `CENTER_SEOCHO`, `CENTER_MAPO`

### 알고리즘 옵션
- `auto` (기본): 주문 수에 따라 자동 선택
- `nearest`: 최근접 이웃 (빠름)
- `genetic`: 유전자 알고리즘 (균형)
- `annealing`: 시뮬레이티드 어닐링 (고품질)

---

## 🔧 고급 설정

### API 키 설정 (선택사항)
`.env` 파일에 추가하면 실시간 데이터 사용:
```bash
OPENWEATHER_API_KEY=your_openweather_api_key
HERE_API_KEY=your_here_api_key
KAKAO_REST_API_KEY=your_kakao_api_key
```

### Docker 서비스 정보
- **MySQL**: `localhost:3306` (tms_user/tms_password)
- **phpMyAdmin**: `localhost:8080`
- **Redis**: `localhost:6379`

### 설정 파일 (config.json)
```json
{
  "cache": {
    "memory_size_mb": 200,
    "weather_cache_ttl": 30,
    "traffic_cache_ttl": 15
  },
  "algorithm": {
    "small_order_threshold": 30,
    "medium_order_threshold": 100
  }
}
```

---

## 🏗️ 아키텍처

### 프로젝트 구조
```
tms-router-hybrid/
├── core/                 # 독립적인 배차 엔진
│   ├── main.py          # CLI 인터페이스
│   ├── models/          # 도메인 모델
│   ├── services/        # 비즈니스 로직
│   ├── algorithms/      # 최적화 알고리즘
│   ├── external/        # 외부 API & 캐싱
│   └── database/        # DB 연결 & 모델
└── temp_ui/             # 독립적인 UI 레이어
    ├── run_web.py       # Streamlit 서버
    ├── pages/           # 웹 페이지
    └── web_api/         # Core 통신 API
```

### 처리 흐름
1. **UI 요청** → subprocess로 Core 엔진 호출
2. **Core 엔진** → 데이터 수집, 조건 분석, 알고리즘 실행
3. **JSON 응답** → UI에서 시각화 (지도, 차트, 테이블)

### 아키텍처 장점
- **독립성**: 각 레이어 독립 개발/배포
- **확장성**: 새 UI 레이어(모바일 등) 추가 용이
- **안전성**: 프로세스 격리, JSON 통신
- **모니터링**: 각 프로세스별 리소스 추적

---

## 🐛 문제 해결

### 자주 발생하는 문제

#### 1. 웹 페이지 접속 불가
```bash
# temp_ui 디렉토리에서 실행 확인
source venv/bin/activate

cd temp_ui && python run_web.py

# 포트 충돌 시 프로세스 종료
# lsof -ti:8504 | xargs kill -9
```

#### 2. 배차할 데이터 없음
```bash
# 데이터 확인
docker exec tms_mysql mysql -u tms_user -ptms_password -D tms_db -e "SELECT center_id, COUNT(*) FROM orders WHERE status='pending' GROUP BY center_id;"

# 완전 초기화
docker compose down -v && docker compose up -d
```

#### 3. Docker 컨테이너 오류
```bash
# 컨테이너 상태 확인
docker compose ps

# 재시작
docker compose restart

# 완전 초기화
docker compose down -v && docker compose up -d
```

### 디버그 모드
```bash
# 상세 로그 출력
python -m core.main --debug dispatch -c CENTER_GANGNAM

# Docker 로그 확인
docker compose logs -f mysql
```

---

## 📊 성능 최적화

### 캐시 활용
- **날씨**: 30분 TTL, API 호출 최소화
- **교통**: 15분 TTL, 실시간성 유지
- **경로**: 60분 TTL, 계산 비용 절약

### 알고리즘 선택 가이드
- **≤30개 주문**: `nearest` (속도 우선)
- **품질 중시**: `annealing` (최고 품질)
- **균형**: `auto` (자동 선택)

---

## 🧪 테스트

```bash
# 단위 테스트
pytest tests/

# 커버리지 포함
pytest --cov=. tests/

# 특정 테스트
pytest tests/test_algorithms.py -v
```

---

## 📚 추가 문서

- [개발 진행상황](DEVELOPMENT_PROGRESS.md)
- [알고리즘 선택 규칙](docs/01-algorithm-selection-rules.md)
- [동적 주문 조정](docs/02-dynamic-order-adjustment.md)
- [경로 최적화](docs/03-route-optimization-rules.md)
- [프로세스 플로우](docs/04-process-flow-integration.md)
- [프로젝트 구조](docs/05-improved-project-structure.md)

---

## 🤝 기여하기

1. Fork 프로젝트
2. Feature 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

---

**TMS Router Hybrid** - 더 스마트한 배송, 더 효율적인 물류 🚚✨