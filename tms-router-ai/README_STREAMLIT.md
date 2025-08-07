# 🚚 TMS AI 배차 시스템 - Streamlit Web Interface

## 📋 개요

사용자 친화적인 웹 인터페이스를 통해 TMS AI 배차 시스템을 쉽게 사용할 수 있습니다.

### 🔥 주요 기능

- **🚛 배차 최적화 요청**: 직관적인 폼을 통한 차량/주문 정보 입력
- **📊 결과 분석**: 시각화된 경로 지도 및 통계 차트
- **💾 이력 관리**: 최적화 이력 저장 및 재실행
- **🔍 시스템 모니터링**: 실시간 API 상태 및 성능 모니터링
- **💬 피드백 시스템**: 결과에 대한 평가 및 학습

## 🚀 빠른 시작

### 1. 패키지 설치

```bash
cd tms-router-ai

# 가상환경 설치
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 의존성 설치 (Streamlit 포함)
pip install -r requirements.txt
```

### 2. Chalice API 서버 시작

```bash
# 터미널 1: Chalice 로컬 서버 실행
chalice local --port 8000
```

### 3. Redis 서버 시작

```bash
# 터미널 2: Docker로 Redis 실행
docker-compose up -d redis

# 또는 로컬 Redis 실행
redis-server
```

### 4. Streamlit 앱 실행

```bash
# 터미널 3: Streamlit 앱 실행
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501`로 접속하세요!

## 🎯 사용 방법

### 📝 배차 요청 탭

1. **시나리오 선택**: 사이드바에서 최적화 시나리오 선택
   - 🤖 자동 선택 (AI 추천)
   - 🚛 다중 차량 경로 최적화 (VRP)
   - 🚐 단일 차량 최적 경로 (TSP)
   - 📦 적재 통합 최적화
   - 🚨 긴급 배송 처리
   - ⚡ 실시간 경로 조정

2. **차량 정보 입력**:
   - 차량 수 선택
   - 각 차량별 ID, 용량, 시작 위치, 특수 능력 설정

3. **주문 정보 입력**:
   - 주문 수 선택
   - 각 주문별 ID, 중량, 우선순위, 픽업/배송 위치, 시간창 설정

4. **제약 조건 설정**:
   - 최대 근무 시간
   - 최대 이동 거리
   - 연료비

5. **최적화 실행**: 🚀 "배차 최적화 실행" 버튼 클릭

### 📊 결과 분석 탭

- **전체 요약**: 사용 차량 수, 총 거리, 비용, 효율성
- **경로별 상세**: 각 차량의 경로 정보 및 경유지 목록
- **지도 시각화**: Plotly 기반 인터랙티브 경로 지도
- **최적화 통계**: 차량별 효율성 및 비용 차트
- **피드백 입력**: 결과에 대한 만족도 및 의견 제출

### 💾 이력 관리 탭

- **이력 요약**: 총 실행 횟수, 성공률, 마지막 실행 시간
- **이력 목록**: 과거 최적화 요청 및 결과 확인
- **재실행 기능**: 이전 설정으로 다시 최적화 실행

### 🔍 시스템 모니터링 탭

- **API 서버 상태**: 연결 상태 및 응답 시간 모니터링
- **AI 패턴 매칭 분석**: 시나리오 선택 분포 및 효과성 트렌드
- **세션 정보**: 현재 세션의 메모리 사용량 및 상태

## ⚙️ 고급 기능

### 🧠 피드백 학습

- 결과에 대한 만족도 평가를 통해 AI가 학습
- 사용자별 선호도를 반영한 개인화된 최적화
- 지속적인 성능 개선

### 💾 설정 저장/불러오기

- 현재 차량/주문 설정을 저장
- 저장된 설정을 불러와서 재사용
- 세션 간 설정 유지

### 🗺️ 지도 시각화

- Google Maps 호환 폴리라인 표시
- 차량별 색상 구분
- 경유지 유형별 마커 (시작/픽업/배송/종료)
- 인터랙티브 줌 및 팬

## 🔧 설정

### 환경 변수

```bash
# .env 파일 생성
OPENAI_API_KEY=your_openai_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Chalice API URL 설정

`streamlit_app.py`에서 API URL 수정:

```python
# 로컬 개발용
CHALICE_API_URL = "http://localhost:8000"

# 배포용 (API Gateway URL로 변경)
# CHALICE_API_URL = "https://your-api-gateway-url"
```

## 🎨 UI 특징

### 🌈 현대적인 디자인

- 그라디언트 헤더
- 반응형 레이아웃
- 직관적인 아이콘 사용
- 색상 구분된 상태 표시

### 📱 반응형 웹

- 데스크톱/태블릿/모바일 지원
- 자동 크기 조정
- 터치 친화적 인터페이스

### 🎯 사용자 경험

- 실시간 피드백
- 진행 상태 표시
- 오류 메시지 안내
- 도움말 툴팁

## 🐛 문제 해결

### 자주 발생하는 문제

1. **API 연결 실패**:
   ```
   ❌ API 연결 실패
   ```
   → Chalice 서버가 실행 중인지 확인: `chalice local --port 8000`

2. **Redis 연결 오류**:
   ```
   Redis connection failed
   ```
   → Redis 서버 실행: `docker-compose up -d redis`

3. **패키지 설치 오류**:
   ```
   ModuleNotFoundError: No module named 'streamlit'
   ```
   → 의존성 재설치: `pip install -r requirements.txt`

4. **OpenAI API 키 오류**:
   ```
   OpenAI API key not configured
   ```
   → 환경 변수 설정: `export OPENAI_API_KEY=your_key`

### 로그 확인

```bash
# Chalice 로그
chalice logs

# Streamlit 로그
# 터미널에서 직접 확인 가능
```

## 🎮 데모 시나리오

### 시나리오 1: 기본 VRP 테스트

1. 차량 3대 설정 (각각 5톤 용량)
2. 주문 8개 설정 (서울 시내 분산)
3. 자동 시나리오 선택
4. 최적화 실행 → VRP 시나리오 자동 선택 확인

### 시나리오 2: 긴급 배송 테스트

1. 기존 VRP 설정에 긴급 주문 추가
2. 우선순위 'URGENT' 설정
3. 시나리오 '긴급 배송 처리' 선택
4. 최적화 실행 → 긴급 주문 우선 처리 확인

### 시나리오 3: 피드백 학습 테스트

1. 동일한 설정으로 최적화 실행
2. 만족도 5점 평가 및 피드백 제출
3. 같은 설정으로 재실행
4. 학습된 선호도 반영 확인

## 📊 성능 지표

- **응답 시간**: < 3초 (일반적인 최적화)
- **메모리 사용량**: < 100MB (세션당)
- **동시 사용자**: 최대 10명 (로컬 개발환경)
- **지원 브라우저**: Chrome, Firefox, Safari, Edge

## 🔮 향후 계획

- [ ] 실시간 차량 위치 추적
- [ ] 배송 상태 실시간 업데이트
- [ ] 모바일 앱 연동
- [ ] 다국어 지원
- [ ] 고급 분석 대시보드
- [ ] 배치 최적화 기능

---

**🎉 축하합니다! TMS AI 배차 시스템의 강력한 웹 인터페이스를 경험해보세요!** 