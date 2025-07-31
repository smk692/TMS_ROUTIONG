# TMS Router AI

AI 기반 TMS(Transportation Management System) 배차 최적화 시스템

## 🎯 개요

이 시스템은 전통적인 알고리즘 구현 대신 **AI(GPT)에게 배차 최적화를 위임**하는 혁신적인 접근 방식을 사용합니다. 파라미터 기반 프롬프트 선택을 통해 다양한 TMS 시나리오에 최적화된 배차 계획을 제공합니다.

## 🏗️ 아키텍처

### Clean Architecture 원칙
```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│                   (API, Validators)                        │
├─────────────────────────────────────────────────────────────┤
│                     Use Cases Layer                        │
│              (Application Business Logic)                  │
├─────────────────────────────────────────────────────────────┤
│                    Interfaces Layer                        │
│                  (Abstract Contracts)                      │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                      │
│           (AI Services, Database, External APIs)           │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                          │
│                 (Business Entities & Rules)                │
└─────────────────────────────────────────────────────────────┘
```

### 기술 스택
- **Framework**: AWS Chalice (서버리스)
- **AI**: LangChain/LangGraph + OpenAI GPT
- **Database**: Redis (대화 메모리)
- **Language**: Python 3.11+
- **Architecture**: Clean Architecture + Clean Code
- **Development**: Docker Compose + Redis

## 🚀 시작하기

### 사전 요구사항
- Python 3.11+
- AWS CLI 설정
- OpenAI API Key

### 로컬 개발 환경 설정

#### 🚀 빠른 시작 (Docker Compose 사용)

1. **개발 환경 자동 설정**
```bash
chmod +x scripts/dev-setup.sh
./scripts/dev-setup.sh
```

2. **환경 변수 설정**
```bash
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정
```

3. **Docker Compose로 전체 스택 실행**
```bash
docker-compose up
```

#### 🔧 수동 설정

1. **Redis 시작**
```bash
docker-compose up -d redis
```

2. **의존성 설치**
```bash
pip install -r requirements.txt
```

3. **환경 변수 설정**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export REDIS_HOST="localhost"
export STORAGE_TYPE="redis"
```

4. **로컬 서버 실행**
```bash
chalice local

source venv/bin/activate && streamlit run streamlit_app.py --server.port 8501 --server.headless true.port 8501 --server.headless true
```

서버가 `http://localhost:8000`에서 실행됩니다.

#### 📊 Redis 관리 도구 (선택사항)
```bash
docker-compose --profile tools up redis-commander
# http://localhost:8081에서 Redis 데이터 확인
```

### API 테스트

#### 헬스 체크
```bash
curl http://localhost:8000/health
```

#### 배차 최적화 요청
```bash
curl -X POST http://localhost:8000/optimize-route \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_type": "vrp",
    "vehicles": [
      {
        "vehicle_id": "V001",
        "capacity": 5.0,
        "current_location": {"lat": 37.5665, "lng": 126.9780},
        "status": "AVAILABLE"
      }
    ],
    "orders": [
      {
        "order_id": "O001",
        "pickup_location": {"lat": 37.5547, "lng": 126.9706},
        "delivery_location": {"lat": 37.5172, "lng": 127.0473},
        "weight": 2.5,
        "priority": "HIGH"
      }
    ]
  }'
```

## 📁 프로젝트 구조

```
tms-router-ai/
├── app.py                      # Chalice 메인 애플리케이션
├── docker-compose.yml          # 로컬 개발 환경
├── Dockerfile.dev              # 개발용 Docker 이미지
├── redis.conf                  # Redis 설정
├── .env.example                # 환경 변수 예시
├── .chalice/                   # Chalice 설정
│   └── config.json
├── requirements.txt            # 의존성
├── scripts/                    # 개발 도구
│   ├── dev-setup.sh           # 개발 환경 설정
│   └── test-redis.py          # Redis 테스트
├── src/                        # Clean Architecture 소스
│   ├── domain/                 # 비즈니스 엔티티 (최내부)
│   ├── use_cases/              # 애플리케이션 로직
│   ├── interfaces/             # 추상화 계층
│   ├── infrastructure/         # 외부 시스템 구현체
│   │   ├── ai/                # AI 서비스 (LangChain/GPT)
│   │   └── memory/            # 메모리 저장소 (Redis/DynamoDB)
│   ├── presentation/           # API 처리
│   └── shared/                 # 공통 유틸리티
├── tests/                      # 테스트
└── .cursor/                    # Cursor IDE 규칙
    └── rules/
```

## 🎪 TMS 시나리오

### 지원되는 배차 시나리오
- **VRP (Vehicle Routing Problem)**: 다중 차량 경로 최적화
- **TSP (Traveling Salesman)**: 단일 차량 최적 경로
- **Load Consolidation**: 소량 주문 통합 배송
- **Emergency Dispatch**: 긴급 배송 처리
- **Real-time Adjustment**: 실시간 경로 수정

### AI 프롬프트 패턴
각 시나리오별로 최적화된 AI 프롬프트가 자동 선택되어 GPT에게 전달됩니다.

## 🔄 피드백 시스템

배차 결과에 대한 피드백을 통해 AI가 지속적으로 학습하고 개선됩니다:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "feedback_type": "IMPROVEMENT_REQUEST",
    "message": "다음에는 더 짧은 경로로 제안해주세요",
    "rating": 3
  }'
```

## 🚀 배포

### 개발 환경 배포
```bash
# 개발 스테이지 배포 (Redis → ElastiCache)
export STORAGE_TYPE=redis
chalice deploy --stage dev

# 배포된 API URL 확인
chalice url --stage dev
```

### 프로덕션 배포
```bash
# 프로덕션 배포 (ElastiCache Redis 사용)
export STORAGE_TYPE=redis
chalice deploy --stage prod

# 로그 확인
chalice logs --stage prod

# 배포 상태 확인
chalice status --stage prod
```

### 메모리 저장소 구성
- **로컬 개발**: Docker Redis
- **개발/테스트**: ElastiCache Redis
- **프로덕션**: ElastiCache Redis

## 📊 모니터링

### 로그 확인
```bash
chalice logs --stage dev
```

### 구조화된 로깅
모든 요청과 AI 상호작용이 구조화된 JSON 형태로 CloudWatch에 기록됩니다.

## 🧪 테스트

```bash
# Redis 연결 및 메모리 저장소 테스트
python scripts/test-redis.py

# 단위 테스트
pytest tests/unit/

# 통합 테스트  
pytest tests/integration/

# 로컬 API 테스트
python tests/test_local.py
```

### Docker 환경에서 테스트
```bash
# 컨테이너 내부에서 테스트 실행
docker-compose exec app python scripts/test-redis.py
docker-compose exec app python tests/test_local.py
```

## 📖 개발 가이드

### Clean Code 원칙
- 모든 함수는 최대 20줄 제한
- 매개변수는 4개 이하
- 타입 힌트 100% 적용
- Docstring 모든 public 메서드

### Clean Architecture 규칙
- Domain → 어떤 레이어에도 의존하지 않음
- Use Cases → Domain만 의존
- Infrastructure → Domain, Use Cases 의존 가능
- Presentation → Use Cases만 의존

## 📝 라이선스

MIT License

## 🤝 기여

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 