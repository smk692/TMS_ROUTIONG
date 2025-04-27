# TMS Routing Project Skeleton

## Directory Structure

```
TMS_ROUTING/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── optimizer.py        # 메인 TMS 최적화 로직
│   │   ├── route_planner.py    # 경로 계획
│   │   └── validator.py        # 경로 검증
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── delivery_point.py   # 배송지점 모델
│   │   ├── vehicle.py         # 차량 모델
│   │   └── route.py           # 경로 모델
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── area_divider.py    # 지역 분할
│   │   ├── time_optimizer.py   # 시간 최적화
│   │   ├── vehicle_optimizer.py # 차량 최적화
│   │   └── constraint_handler.py # 제약조건 처리
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── distance.py        # 거리 계산
│   │   ├── geo_utils.py       # 지리정보 유틸리티
│   │   └── time_utils.py      # 시간 관련 유틸리티
│   │
│   └── db/                    # 기존 DB 관련 코드
│
├── tests/                     # 테스트 코드
│   └── ...
│
├── notebooks/                 # 기존 노트북 파일들
│   └── ...
│
└── config/                    # 설정 파일
    ├── vehicle_types.json     # 차량 유형 정의
    └── constraints.json       # 제약조건 정의
```

## Setup

   pip install -e .
   
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter notebooks in the `notebooks/` directory for experiments.
