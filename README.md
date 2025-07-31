# TMS (Transportation Management System) Projects

이 저장소는 운송 관리 시스템(TMS)과 관련된 여러 독립적인 프로젝트들을 포함합니다.

## 📁 프로젝트 구조

### 🚛 tms-legacy/
기존 TMS 라우팅 시스템
- **기술 스택**: Python, 기존 알고리즘 기반
- **주요 기능**: 전통적인 라우팅 알고리즘, 시각화, 모니터링
- **실행 방법**: `cd tms-legacy && python run_all.py`

### 🤖 tms-router-ai/
AI 기반 TMS 라우터 시스템
- **기술 스택**: Python, LangChain, OpenAI, Redis, Streamlit
- **주요 기능**: AI 기반 경로 최적화, 대화형 인터페이스, 메모리 기반 학습
- **실행 방법**: `cd tms-router-ai && python app.py`

## 🚀 시작하기

각 프로젝트는 독립적으로 실행됩니다. 원하는 프로젝트 디렉토리로 이동하여 해당 README를 참조하세요.

```bash
# 기존 TMS 시스템 실행
cd tms-legacy
python run_all.py

# AI 기반 라우터 실행
cd tms-router-ai
python app.py
```

## 📋 요구사항

각 프로젝트는 독립적인 의존성을 가지고 있습니다:
- `tms-legacy/requirements.txt`
- `tms-router-ai/requirements.txt`

## 📚 문서

- `TMS_ROUTING_시스템_구조_분석.md`: 기존 시스템 구조 분석 문서

## 🤝 기여하기

각 프로젝트는 독립적으로 관리됩니다. 기여하고 싶은 프로젝트의 디렉토리로 이동하여 해당 프로젝트의 가이드라인을 따르세요.

## 📄 라이센스

각 프로젝트별로 라이센스가 다를 수 있습니다. 해당 프로젝트 디렉토리를 확인하세요.
