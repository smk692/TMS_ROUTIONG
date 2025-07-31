#!/bin/bash

# TMS Router AI 로컬 개발 환경 설정 스크립트

set -e

echo "🚀 TMS Router AI 로컬 개발 환경 설정을 시작합니다..."

# .env 파일 확인
if [ ! -f ".env" ]; then
    echo "📄 .env 파일을 생성합니다..."
    cp .env.example .env
    echo "⚠️  .env 파일을 편집하여 OPENAI_API_KEY를 설정해주세요!"
    echo "   nano .env"
    echo ""
fi

# Docker Compose로 Redis 시작
echo "🐳 Redis 컨테이너를 시작합니다..."
docker-compose up -d redis

# Redis 연결 대기
echo "⏳ Redis 연결을 확인합니다..."
for i in {1..30}; do
    if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis가 성공적으로 시작되었습니다!"
        break
    fi
    echo "   대기 중... ($i/30)"
    sleep 2
done

# Python 의존성 설치
echo "📦 Python 의존성을 설치합니다..."
pip install -r requirements.txt

# 로컬 서버 시작 옵션 안내
echo ""
echo "🎉 설정이 완료되었습니다!"
echo ""
echo "🔧 다음 중 하나를 선택하여 개발 서버를 시작하세요:"
echo ""
echo "1️⃣  Docker Compose로 실행 (권장):"
echo "   docker-compose up app"
echo ""
echo "2️⃣  로컬에서 직접 실행:"
echo "   source .env && chalice local"
echo ""
echo "3️⃣  Redis 관리 도구 사용:"
echo "   docker-compose --profile tools up redis-commander"
echo "   브라우저에서 http://localhost:8081 접속"
echo ""
echo "📚 API 문서: http://localhost:8000/health"
echo "🧪 테스트: python tests/test_local.py" 