#!/bin/bash

# TMS Router AI 테스트 실행 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 환경 확인
check_environment() {
    print_header "환경 확인"
    
    # Python 버전 확인
    python_version=$(python --version 2>&1)
    echo "Python Version: $python_version"
    
    # pytest 설치 확인
    if ! command -v pytest &> /dev/null; then
        print_error "pytest가 설치되지 않았습니다. pip install pytest를 실행하세요."
        exit 1
    fi
    
    # Redis 연결 확인
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping &> /dev/null; then
            print_success "Redis 연결 성공"
        else
            print_warning "Redis 서버가 실행되지 않고 있습니다. Redis 테스트를 건너뜁니다."
            export SKIP_REDIS_TESTS=true
        fi
    else
        print_warning "redis-cli를 찾을 수 없습니다. Redis 테스트를 건너뜁니다."
        export SKIP_REDIS_TESTS=true
    fi
    
    # 테스트 디렉토리 생성
    mkdir -p tests/reports/coverage
    mkdir -p tests/reports/junit
}

# 단위 테스트 실행
run_unit_tests() {
    print_header "단위 테스트 실행"
    
    if [ "$SKIP_REDIS_TESTS" = "true" ]; then
        pytest tests/unit/ -m "not requires_redis" -v \
            --junitxml=tests/reports/junit/unit_tests.xml \
            --cov-append
    else
        pytest tests/unit/ -v \
            --junitxml=tests/reports/junit/unit_tests.xml \
            --cov-append
    fi
    
    if [ $? -eq 0 ]; then
        print_success "단위 테스트 완료"
    else
        print_error "단위 테스트 실패"
        exit 1
    fi
}

# 통합 테스트 실행
run_integration_tests() {
    print_header "통합 테스트 실행"
    
    # API 서버 실행 확인
    if curl -s -f "http://localhost:8000/health" > /dev/null 2>&1; then
        print_success "API 서버 실행 중"
        pytest tests/integration/ -v \
            --junitxml=tests/reports/junit/integration_tests.xml \
            --cov-append
    else
        print_warning "API 서버가 실행되지 않고 있습니다. 통합 테스트를 건너뜁니다."
        print_warning "chalice local을 실행한 후 다시 시도하세요."
        return 0
    fi
    
    if [ $? -eq 0 ]; then
        print_success "통합 테스트 완료"
    else
        print_error "통합 테스트 실패"
        exit 1
    fi
}

# 성능 테스트 실행
run_performance_tests() {
    print_header "성능 테스트 실행"
    
    pytest tests/performance/ -v \
        --junitxml=tests/reports/junit/performance_tests.xml \
        --cov-append \
        --timeout=300
    
    if [ $? -eq 0 ]; then
        print_success "성능 테스트 완료"
    else
        print_warning "성능 테스트에서 일부 실패가 있었습니다."
    fi
}

# 커버리지 리포트 생성
generate_coverage_report() {
    print_header "커버리지 리포트 생성"
    
    pytest --cov-report=html:tests/reports/coverage/html \
           --cov-report=xml:tests/reports/coverage/coverage.xml \
           --cov-report=term \
           --cov=src \
           --cov-fail-under=0 \
           tests/ > /dev/null 2>&1
    
    print_success "커버리지 리포트가 tests/reports/coverage/에 생성되었습니다."
    echo "HTML 리포트: tests/reports/coverage/html/index.html"
}

# 테스트 결과 요약
show_summary() {
    print_header "테스트 결과 요약"
    
    echo "📊 테스트 리포트 위치:"
    echo "  - 단위 테스트: tests/reports/junit/unit_tests.xml"
    echo "  - 통합 테스트: tests/reports/junit/integration_tests.xml"
    echo "  - 성능 테스트: tests/reports/junit/performance_tests.xml"
    echo "  - 커버리지: tests/reports/coverage/html/index.html"
    
    echo ""
    echo "🔍 테스트 결과를 확인하려면:"
    echo "  python -m http.server 8080 --directory tests/reports/coverage/html"
    echo "  그 후 http://localhost:8080 에서 확인"
}

# 도움말
show_help() {
    echo "TMS Router AI 테스트 실행 스크립트"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  unit          단위 테스트만 실행"
    echo "  integration   통합 테스트만 실행"
    echo "  performance   성능 테스트만 실행"
    echo "  coverage      커버리지 리포트만 생성"
    echo "  all           모든 테스트 실행 (기본값)"
    echo "  help          이 도움말 표시"
    echo ""
    echo "환경 변수:"
    echo "  SKIP_REDIS_TESTS=true    Redis 테스트 건너뛰기"
    echo "  SKIP_INTEGRATION=true    통합 테스트 건너뛰기"
    echo "  SKIP_PERFORMANCE=true    성능 테스트 건너뛰기"
}

# 메인 실행 로직
main() {
    case "${1:-all}" in
        "unit")
            check_environment
            run_unit_tests
            ;;
        "integration")
            check_environment
            run_integration_tests
            ;;
        "performance")
            check_environment
            run_performance_tests
            ;;
        "coverage")
            generate_coverage_report
            ;;
        "all")
            check_environment
            run_unit_tests
            
            if [ "$SKIP_INTEGRATION" != "true" ]; then
                run_integration_tests
            fi
            
            if [ "$SKIP_PERFORMANCE" != "true" ]; then
                run_performance_tests
            fi
            
            generate_coverage_report
            show_summary
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
}

# 스크립트 실행
main "$@" 