#!/usr/bin/env python3
"""
TMS 웹 인터페이스 실행 스크립트 - 실제 TMS 시스템 및 데이터베이스 연동
"""
import subprocess
import sys
import os
import time
from pathlib import Path


def check_docker_containers():
    """Docker 컨테이너 상태 확인"""
    try:
        # Docker compose 프로세스 확인
        result = subprocess.run(['docker', 'compose', 'ps', '--services', '--filter', 'status=running'], 
                              capture_output=True, text=True, timeout=10)
        
        running_services = result.stdout.strip().split('\n') if result.stdout.strip() else []
        required_services = ['mysql', 'redis']
        
        missing_services = [svc for svc in required_services if svc not in running_services]
        
        if missing_services:
            print(f"❌ 필요한 Docker 서비스가 실행되지 않음: {', '.join(missing_services)}")
            return False, missing_services
        else:
            print("✅ Docker 컨테이너가 정상 실행 중입니다.")
            return True, []
            
    except subprocess.TimeoutExpired:
        print("❌ Docker 상태 확인 시간 초과")
        return False, ['timeout']
    except FileNotFoundError:
        print("❌ Docker가 설치되지 않았거나 실행되지 않습니다.")
        return False, ['docker_not_found']
    except Exception as e:
        print(f"❌ Docker 상태 확인 중 오류: {str(e)}")
        return False, ['error']


def check_database_connection():
    """데이터베이스 연결 확인"""
    try:
        # TMS 설정을 사용하여 데이터베이스 연결 테스트
        sys.path.append(str(Path(__file__).parent.parent))
        from core.config.settings import get_settings
        from core.database.connection import get_session
        
        settings = get_settings()
        session = get_session()
        
        # 간단한 쿼리로 연결 테스트
        from sqlalchemy import text
        session.execute(text("SELECT 1"))
        session.close()
        
        print("✅ 데이터베이스 연결이 정상입니다.")
        return True
        
    except ImportError as e:
        print(f"❌ TMS 모듈 import 실패: {str(e)}")
        print("💡 가상환경(venv)이 활성화되어 있는지 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {str(e)}")
        print("💡 Docker 컨테이너가 완전히 시작될 때까지 30-60초 기다려보세요.")
        return False


def start_docker_services():
    """Docker 서비스 시작"""
    print("🐳 Docker 서비스를 시작합니다...")
    try:
        result = subprocess.run(['docker', 'compose', 'up', '-d'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Docker 서비스 시작 완료!")
            print("⏳ 데이터베이스 초기화를 위해 30초 대기합니다...")
            time.sleep(30)
            return True
        else:
            print(f"❌ Docker 서비스 시작 실패: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker 서비스 시작 시간 초과")
        return False
    except Exception as e:
        print(f"❌ Docker 서비스 시작 중 오류: {str(e)}")
        return False


def check_dependencies():
    """웹 의존성 패키지 확인"""
    try:
        import streamlit
        import streamlit_folium
        import folium
        import plotly
        import streamlit_option_menu
        print("✅ 모든 웹 의존성이 설치되어 있습니다.")
        return True
    except ImportError as e:
        print(f"❌ 누락된 패키지: {e}")
        return False


def install_web_dependencies():
    """웹 의존성 패키지 설치"""
    print("📦 웹 의존성 패키지를 설치합니다...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements_web.txt"
        ])
        print("✅ 웹 의존성 패키지 설치 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False


def run_streamlit():
    """Streamlit 앱 실행"""
    app_path = Path("🏠_대시보드.py")
    
    if not app_path.exists():
        print(f"❌ Streamlit 앱을 찾을 수 없습니다: {app_path}")
        return False
    
    print("🚀 TMS 웹 인터페이스를 시작합니다...")
    print("📱 브라우저에서 http://localhost:8504 로 접속하세요")
    print("🛑 종료하려면 Ctrl+C를 누르세요")
    
    try:
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.port=8504",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 TMS 웹 인터페이스가 종료되었습니다.")
    except FileNotFoundError:
        print("❌ streamlit 명령을 찾을 수 없습니다. pip install streamlit을 실행해주세요.")
        return False
    
    return True


def main():
    """메인 실행 함수"""
    print("🚛 TMS 웹 인터페이스 시작 (실제 TMS 시스템 연동)")
    print("=" * 60)
    
    # 현재 디렉토리 확인 (temp_ui에서 실행되므로 core/main.py 확인)
    if not Path("../core/main.py").exists():
        print("❌ TMS 프로젝트의 temp_ui 디렉토리에서 실행해주세요.")
        return
    
    # 1. Docker 컨테이너 상태 확인
    print("🔍 1. Docker 컨테이너 상태 확인...")
    docker_ok, missing_services = check_docker_containers()
    
    if not docker_ok:
        if 'docker_not_found' in missing_services:
            print("❌ Docker가 필요합니다. Docker Desktop을 설치하고 실행해주세요.")
            return
        elif missing_services:
            print("🔧 Docker 서비스를 시작하시겠습니까? (y/n): ", end="")
            response = input().lower()
            if response in ['y', 'yes', '네', 'ㅇ']:
                if not start_docker_services():
                    print("❌ Docker 서비스 시작에 실패했습니다.")
                    return
            else:
                print("❌ Docker 서비스가 필요합니다. 'docker compose up -d' 명령어로 시작해주세요.")
                return
    
    # 2. 데이터베이스 연결 확인
    print("🔍 2. 데이터베이스 연결 확인...")
    if not check_database_connection():
        print("❌ 데이터베이스에 연결할 수 없습니다.")
        print("💡 다음 사항을 확인해보세요:")
        print("   - 가상환경이 활성화되어 있는가? (source venv/bin/activate)")
        print("   - Docker 컨테이너가 완전히 시작되었는가? (30-60초 대기)")
        print("   - TMS 의존성이 설치되어 있는가? (pip install -r requirements.txt)")
        return
    
    # 3. 웹 의존성 확인 및 설치
    print("🔍 3. 웹 의존성 패키지 확인...")
    if not check_dependencies():
        print("🔧 필요한 웹 패키지를 설치하시겠습니까? (y/n): ", end="")
        response = input().lower()
        if response in ['y', 'yes', '네', 'ㅇ']:
            if not install_web_dependencies():
                print("❌ 패키지 설치에 실패했습니다.")
                return
        else:
            print("❌ 웹 인터페이스를 실행하려면 필요한 패키지를 설치해야 합니다.")
            return
    
    # 4. Streamlit 앱 실행
    print("🚀 4. TMS 웹 인터페이스 시작...")
    print("💾 실제 데이터베이스에서 데이터를 가져옵니다.")
    print("🔄 실제 TMS 배차 스크립트를 실행합니다.")
    run_streamlit()


if __name__ == "__main__":
    main()