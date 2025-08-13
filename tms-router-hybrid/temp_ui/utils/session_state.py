"""
Streamlit 세션 상태 관리
"""
import streamlit as st
from datetime import datetime


def init_session_state():
    """세션 상태 초기화"""
    
    # 배차 결과 저장
    if 'last_dispatch_result' not in st.session_state:
        st.session_state.last_dispatch_result = None
    
    # 배차 이력 캐시
    if 'dispatch_history' not in st.session_state:
        st.session_state.dispatch_history = []
    
    # 선택된 센터
    if 'selected_center' not in st.session_state:
        st.session_state.selected_center = None
    
    # 선택된 메뉴 (페이지 네비게이션용)
    if 'selected_menu' not in st.session_state:
        st.session_state.selected_menu = "대시보드"
    
    # 마지막 업데이트 시간
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    # API 서비스 인스턴스 (실제 TMS 시스템)
    if 'api_service' not in st.session_state:
        try:
            # 프로젝트 루트를 Python 경로에 추가
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from web_api import SubprocessTmsApiService
            st.session_state.api_service = SubprocessTmsApiService()
            st.session_state.system_mode = "production"  # 실제 시스템 모드
        except ImportError as e:
            st.error(f"TMS 시스템 모듈을 가져올 수 없습니다: {str(e)}")
            st.error("가상환경이 활성화되어 있고 필요한 패키지가 설치되어 있는지 확인하세요.")
            st.stop()
        except Exception as e:
            st.error(f"TMS 시스템 초기화 중 오류: {str(e)}")
            st.error("Docker 컨테이너가 실행 중인지, 데이터베이스 연결이 가능한지 확인하세요.")
            st.stop()
    
    # 센터 목록 캐시
    if 'centers_list' not in st.session_state:
        st.session_state.centers_list = []
    
    # 통계 정보 캐시
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {}


def update_dispatch_result(result):
    """배차 결과 업데이트"""
    st.session_state.last_dispatch_result = result
    st.session_state.last_update = datetime.now()


def get_dispatch_result():
    """현재 배차 결과 가져오기"""
    return st.session_state.last_dispatch_result


def clear_dispatch_result():
    """배차 결과 초기화"""
    st.session_state.last_dispatch_result = None