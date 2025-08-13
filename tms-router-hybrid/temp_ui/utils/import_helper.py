"""
Import helper utility for robust module loading
"""
import sys
import importlib.util
from pathlib import Path


def setup_imports():
    """Setup Python path and import required modules robustly"""
    
    # 현재 파일의 절대 경로 기반으로 프로젝트 구조 설정
    current_file = Path(__file__).resolve()
    streamlit_app_dir = current_file.parent.parent
    project_root = streamlit_app_dir.parent
    
    # Python 경로에 추가
    if str(streamlit_app_dir) not in sys.path:
        sys.path.insert(0, str(streamlit_app_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 모듈 임포트 시도
    modules = {}
    
    try:
        # 상대 경로로 시도
        from .data_formatter import format_time, create_history_dataframe
        from .session_state import init_session_state
        
        modules.update({
            'format_time': format_time,
            'create_history_dataframe': create_history_dataframe,
            'init_session_state': init_session_state
        })
        
    except ImportError:
        try:
            # 절대 경로로 시도
            from streamlit_app.utils.data_formatter import format_time, create_history_dataframe
            from streamlit_app.utils.session_state import init_session_state
            
            modules.update({
                'format_time': format_time,
                'create_history_dataframe': create_history_dataframe,
                'init_session_state': init_session_state
            })
            
        except ImportError:
            # 직접 파일 로드 방법 (최후의 수단)
            # data_formatter 직접 로드
            spec = importlib.util.spec_from_file_location(
                "data_formatter", 
                streamlit_app_dir / "utils" / "data_formatter.py"
            )
            data_formatter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_formatter)
            
            # session_state 직접 로드  
            spec = importlib.util.spec_from_file_location(
                "session_state",
                streamlit_app_dir / "utils" / "session_state.py" 
            )
            session_state = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(session_state)
            
            modules.update({
                'format_time': data_formatter.format_time,
                'create_history_dataframe': data_formatter.create_history_dataframe,
                'init_session_state': session_state.init_session_state,
                'update_dispatch_result': getattr(session_state, 'update_dispatch_result', None)
            })
    
    return modules


def get_page_setup_paths():
    """Get project paths for page setup"""
    current_file = Path(__file__).resolve()
    streamlit_app_dir = current_file.parent.parent
    project_root = streamlit_app_dir.parent
    
    return {
        'streamlit_app_dir': streamlit_app_dir,
        'project_root': project_root
    }