"""
TMS 대시보드 - 메인 페이지
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import subprocess
import os

# 유틸리티 모듈 임포트
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.import_helper import setup_imports
modules = setup_imports()

format_time = modules['format_time']
create_history_dataframe = modules['create_history_dataframe'] 
init_session_state = modules['init_session_state']


def main():
    """대시보드 메인 페이지"""
    # 페이지 설정
    st.set_page_config(
        page_title="TMS 배차 시스템",
        page_icon="🚛",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS 스타일 적용
    st.markdown("""
        <style>
        .main > div {
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.2);
            padding: 10px 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # 세션 상태 초기화
    init_session_state()
    
    # 사이드바 정보
    with st.sidebar:
        st.title("🚛 TMS 배차 시스템")
        
        # 실제 시스템 연동 표시
        system_mode = st.session_state.get('system_mode', 'unknown')
        if system_mode == 'production':
            st.success("💾 실제 데이터베이스 연동")
            st.success("🔄 실제 배차 엔진 사용")
        else:
            st.warning("⚠️ 시스템 상태 확인 필요")
        
        st.markdown("---")
        
        # 시스템 정보
        st.markdown("### 시스템 정보")
        st.info("""
        **버전**: 1.0.0  
        **상태**: 🟢 정상 운영 중
        """)
        
        # 빠른 통계
        if st.button("🔄 새로고침", use_container_width=True):
            st.rerun()
    
    # 대시보드 내용
    show_dashboard()


def show_dashboard():
    """대시보드 페이지 표시"""
    st.title("📊 TMS 대시보드")
    
    # Docker 재시작 버튼
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🐳 Docker 재시작", type="primary", use_container_width=True):
            restart_docker_services()
    
    st.markdown("---")
    
    # API 서비스 가져오기
    api_service = st.session_state.api_service
    
    # 자동 새로고침 옵션
    col1, col2 = st.columns([6, 1])
    with col2:
        auto_refresh = st.checkbox("자동 새로고침", value=False)
        if auto_refresh:
            st.rerun()
    
    # 통계 정보 가져오기
    try:
        stats = api_service.get_center_statistics()
        st.session_state.statistics = stats
    except Exception as e:
        st.error(f"통계 정보를 불러올 수 없습니다: {str(e)}")
        stats = st.session_state.get('statistics', {})
    
    # 주요 지표 표시
    display_key_metrics(stats)
    
    # 차트 섹션
    col1, col2 = st.columns(2)
    
    with col1:
        display_order_status_chart(stats)
    
    with col2:
        display_vehicle_status_chart(stats)
    
    # 센터별 통계
    display_center_statistics(api_service)
    
    # 최근 배차 이력
    display_recent_dispatch_history(api_service)
    
    # 시스템 상태
    display_system_status()


def display_key_metrics(stats):
    """주요 지표 표시"""
    st.markdown("### 🎯 주요 운영 지표")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_orders = stats.get('total_orders', 0)
        pending_orders = stats.get('pending_orders', 0)
        st.metric(
            label="총 주문 수",
            value=f"{total_orders:,}개",
            delta=f"{pending_orders}개 대기 중",
            delta_color="normal"
        )
    
    with col2:
        assigned_orders = stats.get('assigned_orders', 0)
        assignment_rate = (assigned_orders / total_orders * 100) if total_orders > 0 else 0
        st.metric(
            label="배정 완료",
            value=f"{assigned_orders:,}개",
            delta=f"{assignment_rate:.1f}% 배정률",
            delta_color="normal"
        )
    
    with col3:
        active_vehicles = stats.get('active_vehicles', 0)
        total_vehicles = stats.get('total_vehicles', 0)
        st.metric(
            label="활성 차량",
            value=f"{active_vehicles}대",
            delta=f"총 {total_vehicles}대 중",
            delta_color="normal"
        )
    
    with col4:
        completed_orders = stats.get('completed_orders', 0)
        completion_rate = (completed_orders / total_orders * 100) if total_orders > 0 else 0
        st.metric(
            label="완료된 배송",
            value=f"{completed_orders:,}개",
            delta=f"{completion_rate:.1f}% 완료율",
            delta_color="normal"
        )


def display_order_status_chart(stats):
    """주문 상태 차트"""
    st.markdown("#### 📦 주문 상태 분포")
    
    # 데이터 준비
    order_data = pd.DataFrame([
        {"상태": "대기 중", "수량": stats.get('pending_orders', 0), "색상": "#FFA500"},
        {"상태": "배정 완료", "수량": stats.get('assigned_orders', 0), "색상": "#1E90FF"},
        {"상태": "배송 완료", "수량": stats.get('completed_orders', 0), "색상": "#32CD32"}
    ])
    
    if order_data['수량'].sum() > 0:
        # 파이 차트 생성
        fig = px.pie(
            order_data,
            values='수량',
            names='상태',
            color='상태',
            color_discrete_map={
                "대기 중": "#FFA500",
                "배정 완료": "#1E90FF",
                "배송 완료": "#32CD32"
            },
            hole=0.4
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>수량: %{value}개<br>비율: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            showlegend=True,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("주문 데이터가 없습니다.")


def display_vehicle_status_chart(stats):
    """차량 상태 차트"""
    st.markdown("#### 🚛 차량 운영 현황")
    
    # 데이터 준비
    active_vehicles = stats.get('active_vehicles', 0)
    inactive_vehicles = stats.get('inactive_vehicles', 0)
    
    vehicle_data = pd.DataFrame([
        {"상태": "운행 가능", "수량": active_vehicles},
        {"상태": "운행 불가", "수량": inactive_vehicles}
    ])
    
    if vehicle_data['수량'].sum() > 0:
        # 막대 차트 생성
        fig = go.Figure(data=[
            go.Bar(
                x=vehicle_data['상태'],
                y=vehicle_data['수량'],
                text=vehicle_data['수량'],
                textposition='auto',
                marker_color=['#32CD32', '#DC143C']
            )
        ])
        
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_title="차량 수",
            xaxis_title=""
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("차량 데이터가 없습니다.")


def display_center_statistics(api_service):
    """센터별 통계"""
    st.markdown("### 🏢 물류센터별 현황")
    
    try:
        centers = api_service.get_centers_list()
        
        if centers:
            # 센터별 통계 데이터 수집
            center_stats = []
            for center in centers:
                stats = api_service.get_center_statistics(center['center_id'])
                center_stats.append({
                    '센터명': center['name'],
                    '총 주문': stats.get('total_orders', 0),
                    '대기 중': stats.get('pending_orders', 0),
                    '배정 완료': stats.get('assigned_orders', 0),
                    '배송 완료': stats.get('completed_orders', 0),
                    '활성 차량': stats.get('active_vehicles', 0),
                    '전체 차량': stats.get('total_vehicles', 0)
                })
            
            # DataFrame 생성
            df = pd.DataFrame(center_stats)
            
            # 스타일링된 테이블 표시
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "센터명": st.column_config.TextColumn("센터명", width="medium"),
                    "총 주문": st.column_config.NumberColumn("총 주문", format="%d개"),
                    "대기 중": st.column_config.NumberColumn("대기 중", format="%d개"),
                    "배정 완료": st.column_config.NumberColumn("배정 완료", format="%d개"),
                    "배송 완료": st.column_config.NumberColumn("배송 완료", format="%d개"),
                    "활성 차량": st.column_config.NumberColumn("활성 차량", format="%d대"),
                    "전체 차량": st.column_config.NumberColumn("전체 차량", format="%d대")
                }
            )
        else:
            st.info("센터 정보가 없습니다.")
            
    except Exception as e:
        st.error(f"센터 통계를 불러올 수 없습니다: {str(e)}")


def display_recent_dispatch_history(api_service):
    """최근 배차 이력"""
    st.markdown("### 📋 최근 배차 이력")
    
    try:
        history = api_service.get_dispatch_history(limit=10)
        
        if history:
            # DataFrame 생성
            df = create_history_dataframe(history)
            
            # 상태별 색상 매핑
            def highlight_status(row):
                colors = []
                for col in row.index:
                    if col == '상태':
                        if row[col] == 'success':
                            colors.append('background-color: #d4edda')
                        elif row[col] == 'failed':
                            colors.append('background-color: #f8d7da')
                        else:
                            colors.append('background-color: #fff3cd')
                    else:
                        colors.append('')
                return colors
            
            # 스타일 적용하여 표시
            styled_df = df.style.apply(highlight_status, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("배차 이력이 없습니다.")
            
    except Exception as e:
        st.error(f"배차 이력을 불러올 수 없습니다: {str(e)}")


def display_system_status():
    """시스템 상태"""
    st.markdown("### 🔧 시스템 상태")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #d4edda; border-radius: 10px;'>
            <h4 style='color: #155724;'>데이터베이스</h4>
            <p style='font-size: 24px; margin: 0;'>✅</p>
            <p style='color: #155724; margin: 0;'>정상</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #d4edda; border-radius: 10px;'>
            <h4 style='color: #155724;'>API 서버</h4>
            <p style='font-size: 24px; margin: 0;'>✅</p>
            <p style='color: #155724; margin: 0;'>정상</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #fff3cd; border-radius: 10px;'>
            <h4 style='color: #856404;'>캐시 시스템</h4>
            <p style='font-size: 24px; margin: 0;'>⚠️</p>
            <p style='color: #856404; margin: 0;'>점검 중</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #d1ecf1; border-radius: 10px;'>
            <h4 style='color: #0c5460;'>현재 시간</h4>
            <p style='font-size: 20px; margin: 0; color: #0c5460;'>{current_time}</p>
            <p style='color: #0c5460; margin: 0;'>KST</p>
        </div>
        """, unsafe_allow_html=True)


def restart_docker_services():
    """Docker 서비스 재시작"""
    try:
        # 현재 작업 디렉토리 확인
        current_dir = os.getcwd()
        st.info(f"🔍 현재 작업 디렉토리: {current_dir}")
        
        # 프로젝트 루트 디렉토리로 이동 (docker-compose.yml이 있는 위치)
        # temp_ui 디렉토리에서 상위로 이동하여 프로젝트 루트 찾기
        project_root = os.path.dirname(current_dir)
        st.info(f"🔍 프로젝트 루트 디렉토리: {project_root}")
        
        # docker-compose.yml 파일 존재 확인
        compose_file = os.path.join(project_root, "docker-compose.yml")
        if not os.path.exists(compose_file):
            st.error(f"❌ docker-compose.yml 파일을 찾을 수 없습니다: {compose_file}")
            return
        
        st.info(f"✅ docker-compose.yml 파일 발견: {compose_file}")
        
        st.info("🐳 Docker 서비스 재시작을 시작합니다...")
        
        # Docker Compose down -v 실행
        st.info("📥 Docker 컨테이너 중지 및 볼륨 제거 중...")
        result_down = subprocess.run(
            ["docker", "compose", "down", "-v"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result_down.returncode != 0:
            st.error(f"❌ Docker Compose down 실패: {result_down.stderr}")
            return
        
        st.success("✅ Docker 컨테이너 중지 완료")
        
        # Docker Compose up -d 실행
        st.info("📤 Docker 컨테이너 시작 중...")
        result_up = subprocess.run(
            ["docker", "compose", "up", "-d"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result_up.returncode != 0:
            st.error(f"❌ Docker Compose up 실패: {result_up.stderr}")
            return
        
        st.success("✅ Docker 서비스 재시작 완료!")
        st.info("🔄 페이지를 새로고침하여 변경사항을 확인하세요.")
        
        # 성공 메시지와 함께 새로고침 버튼 표시
        if st.button("🔄 페이지 새로고침", type="secondary"):
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Docker 재시작 중 오류 발생: {str(e)}")
        st.error("**문제 해결 방법:**")
        st.error("1. Docker가 실행 중인지 확인")
        st.error("2. docker-compose.yml 파일이 올바른 위치에 있는지 확인")
        st.error("3. Docker 권한이 있는지 확인")


if __name__ == "__main__":
    main()