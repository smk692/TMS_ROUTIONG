"""
배차 실행 페이지
"""
import sys
from pathlib import Path

# 프로젝트 루트와 streamlit_app 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
streamlit_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(streamlit_root) not in sys.path:
    sys.path.insert(0, str(streamlit_root))

import streamlit as st
import time
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.session_state import update_dispatch_result, init_session_state
from utils.data_formatter import (
    create_dispatch_dataframe, 
    format_time, 
    format_distance,
    format_percentage
)


def main():
    """배차 실행 페이지 메인"""
    st.set_page_config(
        page_title="배차 실행 - TMS",
        page_icon="🚚",
        layout="wide"
    )
    
    # 세션 상태 초기화
    init_session_state()
    
    # 배차 실행 페이지 표시
    show_dispatch_page()


def show_dispatch_page():
    """배차 실행 페이지 표시"""
    st.title("🚚 배차 실행")
    st.markdown("---")
    
    # API 서비스 가져오기
    api_service = st.session_state.api_service
    
    # 센터 목록 가져오기
    if not st.session_state.centers_list:
        with st.spinner("센터 정보 로딩 중..."):
            st.session_state.centers_list = api_service.get_centers_list()
    
    centers = st.session_state.centers_list
    
    if not centers:
        st.error("물류센터 정보를 불러올 수 없습니다.")
        return
    
    # 배차 설정 섹션
    st.markdown("### 📋 배차 설정")
    
    # 센터 선택을 폼 밖으로 이동하여 실시간 반응 가능
    col1, col2 = st.columns([1, 2])
    
    with col1:
        center_options = {f"{c['name']} ({c['center_id']})": c['center_id'] 
                        for c in centers}
        selected_center_display = st.selectbox(
            "물류센터 선택",
            options=list(center_options.keys()),
            help="배차를 실행할 물류센터를 선택하세요",
            key="center_selectbox"
        )
        selected_center_id = center_options[selected_center_display]
    
    with col2:
        st.empty()  # 빈 공간 유지
    
    # 선택된 센터의 주문 정보 표시 (센터 변경 시 즉시 업데이트)
    if selected_center_id:
        display_center_order_info(api_service, selected_center_id)
    
    # 배차 실행 설정 폼 (알고리즘 선택과 실행 버튼만)
    st.markdown("### 🚀 배차 실행 설정")
    
    with st.form("dispatch_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 알고리즘 선택
            algorithm = st.selectbox(
                "최적화 알고리즘",
                options=["auto", "nearest", "genetic", "annealing"],
                index=0,
                help="auto를 선택하면 주문량에 따라 자동으로 알고리즘이 선택됩니다"
            )
        
        with col2:
            # 실행 모드
            dry_run = st.checkbox(
                "시뮬레이션 모드",
                value=False,
                help="실제 배차를 실행하지 않고 시뮬레이션만 수행합니다"
            )
        
        with col3:
            st.empty()  # 빈 공간 유지
        
        # 실행 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button(
                "🚀 실제 배차 실행 (TMS 엔진)",
                use_container_width=True,
                type="primary"
            )
    
    # 배차 실행
    if submit_button:
        execute_dispatch(api_service, selected_center_id, algorithm, dry_run)
    
    # 최근 배차 결과 표시
    if st.session_state.last_dispatch_result:
        st.markdown("---")
        display_last_result()


def display_center_order_info(api_service, center_id):
    """선택된 센터의 주문 정보 표시"""
    st.markdown("### 📦 선택된 센터 주문 현황")
    
    try:
        # 센터별 통계 정보 조회
        stats = api_service.get_center_statistics(center_id)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pending_orders = stats.get('pending_orders', 0)
            st.metric(
                label="배차 가능 주문",
                value=f"{pending_orders:,}개",
                delta="대기 중",
                delta_color="normal",
                help="배차를 실행할 수 있는 대기 중인 주문 수"
            )
        
        with col2:
            assigned_orders = stats.get('assigned_orders', 0)
            st.metric(
                label="배차 완료 주문",
                value=f"{assigned_orders:,}개",
                delta="이미 배정됨",
                delta_color="normal",
                help="이미 배차가 완료된 주문 수"
            )
        
        with col3:
            total_orders = stats.get('total_orders', 0)
            assignment_rate = (assigned_orders / total_orders * 100) if total_orders > 0 else 0
            st.metric(
                label="총 주문 수",
                value=f"{total_orders:,}개",
                delta=f"{assignment_rate:.1f}% 배정률",
                delta_color="normal",
                help="해당 센터의 전체 주문 수"
            )
        
        with col4:
            active_vehicles = stats.get('active_vehicles', 0)
            st.metric(
                label="사용 가능 차량",
                value=f"{active_vehicles:,}대",
                delta="운행 가능",
                delta_color="normal",
                help="배차에 사용할 수 있는 활성 차량 수"
            )
        
        # 배차 가능 여부 표시
        if pending_orders > 0 and active_vehicles > 0:
            st.success(f"✅ 배차 실행 가능: {pending_orders}개 주문을 {active_vehicles}대 차량으로 배정할 수 있습니다.")
        elif pending_orders == 0:
            st.info("ℹ️ 배차할 대기 중인 주문이 없습니다.")
        elif active_vehicles == 0:
            st.warning("⚠️ 사용 가능한 차량이 없습니다.")
        else:
            st.error("❌ 배차를 실행할 수 없습니다.")
            
    except Exception as e:
        st.error(f"센터 주문 정보를 불러올 수 없습니다: {str(e)}")


def execute_dispatch(api_service, center_id, algorithm, dry_run):
    """배차 실행"""
    
    # 프로그레스 바 표시
    progress_bar = st.progress(0, text="배차 준비 중...")
    status_text = st.empty()
    
    try:
        # 단계별 진행 상황 표시
        progress_bar.progress(20, text="데이터 수집 중...")
        status_text.info("📊 주문 및 차량 데이터를 수집하고 있습니다...")
        time.sleep(0.5)
        
        progress_bar.progress(40, text="외부 조건 분석 중...")
        status_text.info("🌤️ 날씨 및 교통 정보를 분석하고 있습니다...")
        time.sleep(0.5)
        
        progress_bar.progress(60, text="최적화 알고리즘 실행 중...")
        status_text.info("🔄 최적 경로를 계산하고 있습니다...")
        
        # 실제 TMS 배차 실행 (데이터베이스 및 알고리즘 엔진 사용)
        result = api_service.execute_dispatch(center_id, algorithm)
        
        progress_bar.progress(80, text="결과 처리 중...")
        status_text.info("📝 배차 결과를 저장하고 있습니다...")
        time.sleep(0.5)
        
        progress_bar.progress(100, text="완료!")
        
        # 결과 저장
        update_dispatch_result(result)
        
        # 진행 상황 UI 제거
        progress_bar.empty()
        status_text.empty()
        
        # 결과 표시
        if result.is_successful():
            st.success(f"✅ 실제 TMS 배차가 성공적으로 완료되었습니다! (실행 시간: {format_time(result.execution_time)})")
            st.info("💾 배차 결과가 데이터베이스에 저장되었습니다.")
            display_dispatch_result(result)
        else:
            st.error(f"❌ 배차 실행 중 오류가 발생했습니다: {result.error_message}")
            
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ 배차 실행 중 오류가 발생했습니다: {str(e)}")


def display_dispatch_result(result):
    """배차 결과 표시"""
    
    # 주요 지표 표시
    st.markdown("### 📊 배차 결과 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "총 주문 수",
            f"{result.total_orders}개",
            f"{result.assigned_orders}개 배정"
        )
    
    with col2:
        assignment_rate = (result.assigned_orders / result.total_orders * 100) if result.total_orders > 0 else 0
        st.metric(
            "배정률",
            f"{assignment_rate:.1f}%",
            f"{result.assigned_orders - len(result.unassigned_orders)}개 성공"
        )
    
    with col3:
        st.metric(
            "사용 차량",
            f"{result.used_vehicles}대",
            f"총 {result.total_vehicles}대 중"
        )
    
    with col4:
        st.metric(
            "총 예상 거리",
            format_distance(result.total_distance),
            f"품질 점수: {result.quality_score:.2f}"
        )
    
    # 차량별 배정 현황
    if result.vehicle_assignments:
        st.markdown("### 🚛 차량별 배정 현황")
        
        # DataFrame 생성
        df = create_dispatch_dataframe(result.vehicle_assignments)
        
        # 인터랙티브 테이블 표시
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "차량 ID": st.column_config.TextColumn("차량 ID", width="small"),
                "기사명": st.column_config.TextColumn("기사명", width="small"),
                "차량 유형": st.column_config.TextColumn("차량 유형", width="small"),
                "권역": st.column_config.TextColumn("권역", width="medium"),
                "배정 주문 수": st.column_config.NumberColumn("주문 수", width="small"),
                "예상 거리": st.column_config.TextColumn("예상 거리", width="small"),
                "예상 시간": st.column_config.TextColumn("예상 시간", width="small"),
                "용량 활용도": st.column_config.ProgressColumn(
                    "용량 활용도",
                    help="차량 용량 대비 활용도",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                    width="small"
                )
            }
        )
    
    # 미배정 주문 표시
    if result.unassigned_orders:
        st.markdown("### ⚠️ 미배정 주문")
        st.warning(f"{len(result.unassigned_orders)}개의 주문이 배정되지 않았습니다.")
        
        with st.expander("미배정 주문 상세 보기"):
            for order in result.unassigned_orders[:10]:  # 최대 10개만 표시
                st.text(f"• {order.order_id}: {order.address}")
            if len(result.unassigned_orders) > 10:
                st.text(f"... 외 {len(result.unassigned_orders) - 10}개")
    
    # 경고 메시지 표시
    if result.warnings:
        st.markdown("### ⚠️ 경고 사항")
        for warning in result.warnings:
            st.warning(warning)
    
    # 액션 버튼
    st.markdown("### 🎯 다음 작업")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗺️ 지도에서 보기", use_container_width=True, key="go_to_map"):
            st.switch_page("pages/🗺️_지도보기.py")
    
    with col2:
        if st.button("📥 결과 다운로드", use_container_width=True):
            # CSV 다운로드 기능
            df = create_dispatch_dataframe(result.vehicle_assignments)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="CSV 다운로드",
                data=csv,
                file_name=f"dispatch_result_{result.batch_id}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔄 새 배차 실행", use_container_width=True):
            st.rerun()


def display_last_result():
    """최근 배차 결과 간단 표시"""
    result = st.session_state.last_dispatch_result
    
    st.markdown("### 📌 최근 배차 결과")
    
    # 정보 박스로 표시
    info_text = f"""
    **배치 ID**: {result.batch_id}  
    **실행 시간**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
    **상태**: {'✅ 성공' if result.is_successful() else '❌ 실패'}  
    **배정 현황**: {result.assigned_orders}/{result.total_orders}개 주문, {result.used_vehicles}/{result.total_vehicles}대 차량  
    **총 거리**: {format_distance(result.total_distance)}
    """
    
    if result.is_successful():
        st.info(info_text)
    else:
        st.error(info_text)


if __name__ == "__main__":
    main()