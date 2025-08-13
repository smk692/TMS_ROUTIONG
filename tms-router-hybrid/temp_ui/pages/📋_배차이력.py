"""
배차 이력 페이지
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
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.data_formatter import create_history_dataframe, format_time
from utils.session_state import init_session_state
import plotly.express as px


def main():
    """배차 이력 페이지 메인"""
    st.set_page_config(
        page_title="배차 이력 - TMS",
        page_icon="📋",
        layout="wide"
    )
    
    # 세션 상태 초기화
    init_session_state()
    
    # 배차 이력 페이지 표시
    show_history_page()


def show_history_page():
    """배차 이력 페이지 표시"""
    st.title("📋 배차 이력")
    st.markdown("---")
    
    # API 서비스 가져오기
    api_service = st.session_state.api_service
    
    # 필터 섹션
    with st.expander("🔍 필터 옵션", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 날짜 범위 필터
            date_range = st.date_input(
                "날짜 범위",
                value=(datetime.now() - timedelta(days=7), datetime.now()),
                max_value=datetime.now()
            )
        
        with col2:
            # 센터 필터
            centers = api_service.get_centers_list()
            center_options = ["전체"] + [c['name'] for c in centers]
            selected_center = st.selectbox("물류센터", center_options)
        
        with col3:
            # 상태 필터
            status_options = ["전체", "success", "partial_success", "failed", "cancelled"]
            selected_status = st.selectbox("상태", status_options)
        
        with col4:
            # 표시 개수
            limit_options = [10, 25, 50, 100, 200]
            selected_limit = st.selectbox("표시 개수", limit_options, index=2)
    
    # 데이터 조회
    try:
        history = api_service.get_dispatch_history(limit=selected_limit)
        
        if history:
            # 필터 적용
            filtered_history = apply_filters(
                history, 
                date_range, 
                selected_center, 
                selected_status,
                centers
            )
            
            if filtered_history:
                # 통계 표시
                display_history_statistics(filtered_history)
                
                # 차트 표시
                display_history_charts(filtered_history)
                
                # 테이블 표시
                display_history_table(filtered_history)
                
                # 다운로드 버튼
                provide_download_options(filtered_history)
            else:
                st.info("선택한 필터 조건에 맞는 배차 이력이 없습니다.")
        else:
            st.info("배차 이력이 없습니다.")
            
    except Exception as e:
        st.error(f"배차 이력을 불러올 수 없습니다: {str(e)}")


def apply_filters(history, date_range, selected_center, selected_status, centers):
    """필터 적용"""
    filtered = history.copy()
    
    # 날짜 필터
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + timedelta(days=1)
        filtered = [
            h for h in filtered 
            if h.get('created_at') and 
            start_date <= pd.to_datetime(h['created_at']) < end_date
        ]
    
    # 센터 필터
    if selected_center != "전체":
        center_id = next((c['center_id'] for c in centers if c['name'] == selected_center), None)
        if center_id:
            filtered = [h for h in filtered if h.get('center_id') == center_id]
    
    # 상태 필터
    if selected_status != "전체":
        filtered = [h for h in filtered if h.get('status') == selected_status]
    
    return filtered


def display_history_statistics(history):
    """이력 통계 표시"""
    st.markdown("### 📊 배차 이력 통계")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # 총 배차 수
    with col1:
        total_dispatches = len(history)
        st.metric("총 배차 수", f"{total_dispatches:,}건")
    
    # 성공률
    with col2:
        success_count = sum(1 for h in history if h.get('status') == 'success')
        success_rate = (success_count / total_dispatches * 100) if total_dispatches > 0 else 0
        st.metric("성공률", f"{success_rate:.1f}%", f"{success_count}건 성공")
    
    # 평균 처리 시간
    with col3:
        avg_time = sum(h.get('execution_time', 0) for h in history) / len(history) if history else 0
        st.metric("평균 처리 시간", format_time(avg_time))
    
    # 평균 배정률
    with col4:
        avg_assignment_rate = calculate_avg_assignment_rate(history)
        st.metric("평균 배정률", f"{avg_assignment_rate:.1f}%")


def calculate_avg_assignment_rate(history):
    """평균 배정률 계산"""
    total_rate = 0
    count = 0
    
    for h in history:
        total_orders = h.get('total_orders', 0)
        assigned_orders = h.get('assigned_orders', 0)
        if total_orders > 0:
            rate = (assigned_orders / total_orders) * 100
            total_rate += rate
            count += 1
    
    return total_rate / count if count > 0 else 0


def display_history_charts(history):
    """이력 차트 표시"""
    st.markdown("### 📈 배차 추이 분석")
    
    # DataFrame 생성
    df = pd.DataFrame(history)
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 시간별 배차 수 차트
        if 'created_at' in df.columns:
            # 일별 그룹화
            daily_counts = df.groupby(df['created_at'].dt.date).size().reset_index()
            daily_counts.columns = ['날짜', '배차 수']
            
            fig = px.line(
                daily_counts,
                x='날짜',
                y='배차 수',
                title='일별 배차 수 추이',
                markers=True
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                xaxis_title="날짜",
                yaxis_title="배차 수"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 상태별 분포 차트
        if 'status' in df.columns:
            status_counts = df['status'].value_counts().reset_index()
            status_counts.columns = ['상태', '건수']
            
            # 상태 한글 변환
            status_map = {
                'success': '성공',
                'partial_success': '부분 성공',
                'failed': '실패',
                'cancelled': '취소'
            }
            status_counts['상태'] = status_counts['상태'].map(status_map).fillna(status_counts['상태'])
            
            fig = px.pie(
                status_counts,
                values='건수',
                names='상태',
                title='상태별 배차 분포',
                color_discrete_map={
                    '성공': '#28a745',
                    '부분 성공': '#ffc107',
                    '실패': '#dc3545',
                    '취소': '#6c757d'
                }
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


def display_history_table(history):
    """이력 테이블 표시"""
    st.markdown("### 📋 배차 이력 상세")
    
    # DataFrame 생성
    df = create_history_dataframe(history)
    
    # 상태 한글 변환
    status_map = {
        'success': '✅ 성공',
        'partial_success': '⚠️ 부분 성공',
        'failed': '❌ 실패',
        'cancelled': '🚫 취소'
    }
    df['상태'] = df['상태'].map(status_map).fillna(df['상태'])
    
    # 인터랙티브 테이블 표시
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "배치 ID": st.column_config.TextColumn("배치 ID", width="medium"),
            "센터": st.column_config.TextColumn("센터", width="small"),
            "상태": st.column_config.TextColumn("상태", width="small"),
            "총 주문": st.column_config.NumberColumn("총 주문", format="%d개"),
            "배정 주문": st.column_config.NumberColumn("배정 주문", format="%d개"),
            "차량 수": st.column_config.TextColumn("차량 수", width="small"),
            "실행 시간": st.column_config.TextColumn("실행 시간", width="small"),
            "실행 일시": st.column_config.DatetimeColumn(
                "실행 일시",
                format="YYYY-MM-DD HH:mm:ss",
                width="medium"
            )
        }
    )
    
    # 페이지네이션 정보
    st.caption(f"총 {len(history)}개 항목 표시 중")


def provide_download_options(history):
    """다운로드 옵션 제공"""
    st.markdown("### 💾 데이터 내보내기")
    
    col1, col2, col3 = st.columns(3)
    
    # DataFrame 생성
    df = create_history_dataframe(history)
    
    with col1:
        # CSV 다운로드
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📄 CSV 다운로드",
            data=csv,
            file_name=f"dispatch_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel 다운로드
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='배차이력')
            
            st.download_button(
                label="📊 Excel 다운로드",
                data=buffer.getvalue(),
                file_name=f"dispatch_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.info("Excel 다운로드를 위해 openpyxl 패키지가 필요합니다.")
    
    with col3:
        # JSON 다운로드
        import json
        from utils.data_formatter import DecimalEncoder
        
        json_data = json.dumps(history, ensure_ascii=False, indent=2, cls=DecimalEncoder)
        st.download_button(
            label="🔧 JSON 다운로드",
            data=json_data,
            file_name=f"dispatch_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )


if __name__ == "__main__":
    main()