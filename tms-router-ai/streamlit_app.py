"""
TMS AI 배차 시스템 - Streamlit Web Interface

사용자가 쉽게 배차 최적화를 요청하고 결과를 확인할 수 있는 웹 인터페이스입니다.
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid

# 페이지 설정
st.set_page_config(
    page_title="TMS AI 배차 시스템",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 전역 설정
# API URL 설정 (환경변수 우선, 로컬 개발 시 localhost)
CHALICE_API_URL = os.getenv("CHALICE_API_URL", "http://localhost:8000")

# CSS 스타일링
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚚 TMS AI 배차 시스템</h1>
        <p>인공지능 기반 차량 경로 최적화 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 시스템 설정")
        
        # API 연결 상태 확인
        api_status = check_api_connection()
        if api_status:
            st.success("✅ API 연결 성공")
        else:
            st.error("❌ API 연결 실패")
            st.info("Chalice 서버를 시작해주세요: `chalice local`")
        
        st.divider()
        
        # 시나리오 선택
        st.subheader("📋 시나리오 선택")
        scenario_type = st.selectbox(
            "최적화 시나리오",
            ["auto", "vrp", "tsp", "load_consolidation", "emergency_dispatch", "realtime_adjustment"],
            format_func=lambda x: {
                "auto": "🤖 자동 선택 (AI 추천)",
                "vrp": "🚛 다중 차량 경로 최적화 (VRP)",
                "tsp": "🚐 단일 차량 최적 경로 (TSP)",
                "load_consolidation": "📦 적재 통합 최적화",
                "emergency_dispatch": "🚨 긴급 배송 처리",
                "realtime_adjustment": "⚡ 실시간 경로 조정"
            }[x]
        )
        
        st.divider()
        
        # 고급 옵션
        st.subheader("🔧 고급 옵션")
        use_feedback = st.checkbox("피드백 학습 활용", value=True)
        conversation_id = st.text_input("대화 ID (선택사항)", value="")
        
        if st.button("🗑️ 세션 초기화"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # 메인 콘텐츠 영역
    tab1, tab2, tab3, tab4 = st.tabs(["📝 배차 요청", "📊 결과 분석", "💾 이력 관리", "🔍 시스템 모니터링"])
    
    with tab1:
        optimization_request_tab(scenario_type, use_feedback, conversation_id, api_status)
    
    with tab2:
        results_analysis_tab()
    
    with tab3:
        history_management_tab()
    
    with tab4:
        system_monitoring_tab()


def check_api_connection() -> bool:
    """API 연결 상태 확인"""
    try:
        response = requests.get(f"{CHALICE_API_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False


def optimization_request_tab(scenario_type: str, use_feedback: bool, conversation_id: str, api_connected: bool):
    """배차 요청 탭"""
    
    st.header("🚛 배차 최적화 요청")
    
    if not api_connected:
        st.warning("⚠️ API 서버에 연결할 수 없습니다. Chalice 서버를 시작해주세요.")
        return
    
    # 입력 폼
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🚐 차량 정보")
        
        # 차량 수 선택
        num_vehicles = st.number_input("차량 수", min_value=1, max_value=20, value=3)
        
        # 차량 정보 입력
        vehicles = []
        for i in range(num_vehicles):
            with st.expander(f"차량 {i+1} 정보"):
                vehicle_id = st.text_input(f"차량 ID", value=f"V{i+1:03d}", key=f"vehicle_id_{i}")
                capacity = st.number_input(f"적재 용량 (톤)", min_value=0.1, max_value=50.0, value=5.0, step=0.1, key=f"capacity_{i}")
                
                col_lat, col_lng = st.columns(2)
                with col_lat:
                    start_lat = st.number_input(f"시작 위도", value=37.5665, format="%.4f", key=f"start_lat_{i}")
                with col_lng:
                    start_lng = st.number_input(f"시작 경도", value=126.9780, format="%.4f", key=f"start_lng_{i}")
                
                # 특수 능력
                special_capabilities = st.multiselect(
                    f"특수 능력", 
                    ["냉장", "냉동", "위험물", "대형화물", "급송"],
                    key=f"capabilities_{i}"
                )
                
                vehicles.append({
                    "id": vehicle_id,
                    "capacity_tons": capacity,
                    "start_location": {"lat": start_lat, "lng": start_lng},
                    "special_capabilities": special_capabilities
                })
    
    with col2:
        st.subheader("📦 주문 정보")
        
        # 주문 수 선택
        num_orders = st.number_input("주문 수", min_value=1, max_value=50, value=5)
        
        # 주문 정보 입력
        orders = []
        for i in range(num_orders):
            with st.expander(f"주문 {i+1} 정보"):
                order_id = st.text_input(f"주문 ID", value=f"O{i+1:03d}", key=f"order_id_{i}")
                weight = st.number_input(f"중량 (톤)", min_value=0.01, max_value=30.0, value=1.0, step=0.01, key=f"weight_{i}")
                priority = st.selectbox(f"우선순위", ["LOW", "MEDIUM", "HIGH", "URGENT"], index=1, key=f"priority_{i}")
                
                # 픽업 위치
                st.write("📍 픽업 위치")
                col_p_lat, col_p_lng = st.columns(2)
                with col_p_lat:
                    pickup_lat = st.number_input(f"픽업 위도", value=37.5665 + i*0.01, format="%.4f", key=f"pickup_lat_{i}")
                with col_p_lng:
                    pickup_lng = st.number_input(f"픽업 경도", value=126.9780 + i*0.01, format="%.4f", key=f"pickup_lng_{i}")
                
                # 배송 위치
                st.write("🎯 배송 위치")
                col_d_lat, col_d_lng = st.columns(2)
                with col_d_lat:
                    delivery_lat = st.number_input(f"배송 위도", value=37.6665 + i*0.01, format="%.4f", key=f"delivery_lat_{i}")
                with col_d_lng:
                    delivery_lng = st.number_input(f"배송 경도", value=127.0780 + i*0.01, format="%.4f", key=f"delivery_lng_{i}")
                
                # 시간 창 (선택사항)
                use_time_window = st.checkbox(f"시간 창 설정", key=f"use_time_window_{i}")
                time_window = None
                if use_time_window:
                    col_start, col_end = st.columns(2)
                    with col_start:
                        start_time = st.time_input(f"시작 시간", key=f"start_time_{i}")
                    with col_end:
                        end_time = st.time_input(f"종료 시간", key=f"end_time_{i}")
                    
                    time_window = {
                        "start": f"2024-01-01T{start_time}:00",
                        "end": f"2024-01-01T{end_time}:00"
                    }
                
                orders.append({
                    "id": order_id,
                    "weight_tons": weight,
                    "priority": priority,
                    "pickup_location": {"lat": pickup_lat, "lng": pickup_lng},
                    "delivery_location": {"lat": delivery_lat, "lng": delivery_lng},
                    "time_window": time_window
                })
    
    # 제약 조건
    st.subheader("⚙️ 제약 조건")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_working_hours = st.number_input("최대 근무 시간", min_value=1, max_value=24, value=8)
    with col2:
        max_distance_km = st.number_input("최대 이동 거리 (km)", min_value=10, max_value=1000, value=200)
    with col3:
        fuel_cost_per_km = st.number_input("연료비 (원/km)", min_value=100, max_value=2000, value=500)
    
    constraints = {
        "max_working_hours": max_working_hours,
        "max_distance_km": max_distance_km,
        "fuel_cost_per_km": fuel_cost_per_km
    }
    
    # 최적화 실행
    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🚀 배차 최적화 실행", type="primary", use_container_width=True):
            run_optimization(vehicles, orders, constraints, scenario_type, use_feedback, conversation_id)
    
    with col2:
        if st.button("💾 현재 설정 저장", use_container_width=True):
            save_current_settings(vehicles, orders, constraints)
    
    with col3:
        if st.button("📂 설정 불러오기", use_container_width=True):
            load_saved_settings()


def run_optimization(vehicles: List[Dict], orders: List[Dict], constraints: Dict, 
                    scenario_type: str, use_feedback: bool, conversation_id: str):
    """배차 최적화 실행"""
    
    # 데이터 변환 (Streamlit → API 형식)
    converted_vehicles = []
    for vehicle in vehicles:
        converted_vehicle = {
            "vehicle_id": vehicle["id"],  # id → vehicle_id
            "capacity_tons": vehicle["capacity_tons"],
            "current_location": vehicle["start_location"],  # start_location → current_location
            "special_capabilities": vehicle["special_capabilities"]
        }
        converted_vehicles.append(converted_vehicle)
    
    converted_orders = []
    for order in orders:
        # time_window 처리
        time_window = order.get("time_window")
        if time_window is None:
            # 기본 시간 창 설정 (전체 운영 시간)
            time_window = {
                "start": "2024-01-01T08:00:00",
                "end": "2024-01-01T18:00:00"
            }
        
        converted_order = {
            "order_id": order["id"],  # id → order_id
            "weight_tons": order["weight_tons"],
            "priority": order["priority"],
            "pickup_location": order["pickup_location"],
            "delivery_location": order["delivery_location"],
            "time_window": time_window
        }
        converted_orders.append(converted_order)
    
    # 요청 데이터 구성
    request_data = {
        "vehicles": converted_vehicles,
        "orders": converted_orders,
        "constraints": constraints
    }
    
    if conversation_id:
        request_data["conversation_id"] = conversation_id
    
    # API 호출
    with st.spinner("🤖 AI가 최적화를 계산하고 있습니다..."):
        try:
            response = requests.post(
                f"{CHALICE_API_URL}/optimize-route",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 결과 저장 (세션 상태)
                st.session_state['latest_result'] = result
                st.session_state['latest_request'] = request_data
                st.session_state['optimization_history'] = st.session_state.get('optimization_history', [])
                st.session_state['optimization_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'request': request_data,
                    'result': result
                })
                
                # 성공 메시지
                st.markdown("""
                <div class="success-box">
                    <h4>✅ 최적화 완료!</h4>
                    <p>결과를 확인하려면 '📊 결과 분석' 탭을 클릭하세요.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 간단한 요약 표시
                if result.get('success') and 'solution' in result:
                    solution = result['solution']
                    summary = solution.get('summary', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("사용 차량 수", summary.get('total_vehicles_used', 0))
                    with col2:
                        st.metric("배정 주문 수", summary.get('total_orders_assigned', 0))
                    with col3:
                        st.metric("총 거리 (km)", f"{summary.get('total_distance_km', 0):.1f}")
                    with col4:
                        st.metric("예상 비용 (원)", f"{summary.get('total_cost', 0):,}")
                
            else:
                st.error(f"❌ 최적화 실패: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("❌ 요청 시간 초과. 다시 시도해주세요.")
        except requests.exceptions.ConnectionError:
            st.error("❌ API 서버에 연결할 수 없습니다.")
        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")


def results_analysis_tab():
    """결과 분석 탭"""
    
    st.header("📊 최적화 결과 분석")
    
    if 'latest_result' not in st.session_state:
        st.info("아직 최적화 결과가 없습니다. '📝 배차 요청' 탭에서 최적화를 실행해주세요.")
        return
    
    result = st.session_state['latest_result']
    request_data = st.session_state.get('latest_request', {})
    
    if not result.get('success'):
        st.error("❌ 최적화가 실패했습니다.")
        st.json(result)
        return
    
    solution = result.get('solution', {})
    routes = solution.get('routes', [])
    summary = solution.get('summary', {})
    
    # 전체 요약
    st.subheader("📈 전체 요약")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("사용 차량", f"{summary.get('total_vehicles_used', 0)}대")
    with col2:
        st.metric("배정 주문", f"{summary.get('total_orders_assigned', 0)}건")
    with col3:
        st.metric("총 거리", f"{summary.get('total_distance_km', 0):.1f}km")
    with col4:
        st.metric("총 비용", f"{summary.get('total_cost', 0):,}원")
    with col5:
        st.metric("평균 효율성", f"{summary.get('average_efficiency', 0):.1f}%")
    
    # 경로별 상세 정보
    st.subheader("🗺️ 경로별 상세 정보")
    
    for i, route in enumerate(routes):
        with st.expander(f"🚛 {route.get('vehicle_id', f'차량 {i+1}')} 경로"):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 경로 정보
                st.write("**경로 정보:**")
                route_info = {
                    "총 거리": f"{route.get('total_distance_km', 0):.1f} km",
                    "총 시간": f"{route.get('total_duration_hours', 0):.1f} 시간",
                    "예상 비용": f"{route.get('estimated_cost', 0):,} 원",
                    "효율성 점수": f"{route.get('efficiency_score', 0):.1f}%",
                    "배정 주문": ", ".join(route.get('orders', []))
                }
                
                for key, value in route_info.items():
                    st.write(f"- {key}: {value}")
            
            with col2:
                # 경로 통계
                waypoints = route.get('waypoints', [])
                if waypoints:
                    waypoint_types = [wp.get('type', 'unknown') for wp in waypoints]
                    type_counts = pd.Series(waypoint_types).value_counts()
                    
                    fig = px.pie(
                        values=type_counts.values,
                        names=type_counts.index,
                        title="경유지 유형 분포"
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
            
            # 경유지 상세
            if waypoints:
                st.write("**경유지 상세:**")
                waypoint_df = pd.DataFrame([
                    {
                        "순서": i + 1,
                        "유형": wp.get('type', 'unknown'),
                        "주문 ID": wp.get('order_id', '-'),
                        "위도": wp.get('location', {}).get('lat', 0),
                        "경도": wp.get('location', {}).get('lng', 0),
                        "예상 도착": wp.get('estimated_arrival', '-'),
                        "소요 시간": f"{wp.get('estimated_duration_minutes', 0)}분"
                    }
                    for i, wp in enumerate(waypoints)
                ])
                st.dataframe(waypoint_df, use_container_width=True)
    
    # 지도 시각화
    st.subheader("🗺️ 경로 지도")
    
    if routes and all(route.get('waypoints') for route in routes):
        map_fig = create_route_map(routes, request_data.get('vehicles', []))
        st.plotly_chart(map_fig, use_container_width=True)
    
    # 최적화 통계
    st.subheader("📊 최적화 통계")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 차량별 효율성
        if routes:
            vehicle_efficiency = [(route.get('vehicle_id', f'차량{i}'), route.get('efficiency_score', 0)) 
                                for i, route in enumerate(routes)]
            
            df_efficiency = pd.DataFrame(vehicle_efficiency, columns=['차량', '효율성'])
            fig_efficiency = px.bar(df_efficiency, x='차량', y='효율성', title="차량별 효율성 점수")
            st.plotly_chart(fig_efficiency, use_container_width=True)
    
    with col2:
        # 비용 분석
        if routes:
            cost_data = [(route.get('vehicle_id', f'차량{i}'), route.get('estimated_cost', 0)) 
                        for i, route in enumerate(routes)]
            
            df_cost = pd.DataFrame(cost_data, columns=['차량', '비용'])
            fig_cost = px.bar(df_cost, x='차량', y='비용', title="차량별 예상 비용")
            st.plotly_chart(fig_cost, use_container_width=True)
    
    # 피드백 입력
    st.subheader("💬 결과 피드백")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        feedback_text = st.text_area("피드백 내용 (선택사항)", placeholder="이 결과에 대한 의견을 남겨주세요...")
    
    with col2:
        satisfaction_score = st.selectbox("만족도", [1, 2, 3, 4, 5], index=4, format_func=lambda x: f"{x}점 {'★' * x}")
    
    with col3:
        if st.button("📤 피드백 전송", type="primary", use_container_width=True):
            send_feedback(feedback_text, satisfaction_score, result)


def create_route_map(routes: List[Dict], vehicles: List[Dict]) -> go.Figure:
    """경로 지도 생성"""
    
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan']
    
    for i, route in enumerate(routes):
        waypoints = route.get('waypoints', [])
        if not waypoints:
            continue
        
        color = colors[i % len(colors)]
        vehicle_id = route.get('vehicle_id', f'차량{i+1}')
        
        # 경로 라인
        lats = [wp.get('location', {}).get('lat', 0) for wp in waypoints]
        lngs = [wp.get('location', {}).get('lng', 0) for wp in waypoints]
        
        fig.add_trace(go.Scatter(
            x=lngs,
            y=lats,
            mode='lines+markers',
            name=f'{vehicle_id} 경로',
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color)
        ))
        
        # 경유지 표시
        for j, wp in enumerate(waypoints):
            lat = wp.get('location', {}).get('lat', 0)
            lng = wp.get('location', {}).get('lng', 0)
            wp_type = wp.get('type', 'unknown')
            order_id = wp.get('order_id', '')
            
            symbol = {
                'start': 'star',
                'pickup': 'circle',
                'delivery': 'square',
                'end': 'star'
            }.get(wp_type, 'circle')
            
            fig.add_trace(go.Scatter(
                x=[lng],
                y=[lat],
                mode='markers',
                name=f'{vehicle_id} - {wp_type}',
                marker=dict(
                    size=12,
                    color=color,
                    symbol=symbol,
                    line=dict(width=2, color='white')
                ),
                text=f'{vehicle_id}<br>{wp_type}<br>{order_id}',
                hovertemplate='<b>%{text}</b><br>위도: %{y}<br>경도: %{x}<extra></extra>',
                showlegend=False
            ))
    
    fig.update_layout(
        title="배차 경로 지도",
        xaxis_title="경도",
        yaxis_title="위도",
        hovermode='closest',
        showlegend=True
    )
    
    return fig


def send_feedback(feedback_text: str, satisfaction_score: int, result: Dict):
    """피드백 전송"""
    
    feedback_data = {
        "feedback_text": feedback_text,
        "satisfaction_score": satisfaction_score,
        "optimization_result": result,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(
            f"{CHALICE_API_URL}/feedback",
            json=feedback_data,
            timeout=10
        )
        
        if response.status_code == 200:
            st.success("✅ 피드백이 성공적으로 전송되었습니다!")
        else:
            st.error(f"❌ 피드백 전송 실패: {response.text}")
            
    except Exception as e:
        st.error(f"❌ 피드백 전송 중 오류 발생: {str(e)}")


def save_current_settings(vehicles: List[Dict], orders: List[Dict], constraints: Dict):
    """현재 설정 저장"""
    
    settings = {
        "vehicles": vehicles,
        "orders": orders,
        "constraints": constraints,
        "saved_at": datetime.now().isoformat()
    }
    
    st.session_state['saved_settings'] = st.session_state.get('saved_settings', [])
    st.session_state['saved_settings'].append(settings)
    
    st.success("✅ 현재 설정이 저장되었습니다!")


def load_saved_settings():
    """저장된 설정 불러오기"""
    
    saved_settings = st.session_state.get('saved_settings', [])
    
    if not saved_settings:
        st.info("저장된 설정이 없습니다.")
        return
    
    st.selectbox(
        "저장된 설정 선택",
        range(len(saved_settings)),
        format_func=lambda i: f"설정 {i+1} - {saved_settings[i]['saved_at'][:19]}"
    )


def history_management_tab():
    """이력 관리 탭"""
    
    st.header("💾 최적화 이력 관리")
    
    history = st.session_state.get('optimization_history', [])
    
    if not history:
        st.info("아직 최적화 이력이 없습니다.")
        return
    
    # 이력 요약
    st.subheader("📈 이력 요약")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 실행 횟수", len(history))
    with col2:
        if history:
            latest_time = datetime.fromisoformat(history[-1]['timestamp'])
            st.metric("마지막 실행", latest_time.strftime("%m/%d %H:%M"))
    with col3:
        success_count = sum(1 for h in history if h.get('result', {}).get('success', False))
        st.metric("성공률", f"{success_count/len(history)*100:.1f}%")
    
    # 이력 목록
    st.subheader("📋 이력 목록")
    
    for i, record in enumerate(reversed(history)):
        timestamp = datetime.fromisoformat(record['timestamp'])
        result = record.get('result', {})
        request_data = record.get('request', {})
        
        with st.expander(f"실행 {len(history)-i}: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**요청 정보:**")
                st.write(f"- 차량 수: {len(request_data.get('vehicles', []))}대")
                st.write(f"- 주문 수: {len(request_data.get('orders', []))}건")
                st.write(f"- 시나리오: {request_data.get('scenario_type', 'auto')}")
                
                if result.get('success'):
                    st.success("✅ 성공")
                else:
                    st.error("❌ 실패")
            
            with col2:
                if result.get('success') and 'solution' in result:
                    summary = result['solution'].get('summary', {})
                    st.write("**결과 요약:**")
                    st.write(f"- 사용 차량: {summary.get('total_vehicles_used', 0)}대")
                    st.write(f"- 총 거리: {summary.get('total_distance_km', 0):.1f}km")
                    st.write(f"- 총 비용: {summary.get('total_cost', 0):,}원")
                    st.write(f"- 평균 효율성: {summary.get('average_efficiency', 0):.1f}%")
            
            # 재실행 버튼
            if st.button(f"🔄 다시 실행", key=f"rerun_{i}"):
                st.session_state['rerun_request'] = request_data
                st.info("'📝 배차 요청' 탭에서 설정이 로드되었습니다.")


def system_monitoring_tab():
    """시스템 모니터링 탭"""
    
    st.header("🔍 시스템 모니터링")
    
    # API 상태 모니터링
    st.subheader("🌐 API 서버 상태")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 상태 새로고침"):
            st.rerun()
        
        # API 연결 테스트
        api_status = check_api_connection()
        if api_status:
            st.success("✅ API 서버 정상")
        else:
            st.error("❌ API 서버 연결 불가")
    
    with col2:
        # 응답 시간 테스트
        if api_status:
            start_time = datetime.now()
            try:
                response = requests.get(f"{CHALICE_API_URL}/health", timeout=10)
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds() * 1000
                st.metric("응답 시간", f"{response_time:.0f}ms")
            except:
                st.metric("응답 시간", "타임아웃")
    
    # 패턴 매칭 분석
    st.subheader("🧠 AI 패턴 매칭 분석")
    
    try:
        response = requests.get(f"{CHALICE_API_URL}/analytics", timeout=10)
        if response.status_code == 200:
            analytics = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**시나리오 선택 분포:**")
                scenario_stats = analytics.get('scenario_selection_stats', {})
                if scenario_stats:
                    scenario_df = pd.DataFrame([
                        {"시나리오": k.replace('_selection_rate', ''), "비율": v} 
                        for k, v in scenario_stats.items() 
                        if k.endswith('_selection_rate')
                    ])
                    fig = px.pie(scenario_df, values='비율', names='시나리오', title="시나리오 선택 분포")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**효과성 트렌드:**")
                trend = analytics.get('effectiveness_trend', {})
                st.metric("현재 평균 효과성", f"{trend.get('current_avg_effectiveness', 0):.1%}")
                st.metric("지난주 대비", f"{trend.get('improvement_rate', 0):+.1%}")
                
        else:
            st.warning("분석 데이터를 가져올 수 없습니다.")
            
    except Exception as e:
        st.error(f"분석 데이터 조회 실패: {str(e)}")
    
    # 세션 정보
    st.subheader("📊 세션 정보")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**현재 세션:**")
        st.write(f"- 최적화 실행 횟수: {len(st.session_state.get('optimization_history', []))}")
        st.write(f"- 저장된 설정 수: {len(st.session_state.get('saved_settings', []))}")
        st.write(f"- 최근 결과 존재: {'✅' if 'latest_result' in st.session_state else '❌'}")
    
    with col2:
        st.write("**메모리 사용량:**")
        import sys
        session_size = sys.getsizeof(str(st.session_state))
        st.write(f"- 세션 상태 크기: {session_size:,} bytes")
        
        # 세션 상태 키 목록
        if st.checkbox("세션 키 상세 보기"):
            for key in st.session_state.keys():
                value_size = sys.getsizeof(st.session_state[key])
                st.write(f"  - {key}: {value_size:,} bytes")


if __name__ == "__main__":
    main() 