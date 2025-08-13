"""
지도 시각화 페이지
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
import folium
from streamlit_folium import st_folium
from folium import plugins
import pandas as pd

from utils.data_formatter import format_distance
from utils.session_state import init_session_state
from utils.dispatch_history import DispatchHistoryManager
from utils.coordinate_helper import get_order_coordinates, get_order_id


def get_darker_color(color):
    """색상을 어둡게 만드는 헬퍼 함수"""
    color_map = {
        'blue': '#1e3a8a',
        'green': '#166534', 
        'purple': '#7c2d12',
        'orange': '#c2410c',
        'darkred': '#7f1d1d',
        'lightred': '#b91c1c',
        'beige': '#a16207',
        'darkblue': '#1e40af'
    }
    return color_map.get(color, '#374151')  # 기본값: 회색


def get_color_emoji(color):
    """색상에 대응하는 이모지 반환"""
    emoji_map = {
        'blue': '🔵',
        'green': '🟢',
        'purple': '🟣',
        'orange': '🟠',
        'darkred': '🔴',
        'lightred': '🔴',
        'beige': '🟤',
        'darkblue': '🔵'
    }
    return emoji_map.get(color, '⚫')  # 기본값: 검은색


def add_direction_arrows(route_coordinates, color, feature_group):
    """경로에 방향 화살표 마커 추가"""
    import math
    
    if len(route_coordinates) < 2:
        return
    
    # 경로가 길 경우 중간 지점들에 화살표 추가
    num_arrows = min(len(route_coordinates) - 1, 5)  # 최대 5개의 화살표
    
    for i in range(1, len(route_coordinates), max(1, len(route_coordinates) // num_arrows)):
        if i >= len(route_coordinates):
            break
            
        prev_point = route_coordinates[i-1]
        curr_point = route_coordinates[i]
        
        # 방향 계산 (각도)
        lat1, lon1 = math.radians(prev_point[0]), math.radians(prev_point[1])
        lat2, lon2 = math.radians(curr_point[0]), math.radians(curr_point[1])
        
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.degrees(math.atan2(y, x))
        bearing = (bearing + 360) % 360  # 0-360도로 정규화
        
        # 중간 지점 좌표
        mid_lat = (prev_point[0] + curr_point[0]) / 2
        mid_lon = (prev_point[1] + curr_point[1]) / 2
        
        # 화살표 마커 추가
        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    transform: rotate({bearing}deg);
                    font-size: 12px;
                    color: {color};
                    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
                ">▶</div>
                """,
                icon_size=(12, 12),
                icon_anchor=(6, 6)
            )
        ).add_to(feature_group)


def main():
    """지도 보기 페이지 메인"""
    st.set_page_config(
        page_title="지도 보기 - TMS",
        page_icon="🗺️",
        layout="wide"
    )
    
    # 세션 상태 초기화
    init_session_state()
    
    # 지도 시각화 페이지 표시
    show_map_page()


def show_map_page():
    """지도 시각화 페이지 표시"""
    st.title("🗺️ 배차 결과 지도")
    st.markdown("---")
    
    # 배차 이력 관리자 초기화
    history_manager = DispatchHistoryManager()
    
    # 배차 데이터 선택 섹션 (다중 선택 지원)
    st.markdown("### 📋 배차 데이터 선택 (다중 선택 가능)")
    
    # 선택된 배차 결과들을 저장할 리스트
    selected_results = []
    
    # 배차 이력 섹션
    with st.expander("📚 배차 이력", expanded=True):
        # 고급 필터 옵션
        with st.container():
            st.markdown("**🔍 고급 필터 옵션**")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # 센터 선택
                centers = history_manager.get_centers_list()
                if centers:
                    center_options = {f"{c['name']} ({c['id']})": c['id'] for c in centers}
                    selected_center = st.selectbox(
                        "물류센터 선택",
                        options=list(center_options.keys()),
                        key="center_select"
                    )
                    center_id = center_options[selected_center] if selected_center else None
                else:
                    center_id = None
                    st.warning("사용 가능한 센터가 없습니다.")
            
            with col2:
                # 날짜 범위 필터
                from datetime import datetime, timedelta
                
                date_range = st.date_input(
                    "날짜 범위",
                    value=(datetime.now() - timedelta(days=7), datetime.now()),
                    max_value=datetime.now(),
                    key="history_date_range",
                    help="배차 이력을 조회할 날짜 범위를 선택하세요"
                )
            
            with col3:
                # 상태별 필터
                status_options = {
                    "전체": None,
                    "✅ 성공": "success",
                    "⚠️ 부분성공": "partial_success", 
                    "❌ 실패": "failed",
                    "🚫 취소": "cancelled"
                }
                selected_status_label = st.selectbox(
                    "배차 상태",
                    options=list(status_options.keys()),
                    key="history_status_filter"
                )
                selected_status = status_options[selected_status_label]
        
        # 추가 필터 옵션
        col1, col2 = st.columns([1, 1])
        
        with col1:
            max_history = st.slider("최대 이력 수", 1, 20, 10, key="max_history")
        
        with col2:
            # 정렬 옵션
            sort_options = {
                "📅 최신순": "created_at_desc",
                "📅 오래된순": "created_at_asc",
                "📦 주문 많은순": "total_orders_desc",
                "🚛 차량 많은순": "used_vehicles_desc",
                "📏 거리 긴순": "total_distance_desc"
            }
            selected_sort_label = st.selectbox(
                "정렬 방식",
                options=list(sort_options.keys()),
                key="history_sort"
            )
            selected_sort = sort_options[selected_sort_label]
        
        if center_id:
            # 필터링된 배차 이력 조회
            selected_batches = []
            
            try:
                batches = history_manager.get_recent_dispatch_batches(center_id=center_id, limit=50)
                
                if batches:
                    # 단순화된 필터 적용
                    filtered_batches = apply_simple_filters(
                        batches, date_range, selected_status, selected_sort
                    )
                    
                    st.success(f"✅ 총 {len(filtered_batches)}개 이력 표시")
                    
                    if filtered_batches:
                        st.markdown("**배차 이력 선택 (다중 선택 가능)**")
                        st.caption(f"총 {len(filtered_batches)}개 결과 (필터 적용 후)")
                        
                        # 각 배치에 대해 체크박스 생성
                        for i, batch in enumerate(filtered_batches[:max_history]):
                            created_at = batch['created_at'].strftime("%Y-%m-%d %H:%M")
                            
                            # 상태 이모지 추가
                            status_emoji = {
                                'success': '✅',
                                'partial_success': '⚠️',
                                'failed': '❌',
                                'cancelled': '🚫'
                            }.get(batch['status'], '❓')
                            
                            # 안전한 거리 정보 접근
                            distance_info = ""
                            if batch.get('total_distance') is not None:
                                distance_info = f" ({batch['total_distance']:.1f}km)"
                            
                            label = f"{status_emoji} [{created_at}] {batch['assigned_orders']}/{batch['total_orders']}개 주문{distance_info}"
                            
                            include_batch = st.checkbox(
                                label,
                                key=f"batch_{batch['batch_id']}",
                                value=False,
                                help=f"알고리즘: {batch.get('algorithm_used', 'N/A')} | 실행시간: {batch.get('execution_time_seconds', 0):.1f}초"
                            )
                            
                            if include_batch:
                                selected_batches.append(batch)
                    else:
                        st.info("🔍 필터 조건에 맞는 배차 이력이 없습니다.")
                else:
                    st.info("선택된 센터에 배차 이력이 없습니다.")
                    
            except Exception as e:
                st.error(f"❌ 배차 이력 조회 중 오류 발생:")
                st.error(f"   오류 타입: {type(e).__name__}")
                st.error(f"   오류 메시지: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            # 선택된 배치들을 결과 리스트에 추가
            for i, batch in enumerate(selected_batches):
                try:
                    batch_result = history_manager.get_dispatch_result_by_batch_id(batch['batch_id'])
                    
                    if batch_result:
                        total_orders = getattr(batch_result, 'total_orders', 0)
                        vehicle_assignments = getattr(batch_result, 'vehicle_assignments', [])
                        
                        # 빈 배차 결과 체크
                        if total_orders == 0 and len(vehicle_assignments) == 0:
                            st.warning(f"⚠️ 배치 {batch['batch_id']}: 주문과 차량 배정이 없어 지도에 표시되지 않습니다")
                            continue  # 빈 결과는 추가하지 않음
                        
                        # 배차 이력에 식별 정보 추가
                        created_at = batch['created_at'].strftime("%m-%d %H:%M")
                        batch_result.display_name = f"📚 {created_at}"
                        batch_result.result_type = "history"
                        batch_result.batch_id = batch['batch_id']
                        batch_result.style_config = {
                            "opacity": 0.7 - (i * 0.1),  # 점진적으로 투명하게
                            "line_style": "dashed" if i % 2 else "solid",
                            "color_offset": (i + 1) * 50  # 색상 오프셋
                        }
                        selected_results.append(batch_result)
                    else:
                        st.error(f"❌ 배치 {batch['batch_id']}: get_dispatch_result_by_batch_id 반환값이 None")
                        
                except Exception as e:
                    st.error(f"❌ 배치 {batch['batch_id']} 로딩 실패:")
                    st.error(f"   오류 타입: {type(e).__name__}")
                    st.error(f"   오류 메시지: {str(e)}")
                    
                    # 상세 스택 트레이스 표시
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("물류센터를 선택해주세요.")
    
    if not selected_results:
        st.info("📍 표시할 배차 결과를 선택해주세요.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚚 배차 실행 페이지로 이동", use_container_width=True):
                st.switch_page("pages/🚚_배차실행.py")
        return
    
    # 선택된 결과 요약 표시
    st.markdown("### 📊 선택된 배차 결과")
    cols = st.columns(min(len(selected_results), 4))
    
    for i, result in enumerate(selected_results):
        with cols[i % len(cols)]:
            st.metric(
                result.display_name,
                f"{result.total_orders}개 주문",
                f"{result.used_vehicles}대 차량"
            )
    
    # 지도 컨트롤
    st.markdown("### 🎛️ 지도 설정")
    
    # 기본 표시 옵션
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        show_centers = st.checkbox("물류센터 표시", value=True)
    with col2:
        show_orders = st.checkbox("배송지 표시", value=True)
    with col3:
        show_routes = st.checkbox("경로 표시", value=True)
    with col4:
        show_unassigned = st.checkbox("미배정 주문 표시", value=False)
    with col5:
        show_order_numbers = st.checkbox("순서 번호 표시", value=True)
    
    # 고급 시각화 옵션
    with st.expander("🎨 고급 시각화 옵션"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_direction_arrows = st.checkbox("방향 화살표 표시", value=True)
            show_route_animation = st.checkbox("경로 애니메이션", value=True)
        
        with col2:
            marker_size = st.select_slider(
                "마커 크기",
                options=["작음", "보통", "큼"],
                value="보통"
            )
            route_width = st.slider("경로 선 두께", 1, 8, 4)
        
        with col3:
            map_style = st.selectbox(
                "지도 스타일",
                ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"],
                index=0
            )
    
    # 여러 결과별 개별 토글 옵션
    with st.expander("🎛️ 배차 결과별 표시 제어"):
        result_controls = {}
        for result in selected_results:
            col1, col2 = st.columns([2, 1])
            with col1:
                show_result = st.checkbox(
                    f"{result.display_name} 표시",
                    value=True,
                    key=f"show_{result.result_type}_{getattr(result, 'batch_id', 'current')}"
                )
            with col2:
                if show_result and hasattr(result, 'style_config'):
                    opacity = st.slider(
                        "투명도",
                        0.1, 1.0, result.style_config['opacity'],
                        key=f"opacity_{result.result_type}_{getattr(result, 'batch_id', 'current')}"
                    )
                    result.style_config['opacity'] = opacity
            
            result_controls[result.display_name] = show_result
    
    # 차량별 필터링 제거 (OpenStreetMap 레이어 컨트롤 사용)
    
    # 주문별 고급 필터링
    with st.expander("📦 주문별 고급 필터링"):
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # 우선순위별 필터
            priority_options = {
                "전체": None,
                "🔴 높음 (High)": "high",
                "🟡 보통 (Medium)": "medium", 
                "🟢 낮음 (Low)": "low"
            }
            selected_priority_label = st.selectbox(
                "우선순위 필터",
                options=list(priority_options.keys()),
                key="priority_filter"
            )
            priority_filter = priority_options[selected_priority_label]
        
        with col2:
            # 주소 검색
            search_address = st.text_input(
                "주소 검색",
                placeholder="주소나 지역명 입력",
                key="address_search",
                help="주문 주소에서 특정 키워드를 포함한 주문만 표시"
            )
        
        with col3:
            # 주문 ID 검색
            search_order_id = st.text_input(
                "주문 ID 검색",
                placeholder="주문 ID 입력",
                key="order_id_search",
                help="특정 주문 ID를 포함한 주문만 표시"
            )
    
    # 고급 지도 옵션
    with st.expander("🗺️ 고급 지도 옵션"):
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # 마커 클러스터링
            enable_clustering = st.checkbox(
                "마커 클러스터링",
                value=False,
                key="enable_clustering",
                help="가까운 마커들을 클러스터로 그룹화하여 성능 향상"
            )
            
            # 히트맵 표시
            enable_heatmap = st.checkbox(
                "주문 밀도 히트맵",
                value=False,
                key="enable_heatmap",
                help="주문 밀도를 히트맵으로 표시"
            )
        
        with col2:
            # 거리 측정 도구
            enable_measure = st.checkbox(
                "거리 측정 도구",
                value=True,
                key="enable_measure",
                help="지도에서 거리를 측정할 수 있는 도구 추가"
            )
            
            # 전체화면 버튼
            enable_fullscreen = st.checkbox(
                "전체화면 버튼",
                value=True,
                key="enable_fullscreen",
                help="지도를 전체화면으로 볼 수 있는 버튼 추가"
            )
        
        with col3:
            # 미니맵 표시
            enable_minimap = st.checkbox(
                "미니맵 표시",
                value=False,
                key="enable_minimap",
                help="우하단에 작은 미니맵 표시"
            )
            
            # 좌표 표시
            enable_coordinates = st.checkbox(
                "마우스 좌표 표시",
                value=False,
                key="enable_coordinates",
                help="마우스 위치의 좌표를 실시간 표시"
            )
    
    # 활성화된 결과만 필터링
    active_results = []
    for result in selected_results:
        if result_controls.get(result.display_name, True):
            active_results.append(result)
    
    # 주문 필터링 적용
    if priority_filter or search_address or search_order_id:
        active_results = apply_order_filters(
            active_results, priority_filter, search_address, search_order_id
        )
        st.write(f"🔍 디버그: 주문 필터 후 결과 수: {len(active_results)}")
    
    # 지도 생성 및 표시
    map_options = {
        'show_centers': show_centers,
        'show_orders': show_orders,
        'show_routes': show_routes,
        'show_unassigned': show_unassigned,
        'show_order_numbers': show_order_numbers,
        'show_direction_arrows': show_direction_arrows if 'show_direction_arrows' in locals() else True,
        'show_route_animation': show_route_animation if 'show_route_animation' in locals() else True,
        'marker_size': marker_size if 'marker_size' in locals() else "보통",
        'route_width': route_width if 'route_width' in locals() else 4,
        'map_style': map_style if 'map_style' in locals() else "OpenStreetMap",

        'result_controls': result_controls,
        # 고급 지도 옵션
        'enable_clustering': enable_clustering if 'enable_clustering' in locals() else False,
        'enable_heatmap': enable_heatmap if 'enable_heatmap' in locals() else False,
        'enable_measure': enable_measure if 'enable_measure' in locals() else True,
        'enable_fullscreen': enable_fullscreen if 'enable_fullscreen' in locals() else True,
        'enable_minimap': enable_minimap if 'enable_minimap' in locals() else False,
        'enable_coordinates': enable_coordinates if 'enable_coordinates' in locals() else False,
        # 필터링 정보
        'priority_filter': priority_filter if 'priority_filter' in locals() else None,
        'search_address': search_address if 'search_address' in locals() else "",
        'search_order_id': search_order_id if 'search_order_id' in locals() else ""
    }
    
    # 지도 생성
    m = create_multiple_dispatch_map(active_results, **map_options)
    
    if m is None:
        st.error("❌ 지도 생성 실패: 배차 데이터를 확인해주세요")
        return
    
    # Folium 지도 표시
    map_data = st_folium(
        m,
        width='100%',
        height=600,
        returned_objects=["last_object_clicked"],
        key="dispatch_map"
    )
    
    # 클릭된 객체 정보 표시
    if map_data['last_object_clicked']:
        clicked = map_data['last_object_clicked']
        if 'popup' in clicked:
            st.info(f"선택된 위치: {clicked['popup']}")
    
    # 간단한 필터 정보만 표시 (필요시)
    active_filters = []
    if 'priority_filter' in locals() and priority_filter:
        active_filters.append(f"우선순위: {priority_filter}")
    if 'search_address' in locals() and search_address:
        active_filters.append(f"주소: {search_address}")
    if 'selected_status' in locals() and selected_status and selected_status != "전체":
        active_filters.append(f"상태: {selected_status}")
    
    if active_filters:
        st.info(f"🔍 적용된 필터: {', '.join(active_filters)}")
    
    # 범례 및 통계 표시
    st.markdown("---")
    display_multiple_results_legend_and_stats(active_results)


def create_dispatch_map(dispatch_result, show_centers=True, show_orders=True, 
                        show_routes=True, show_unassigned=False, show_order_numbers=True,
                        show_direction_arrows=True, show_route_animation=True,
                        marker_size="보통", route_width=4, map_style="OpenStreetMap"):
    """배차 결과를 기반으로 지도 생성"""
    import streamlit as st
    
    if dispatch_result is None:
        st.error("❌ 배차 결과가 None입니다")
        raise ValueError("dispatch_result가 None입니다")
    
    # 지도 중심점 설정 (센터 위치 기준)
    if hasattr(dispatch_result, 'center') and dispatch_result.center:
        center_lat = getattr(dispatch_result.center, 'latitude', 37.5665)
        center_lon = getattr(dispatch_result.center, 'longitude', 126.9780)
    else:
        # 기본값 (서울 중심)
        center_lat = 37.5665
        center_lon = 126.9780
    
    # 마커 크기 설정
    marker_sizes = {
        "작음": {"icon": 28, "number": 12},
        "보통": {"icon": 32, "number": 14},
        "큼": {"icon": 40, "number": 16}
    }
    size_config = marker_sizes.get(marker_size, marker_sizes["보통"])
    
    # 지도 스타일 변환
    tile_styles = {
        "OpenStreetMap": "OpenStreetMap",
        "CartoDB positron": "CartoDB positron",
        "CartoDB dark_matter": "CartoDB dark_matter"
    }
    tiles = tile_styles.get(map_style, "OpenStreetMap")
    
    # 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=tiles,
        prefer_canvas=True
    )
    
    # 물류센터 마커 (시작점)
    if show_centers and dispatch_result.center:
        # 향상된 시작점 마커
        folium.Marker(
            location=[dispatch_result.center.latitude, dispatch_result.center.longitude],
            popup=folium.Popup(
                f"""
                <div style='width: 280px; padding: 10px;'>
                <h4 style='margin-top: 0; color: #d63031;'>🏢 {dispatch_result.center.name}</h4>
                <hr style='margin: 8px 0;'>
                <table style='width: 100%; font-size: 12px;'>
                <tr><td><b>센터 ID:</b></td><td>{dispatch_result.center.center_id}</td></tr>
                <tr><td><b>주소:</b></td><td>{dispatch_result.center.address}</td></tr>
                <tr><td><b>총 주문:</b></td><td><span style='color: #00b894;'>{dispatch_result.total_orders}개</span></td></tr>
                <tr><td><b>사용 차량:</b></td><td><span style='color: #0984e3;'>{dispatch_result.used_vehicles}대</span></td></tr>
                <tr><td><b>총 거리:</b></td><td><span style='color: #e17055;'>{getattr(dispatch_result, 'total_distance', 0):.1f}km</span></td></tr>
                </table>
                </div>
                """,
                max_width=300
            ),
            tooltip="🏢 물류센터 (출발지)",
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    background: linear-gradient(135deg, #d63031 0%, #e17055 100%);
                    border: 3px solid white;
                    border-radius: 50%;
                    color: white;
                    font-weight: bold;
                    font-size: 16px;
                    text-align: center;
                    width: 40px;
                    height: 40px;
                    line-height: 34px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    position: relative;
                ">
                🏢
                </div>
                """,
                icon_size=(40, 40),
                icon_anchor=(20, 20)
            )
        ).add_to(m)
    
    # 차량별 경로 및 배송지 표시
    if hasattr(dispatch_result, 'vehicle_assignments') and dispatch_result.vehicle_assignments:
        # 차량별 레이어 그룹 생성
        for assignment in dispatch_result.vehicle_assignments:
            # 차량별 고유 색상
            color = assignment.color
            
            # 차량 레이어 그룹
            vehicle_group = folium.FeatureGroup(
                name=f"🚛 {assignment.driver_name} ({assignment.vehicle_id})"
            )
            
            # 경로 표시
            if show_routes and len(assignment.route_coordinates) > 1:
                # 메인 경로 선
                folium.PolyLine(
                    locations=assignment.route_coordinates,
                    color=color,
                    weight=route_width,
                    opacity=0.8,
                    popup=f"🚛 {assignment.driver_name} 경로",
                    smooth_factor=1
                ).add_to(vehicle_group)
                
                # 경로 방향 표시 (애니메이션) - 옵션에 따라
                if show_route_animation:
                    plugins.AntPath(
                        locations=assignment.route_coordinates,
                        color=color,
                        weight=max(1, route_width - 2),
                        opacity=0.8,
                        delay=800,
                        dash_array=[10, 20],
                        pulse_color='white'
                    ).add_to(vehicle_group)
                
                # 방향 화살표 마커 추가 (옵션에 따라)
                if show_direction_arrows:
                    add_direction_arrows(assignment.route_coordinates, color, vehicle_group)
            
            # 배송지 마커
            if show_orders:
                for j, order in enumerate(assignment.assigned_orders):
                    order_number = j + 1
                    
                    # 순서 번호 표시 여부에 따른 마커 생성
                    if show_order_numbers:
                        # 숫자가 표시된 DivIcon 사용
                        folium.Marker(
                            location=get_order_coordinates(order),
                            popup=folium.Popup(
                                f"""
                                <div style='width: 250px'>
                                <b>📦 주문 정보 (순서: {order_number})</b><br>
                                주문 ID: {get_order_id(order)}<br>
                                주소: {order.address}<br>
                                우선순위: {order.priority}<br>
                                차량: {assignment.driver_name}<br>
                                배송 순서: {order_number}/{len(assignment.assigned_orders)}<br>
                                예상 배송시간: {getattr(order, 'estimated_delivery_time', None) or 'N/A'}분
                                </div>
                                """,
                                max_width=270
                            ),
                            tooltip=f"[{order_number}] {get_order_id(order)}",
                            icon=folium.DivIcon(
                                html=f"""
                                <div style="
                                    background: linear-gradient(135deg, {color} 0%, {get_darker_color(color)} 100%);
                                    border: 3px solid white;
                                    border-radius: 50%;
                                    color: white;
                                    font-weight: bold;
                                    font-size: {size_config['number']}px;
                                    text-align: center;
                                    width: {size_config['icon']}px;
                                    height: {size_config['icon']}px;
                                    line-height: {size_config['icon'] - 6}px;
                                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                                    position: relative;
                                    z-index: 1000;
                                ">{order_number}</div>
                                <div style="
                                    position: absolute;
                                    top: {size_config['icon'] - 4}px;
                                    left: 50%;
                                    transform: translateX(-50%);
                                    width: 0;
                                    height: 0;
                                    border-left: 6px solid transparent;
                                    border-right: 6px solid transparent;
                                    border-top: 8px solid {color};
                                    filter: drop-shadow(0 2px 2px rgba(0,0,0,0.2));
                                "></div>
                                """,
                                icon_size=(size_config['icon'], size_config['icon'] + 8),
                                icon_anchor=(size_config['icon'] // 2, size_config['icon'] + 8)
                            )
                        ).add_to(vehicle_group)
                    else:
                        # 기본 CircleMarker 사용
                        folium.CircleMarker(
                            location=get_order_coordinates(order),
                            radius=8,
                            popup=folium.Popup(
                                f"""
                                <div style='width: 250px'>
                                <b>📦 주문 정보</b><br>
                                주문 ID: {get_order_id(order)}<br>
                                주소: {order.address}<br>
                                우선순위: {order.priority}<br>
                                차량: {assignment.driver_name}<br>
                                배송 순서: {order_number}/{len(assignment.assigned_orders)}<br>
                                예상 배송시간: {getattr(order, 'estimated_delivery_time', None) or 'N/A'}분
                                </div>
                                """,
                                max_width=270
                        ),
                            tooltip=f"주문 {order_number}: {get_order_id(order)}",
                        color=color,
                        fill=True,
                        fillColor=color,
                            fillOpacity=0.8,
                        weight=2
                    ).add_to(vehicle_group)
            
            vehicle_group.add_to(m)
    
    # 미배정 주문 표시
    if show_unassigned and hasattr(dispatch_result, 'unassigned_orders') and dispatch_result.unassigned_orders:
        unassigned_group = folium.FeatureGroup(name="⚠️ 미배정 주문")
        
        for order in dispatch_result.unassigned_orders:
            folium.CircleMarker(
                location=get_order_coordinates(order),
                radius=6,
                popup=folium.Popup(
                    f"""
                    <div style='width: 200px'>
                    <b>⚠️ 미배정 주문</b><br>
                    주문 ID: {get_order_id(order)}<br>
                    주소: {order.address}<br>
                    우선순위: {order.priority}<br>
                    상태: 미배정
                    </div>
                    """,
                    max_width=250
                ),
                tooltip=f"미배정: {get_order_id(order)}",
                color='gray',
                fill=True,
                fillColor='gray',
                fillOpacity=0.5,
                weight=2
            ).add_to(unassigned_group)
        
        unassigned_group.add_to(m)
    
    # 레이어 컨트롤은 create_multiple_dispatch_map에서 추가함
    
    # 전체 화면 버튼 추가
    plugins.Fullscreen(
        position='topleft',
        title='전체 화면',
        title_cancel='전체 화면 종료',
        force_separate_button=True
    ).add_to(m)
    
    # 측정 도구 추가
    plugins.MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        secondary_length_unit='meters',
        primary_area_unit='sqkilometers',
        secondary_area_unit='sqmeters'
    ).add_to(m)
    
    return m


def display_map_legend_and_stats(result):
    """지도 범례 및 통계 표시"""
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📍 범례")
        
        # 시작점 범례
        st.markdown("**🏢 물류센터 (시작점)**")
        st.markdown("- 그라데이션 원형 마커")
        st.markdown("- 모든 배송 경로의 출발지")
        
        st.markdown("---")
        
        # 차량별 동적 범례
        if result.vehicle_assignments:
            st.markdown("**🚛 차량별 배송 정보**")
            for i, assignment in enumerate(result.vehicle_assignments):
                color_emoji = get_color_emoji(assignment.color)
                st.markdown(f"""
                **{color_emoji} {assignment.driver_name}**
                - 차량 ID: `{assignment.vehicle_id}`
                - 배송 주문: {len(assignment.assigned_orders)}개
                - 예상 거리: {assignment.estimated_distance_km:.1f}km
                """)
        
        st.markdown("---")
        
        # 기타 범례
        legend_items = [
            ("📍", "순서 번호가 표시된 배송지"),
            ("➡️", "배송 경로 (실선)"),
            ("⚡", "경로 방향 (애니메이션)"),
            ("▶", "방향 화살표")
        ]
        
        for icon, desc in legend_items:
            st.markdown(f"**{icon}** {desc}")
        
        if result.unassigned_orders:
            st.markdown("**⚫ 미배정 주문**")
            st.markdown(f"- {len(result.unassigned_orders)}개 주문")
    
    with col2:
        st.markdown("### 📊 배차 통계")
        
        # 통계 데이터 생성
        stats_df = pd.DataFrame([
            {"항목": "📦 총 주문 수", "값": f"{result.total_orders}개"},
            {"항목": "✅ 배정 주문", "값": f"{result.assigned_orders}개"},
            {"항목": "❌ 미배정 주문", "값": f"{len(result.unassigned_orders)}개"},
            {"항목": "🚛 사용 차량", "값": f"{result.used_vehicles}/{result.total_vehicles}대"},
            {"항목": "📏 총 예상 거리", "값": format_distance(getattr(result, 'total_distance', 0))},
            {"항목": "⏱️ 총 예상 시간", "값": f"{getattr(result, 'total_time', 0)}분"},
            {"항목": "📈 배정률", "값": f"{(result.assigned_orders/result.total_orders*100):.1f}%" if result.total_orders > 0 else "0.0%"},
            {"항목": "⚙️ 사용 알고리즘", "값": getattr(result, 'algorithm_used', 'N/A')}
        ])
        
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "항목": st.column_config.TextColumn("항목", width="medium"),
                "값": st.column_config.TextColumn("값", width="medium")
            }
        )
    
    # 차량별 상세 정보
    if result.vehicle_assignments:
        st.markdown("### 🚛 차량별 배송 정보")
        
        for assignment in result.vehicle_assignments:
            with st.expander(f"{assignment.driver_name} ({assignment.vehicle_id}) - {len(assignment.assigned_orders)}개 주문"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("배정 주문", f"{len(assignment.assigned_orders)}개")
                with col2:
                    st.metric("예상 거리", format_distance(assignment.estimated_distance_km))
                with col3:
                    st.metric("용량 활용도", f"{assignment.capacity_utilization:.1%}")
                
                # 주문 목록
                st.markdown("**배송 순서:**")
                for i, order in enumerate(assignment.assigned_orders[:5], 1):
                    st.text(f"{i}. {get_order_id(order)}: {order.address[:30]}...")
                if len(assignment.assigned_orders) > 5:
                    st.text(f"... 외 {len(assignment.assigned_orders) - 5}개")


def create_multiple_dispatch_map(dispatch_results, show_centers=True, show_orders=True, 
                                show_routes=True, show_unassigned=False, show_order_numbers=True,
                                show_direction_arrows=True, show_route_animation=True,
                                marker_size="보통", route_width=4, map_style="OpenStreetMap", 
                                result_controls=None,
                                enable_clustering=False, enable_heatmap=False,
                                enable_measure=True, enable_fullscreen=True,
                                enable_minimap=False, enable_coordinates=False,
                                priority_filter=None, search_address="", search_order_id=""):
    """여러 배차 결과를 기반으로 지도 생성"""
    import streamlit as st
    
    if not dispatch_results:
        st.warning("⚠️ 표시할 배차 결과가 없습니다")
        return None
    
    # 첫 번째 결과를 기반으로 기본 지도 생성
    first_result = dispatch_results[0]
    
    try:
        # 기존 단일 결과 지도 생성 함수 재사용
        base_map = create_dispatch_map(
            first_result,
            show_centers=show_centers,
            show_orders=show_orders,
            show_routes=show_routes,
            show_unassigned=show_unassigned,
            show_order_numbers=show_order_numbers,
            show_direction_arrows=show_direction_arrows,
            show_route_animation=show_route_animation,
            marker_size=marker_size,
            route_width=route_width,
            map_style=map_style,

        )
        
        if base_map is None:
            st.error("❌ 기본 지도 생성 실패")
            raise ValueError("create_dispatch_map에서 None 반환")
        
    except Exception as e:
        st.error(f"❌ 기본 지도 생성 중 오류: {str(e)}")
        st.error("**에러 상세 정보:**")
        import traceback
        st.code(traceback.format_exc())
        raise e  # 에러를 다시 발생시켜서 상위에서 확인 가능
    
    # 추가 결과들을 레이어로 추가
    if len(dispatch_results) > 1:
        # 결과별 색상 세트
        result_color_sets = [
            ['darkred', 'lightred', 'beige', 'darkblue'],
            ['cadetblue', 'lightgreen', 'pink', 'gray'],
            ['red', 'darkgreen', 'black', 'lightblue']
        ]
        
        for result_idx, result in enumerate(dispatch_results[1:], 1):
            if not result_controls.get(result.display_name, True):
                continue
                
            color_set = result_color_sets[(result_idx-1) % len(result_color_sets)]
            opacity = result.style_config.get('opacity', 0.7)
            
            # 추가 결과의 차량들을 별도 레이어로 추가
            if hasattr(result, 'vehicle_assignments') and result.vehicle_assignments:
                for assignment in result.vehicle_assignments:
                    
                    vehicle_idx = result.vehicle_assignments.index(assignment)
                    color = color_set[vehicle_idx % len(color_set)]
                    
                    layer_name = f"{result.display_name} - {assignment.driver_name}"
                    vehicle_group = folium.FeatureGroup(name=layer_name)
                    
                    # 경로 추가 (점선, 투명도 적용)
                    if show_routes and hasattr(assignment, 'route_coordinates') and len(assignment.route_coordinates) > 1:
                        folium.PolyLine(
                            locations=assignment.route_coordinates,
                            color=color,
                            weight=route_width,
                            opacity=opacity,
                            dash_array=[10, 5],  # 점선
                            popup=f"🚛 {layer_name}",
                            smooth_factor=1
                        ).add_to(vehicle_group)
                    
                    # 마커 추가 (투명도 적용)
                    if show_orders and hasattr(assignment, 'assigned_orders'):
                        for j, order in enumerate(assignment.assigned_orders):
                            order_number = j + 1
                            folium.CircleMarker(
                                location=get_order_coordinates(order),
                                radius=6,
                                popup=folium.Popup(
                                    f"""
                                    <div style='width: 250px'>
                                    <b>📦 {result.display_name} (순서: {order_number})</b><br>
                                    주문 ID: {get_order_id(order)}<br>
                                    차량: {assignment.driver_name}<br>
                                    </div>
                                    """,
                                    max_width=270
                                ),
                                tooltip=f"[{result.display_name}] {order_number}",
                                color=color,
                                fill=True,
                                fillColor=color,
                                fillOpacity=opacity,
                                weight=2,
                                opacity=opacity
                            ).add_to(vehicle_group)
                    
                    vehicle_group.add_to(base_map)
    
    # 고급 지도 옵션 적용
    
    # 1. 히트맵 레이어 추가
    if enable_heatmap and dispatch_results:
        try:
            from folium.plugins import HeatMap
            
            # 모든 주문 위치 수집
            heat_data = []
            for result in dispatch_results:
                if hasattr(result, 'vehicle_assignments') and result.vehicle_assignments:
                    for assignment in result.vehicle_assignments:
                        if hasattr(assignment, 'assigned_orders'):
                            for order in assignment.assigned_orders:
                                coords = get_order_coordinates(order)
                                heat_data.append([coords[0], coords[1], 1.0])
            
            if heat_data:
                heat_layer = HeatMap(
                    heat_data,
                    name="주문 밀도 히트맵",
                    radius=15,
                    blur=10,
                    max_zoom=1,
                    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
                )
                heat_layer.add_to(base_map)
        except ImportError:
            pass  # HeatMap 플러그인이 없는 경우 무시
    
    # 2. 마커 클러스터링 (추가 결과용)
    if enable_clustering and len(dispatch_results) > 1:
        try:
            from folium.plugins import MarkerCluster
            
            # 클러스터 그룹 생성
            marker_cluster = MarkerCluster(
                name="클러스터된 마커",
                overlay=True,
                control=True
            )
            
            # 추가 결과들의 마커를 클러스터에 추가
            for result_idx, result in enumerate(dispatch_results[1:], 1):
                if hasattr(result, 'vehicle_assignments') and result.vehicle_assignments:
                    for assignment in result.vehicle_assignments:
                        if hasattr(assignment, 'assigned_orders'):
                            for order in assignment.assigned_orders:
                                coords = get_order_coordinates(order)
                                folium.Marker(
                                    location=coords,
                                    popup=f"[클러스터] {get_order_id(order)}",
                                    tooltip=f"{result.display_name}",
                                    icon=folium.Icon(color='gray', icon='info-sign')
                                ).add_to(marker_cluster)
            
            marker_cluster.add_to(base_map)
        except ImportError:
            pass  # MarkerCluster 플러그인이 없는 경우 무시
    
    # 3. 미니맵 추가
    if enable_minimap:
        try:
            from folium.plugins import MiniMap
            minimap = MiniMap(
                tile_layer="OpenStreetMap",
                position="bottomright",
                width=150,
                height=150,
                zoom_level_offset=-5,
                toggle_display=True
            )
            minimap.add_to(base_map)
        except ImportError:
            pass
    
    # 4. 마우스 좌표 표시
    if enable_coordinates:
        try:
            from folium.plugins import MousePosition
            MousePosition(
                position='topright',
                separator=' | ',
                empty_string='',
                lng_first=False,
                num_digits=6,
                prefix='좌표: ',
                lat_formatter="function(num) {return L.Util.formatNum(num, 6) + '° N';}",
                lng_formatter="function(num) {return L.Util.formatNum(num, 6) + '° E';}"
            ).add_to(base_map)
        except ImportError:
            pass
    
    # 5. 기본 도구들 (이미 있는지 확인 후 추가)
    if enable_fullscreen:
        plugins.Fullscreen(
            position='topleft',
            title='전체 화면',
            title_cancel='전체 화면 종료',
            force_separate_button=True
        ).add_to(base_map)
    
    if enable_measure:
        plugins.MeasureControl(
            position='topleft',
            primary_length_unit='kilometers',
            secondary_length_unit='meters',
            primary_area_unit='sqkilometers',
            secondary_area_unit='sqmeters'
        ).add_to(base_map)
    
    # 레이어 컨트롤 추가 (항상 마지막에)
    folium.LayerControl(collapsed=False).add_to(base_map)
    
    return base_map


def display_multiple_results_legend_and_stats(results):
    """여러 배차 결과에 대한 범례 및 통계 표시"""
    
    if not results:
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📍 범례 (다중 결과)")
        
        # 공통 범례
        st.markdown("**🏢 물류센터 (시작점)**")
        st.markdown("- 그라데이션 원형 마커")
        st.markdown("- 모든 배송 경로의 출발지")
        
        st.markdown("---")
        
        # 결과별 범례
        for i, result in enumerate(results):
            st.markdown(f"**{result.display_name}**")
            
            if hasattr(result, 'vehicle_assignments') and result.vehicle_assignments:
                style_info = ""
                if hasattr(result, 'style_config'):
                    opacity = result.style_config.get('opacity', 1.0)
                    line_style = result.style_config.get('line_style', 'solid')
                    style_info = f" (투명도: {opacity:.1f}, {line_style})"
                
                st.markdown(f"- 차량 수: {len(result.vehicle_assignments)}대")
                st.markdown(f"- 스타일: {'실선' if i == 0 else '점선'}{style_info}")
                
                for j, assignment in enumerate(result.vehicle_assignments[:3]):  # 최대 3개만 표시
                    color_emoji = get_color_emoji(assignment.color if hasattr(assignment, 'color') else 'blue')
                    st.markdown(f"  {color_emoji} {assignment.driver_name}: {len(getattr(assignment, 'assigned_orders', []))}개")
                
                if len(result.vehicle_assignments) > 3:
                    st.markdown(f"  ... 외 {len(result.vehicle_assignments) - 3}대")
            
            st.markdown("---")
        
        # 기타 범례
        legend_items = [
            ("📍", "순서 번호 마커"),
            ("➡️", "배송 경로"),
            ("⚡", "경로 애니메이션"),
            ("▶", "방향 화살표")
        ]
        
        for icon, desc in legend_items:
            st.markdown(f"**{icon}** {desc}")
    
    with col2:
        st.markdown("### 📊 다중 배차 통계")
        
        # 전체 통계 계산
        total_orders = sum(getattr(result, 'total_orders', 0) for result in results)
        total_assigned = sum(getattr(result, 'assigned_orders', 0) for result in results)
        total_unassigned = sum(len(getattr(result, 'unassigned_orders', [])) for result in results)
        total_vehicles = sum(getattr(result, 'used_vehicles', 0) for result in results)
        total_distance = sum(getattr(result, 'total_distance', 0) for result in results)
        total_time = sum(getattr(result, 'total_time', 0) for result in results)
        
        # 전체 요약 표
        summary_df = pd.DataFrame([
            {"항목": "📊 표시 결과 수", "값": f"{len(results)}개"},
            {"항목": "📦 총 주문", "값": f"{total_orders}개"},
            {"항목": "✅ 총 배정", "값": f"{total_assigned}개"},
            {"항목": "❌ 총 미배정", "값": f"{total_unassigned}개"},
            {"항목": "🚛 총 사용 차량", "값": f"{total_vehicles}대"},
            {"항목": "📏 총 거리", "값": format_distance(total_distance)},
            {"항목": "⏱️ 총 시간", "값": f"{total_time}분"},
            {"항목": "📈 평균 배정률", "값": f"{(total_assigned/total_orders*100):.1f}%" if total_orders > 0 else "0.0%"}
        ])
        
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "항목": st.column_config.TextColumn("항목", width="medium"),
                "값": st.column_config.TextColumn("값", width="medium")
            }
        )
        
        # 결과별 상세 비교
        if len(results) > 1:
            st.markdown("#### 📈 결과별 비교")
            
            comparison_data = []
            for result in results:
                comparison_data.append({
                    "결과": result.display_name,
                    "주문": getattr(result, 'total_orders', 0),
                    "배정": getattr(result, 'assigned_orders', 0),
                    "차량": getattr(result, 'used_vehicles', 0),
                    "거리(km)": f"{getattr(result, 'total_distance', 0):.1f}",
                    "배정률": f"{(getattr(result, 'assigned_orders', 0)/getattr(result, 'total_orders', 1)*100):.1f}%" if getattr(result, 'total_orders', 0) > 0 else "0.0%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )


def apply_advanced_filters(batches, date_range, status_filter, min_orders, sort_option):
    """배차 이력에 고급 필터를 적용"""
    from datetime import datetime, date
    import streamlit as st
    
    filtered = batches.copy()
    original_count = len(filtered)
    
    # 날짜 범위 필터 (좀 더 관대하게)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        
        try:
            # date 객체를 datetime으로 변환
            if isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_datetime = datetime.combine(start_date, datetime.min.time())
            else:
                start_datetime = start_date
                
            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_datetime = datetime.combine(end_date, datetime.max.time())
            else:
                end_datetime = end_date
            
            # 날짜 필터링 (더 관대하게)
            filtered = [
                batch for batch in filtered
                if batch.get('created_at') and start_datetime <= batch['created_at'] <= end_datetime
            ]
            
        except Exception as e:
            st.warning(f"날짜 필터 오류 (무시됨): {e}")
    
    # 상태 필터 (None과 "전체" 구분)
    if status_filter and status_filter != "전체":
        filtered = [
            batch for batch in filtered
            if batch.get('status') == status_filter
        ]
    
    # 최소 주문 수 필터 (더 관대하게 - 기본값 0이면 무시)
    if min_orders > 0:  # 0보다 클 때만 적용
        filtered = [
            batch for batch in filtered
            if batch.get('total_orders', 0) >= min_orders
        ]
    
    # 정렬 적용
    if sort_option and sort_option != "created_at_desc":  # 기본 정렬이 아닐 때만
        if sort_option == "created_at_asc":
            filtered.sort(key=lambda x: x['created_at'], reverse=False)
        elif sort_option == "total_orders_desc":
            filtered.sort(key=lambda x: x.get('total_orders', 0), reverse=True)
        elif sort_option == "used_vehicles_desc":
            filtered.sort(key=lambda x: x.get('used_vehicles', 0), reverse=True)
        elif sort_option == "total_distance_desc":
            filtered.sort(key=lambda x: x.get('total_distance', 0), reverse=True)
    else:
        # 기본 정렬: 최신순
        filtered.sort(key=lambda x: x['created_at'], reverse=True)
    
    return filtered


def apply_simple_filters(batches, date_range, status_filter, sort_option):
    """단순화된 배차 이력 필터 적용"""
    from datetime import datetime, date
    
    filtered = batches.copy()
    
    # 날짜 범위 필터 (옵션)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        
        try:
            # date 객체를 datetime으로 변환
            if isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_datetime = datetime.combine(start_date, datetime.min.time())
            else:
                start_datetime = start_date
                
            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_datetime = datetime.combine(end_date, datetime.max.time())
            else:
                end_datetime = end_date
            
            # 날짜 필터링
            filtered = [
                batch for batch in filtered
                if batch.get('created_at') and start_datetime <= batch['created_at'] <= end_datetime
            ]
            
        except Exception:
            pass  # 날짜 필터 오류 시 무시
    
    # 상태 필터 (옵션)
    if status_filter and status_filter != "전체":
        filtered = [
            batch for batch in filtered
            if batch.get('status') == status_filter
        ]
    
    # 정렬 적용
    if sort_option == "created_at_asc":
        filtered.sort(key=lambda x: x['created_at'], reverse=False)
    elif sort_option == "total_orders_desc":
        filtered.sort(key=lambda x: x.get('total_orders', 0), reverse=True)
    elif sort_option == "used_vehicles_desc":
        filtered.sort(key=lambda x: x.get('used_vehicles', 0), reverse=True)
    elif sort_option == "total_distance_desc":
        filtered.sort(key=lambda x: x.get('total_distance', 0), reverse=True)
    else:
        # 기본 정렬: 최신순
        filtered.sort(key=lambda x: x['created_at'], reverse=True)
    
    return filtered


def apply_order_filters(results, priority_filter, search_address, search_order_id):
    """선택된 결과들에 주문별 필터를 적용"""
    
    if not results:
        return results
    
    filtered_results = []
    
    for result in results:
        if not hasattr(result, 'vehicle_assignments') or not result.vehicle_assignments:
            filtered_results.append(result)
            continue
        
        # 결과 복사
        filtered_result = result.__class__(**{
            k: v for k, v in result.__dict__.items()
        })
        
        filtered_assignments = []
        
        for assignment in result.vehicle_assignments:
            if not hasattr(assignment, 'assigned_orders') or not assignment.assigned_orders:
                filtered_assignments.append(assignment)
                continue
            
            # 주문 필터링
            filtered_orders = []
            
            for order in assignment.assigned_orders:
                # 우선순위 필터
                if priority_filter:
                    order_priority = getattr(order, 'priority', '').lower()
                    if order_priority != priority_filter:
                        continue
                
                # 주소 검색
                if search_address:
                    order_address = getattr(order, 'address', '').lower()
                    if search_address.lower() not in order_address:
                        continue
                
                # 주문 ID 검색
                if search_order_id:
                    order_id = str(get_order_id(order)).lower()
                    if search_order_id.lower() not in order_id:
                        continue
                
                filtered_orders.append(order)
            
            # 필터링된 주문이 있는 경우만 차량 배정 포함
            if filtered_orders:
                # 배정 복사 및 주문 교체
                filtered_assignment = assignment.__class__(**{
                    k: v for k, v in assignment.__dict__.items()
                })
                filtered_assignment.assigned_orders = filtered_orders
                
                # 경로 좌표도 업데이트 (필터링된 주문만)
                if hasattr(assignment, 'route_coordinates'):
                    filtered_route = []
                    for order in filtered_orders:
                        filtered_route.append(get_order_coordinates(order))
                    filtered_assignment.route_coordinates = filtered_route
                
                filtered_assignments.append(filtered_assignment)
        
        # 필터링된 차량 배정 적용
        filtered_result.vehicle_assignments = filtered_assignments
        
        # 통계 업데이트
        total_filtered_orders = sum(len(getattr(a, 'assigned_orders', [])) for a in filtered_assignments)
        filtered_result.assigned_orders = total_filtered_orders
        filtered_result.used_vehicles = len(filtered_assignments)
        
        filtered_results.append(filtered_result)
    
    return filtered_results


def display_applied_filters_summary(priority_filter, search_address, search_order_id, 
                                  selected_status, date_range, min_orders,
                                  enable_clustering, enable_heatmap, enable_minimap, enable_coordinates):
    """적용된 필터 및 지도 옵션 요약 표시"""
    
    # 활성화된 필터 수집
    active_filters = []
    active_options = []
    
    # 주문 필터
    if priority_filter:
        priority_labels = {"high": "🔴 높음", "medium": "🟡 보통", "low": "🟢 낮음"}
        active_filters.append(f"**우선순위**: {priority_labels.get(priority_filter, priority_filter)}")
    
    if search_address:
        active_filters.append(f"**주소 검색**: '{search_address}'")
    
    if search_order_id:
        active_filters.append(f"**주문 ID**: '{search_order_id}'")
    
    # 배차 이력 필터
    if selected_status:
        status_labels = {
            "success": "✅ 성공", 
            "partial_success": "⚠️ 부분성공", 
            "failed": "❌ 실패", 
            "cancelled": "🚫 취소"
        }
        active_filters.append(f"**배차 상태**: {status_labels.get(selected_status, selected_status)}")
    
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        active_filters.append(f"**날짜 범위**: {start_date} ~ {end_date}")
    
    if min_orders and min_orders > 1:
        active_filters.append(f"**최소 주문 수**: {min_orders}개 이상")
    
    # 고급 지도 옵션
    if enable_clustering:
        active_options.append("🔗 마커 클러스터링")
    
    if enable_heatmap:
        active_options.append("🌡️ 주문 밀도 히트맵")
    
    if enable_minimap:
        active_options.append("🗺️ 미니맵")
    
    if enable_coordinates:
        active_options.append("📍 마우스 좌표 표시")
    
    # 필터 요약 표시
    if active_filters or active_options:
        with st.expander("🔍 적용된 필터 및 옵션 요약", expanded=False):
            
            if active_filters:
                st.markdown("### 📋 활성 필터")
                for filter_info in active_filters:
                    st.markdown(f"• {filter_info}")
            
            if active_options:
                st.markdown("### 🛠️ 활성 지도 옵션")
                for option_info in active_options:
                    st.markdown(f"• {option_info}")
            
            # 필터 초기화 버튼들
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🔄 주문 필터 초기화", key="reset_order_filters"):
                    # 세션 상태 초기화
                    for key in ['priority_filter', 'address_search', 'order_id_search']:
                        if key in st.session_state:
                            st.session_state[key] = "" if "search" in key else 0
                    st.rerun()
            
            with col2:
                if st.button("📅 이력 필터 초기화", key="reset_history_filters"):
                    # 세션 상태 초기화
                    for key in ['history_status_filter', 'history_date_range', 'min_orders_filter']:
                        if key in st.session_state:
                            if "date" in key:
                                from datetime import datetime, timedelta
                                st.session_state[key] = (datetime.now() - timedelta(days=7), datetime.now())
                            elif "status" in key:
                                st.session_state[key] = 0  # "전체" 선택
                            else:
                                st.session_state[key] = 1
                    st.rerun()
            
            with col3:
                if st.button("🗺️ 지도 옵션 초기화", key="reset_map_options"):
                    # 지도 옵션 초기화
                    for key in ['enable_clustering', 'enable_heatmap', 'enable_minimap', 'enable_coordinates']:
                        if key in st.session_state:
                            st.session_state[key] = False
                    st.rerun()
    
    else:
        st.info("🔍 현재 적용된 필터가 없습니다. 위의 옵션들을 사용하여 데이터를 필터링해보세요.")


if __name__ == "__main__":
    main()