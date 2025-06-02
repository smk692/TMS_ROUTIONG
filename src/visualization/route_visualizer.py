#!/usr/bin/env python3
"""
Route Visualizer Service - current_route_map_new.html 완전 복제
155,012줄의 원본 HTML 파일을 정확히 반영한 경로 시각화 서비스
"""

import json
import folium
from folium import plugins
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import os
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RouteVisualizerService:
    """경로 시각화 서비스 클래스 - current_route_map_new.html 기반"""
    
    def __init__(self, config=None):
        """초기화"""
        self.config = config
        self.map = None
        self.vehicle_colors = {}
        self.tc_vehicle_mapping = {}
        self.depot_info = {}
        
        # 기본 색상 팔레트
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#F4D03F'
        ]
        
        # TC 데이터 (원본과 동일)
        self.tc_data = {
            "incheon_center": "인천센터",
            "icheon_center": "이천센터", 
            "hwaseong_center": "화성센터",
            "hanam_center": "하남센터",
            "gwangju_center": "광주센터",
            "ilsan_center": "일산센터",
            "namyangju_center": "남양주센터",
            "gunpo_center": "군포센터"
        }
        
        # 실제 통계 데이터 (원본과 동일)
        self.statistics = {
            "total_deliveries": 2015,
            "total_vehicles": 45,
            "total_tcs": 8,
            "total_distance": "4585.9km",
            "avg_distance": "101.9km"
        } 

    def add_ui_controls(self, map_obj: folium.Map, route_data: Dict[str, Any] = None) -> None:
        """인터랙티브 UI 컨트롤 추가 (원본과 동일한 JavaScript 로직)"""
        try:
            logger.info("UI 컨트롤 추가 시작...")
            
            # CSS 스타일 (원본과 동일)
            css_style = """
            <style>
                /* 컨트롤 패널 기본 스타일 */
                .route-control-panel {
                    position: fixed;
                    top: 10px;
                    left: 10px;
                    width: 320px;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                    z-index: 1000;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                
                /* 헤더 스타일 */
                .control-header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px 20px;
                    border-radius: 12px 12px 0 0;
                    font-weight: 600;
                    font-size: 16px;
                    text-align: center;
                    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
                }
                
                /* 컨텐츠 영역 */
                .control-content {
                    padding: 20px;
                    max-height: 70vh;
                    overflow-y: auto;
                }
                
                /* 통계 섹션 */
                .stats-section {
                    margin-bottom: 20px;
                    padding: 15px;
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    border-radius: 8px;
                    border-left: 4px solid #007bff;
                }
                
                .stats-title {
                    font-weight: 600;
                    color: #495057;
                    margin-bottom: 10px;
                    font-size: 14px;
                }
                
                .stats-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 8px;
                }
                
                .stat-item {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 5px 0;
                }
                
                .stat-label {
                    font-size: 12px;
                    color: #6c757d;
                    font-weight: 500;
                }
                
                .stat-value {
                    font-size: 12px;
                    font-weight: 600;
                    color: #495057;
                }
                
                /* TC 목록 스타일 */
                .tc-section {
                    margin-bottom: 20px;
                }
                
                .section-title {
                    font-weight: 600;
                    color: #495057;
                    margin-bottom: 12px;
                    font-size: 14px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .tc-item {
                    background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
                    color: #6c757d;
                    padding: 12px;
                    margin: 8px 0;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 2px 8px rgba(108, 117, 125, 0.2);
                    border: 2px solid #dee2e6;
                }
                
                .tc-item:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
                }
                
                .tc-item.active {
                    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                    color: white;
                    border: 2px solid #28a745;
                    box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
                }
                
                .tc-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 8px;
                }
                
                .tc-name {
                    font-weight: 600;
                    font-size: 13px;
                }
                
                .tc-toggle {
                    font-size: 16px;
                    transition: transform 0.3s ease;
                }
                
                .tc-item.active .tc-toggle {
                    transform: rotate(0deg);
                }
                
                .tc-item:not(.active) .tc-toggle {
                    transform: rotate(45deg);
                }
                
                .tc-stats {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 6px;
                    font-size: 11px;
                    opacity: 0.9;
                }
                
                /* 차량 제어 버튼 */
                .vehicle-controls {
                    margin-bottom: 20px;
                }
                
                .vehicle-toggle-btn {
                    width: 100%;
                    padding: 12px;
                    background: linear-gradient(135deg, #fd7e14 0%, #e63946 100%);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 13px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    box-shadow: 0 2px 8px rgba(253, 126, 20, 0.2);
                }
                
                .vehicle-toggle-btn:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(253, 126, 20, 0.3);
                }
                
                /* 사용법 안내 */
                .usage-section {
                    padding: 15px;
                    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                    border-radius: 8px;
                    border-left: 4px solid #ffc107;
                }
                
                .usage-title {
                    font-weight: 600;
                    color: #856404;
                    margin-bottom: 8px;
                    font-size: 13px;
                }
                
                .usage-list {
                    font-size: 11px;
                    color: #856404;
                    line-height: 1.4;
                    margin: 0;
                    padding-left: 15px;
                }
                
                /* 스크롤바 스타일 */
                .control-content::-webkit-scrollbar {
                    width: 6px;
                }
                
                .control-content::-webkit-scrollbar-track {
                    background: #f1f1f1;
                    border-radius: 3px;
                }
                
                .control-content::-webkit-scrollbar-thumb {
                    background: #c1c1c1;
                    border-radius: 3px;
                }
                
                .control-content::-webkit-scrollbar-thumb:hover {
                    background: #a8a8a8;
                }
                
                /* 반응형 디자인 */
                @media (max-width: 768px) {
                    .route-control-panel {
                        width: 280px;
                        left: 5px;
                        top: 5px;
                    }
                    
                    .control-content {
                        padding: 15px;
                        max-height: 60vh;
                    }
                }
            </style>
            """
            
            # HTML 컨트롤 패널 (실제 데이터 사용)
            tc_buttons = self._generate_tc_buttons(route_data)
            
            html_content = f"""
            {css_style}
            <div class="route-control-panel">
                <div class="control-header">
                    🗺️ TMS 배송 경로 제어판
                </div>
                <div class="control-content">
                    <!-- 실시간 통계 -->
                    <div class="stats-section">
                        <div class="stats-title">📊 실시간 통계</div>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">총 배송지</span>
                                <span class="stat-value" id="total-deliveries">2,015개</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">투입 차량</span>
                                <span class="stat-value" id="total-vehicles">45대</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">운영 센터</span>
                                <span class="stat-value" id="total-centers">8개</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">총 거리</span>
                                <span class="stat-value" id="total-distance">1,328km</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- 차량 전체 제어 -->
                    <div class="vehicle-controls">
                        <button class="vehicle-toggle-btn" onclick="toggleAllVehicleLayers()">
                            🚛 모든 차량 표시/숨김
                        </button>
                    </div>
                    
                    <!-- TC 목록 -->
                    <div class="tc-section">
                        <div class="section-title">
                            🏢 터미널 센터 목록
                        </div>
                        {tc_buttons}
                    </div>
                    
                    <!-- 사용법 안내 -->
                    <div class="usage-section">
                        <div class="usage-title">💡 사용법</div>
                        <ul class="usage-list">
                            <li>TC 버튼 클릭으로 해당 센터 차량 표시/숨김</li>
                            <li>레이어 컨트롤에서 개별 차량 제어 가능</li>
                            <li>마커 클릭으로 상세 정보 확인</li>
                            <li>폴리라인은 실제 도로 경로를 표시</li>
                        </ul>
                    </div>
                </div>
            </div>
            """
            
            # HTML 요소를 맵에 추가
            map_obj.get_root().html.add_child(folium.Element(html_content))

            # 차량 필터링 및 토글 함수들 추가
            js_code_2 = f"""
            <script>
            // 전역 변수
            let selectedTCs = new Set();
            let allVehicleLayersVisible = false;
            let allLayersVisible = false;
            
            // 실제 데이터에서 가져온 TC 정보
            const actualTCData = {json.dumps({depot['id']: depot.get('name', depot.get('label', depot['id'])) for depot in route_data.get('depots', [])})};
            
            console.log('실제 TC 데이터:', actualTCData);
            
            // 페이지 로드 시 초기화
            document.addEventListener('DOMContentLoaded', function() {{
                console.log('🚀 TMS 제어판 초기화 시작...');
                initializeTCButtons();
            }});
            
            // TC 버튼 초기화
            function initializeTCButtons() {{
                const tcItems = document.querySelectorAll('.tc-item');
                console.log(`📋 TC 버튼 수: ${{tcItems.length}}`);
                
                tcItems.forEach(function(tcItem) {{
                    const tcId = tcItem.getAttribute('data-tc-id');
                    const tcNameElement = tcItem.querySelector('.tc-name');
                    const tcName = tcNameElement ? tcNameElement.textContent.trim() : tcId;
                    
                    if (tcId === 'error') {{
                        console.log('⚠️ 오류 상태 TC 발견, 건너뛰기');
                        return;
                    }}
                    
                    console.log(`🏢 TC 버튼 초기화: ${{tcName}} (ID: ${{tcId}})`);
                    
                    // 클릭 이벤트 리스너 추가
                    tcItem.addEventListener('click', function() {{
                        toggleTCSelection(tcId, tcName, tcItem);
                    }});
                    
                    // 호버 효과 추가
                    tcItem.addEventListener('mouseenter', function() {{
                        if (!tcItem.classList.contains('active')) {{
                            tcItem.style.transform = 'translateY(-2px)';
                        }}
                    }});
                    
                    tcItem.addEventListener('mouseleave', function() {{
                        if (!tcItem.classList.contains('active')) {{
                            tcItem.style.transform = 'translateY(0)';
                        }}
                    }});
                }});
                
                // 초기 상태: 모든 차량 숨김
                hideAllVehicles();
                console.log('✅ TC 버튼 초기화 완료');
            }}
            
            // TC 선택/해제 토글
            function toggleTCSelection(tcId, tcName, tcItem) {{
                console.log(`🔄 TC 토글: ${{tcName}} (ID: ${{tcId}})`);
                
                if (selectedTCs.has(tcId)) {{
                    // 선택 해제
                    selectedTCs.delete(tcId);
                    tcItem.classList.remove('active');
                    console.log(`❌ TC 선택 해제: ${{tcName}}`);
                }} else {{
                    // 선택
                    selectedTCs.add(tcId);
                    tcItem.classList.add('active');
                    console.log(`✅ TC 선택: ${{tcName}}`);
                }}
                
                // 차량 필터링 적용
                filterVehiclesBySelectedTCs();
                updateStatistics();
            }}
            
            // 선택된 TC에 해당하는 차량만 표시
            function filterVehiclesBySelectedTCs() {{
                console.log('🔍 차량 필터링 시작...');
                console.log('선택된 TC들:', Array.from(selectedTCs));
                
                const layerControl = document.querySelector('.leaflet-control-layers');
                if (!layerControl) {{
                    console.error('❌ 레이어 컨트롤을 찾을 수 없습니다.');
                    return;
                }}
                
                const checkboxes = layerControl.querySelectorAll('input[type="checkbox"]');
                const labels = layerControl.querySelectorAll('.leaflet-control-layers-overlays label');
                
                let visibleCount = 0;
                let hiddenCount = 0;
                
                labels.forEach(function(label, index) {{
                    const checkbox = checkboxes[index];
                    const labelText = label.textContent || label.innerText;
                    
                    // 레이블에서 TC 이름 추출하여 매칭
                    let shouldShow = false;
                    
                    if (selectedTCs.size === 0) {{
                        shouldShow = false;
                    }} else {{
                        selectedTCs.forEach(function(tcId) {{
                            const tcName = actualTCData[tcId] || tcId;
                            // TC 이름이 레이블에 포함되어 있는지 확인
                            if (labelText.includes(tcName) || labelText.includes(tcId)) {{
                                shouldShow = true;
                            }}
                        }});
                    }}
                    
                    // 차량 레이어 표시/숨김 처리
                    if (shouldShow) {{
                        label.style.display = 'block';
                        label.style.height = 'auto';
                        label.style.margin = '0 0 8px 0';
                        label.style.padding = '6px';
                        label.style.opacity = '1';
                        label.style.visibility = 'visible';
                        if (!checkbox.checked) {{
                            checkbox.click();
                        }}
                        visibleCount++;
                    }} else {{
                        label.style.display = 'none';
                        label.style.height = '0';
                        label.style.margin = '0';
                        label.style.padding = '0';
                        label.style.opacity = '0';
                        label.style.visibility = 'hidden';
                        if (checkbox.checked) {{
                            checkbox.click();
                        }}
                        hiddenCount++;
                    }}
                }});
                
                console.log(`📊 필터링 결과: 표시 ${{visibleCount}}개, 숨김 ${{hiddenCount}}개`);
            }}
            
            // 모든 차량 숨김
            function hideAllVehicles() {{
                console.log('🚫 모든 차량 숨김 처리...');
                const layerControl = document.querySelector('.leaflet-control-layers');
                if (!layerControl) return;
                
                const checkboxes = layerControl.querySelectorAll('input[type="checkbox"]');
                checkboxes.forEach(function(checkbox) {{
                    if (checkbox.checked) {{
                        checkbox.click();
                    }}
                }});
            }}
            
            // 차량 레이어 전체 선택/해제
            function toggleAllVehicleLayers() {{
                console.log('🚗 toggleAllVehicleLayers 함수 호출됨');
                const layerControl = document.querySelector('.leaflet-control-layers');
                if (!layerControl) {{
                    console.error('❌ 레이어 컨트롤을 찾을 수 없습니다.');
                    return;
                }}
                
                const checkboxes = layerControl.querySelectorAll('input[type="checkbox"]');
                const labels = layerControl.querySelectorAll('.leaflet-control-layers-overlays label');
                
                // 현재 보이는 차량들만 대상으로 함
                let visibleCheckboxes = [];
                labels.forEach(function(label, index) {{
                    if (label.style.display !== 'none') {{
                        visibleCheckboxes.push(checkboxes[index]);
                    }}
                }});
                
                console.log(`👁️ 보이는 차량 수: ${{visibleCheckboxes.length}}`);
                
                // 현재 상태 확인
                let checkedCount = 0;
                visibleCheckboxes.forEach(function(checkbox) {{
                    if (checkbox.checked) checkedCount++;
                }});
                
                console.log(`✅ 체크된 차량 수: ${{checkedCount}}`);
                
                if (checkedCount === visibleCheckboxes.length && visibleCheckboxes.length > 0) {{
                    // 모두 체크되어 있으면 전체 해제
                    visibleCheckboxes.forEach(function(checkbox) {{
                        if (checkbox.checked) {{
                            checkbox.click();
                        }}
                    }});
                    allVehicleLayersVisible = false;
                    console.log('❌ 모든 차량 레이어 해제됨');
                }} else {{
                    // 일부만 체크되어 있거나 아무것도 없으면 전체 선택
                    visibleCheckboxes.forEach(function(checkbox) {{
                        if (!checkbox.checked) {{
                            checkbox.click();
                        }}
                    }});
                    allVehicleLayersVisible = true;
                    console.log('✅ 모든 차량 레이어 선택됨');
                }}
            }}
            
            // 통계 업데이트
            function updateStatistics() {{
                // 선택된 TC들의 통계를 계산하여 업데이트
                console.log('📊 통계 업데이트:', Array.from(selectedTCs));
                
                // 실제 구현에서는 선택된 TC의 통계만 계산하여 표시
                const totalDeliveriesElement = document.getElementById('total-deliveries');
                const totalVehiclesElement = document.getElementById('total-vehicles');
                const totalCentersElement = document.getElementById('total-centers');
                
                if (totalCentersElement) {{
                    totalCentersElement.textContent = `${{selectedTCs.size}}개`;
                }}
            }}
            </script>
            """
            
            # 두 번째 JavaScript 코드 추가
            map_obj.get_root().html.add_child(folium.Element(js_code_2))

        except Exception as e:
            logger.error(f"UI 컨트롤 추가 실패: {e}")

    def _generate_tc_buttons(self, route_data: Dict[str, Any]) -> str:
        """TC 센터 버튼 생성 - 실제 데이터 기반 (JavaScript와 일치하는 구조)"""
        try:
            # 실제 데이터에서 depots 정보 추출
            depots = route_data.get('depots', [])
            if not depots:
                # 하위 호환성: 단일 depot 처리
                depot = route_data.get('depot', {})
                if depot:
                    depots = [depot]
            
            if not depots:
                logger.warning("TC 센터 정보를 찾을 수 없습니다.")
                return ""
            
            # routes에서 실제 사용된 TC 정보 추출
            routes = route_data.get('routes', [])
            used_tc_ids = set()
            tc_stats = {}
            
            for route in routes:
                depot_id = route.get('depot_id')
                if depot_id:
                    used_tc_ids.add(depot_id)
                    if depot_id not in tc_stats:
                        tc_stats[depot_id] = {
                            'vehicle_count': 0,
                            'delivery_count': 0,
                            'total_distance': 0
                        }
                    
                    tc_stats[depot_id]['vehicle_count'] += 1
                    tc_stats[depot_id]['delivery_count'] += len(route.get('coordinates', []))
                    tc_stats[depot_id]['total_distance'] += route.get('distance', 0)
            
            # TC 버튼 HTML 생성 (JavaScript가 기대하는 .tc-item 구조)
            buttons_html = []
            
            # 각 TC별 버튼 생성
            for depot in depots:
                depot_id = depot.get('id', depot.get('depot_id', ''))
                depot_name = depot.get('name', depot.get('label', f'TC-{depot_id}'))
                
                # 실제 사용된 TC만 표시
                if depot_id not in used_tc_ids:
                    continue
                
                stats = tc_stats.get(depot_id, {})
                vehicle_count = stats.get('vehicle_count', 0)
                delivery_count = stats.get('delivery_count', 0)
                total_distance = stats.get('total_distance', 0)
                
                # TC 아이콘 선택
                tc_icon = self._get_tc_icon(depot_name)
                
                # JavaScript가 기대하는 구조로 생성
                button_html = f"""
                    <div class="tc-item" data-tc-id="{depot_id}">
                        <div class="tc-header">
                            <div class="tc-name">{tc_icon} {depot_name}</div>
                            <div class="tc-toggle">✕</div>
                        </div>
                        <div class="tc-stats">
                            <span>차량: {vehicle_count}대</span>
                            <span>배송: {delivery_count}개</span>
                            <span>거리: {total_distance:.1f}km</span>
                            <span>평균: {total_distance/vehicle_count if vehicle_count > 0 else 0:.1f}km</span>
                        </div>
                    </div>
                """
                buttons_html.append(button_html)
            
            return '\n'.join(buttons_html)
            
        except Exception as e:
            logger.error(f"TC 버튼 생성 중 오류: {str(e)}")
            return """
                <div class="tc-item" data-tc-id="error">
                    <div class="tc-header">
                        <div class="tc-name">⚠️ 데이터 로드 실패</div>
                        <div class="tc-toggle">✕</div>
                    </div>
                    <div class="tc-stats">
                        <span>데이터를 확인해주세요</span>
                    </div>
                </div>
            """
    
    def _get_tc_icon(self, tc_name: str) -> str:
        """TC 이름에 따른 아이콘 반환"""
        tc_name_lower = tc_name.lower()
        
        if '인천' in tc_name_lower or 'incheon' in tc_name_lower:
            return '🏭'
        elif '이천' in tc_name_lower or 'icheon' in tc_name_lower:
            return '🏢'
        elif '화성' in tc_name_lower or 'hwaseong' in tc_name_lower:
            return '🏗️'
        elif '하남' in tc_name_lower or 'hanam' in tc_name_lower:
            return '🏪'
        elif '광주' in tc_name_lower or 'gwangju' in tc_name_lower:
            return '🏬'
        elif '일산' in tc_name_lower or 'ilsan' in tc_name_lower:
            return '🏭'
        elif '남양주' in tc_name_lower or 'namyangju' in tc_name_lower:
            return '🏢'
        elif '군포' in tc_name_lower or 'gunpo' in tc_name_lower:
            return '🏗️'
        else:
            return '🏢'

    def create_depot_marker(self, map_obj: folium.Map, depot_info: Dict[str, Any]) -> None:
        """TC 위치 마커 생성 (원본과 동일한 스타일)"""
        try:
            lat = float(depot_info.get('latitude', 37.4788353))
            lng = float(depot_info.get('longitude', 126.6648639))
            name = depot_info.get('name', 'TC 센터')
            
            # 원본과 동일한 검은색 홈 아이콘
            depot_marker = folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(
                    f"""
                    <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 200px;">
                        <h4 style="margin: 0 0 10px 0; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px;">
                            🏢 {name}
                        </h4>
                        <div style="font-size: 13px; line-height: 1.5;">
                            <p style="margin: 5px 0;"><strong>📍 위치:</strong> {lat:.6f}, {lng:.6f}</p>
                            <p style="margin: 5px 0;"><strong>🏢 유형:</strong> 물류센터</p>
                            <p style="margin: 5px 0;"><strong>📊 상태:</strong> 운영중</p>
                        </div>
                    </div>
                    """,
                    max_width=250
                ),
                icon=folium.Icon(
                    color='black',
                    icon='home',
                    prefix='fa'
                )
            )
            
            depot_marker.add_to(map_obj)
            logger.info(f"TC 마커 생성 완료: {name} at ({lat}, {lng})")
            
        except Exception as e:
            logger.error(f"TC 마커 생성 실패: {e}")
    
    def create_polylines(self, map_obj: folium.Map, route_data: List[Dict], vehicle_id: str, tc_name: str) -> None:
        """경로 폴리라인 생성 (원본과 동일한 스타일)"""
        try:
            if not route_data:
                return
                
            # 차량별 색상 할당
            if vehicle_id not in self.vehicle_colors:
                color_index = len(self.vehicle_colors) % len(self.color_palette)
                self.vehicle_colors[vehicle_id] = self.color_palette[color_index]
            
            color = self.vehicle_colors[vehicle_id]
            
            # 좌표 추출
            coordinates = []
            for point in route_data:
                try:
                    lat = float(point.get('latitude', 0))
                    lng = float(point.get('longitude', 0))
                    if lat != 0 and lng != 0:
                        coordinates.append([lat, lng])
                except (ValueError, TypeError):
                    continue
            
            if len(coordinates) < 2:
                return
            
            # 폴리라인 생성
            polyline = folium.PolyLine(
                locations=coordinates,
                color=color,
                weight=4,
                opacity=0.8,
                popup=f"{tc_name} - {vehicle_id}"
            )
            
            # 차량별 레이어 그룹에 추가
            layer_name = f"{tc_name} - {vehicle_id}"
            if layer_name not in self.tc_vehicle_mapping or self.tc_vehicle_mapping[layer_name] is None:
                vehicle_group = folium.FeatureGroup(name=layer_name)
                self.tc_vehicle_mapping[layer_name] = vehicle_group
                vehicle_group.add_to(map_obj)  # 지도에 먼저 추가
            
            polyline.add_to(self.tc_vehicle_mapping[layer_name])
            logger.info(f"폴리라인 생성 완료: {vehicle_id} ({len(coordinates)}개 좌표)")
            
        except Exception as e:
            logger.error(f"폴리라인 생성 실패 {vehicle_id}: {e}")
    
    def create_delivery_markers(self, map_obj: folium.Map, deliveries: List[Dict], vehicle_id: str, tc_name: str) -> None:
        """배송지 마커 생성 (원본과 동일한 스타일)"""
        try:
            if not deliveries:
                return
                
            color = self.vehicle_colors.get(vehicle_id, '#FF6B6B')
            layer_name = f"{tc_name} - {vehicle_id}"
            
            # 레이어 그룹 확인 및 생성
            if layer_name not in self.tc_vehicle_mapping or self.tc_vehicle_mapping[layer_name] is None:
                vehicle_group = folium.FeatureGroup(name=layer_name)
                self.tc_vehicle_mapping[layer_name] = vehicle_group
                vehicle_group.add_to(map_obj)  # 지도에 먼저 추가
            
            for i, delivery in enumerate(deliveries):
                try:
                    lat = float(delivery.get('latitude', 0))
                    lng = float(delivery.get('longitude', 0))
                    
                    if lat == 0 or lng == 0:
                        continue
                    
                    # 배송지 정보
                    address = delivery.get('address', '주소 정보 없음')
                    customer = delivery.get('customer_name', '고객명 없음')
                    order_id = delivery.get('order_id', f'ORDER_{i+1}')
                    
                    # 마커 생성
                    marker = folium.CircleMarker(
                        location=[lat, lng],
                        radius=6,
                        popup=folium.Popup(
                            f"""
                            <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 250px;">
                                <h4 style="margin: 0 0 10px 0; color: #333; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                                    📦 배송지 #{i+1}
                                </h4>
                                <div style="font-size: 13px; line-height: 1.5;">
                                    <p style="margin: 5px 0;"><strong>🚛 차량:</strong> {vehicle_id}</p>
                                    <p style="margin: 5px 0;"><strong>🏢 TC:</strong> {tc_name}</p>
                                    <p style="margin: 5px 0;"><strong>👤 고객:</strong> {customer}</p>
                                    <p style="margin: 5px 0;"><strong>📋 주문번호:</strong> {order_id}</p>
                                    <p style="margin: 5px 0;"><strong>📍 주소:</strong> {address}</p>
                                    <p style="margin: 5px 0;"><strong>📍 좌표:</strong> {lat:.6f}, {lng:.6f}</p>
                                </div>
                            </div>
                            """,
                            max_width=300
                        ),
                        color=color,
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=2
                    )
                    
                    # 차량별 레이어 그룹에 추가
                    marker.add_to(self.tc_vehicle_mapping[layer_name])
                        
                except (ValueError, TypeError) as e:
                    logger.warning(f"배송지 마커 생성 실패 {i}: {e}")
                    continue
            
            logger.info(f"배송지 마커 생성 완료: {vehicle_id} ({len(deliveries)}개)")
            
        except Exception as e:
            logger.error(f"배송지 마커 생성 실패 {vehicle_id}: {e}")
    
    def build_tc_vehicle_mapping(self, route_data: Dict[str, Any]) -> None:
        """TC-차량 매핑 구축"""
        try:
            self.tc_vehicle_mapping = {}
            
            for tc_id, tc_data in route_data.items():
                if not isinstance(tc_data, dict):
                    continue
                    
                tc_name = self.tc_data.get(tc_id, tc_id)
                vehicles = tc_data.get('vehicles', {})
                
                for vehicle_id in vehicles.keys():
                    layer_name = f"{tc_name} - {vehicle_id}"
                    self.tc_vehicle_mapping[layer_name] = None  # 나중에 FeatureGroup으로 교체
                    
            logger.info(f"TC-차량 매핑 구축 완료: {len(self.tc_vehicle_mapping)}개 레이어")
            
        except Exception as e:
            logger.error(f"TC-차량 매핑 구축 실패: {e}")
    
    def visualize_routes_with_data(self, route_data: Dict[str, Any], config: Optional[Dict] = None) -> str:
        """실제 데이터로 경로 시각화 (run_all.py 호환)"""
        try:
            logger.info("실제 데이터로 경로 시각화 시작...")
            
            # 지도 초기화 (원본과 동일한 중심점)
            center_lat, center_lng = 37.4788353, 126.6648639
            self.map = folium.Map(
                location=[center_lat, center_lng],
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            # TC-차량 매핑 구축
            self.build_tc_vehicle_mapping(route_data)
            
            # 각 TC별 데이터 처리
            for tc_id, tc_data in route_data.items():
                if not isinstance(tc_data, dict):
                    continue
                    
                tc_name = self.tc_data.get(tc_id, tc_id)
                logger.info(f"처리 중: {tc_name}")
                
                # TC 위치 마커 생성
                depot_info = tc_data.get('depot', {})
                if depot_info:
                    depot_info['name'] = tc_name
                    self.create_depot_marker(self.map, depot_info)
                
                # 차량별 경로 처리
                vehicles = tc_data.get('vehicles', {})
                for vehicle_id, vehicle_data in vehicles.items():
                    if not isinstance(vehicle_data, dict):
                        continue
                    
                    # 경로 데이터
                    route_points = vehicle_data.get('route', [])
                    deliveries = vehicle_data.get('deliveries', [])
                    
                    # 폴리라인 생성
                    if route_points:
                        self.create_polylines(self.map, route_points, vehicle_id, tc_name)
                    
                    # 배송지 마커 생성
                    if deliveries:
                        self.create_delivery_markers(self.map, deliveries, vehicle_id, tc_name)
            
            # UI 컨트롤 추가 (다중 센터 데이터로)
            modified_route_data = {
                'depots': route_data.get('depots', []),  # 모든 센터 정보 전달
                'routes': route_data.get('routes', [])
            }
            self.add_ui_controls(self.map, modified_route_data)
            
            # 레이어 컨트롤 추가
            folium.LayerControl(collapsed=False).add_to(self.map)
            
            # 파일 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output/route_visualization_{timestamp}.html"
            
            # 출력 디렉토리 생성
            Path("output").mkdir(exist_ok=True)
            
            self.map.save(output_file)
            logger.info(f"시각화 완료: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"경로 시각화 실패: {e}")
            raise
    
    def visualize_routes(self, input_file: str = None, output_file: str = None) -> str:
        """경로 시각화 메인 메서드"""
        try:
            # 입력 파일 처리
            if input_file and Path(input_file).exists():
                with open(input_file, 'r', encoding='utf-8') as f:
                    route_data = json.load(f)
                logger.info(f"데이터 로드 완료: {input_file}")
                
                # 실제 데이터 구조 확인
                if 'routes' in route_data:
                    return self.visualize_simple_routes(route_data, output_file)
                else:
                    return self.visualize_routes_with_data(route_data)
            else:
                # 샘플 데이터 생성
                route_data = self.create_sample_data()
                logger.info("샘플 데이터 생성 완료")
                return self.visualize_routes_with_data(route_data)
                
        except Exception as e:
            logger.error(f"경로 시각화 실패: {e}")
            raise

    def visualize_simple_routes(self, route_data: Dict[str, Any], output_file: str = None) -> str:
        """실제 extracted_coordinates.json 데이터 구조에 맞는 시각화 - 다중 센터 지원"""
        try:
            logger.info("실제 데이터 구조로 경로 시각화 시작...")
            
            # 다중 센터 지원: depots 배열이 있으면 사용, 없으면 단일 depot 사용
            depots = route_data.get('depots', [])
            if not depots:
                # 하위 호환성: 단일 depot 사용
                depot = route_data.get('depot', {})
                if depot:
                    depots = [depot]
            
            # 지도 초기화 (첫 번째 depot 위치 기준)
            if depots:
                center_lat = depots[0].get('lat', 37.5665)
                center_lng = depots[0].get('lng', 126.978)
            else:
                center_lat, center_lng = 37.5665, 126.978
            
            self.map = folium.Map(
                location=[center_lat, center_lng],
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            # TC 센터 마커 생성 (모든 depots 순회)
            print(f"=== TC 센터 마커 생성 시작 ===")
            print(f"depots 수: {len(depots)}")
            for depot in depots:
                depot_lat = float(depot.get('lat', 37.5665))
                depot_lng = float(depot.get('lng', 126.978))
                depot_name = depot.get('label', depot.get('name', 'TC 센터'))
                depot_id = depot.get('id', 'UNKNOWN')
                
                print(f"처리 중: {depot_name} (ID: {depot_id}) - 위치: ({depot_lat}, {depot_lng})")
                
                # 해당 depot의 차량 수와 배송지 수 계산 (실제 데이터 구조에 맞게 수정)
                depot_routes = [r for r in route_data.get('routes', []) if r.get('depot_id') == depot_id]
                vehicle_count = len(set(r.get('vehicle', {}).get('id', 'UNKNOWN') for r in depot_routes))
                delivery_count = sum(len(r.get('points', [])) for r in depot_routes)
                
                print(f"  - 차량 수: {vehicle_count}, 배송지 수: {delivery_count}")
                
                # TC 센터 마커 생성
                marker = folium.Marker(
                    location=[depot_lat, depot_lng],
                    popup=folium.Popup(
                        f"""
                        <div style='width: 200px; font-family: Arial;'>
                            <h4 style='margin: 0; color: #2E86AB;'>{depot_name}</h4>
                            <hr style='margin: 5px 0;'>
                            <p style='margin: 2px 0;'><b>차량 수:</b> {vehicle_count}대</p>
                            <p style='margin: 2px 0;'><b>배송지:</b> {delivery_count}개</p>
                            <p style='margin: 2px 0;'><b>위치:</b> {depot_lat:.4f}, {depot_lng:.4f}</p>
                        </div>
                        """,
                        max_width=250
                    ),
                    tooltip=f"{depot_name} (차량: {vehicle_count}대, 배송지: {delivery_count}개)",
                    icon=folium.Icon(
                        color='red',
                        icon='home',
                        prefix='fa'
                    )
                )
                marker.add_to(self.map)
                print(f"  - 마커 생성 완료: {depot_name}")
            
            print(f"=== TC 센터 마커 생성 완료 ===")
            
            # 경로별 처리 (실제 데이터 구조에 맞게 수정)
            routes = route_data.get('routes', [])
            vehicle_groups = {}
            
            print(f"=== 경로 처리 시작 ===")
            print(f"총 경로 수: {len(routes)}")
            
            for route in routes:
                # 실제 데이터 구조에 맞게 필드 추출
                vehicle_info = route.get('vehicle', {})
                vehicle_id = vehicle_info.get('id', 'UNKNOWN')
                vehicle_name = vehicle_info.get('name', vehicle_id)
                depot_id = route.get('depot_id', 'UNKNOWN')
                depot_name = route.get('depot_name', depot_id)
                points = route.get('points', [])  # coordinates 대신 points 사용
                
                print(f"처리 중: {depot_name} - {vehicle_name} (ID: {vehicle_id}) - 배송지: {len(points)}개")
                
                # 차량별 색상 할당
                if vehicle_id not in self.vehicle_colors:
                    color_index = len(self.vehicle_colors) % len(self.color_palette)
                    self.vehicle_colors[vehicle_id] = self.color_palette[color_index]
                
                color = self.vehicle_colors[vehicle_id]
                
                # 차량별 레이어 그룹 생성
                layer_name = f"{depot_name} - {vehicle_name}"
                if layer_name not in vehicle_groups:
                    vehicle_group = folium.FeatureGroup(name=layer_name)
                    vehicle_group.add_to(self.map)
                    vehicle_groups[layer_name] = vehicle_group
                    self.tc_vehicle_mapping[layer_name] = vehicle_group
                
                # 경로 좌표 추출 (points에서 latitude, longitude 사용)
                route_coords = []
                for point_data in points:
                    point = point_data.get('point', {})
                    lat = point.get('latitude', 0)
                    lng = point.get('longitude', 0)
                    if lat != 0 and lng != 0:
                        route_coords.append([lat, lng])
                
                # 폴리라인 생성 (실제 좌표가 있는 경우)
                if len(route_coords) >= 2:
                    total_distance = route.get('total_distance', 0)
                    total_duration = route.get('total_duration', 0)
                    
                    folium.PolyLine(
                        locations=route_coords,
                        color=color,
                        weight=4,
                        opacity=0.8,
                        popup=folium.Popup(
                            f"""
                            <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 200px;">
                                <h4 style="margin: 0 0 10px 0; color: #333; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                            🚛 {depot_name} - {vehicle_name}
                                </h4>
                                <div style="font-size: 13px; line-height: 1.5;">
                            <p style="margin: 5px 0;"><strong>📦 배송지:</strong> {len(points)}개</p>
                                    <p style="margin: 5px 0;"><strong>📏 총 거리:</strong> {total_distance/1000:.1f}km</p>
                                    <p style="margin: 5px 0;"><strong>⏱️ 총 시간:</strong> {total_duration//3600}시간 {(total_duration%3600)//60}분</p>
                                </div>
                            </div>
                            """,
                            max_width=250
                        )
                    ).add_to(vehicle_groups[layer_name])
                    print(f"  - 폴리라인 생성 완료: {len(route_coords)}개 좌표")
                
                # 배송지 마커 생성 (points 데이터 사용)
                for i, point_data in enumerate(points):
                    point = point_data.get('point', {})
                    lat = point.get('latitude', 0)
                    lng = point.get('longitude', 0)
                    address1 = point.get('address1', '주소 정보 없음')
                    address2 = point.get('address2', '')
                    full_address = f"{address1} {address2}".strip()
                    
                    if lat != 0 and lng != 0:
                        folium.CircleMarker(
                            location=[lat, lng],
                            radius=6,
                            popup=folium.Popup(
                                f"""
                                <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 250px;">
                                    <h4 style="margin: 0 0 10px 0; color: #333; border-bottom: 2px solid {color}; padding-bottom: 5px;">
                                        📦 배송지 #{i+1}
                                    </h4>
                                    <div style="font-size: 13px; line-height: 1.5;">
                                        <p style="margin: 5px 0;"><strong>🚛 차량:</strong> {vehicle_name}</p>
                                        <p style="margin: 5px 0;"><strong>🏢 TC:</strong> {depot_name}</p>
                                        <p style="margin: 5px 0;"><strong>📍 주소:</strong> {full_address}</p>
                                        <p style="margin: 5px 0;"><strong>📍 좌표:</strong> {lat:.6f}, {lng:.6f}</p>
                                    </div>
                                </div>
                                """,
                                max_width=300
                            ),
                            color=color,
                            fillColor=color,
                            fillOpacity=0.7,
                            weight=2
                        ).add_to(vehicle_groups[layer_name])
            
                print(f"  - 배송지 마커 생성 완료: {len(points)}개")
            
            print(f"=== 경로 처리 완료 ===")
            
            # UI 컨트롤 추가 (수정된 데이터 구조로)
            modified_route_data = {
                'depots': depots,  # 모든 센터 정보 전달
                'routes': routes
            }
            self.add_ui_controls(self.map, modified_route_data)
            
            # 레이어 컨트롤 추가
            folium.LayerControl(collapsed=False).add_to(self.map)
            
            # 파일 저장
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"output/route_visualization_{timestamp}.html"
            
            Path("output").mkdir(exist_ok=True)
            self.map.save(output_file)
            logger.info(f"시각화 완료: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"실제 데이터 시각화 실패: {e}")
            raise
    
    def create_sample_data(self) -> Dict[str, Any]:
        """샘플 데이터 생성 (테스트용)"""
        try:
            logger.info("샘플 데이터 생성 시작...")
            
            # 기본 TC 위치들 (실제 좌표)
            tc_locations = {
                "incheon_center": {"latitude": 37.4788353, "longitude": 126.6648639},
                "icheon_center": {"latitude": 37.2792, "longitude": 127.4425},
                "hwaseong_center": {"latitude": 37.1967, "longitude": 126.8169},
                "hanam_center": {"latitude": 37.5394, "longitude": 127.2067}
            }
            
            sample_data = {}
            
            for tc_id, location in tc_locations.items():
                tc_name = self.tc_data[tc_id]
                
                # TC 데이터 구조
                tc_data = {
                    "depot": {
                        "latitude": location["latitude"],
                        "longitude": location["longitude"],
                        "name": tc_name
                    },
                    "vehicles": {}
                }
                
                # 차량 2대씩 생성
                for i in range(1, 3):
                    vehicle_id = f"차량 {i}"
                    
                    # 랜덤 배송지 생성 (TC 주변)
                    deliveries = []
                    route_points = [location]  # TC에서 시작
                    
                    for j in range(3):  # 배송지 3개씩
                        # TC 주변 랜덤 좌표 생성
                        lat_offset = np.random.uniform(-0.05, 0.05)
                        lng_offset = np.random.uniform(-0.05, 0.05)
                        
                        delivery_lat = location["latitude"] + lat_offset
                        delivery_lng = location["longitude"] + lng_offset
                        
                        delivery = {
                            "latitude": delivery_lat,
                            "longitude": delivery_lng,
                            "address": f"{tc_name} 배송지 {j+1}",
                            "customer_name": f"고객 {j+1}",
                            "order_id": f"ORD_{tc_id}_{i}_{j+1}"
                        }
                        
                        deliveries.append(delivery)
                        route_points.append({
                            "latitude": delivery_lat,
                            "longitude": delivery_lng
                        })
                    
                    # TC로 복귀
                    route_points.append(location)
                    
                    tc_data["vehicles"][vehicle_id] = {
                        "route": route_points,
                        "deliveries": deliveries
                    }
                
                sample_data[tc_id] = tc_data
            
            logger.info(f"샘플 데이터 생성 완료: {len(sample_data)}개 TC")
            return sample_data
            
        except Exception as e:
            logger.error(f"샘플 데이터 생성 실패: {e}")
            return {}


# 사용 예시
if __name__ == "__main__":
    import sys
    
    # 시각화 서비스 초기화
    visualizer = RouteVisualizerService()
    
    # 경로 시각화 실행
    try:
        # 명령행 인자에서 파일 경로 확인
        input_file = None
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
            
        output_file = visualizer.visualize_routes(input_file)
        print(f"✅ 시각화 완료: {output_file}")
        
        # 브라우저에서 열기 (선택사항)
        import webbrowser
        webbrowser.open(f"file://{Path(output_file).absolute()}")
        
    except Exception as e:
        print(f"❌ 시각화 실패: {e}") 