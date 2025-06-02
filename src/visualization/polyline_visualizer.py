#!/usr/bin/env python3
"""
폴리라인 시각화 모듈

주요 기능:
- OSRM API를 통한 실제 도로 경로 조회
- 폴리라인 디코딩 및 시각화
- 차량별 색상 구분
- 경로 최적화 표시
"""

import requests
import polyline
import logging
from typing import List, Dict, Tuple, Optional
import json
import time
from dataclasses import dataclass

from src.model.delivery_point import DeliveryPoint
from src.model.route import Route

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """시각화 설정 클래스"""
    # OSRM 설정
    osrm_url: str = "http://router.project-osrm.org"
    osrm_timeout: int = 30
    osrm_retries: int = 3
    
    # 폴리라인 스타일
    polyline_weight: int = 5
    polyline_opacity: float = 0.9
    polyline_hover_weight: int = 7
    polyline_hover_opacity: float = 1.0
    
    # 마커 설정
    delivery_marker_radius: int = 4
    delivery_marker_opacity: float = 0.8
    depot_marker_size: int = 24
    
    # 지도 설정
    default_center_lat: float = 37.5665
    default_center_lng: float = 126.9780
    default_zoom: int = 10
    
    # 색상 팔레트
    colors: List[str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = [
                '#CC0000',  # 진한 빨강
                '#006600',  # 진한 초록
                '#0000CC',  # 진한 파랑
                '#FF8800',  # 진한 주황
                '#CC00CC',  # 진한 자주
                '#008888',  # 진한 청록
                '#990000',  # 진한 적갈색
                '#4B0082',  # 인디고
                '#228B22',  # 포레스트 그린
                '#8B0000',  # 다크 레드
                '#2F4F4F',  # 다크 슬레이트 그레이
                '#191970',  # 미드나잇 블루
                '#8B4513',  # 새들 브라운
                '#556B2F',  # 다크 올리브 그린
                '#800000'   # 마룬
            ]

class PolylineVisualizer:
    """폴리라인 시각화 클래스"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
    def get_route_polylines(self, routes: List[Route]) -> Dict[str, any]:
        """모든 경로의 폴리라인 데이터 생성"""
        logger.info(f"🗺️ {len(routes)}개 경로의 폴리라인 생성 중...")
        
        polyline_data = {
            'routes': [],
            'depots': [],  # 모든 depot 정보 저장
            'total_distance': 0,
            'total_duration': 0,
            'error_count': 0
        }
        
        # depot 중복 제거를 위한 집합
        depot_locations = set()
        
        for i, route in enumerate(routes):
            try:
                route_polyline = self._get_single_route_polyline(route, i)
                if route_polyline:
                    polyline_data['routes'].append(route_polyline)
                    polyline_data['total_distance'] += route_polyline.get('distance', 0)
                    polyline_data['total_duration'] += route_polyline.get('duration', 0)
                    
                    # depot 정보 수집 (중복 제거)
                    depot_info = route_polyline.get('depot')
                    if depot_info:
                        depot_key = (depot_info['lat'], depot_info['lng'])
                        if depot_key not in depot_locations:
                            depot_locations.add(depot_key)
                            polyline_data['depots'].append(depot_info)
                else:
                    polyline_data['error_count'] += 1
                    
            except Exception as e:
                logger.error(f"경로 {i} 폴리라인 생성 실패: {e}")
                polyline_data['error_count'] += 1
                
        logger.info(f"✅ 폴리라인 생성 완료: {len(polyline_data['routes'])}개 성공, {polyline_data['error_count']}개 실패")
        logger.info(f"📍 발견된 TC센터: {len(polyline_data['depots'])}개")
        
        return polyline_data
    
    def _get_single_route_polyline(self, route: Route, route_index: int) -> Optional[Dict]:
        """단일 경로의 폴리라인 데이터 생성"""
        if not route.points:
            logger.warning(f"경로 {route_index}: 배송지점이 없음")
            return None
            
        # 경로 좌표 준비 (depot → 배송지들 → depot)
        coordinates = []
        
        # 시작점 (depot)
        depot = route.depot
        coordinates.append([depot.longitude, depot.latitude])
        
        # 배송지점들 - RoutePoint에서 실제 DeliveryPoint 추출
        delivery_points = []
        for route_point in route.points:
            point = route_point.point  # RoutePoint.point가 실제 DeliveryPoint
            coordinates.append([point.longitude, point.latitude])
            
            # 배송지점 정보 수집 (datetime 객체 처리)
            delivery_points.append({
                'lat': point.latitude,
                'lng': point.longitude,
                'name': getattr(point, 'name', f'배송지 {len(delivery_points) + 1}'),
                'address': getattr(point, 'address', '주소 정보 없음'),
                'arrival_time': str(route_point.arrival_time) if hasattr(route_point, 'arrival_time') and route_point.arrival_time else None,
                'departure_time': str(route_point.departure_time) if hasattr(route_point, 'departure_time') and route_point.departure_time else None
            })
            
        # 종료점 (depot으로 복귀)
        coordinates.append([depot.longitude, depot.latitude])
        
        # OSRM API 호출
        osrm_response = self._call_osrm_api(coordinates)
        if not osrm_response:
            logger.warning(f"경로 {route_index}: OSRM API 호출 실패")
            return None
            
        try:
            # 폴리라인 디코딩
            encoded_polyline = osrm_response['geometry']
            decoded_polyline = polyline.decode(encoded_polyline)
            
            # 거리와 시간 정보
            distance = osrm_response['distance']  # 미터
            duration = osrm_response['duration']  # 초
            
            # 색상 선택
            color = self.config.colors[route_index % len(self.config.colors)]
            
            # 차량 ID 추출 (안전하게)
            vehicle_id = getattr(route.vehicle, 'id', f'V{route_index + 1}')
            
            return {
                'polyline': decoded_polyline,
                'color': color,
                'vehicle_id': vehicle_id,
                'distance': round(distance / 1000, 1),  # km로 변환
                'duration': round(duration / 60, 1),    # 분으로 변환
                'delivery_count': len(delivery_points),
                'delivery_points': delivery_points,
                'depot': {
                    'lat': depot.latitude,
                    'lng': depot.longitude,
                    'name': getattr(depot, 'name', '물류센터')
                }
            }
            
        except Exception as e:
            logger.error(f"경로 {route_index} 폴리라인 처리 오류: {e}")
            return None
    
    def _call_osrm_api(self, coordinates: List[List[float]], retries: int = 3) -> Optional[Dict]:
        """OSRM API 호출하여 경로 정보 가져오기"""
        if len(coordinates) < 2:
            logger.warning("좌표가 2개 미만입니다.")
            return None
            
        # 좌표를 문자열로 변환 (longitude,latitude 형식)
        coord_str = ";".join([f"{coord[0]},{coord[1]}" for coord in coordinates])
        
        # OSRM API URL 구성
        url = f"{self.config.osrm_url}/route/v1/driving/{coord_str}"
        params = {
            'overview': 'full',
            'geometries': 'polyline',
            'steps': 'false'
        }
        
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=self.config.osrm_timeout)
                response.raise_for_status()
                
                data = response.json()
                
                # OSRM API 응답 구조 확인
                if 'routes' not in data or not data['routes']:
                    logger.warning(f"OSRM API 응답에 경로 정보 없음: {data}")
                    return None
                
                # 첫 번째 경로 반환
                route_data = data['routes'][0]
                
                return {
                    'geometry': route_data.get('geometry', ''),
                    'distance': route_data.get('distance', 0),
                    'duration': route_data.get('duration', 0),
                    'legs': route_data.get('legs', [])
                }
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"OSRM API 호출 실패 (시도 {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1:
                    logger.error(f"OSRM API 호출 최종 실패: {e}")
                    return None
                time.sleep(1)  # 재시도 전 대기
                
            except Exception as e:
                logger.error(f"OSRM API 처리 오류: {e}")
                return None
        
        return None
    
    def generate_polyline_javascript(self, polyline_data: Dict) -> str:
        """새로운 인터랙티브 UI를 위한 JavaScript 코드 생성"""
        config = self.config
        
        js_code = f"""
        // 전역 변수
        let map;
        let allPolylines = {{}};
        let allMarkers = {{}};
        let selectedCenters = new Set();
        let selectedVehicles = new Set();
        let centerData = {{}};
        let vehicleData = {{}};
        
        // 폴리라인 표시 함수
        function displayPolylines(mapInstance, polylineData) {{
            map = mapInstance;
            
            // 센터별 데이터 구성
            polylineData.routes.forEach(function(routeData, index) {{
                const depotKey = routeData.depot.name || 'TC센터';
                
                if (!centerData[depotKey]) {{
                    centerData[depotKey] = {{
                        name: depotKey,
                        vehicles: [],
                        totalDeliveries: 0,
                        totalDistance: 0,
                        totalDuration: 0,
                        depot: routeData.depot
                    }};
                }}
                
                // 고유한 차량 ID 생성 (센터명_차량ID_인덱스)
                const uniqueVehicleId = depotKey + "_" + routeData.vehicle_id + "_" + index;
                
                // 차량 데이터 추가
                const vehicleInfo = {{
                    id: routeData.vehicle_id,
                    uniqueId: uniqueVehicleId,
                    centerName: depotKey,
                    color: routeData.color,
                    deliveryCount: routeData.delivery_count,
                    distance: routeData.distance,
                    duration: routeData.duration,
                    routeData: routeData
                }};
                
                centerData[depotKey].vehicles.push(vehicleInfo);
                centerData[depotKey].totalDeliveries += routeData.delivery_count;
                centerData[depotKey].totalDistance += routeData.distance;
                centerData[depotKey].totalDuration += routeData.duration;
                
                vehicleData[uniqueVehicleId] = vehicleInfo;
                
                // 폴리라인 생성 (초기에는 숨김)
                const polyline = L.polyline(routeData.polyline, {{
                    color: routeData.color,
                    weight: {config.polyline_weight},
                    opacity: {config.polyline_opacity},
                    className: 'route-polyline vehicle-' + uniqueVehicleId
                }});
                
                polyline.setStyle({{opacity: 0}}); // 초기에는 숨김
                polyline.addTo(map);
                
                // 폴리라인 클릭 이벤트
                polyline.on('click', function(e) {{
                    showVehicleInfo(routeData);
                }});
                
                // 폴리라인 호버 효과
                polyline.on('mouseover', function(e) {{
                    if (selectedVehicles.has(uniqueVehicleId)) {{
                        this.setStyle({{
                            weight: {config.polyline_hover_weight},
                            opacity: {config.polyline_hover_opacity}
                        }});
                    }}
                }});
                
                polyline.on('mouseout', function(e) {{
                    if (selectedVehicles.has(uniqueVehicleId)) {{
                        this.setStyle({{
                            weight: {config.polyline_weight},
                            opacity: {config.polyline_opacity}
                        }});
                    }}
                }});
                
                allPolylines[uniqueVehicleId] = polyline;
                
                // 배송지점 마커 추가 (초기에는 숨김)
                const markers = [];
                if (routeData.delivery_points && routeData.delivery_points.length > 0) {{
                    routeData.delivery_points.forEach(function(point, pointIndex) {{
                        const marker = L.circleMarker([point.lat, point.lng], {{
                            radius: {config.delivery_marker_radius},
                            fillColor: routeData.color,
                            color: '#ffffff',
                            weight: 1,
                            opacity: 0, // 초기에는 숨김
                            fillOpacity: 0, // 초기에는 숨김
                            className: 'delivery-marker vehicle-' + uniqueVehicleId
                        }}).addTo(map);
                        
                        marker.on('click', function(e) {{
                            showDeliveryInfo(point, routeData, pointIndex);
                        }});
                        
                        markers.push(marker);
                    }});
                }}
                
                allMarkers[uniqueVehicleId] = markers;
            }});
            
            // Depot 마커 추가
            if (polylineData.depots && polylineData.depots.length > 0) {{
                polylineData.depots.forEach(function(depot, depotIndex) {{
                    const depotMarker = L.marker([depot.lat, depot.lng], {{
                        icon: L.divIcon({{
                            className: 'depot-marker',
                            html: '<div style="background: #ff4444; color: white; border-radius: 50%; width: {config.depot_marker_size}px; height: {config.depot_marker_size}px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 12px; border: 2px solid white;">🏢</div>',
                            iconSize: [{config.depot_marker_size}, {config.depot_marker_size}],
                            iconAnchor: [{config.depot_marker_size//2}, {config.depot_marker_size//2}]
                        }})
                    }}).addTo(map);
                    
                    depotMarker.on('click', function(e) {{
                        showDepotInfo(depot);
                    }});
                }});
            }}
            
            // UI 초기화
            initializeCenterPanel();
            initializeVehiclePanel();
        }}
        
        // 센터 패널 초기화
        function initializeCenterPanel() {{
            const centerPanel = document.getElementById('center-panel');
            let html = '<div class="panel-header"><button id="center-toggle-button" class="toggle-button" onclick="toggleAllCenters()">전체 선택</button><h3>📍 TC 센터</h3></div>';
            
            Object.keys(centerData).forEach(function(centerName) {{
                const center = centerData[centerName];
                const avgDuration = center.vehicles.length > 0 ? (center.totalDuration / center.vehicles.length).toFixed(1) : '0';
                
                html += `
                    <div class="center-item" data-center="${{centerName}}">
                        <div class="center-checkbox">
                            <input type="checkbox" id="center-${{centerName}}" onchange="toggleCenter('${{centerName}}')">
                            <label for="center-${{centerName}}">
                                <div class="center-name">${{centerName}}</div>
                                <div class="center-stats">
                                    <span>🚛 ${{center.vehicles.length}}대</span>
                                    <span>📦 ${{center.totalDeliveries}}개</span>
                                    <span>⏱️ ${{avgDuration}}분</span>
                                </div>
                            </label>
                        </div>
                    </div>
                `;
            }});
            
            centerPanel.innerHTML = html;
        }}
        
        // 차량 패널 초기화
        function initializeVehiclePanel() {{
            const vehiclePanel = document.getElementById('vehicle-panel');
            vehiclePanel.innerHTML = '<div class="panel-header"><button id="vehicle-toggle-button" class="toggle-button" onclick="toggleAllVehicles()">전체 선택</button><h3>🚛 차량</h3></div><div class="no-selection">센터를 선택해주세요</div>';
        }}
        
        // 센터 버튼 상태 업데이트
        function updateCenterButton() {{
            const button = document.getElementById('center-toggle-button');
            const totalCenters = Object.keys(centerData).length;
            const selectedCount = selectedCenters.size;
            
            if (selectedCount === 0) {{
                button.textContent = '전체 선택';
                button.className = 'toggle-button';
            }} else if (selectedCount === totalCenters) {{
                button.textContent = '전체 해제';
                button.className = 'toggle-button active';
            }} else {{
                button.textContent = `${{selectedCount}}개 선택됨`;
                button.className = 'toggle-button partial';
            }}
        }}
        
        // 차량 버튼 상태 업데이트
        function updateVehicleButton() {{
            const button = document.getElementById('vehicle-toggle-button');
            if (!button) return;
            
            // 현재 표시된 차량들의 총 개수 계산
            let totalVehicles = 0;
            selectedCenters.forEach(function(centerName) {{
                totalVehicles += centerData[centerName].vehicles.length;
            }});
            
            const selectedCount = selectedVehicles.size;
            
            if (selectedCount === 0) {{
                button.textContent = '전체 선택';
                button.className = 'toggle-button';
            }} else if (selectedCount === totalVehicles && totalVehicles > 0) {{
                button.textContent = '전체 해제';
                button.className = 'toggle-button active';
            }} else {{
                button.textContent = `${{selectedCount}}개 선택됨`;
                button.className = 'toggle-button partial';
            }}
        }}
        
        // 모든 센터 토글
        function toggleAllCenters() {{
            const totalCenters = Object.keys(centerData).length;
            const selectedCount = selectedCenters.size;
            
            if (selectedCount === totalCenters) {{
                // 전체 해제
                clearAllCenters();
            }} else {{
                // 전체 선택
                selectedCenters.clear();
                Object.keys(centerData).forEach(function(centerName) {{
                    selectedCenters.add(centerName);
                    document.getElementById('center-' + centerName).checked = true;
                }});
                updateVehiclePanel();
                updateCenterButton();
            }}
        }}
        
        // 모든 차량 토글
        function toggleAllVehicles() {{
            // 현재 표시된 차량들의 총 개수 계산
            let totalVehicles = 0;
            const availableVehicles = [];
            selectedCenters.forEach(function(centerName) {{
                centerData[centerName].vehicles.forEach(function(vehicle) {{
                    totalVehicles++;
                    availableVehicles.push(vehicle.uniqueId);
                }});
            }});
            
            const selectedCount = selectedVehicles.size;
            
            if (selectedCount === totalVehicles && totalVehicles > 0) {{
                // 전체 해제
                clearAllVehicles();
            }} else {{
                // 전체 선택
                selectedVehicles.clear();
                availableVehicles.forEach(function(uniqueVehicleId) {{
                    selectedVehicles.add(uniqueVehicleId);
                    showVehicle(uniqueVehicleId);
                    const checkbox = document.getElementById('vehicle-' + uniqueVehicleId);
                    if (checkbox) checkbox.checked = true;
                }});
                updateVehicleButton();
            }}
        }}
        
        // 센터 토글
        function toggleCenter(centerName) {{
            const checkbox = document.getElementById('center-' + centerName);
            
            if (checkbox.checked) {{
                selectedCenters.add(centerName);
            }} else {{
                selectedCenters.delete(centerName);
                // 해당 센터의 모든 차량 선택 해제
                centerData[centerName].vehicles.forEach(function(vehicle) {{
                    selectedVehicles.delete(vehicle.uniqueId);
                    hideVehicle(vehicle.uniqueId);
                }});
            }}
            
            updateVehiclePanel();
            updateCenterButton();
            updateVehicleButton();
        }}
        
        // 차량 패널 업데이트
        function updateVehiclePanel() {{
            const vehiclePanel = document.getElementById('vehicle-panel');
            let html = '<div class="panel-header"><button id="vehicle-toggle-button" class="toggle-button" onclick="toggleAllVehicles()">전체 선택</button><h3>🚛 차량</h3></div>';
            
            if (selectedCenters.size === 0) {{
                html += '<div class="no-selection">센터를 선택해주세요</div>';
            }} else {{
                selectedCenters.forEach(function(centerName) {{
                    const center = centerData[centerName];
                    center.vehicles.forEach(function(vehicle) {{
                        const isSelected = selectedVehicles.has(vehicle.uniqueId);
                        html += `
                            <div class="vehicle-item" data-vehicle="${{vehicle.uniqueId}}">
                                <div class="vehicle-checkbox">
                                    <input type="checkbox" id="vehicle-${{vehicle.uniqueId}}" ${{isSelected ? 'checked' : ''}} onchange="toggleVehicle('${{vehicle.uniqueId}}')">
                                    <label for="vehicle-${{vehicle.uniqueId}}">
                                        <div class="vehicle-header">
                                            <span class="vehicle-color" style="background-color: ${{vehicle.color}}"></span>
                                            <span class="vehicle-name">${{vehicle.id}}</span>
                                        </div>
                                        <div class="vehicle-stats">
                                            <span>📦 ${{vehicle.deliveryCount}}개</span>
                                            <span>📏 ${{vehicle.distance}}km</span>
                                        </div>
                                    </label>
                                </div>
                            </div>
                        `;
                    }});
                }});
            }}
            
            vehiclePanel.innerHTML = html;
            updateVehicleButton();
        }}
        
        // 차량 토글
        function toggleVehicle(uniqueVehicleId) {{
            const checkbox = document.getElementById('vehicle-' + uniqueVehicleId);
            
            if (checkbox.checked) {{
                selectedVehicles.add(uniqueVehicleId);
                showVehicle(uniqueVehicleId);
            }} else {{
                selectedVehicles.delete(uniqueVehicleId);
                hideVehicle(uniqueVehicleId);
            }}
            
            updateVehicleButton();
        }}
        
        // 차량 표시
        function showVehicle(uniqueVehicleId) {{
            if (allPolylines[uniqueVehicleId]) {{
                allPolylines[uniqueVehicleId].setStyle({{opacity: {config.polyline_opacity}}});
            }}
            
            if (allMarkers[uniqueVehicleId]) {{
                allMarkers[uniqueVehicleId].forEach(function(marker) {{
                    marker.setStyle({{
                        opacity: 1,
                        fillOpacity: {config.delivery_marker_opacity}
                    }});
                }});
            }}
        }}
        
        // 차량 숨김
        function hideVehicle(uniqueVehicleId) {{
            if (allPolylines[uniqueVehicleId]) {{
                allPolylines[uniqueVehicleId].setStyle({{opacity: 0}});
            }}
            
            if (allMarkers[uniqueVehicleId]) {{
                allMarkers[uniqueVehicleId].forEach(function(marker) {{
                    marker.setStyle({{
                        opacity: 0,
                        fillOpacity: 0
                    }});
                }});
            }}
        }}
        
        // 모든 센터 해제
        function clearAllCenters() {{
            selectedCenters.clear();
            selectedVehicles.clear();
            
            // 모든 체크박스 해제
            document.querySelectorAll('#center-panel input[type="checkbox"]').forEach(function(checkbox) {{
                checkbox.checked = false;
            }});
            
            // 모든 차량 숨김
            Object.keys(allPolylines).forEach(function(uniqueVehicleId) {{
                hideVehicle(uniqueVehicleId);
            }});
            
            updateVehiclePanel();
            updateCenterButton();
            updateVehicleButton();
        }}
        
        // 모든 차량 해제
        function clearAllVehicles() {{
            selectedVehicles.clear();
            
            // 모든 차량 체크박스 해제
            document.querySelectorAll('#vehicle-panel input[type="checkbox"]').forEach(function(checkbox) {{
                checkbox.checked = false;
            }});
            
            // 모든 차량 숨김
            Object.keys(allPolylines).forEach(function(uniqueVehicleId) {{
                hideVehicle(uniqueVehicleId);
            }});
            
            updateVehicleButton();
        }}
        
        // 정보 표시 함수들
        function showVehicleInfo(routeData) {{
            const popupContent = `
                <div class="route-popup">
                    <h4>🚛 차량 ${{routeData.vehicle_id}}</h4>
                    <p><strong>📍 배송지:</strong> ${{routeData.delivery_count}}개</p>
                    <p><strong>📏 거리:</strong> ${{routeData.distance}}km</p>
                    <p><strong>⏱️ 시간:</strong> ${{routeData.duration}}분</p>
                    <p><strong>🏢 센터:</strong> ${{routeData.depot.name || 'Unknown'}}</p>
                </div>
            `;
            
            L.popup()
                .setLatLng(routeData.polyline[0])
                .setContent(popupContent)
                .openOn(map);
        }}
        
        function showDeliveryInfo(point, routeData, pointIndex) {{
            const popupContent = `
                <div class="route-popup">
                    <h4>📦 배송지 ${{pointIndex + 1}}</h4>
                    <p><strong>🚛 차량:</strong> ${{routeData.vehicle_id}}</p>
                    <p><strong>📍 주소:</strong> ${{point.address || '주소 정보 없음'}}</p>
                    <p><strong>🕐 도착:</strong> ${{point.arrival_time || 'N/A'}}</p>
                    <p><strong>⏰ 출발:</strong> ${{point.departure_time || 'N/A'}}</p>
                </div>
            `;
            
            L.popup()
                .setLatLng([point.lat, point.lng])
                .setContent(popupContent)
                .openOn(map);
        }}
        
        function showDepotInfo(depot) {{
            const depotName = depot.name || 'TC센터';
            const center = centerData[depotName];
            
            const popupContent = `
                <div class="route-popup">
                    <h4>🏢 ${{depotName}}</h4>
                    <p><strong>🚛 배정 차량:</strong> ${{center ? center.vehicles.length : 0}}대</p>
                    <p><strong>📦 총 배송지:</strong> ${{center ? center.totalDeliveries : 0}}개</p>
                    <p><strong>📍 위치:</strong> ${{depot.lat.toFixed(4)}}, ${{depot.lng.toFixed(4)}}</p>
                </div>
            `;
            
            L.popup()
                .setLatLng([depot.lat, depot.lng])
                .setContent(popupContent)
                .openOn(map);
        }}
        """
        
        # 폴리라인 데이터를 JavaScript 변수로 추가
        js_code += f"""
        
        // 폴리라인 데이터
        const polylineData = {json.dumps(polyline_data, ensure_ascii=False, indent=2)};
        """
        
        return js_code
    
    def generate_html_template(self, polyline_data: Dict, js_code: str) -> str:
        """새로운 3패널 레이아웃 HTML 템플릿 생성"""
        config = self.config
        
        # 지도 중심점 계산 (모든 depot의 중심)
        if polyline_data.get('depots'):
            center_lat = sum(depot['lat'] for depot in polyline_data['depots']) / len(polyline_data['depots'])
            center_lng = sum(depot['lng'] for depot in polyline_data['depots']) / len(polyline_data['depots'])
        else:
            center_lat = config.default_center_lat
            center_lng = config.default_center_lng
        
        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🗺️ TMS 인터랙티브 경로 시각화</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            height: 100vh;
            overflow: hidden;
        }}
        
        .container {{
            display: flex;
            height: 100vh;
        }}
        
        /* 좌측 센터 패널 */
        .left-panel {{
            width: 300px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            overflow-y: auto;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }}
        
        /* 지도 영역 */
        .map-container {{
            flex: 1;
            position: relative;
        }}
        
        #map {{
            height: 100%;
            width: 100%;
        }}
        
        /* 우측 차량 패널 */
        .right-panel {{
            width: 320px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            overflow-y: auto;
            box-shadow: -2px 0 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }}
        
        /* 패널 공통 스타일 */
        .panel-header {{
            padding: 15px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.2);
            position: relative;
        }}
        
        .panel-header h3 {{
            font-size: 14px;
            font-weight: 600;
            text-align: center;
            margin: 0;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }}
        
        .toggle-button {{
            position: absolute;
            top: 10px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 6px 10px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            min-width: 70px;
            text-align: center;
        }}
        
        .toggle-button:hover {{
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}
        
        .toggle-button.active {{
            background: rgba(255, 100, 100, 0.8);
            border-color: rgba(255, 100, 100, 0.9);
        }}
        
        .toggle-button.partial {{
            background: rgba(255, 200, 100, 0.8);
            border-color: rgba(255, 200, 100, 0.9);
        }}
        
        .left-panel .toggle-button {{
            left: 10px;
        }}
        
        .right-panel .toggle-button {{
            right: 10px;
        }}
        
        /* 센터 아이템 스타일 */
        .center-item {{
            margin: 10px 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .center-item:hover {{
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        }}
        
        .center-checkbox {{
            padding: 0;
        }}
        
        .center-checkbox input[type="checkbox"] {{
            display: none;
        }}
        
        .center-checkbox label {{
            display: block;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .center-checkbox input[type="checkbox"]:checked + label {{
            background: rgba(255, 255, 255, 0.2);
        }}
        
        .center-name {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 8px;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }}
        
        .center-stats {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            opacity: 0.9;
        }}
        
        .center-stats span {{
            background: rgba(255, 255, 255, 0.2);
            padding: 4px 8px;
            border-radius: 12px;
            backdrop-filter: blur(5px);
        }}
        
        /* 차량 아이템 스타일 */
        .vehicle-item {{
            margin: 10px 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .vehicle-item:hover {{
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        }}
        
        .vehicle-checkbox {{
            padding: 0;
        }}
        
        .vehicle-checkbox input[type="checkbox"] {{
            display: none;
        }}
        
        .vehicle-checkbox label {{
            display: block;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .vehicle-checkbox input[type="checkbox"]:checked + label {{
            background: rgba(255, 255, 255, 0.2);
        }}
        
        .vehicle-header {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .vehicle-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .vehicle-name {{
            font-size: 16px;
            font-weight: 600;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }}
        
        .vehicle-stats {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            opacity: 0.9;
        }}
        
        .vehicle-stats span {{
            background: rgba(255, 255, 255, 0.2);
            padding: 4px 8px;
            border-radius: 12px;
            backdrop-filter: blur(5px);
        }}
        
        /* 선택 없음 메시지 */
        .no-selection {{
            text-align: center;
            padding: 40px 20px;
            opacity: 0.7;
            font-style: italic;
            font-size: 14px;
        }}
        
        /* 팝업 스타일 */
        .route-popup {{
            font-family: 'Segoe UI', sans-serif;
            font-size: 13px;
            line-height: 1.5;
            max-width: 280px;
        }}
        
        .route-popup h4 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 16px;
            font-weight: 600;
            border-bottom: 2px solid #3498db;
            padding-bottom: 6px;
        }}
        
        .route-popup p {{
            margin: 6px 0;
            color: #34495e;
        }}
        
        .route-popup strong {{
            color: #2c3e50;
        }}
        
        /* 스크롤바 스타일 */
        .left-panel::-webkit-scrollbar,
        .right-panel::-webkit-scrollbar {{
            width: 6px;
        }}
        
        .left-panel::-webkit-scrollbar-track,
        .right-panel::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.1);
        }}
        
        .left-panel::-webkit-scrollbar-thumb,
        .right-panel::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.3);
            border-radius: 3px;
        }}
        
        .left-panel::-webkit-scrollbar-thumb:hover,
        .right-panel::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.5);
        }}
        
        /* 반응형 디자인 */
        @media (max-width: 1200px) {{
            .left-panel, .right-panel {{
                width: 280px;
            }}
        }}
        
        @media (max-width: 768px) {{
            .container {{
                flex-direction: column;
            }}
            
            .left-panel, .right-panel {{
                width: 100%;
                height: 200px;
            }}
            
            .map-container {{
                flex: 1;
                min-height: 400px;
            }}
        }}
        
        /* 로딩 애니메이션 */
        .loading {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
            font-size: 14px;
            opacity: 0.7;
        }}
        
        .loading::after {{
            content: '';
            width: 20px;
            height: 20px;
            margin-left: 10px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top: 2px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 좌측 센터 패널 -->
        <div class="left-panel">
            <div id="center-panel">
                <div class="loading">센터 정보 로딩 중...</div>
            </div>
        </div>
        
        <!-- 지도 영역 -->
        <div class="map-container">
            <div id="map"></div>
        </div>
        
        <!-- 우측 차량 패널 -->
        <div class="right-panel">
            <div id="vehicle-panel">
                <div class="loading">차량 정보 로딩 중...</div>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        {js_code}
        
        // 지도 초기화
        map = L.map('map').setView([{center_lat}, {center_lng}], {config.default_zoom});
        
        // 타일 레이어 추가
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);
        
        // 폴리라인 표시
        displayPolylines(map, polylineData);
        
        console.log('🎉 TMS 인터랙티브 경로 시각화가 로드되었습니다!');
        console.log('📊 총 경로:', polylineData.routes.length, '개');
        console.log('🏢 TC센터:', polylineData.depots.length, '개');
    </script>
</body>
</html>
        """
        
        return html_template

def create_polyline_visualization(routes: List[Route], output_file: str) -> str:
    """
    폴리라인 시각화 HTML 파일 생성
    
    Args:
        routes: Route 객체 리스트
        output_file: 출력 파일 경로
        
    Returns:
        str: 생성된 파일 경로
    """
    try:
        logger.info(f"🎨 폴리라인 시각화 생성 중: {len(routes)}개 경로")
        
        # 시각화 설정 (필요시 커스터마이징 가능)
        config = VisualizationConfig()
        visualizer = PolylineVisualizer(config)
        
        # 폴리라인 데이터 생성
        polyline_data = visualizer.get_route_polylines(routes)
        
        # JavaScript 코드 생성
        js_code = visualizer.generate_polyline_javascript(polyline_data)
        
        # HTML 템플릿 생성
        html_content = visualizer.generate_html_template(polyline_data, js_code)
        
        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"✅ 폴리라인 시각화 완료: {output_file}")
        logger.info(f"📊 생성된 데이터: {len(polyline_data['routes'])}개 경로, {len(polyline_data['depots'])}개 TC센터")
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ 폴리라인 시각화 생성 실패: {e}")
        raise 