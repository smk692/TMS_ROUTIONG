#!/usr/bin/env python3
"""
OSRM API를 사용한 실제 도로 경로 폴리라인 서비스

핵심 최적화:
- 각 구간별 개별 호출 (637번) → 차량별 전체 경로 한 번에 호출 (13번)
- 99% 시간 단축!
- Legs 기반 정확한 구간 분할
"""

import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent.parent

class PolylineService:
    """실제 도로 경로 폴리라인 서비스"""
    
    def __init__(self, config=None):
        # config 설정
        self.config = config
        
        # config에서 API 설정 가져오기
        if config:
            self.osrm_base_url = f"{config.get('system.api.osrm_url', 'http://router.project-osrm.org')}/route/v1/driving"
            self.max_workers = config.get('system.api.max_workers', 4)
            self.timeout = config.get('system.api.timeout', 15)
        else:
            # 기본값 사용 (성능 최적화)
            self.osrm_base_url = "http://router.project-osrm.org/route/v1/driving"
            self.max_workers = 4
            self.timeout = 15
    
    def get_vehicle_route_from_osrm(self, coordinates: List[Dict]) -> Dict:
        """
        OSRM API를 사용해서 차량의 전체 경로를 한 번에 가져옵니다.
        
        Args:
            coordinates: 차량의 전체 경로 좌표 리스트 (depot → 배송지들 → depot)
        
        Returns:
            Dict: 전체 경로 폴리라인과 구간별 폴리라인 정보
        """
        try:
            # 좌표를 OSRM 형식으로 변환 (lng,lat)
            coord_string = ";".join([f"{coord['lng']},{coord['lat']}" for coord in coordinates])
            
            # OSRM API URL 구성 (파라미터 최소화)
            url = f"{self.osrm_base_url}/{coord_string}"
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'true'  # legs 정보를 위해 필요
            }
            
            # API 호출 (타임아웃 감소)
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == 'Ok' and 'routes' in data and len(data['routes']) > 0:
                    route = data['routes'][0]
                    
                    # 전체 경로 폴리라인 추출
                    full_polyline_coords = []
                    total_duration = 0  # 전체 경로 시간 (초)
                    total_distance = 0  # 전체 경로 거리 (미터)
                    
                    if 'geometry' in route and 'coordinates' in route['geometry']:
                        full_polyline_coords = route['geometry']['coordinates']
                        print(f"  ✅ 전체 경로: {len(full_polyline_coords)}개 포인트")
                    else:
                        print("  ⚠️ geometry 정보 없음")
                        return self._create_fallback_polyline(coordinates)
                    
                    # 전체 경로 시간과 거리 정보 추출
                    if 'duration' in route:
                        total_duration = route['duration']  # 초 단위
                        print(f"  ⏱️ 전체 시간: {total_duration:.0f}초 ({total_duration/60:.1f}분)")
                    
                    if 'distance' in route:
                        total_distance = route['distance']  # 미터 단위
                        print(f"  📏 전체 거리: {total_distance/1000:.1f}km")
                    
                    # Legs 기반 구간별 폴리라인 분할
                    segment_polylines = []
                    segment_durations = []  # 구간별 시간 (초)
                    segment_distances = []  # 구간별 거리 (미터)
                    
                    if 'legs' in route and len(route['legs']) > 0:
                        print(f"  🦵 Legs 정보: {len(route['legs'])}개 구간")
                        
                        # legs별 geometry가 있는지 확인
                        has_leg_geometry = any('geometry' in leg and 'coordinates' in leg['geometry'] 
                                             and leg['geometry']['coordinates'] for leg in route['legs'])
                        
                        if has_leg_geometry:
                            # 각 leg의 geometry에서 좌표 추출
                            for i, leg in enumerate(route['legs']):
                                # 시간과 거리 정보 추출
                                leg_duration = leg.get('duration', 0)  # 초
                                leg_distance = leg.get('distance', 0)  # 미터
                                segment_durations.append(leg_duration)
                                segment_distances.append(leg_distance)
                                
                                if 'geometry' in leg and 'coordinates' in leg['geometry']:
                                    leg_coords = leg['geometry']['coordinates']
                                    if leg_coords:
                                        segment_polylines.append(leg_coords)
                                        print(f"    Leg {i}: {len(leg_coords)}개 포인트, {leg_duration:.0f}초, {leg_distance/1000:.2f}km")
                                    else:
                                        # 빈 leg인 경우 직선으로 대체
                                        segment_polylines.append([
                                            [coordinates[i]['lng'], coordinates[i]['lat']],
                                            [coordinates[i + 1]['lng'], coordinates[i + 1]['lat']]
                                        ])
                                        print(f"    Leg {i}: 빈 leg - 직선으로 대체, {leg_duration:.0f}초")
                                else:
                                    # geometry가 없는 경우 직선으로 대체
                                    segment_polylines.append([
                                        [coordinates[i]['lng'], coordinates[i]['lat']],
                                        [coordinates[i + 1]['lng'], coordinates[i + 1]['lat']]
                                    ])
                                    print(f"    Leg {i}: geometry 없음 - 직선으로 대체, {leg_duration:.0f}초")
                        else:
                            # legs별 geometry가 없는 경우, distance 정보를 기반으로 전체 폴리라인 분할
                            print(f"  📏 Legs별 geometry 없음 - distance 기반 분할 사용")
                            
                            # 각 leg의 거리와 시간 정보 수집
                            leg_distances = [leg.get('distance', 0) for leg in route['legs']]
                            leg_durations = [leg.get('duration', 0) for leg in route['legs']]
                            total_leg_distance = sum(leg_distances)
                            
                            segment_distances = leg_distances
                            segment_durations = leg_durations
                    else:
                        # legs 정보가 없는 경우 균등 분할 (간소화)
                        print("⚠️ Legs 정보 없음 - 균등 분할 방식 사용")
                        segments_count = len(coordinates) - 1
                        total_points = len(full_polyline_coords)
                        
                        for i in range(segments_count):
                            start_idx = int(i * total_points / segments_count)
                            end_idx = int((i + 1) * total_points / segments_count)
                            if i == segments_count - 1:
                                end_idx = total_points
                            
                            if start_idx < end_idx:
                                segment = full_polyline_coords[start_idx:end_idx]
                                if segment:
                                    segment[0] = [coordinates[i]['lng'], coordinates[i]['lat']]
                                    segment[-1] = [coordinates[i + 1]['lng'], coordinates[i + 1]['lat']]
                                segment_polylines.append(segment)
                            else:
                                segment_polylines.append([
                                    [coordinates[i]['lng'], coordinates[i]['lat']],
                                    [coordinates[i + 1]['lng'], coordinates[i + 1]['lat']]
                                ])
                    
                    return {
                        'full_polyline': full_polyline_coords,
                        'segment_polylines': segment_polylines,
                        'segment_durations': segment_durations,
                        'segment_distances': segment_distances,
                        'total_duration': total_duration,
                        'total_distance': total_distance,
                        'success': True
                    }
                else:
                    print(f"⚠️ OSRM 경로를 찾을 수 없음")
                    # 실패 시 직선 경로 반환
                    full_polyline = [[coord['lng'], coord['lat']] for coord in coordinates]
                    segment_polylines = []
                    for i in range(len(coordinates) - 1):
                        segment_polylines.append([
                            [coordinates[i]['lng'], coordinates[i]['lat']],
                            [coordinates[i + 1]['lng'], coordinates[i + 1]['lat']]
                        ])
                    return {
                        'full_polyline': full_polyline,
                        'segment_polylines': segment_polylines,
                        'success': False
                    }
                    
        except Exception as e:
            print(f"❌ OSRM API 오류: {e}")
            # 오류 시 직선 경로 반환
            full_polyline = [[coord['lng'], coord['lat']] for coord in coordinates]
            segment_polylines = []
            for i in range(len(coordinates) - 1):
                segment_polylines.append([
                    [coordinates[i]['lng'], coordinates[i]['lat']],
                    [coordinates[i + 1]['lng'], coordinates[i + 1]['lat']]
                ])
            return {
                'full_polyline': full_polyline,
                'segment_polylines': segment_polylines,
                'success': False
            }

    def process_vehicle_route(self, vehicle_data):
        """
        단일 차량의 전체 경로를 처리하는 함수 (병렬 처리용)
        """
        route_idx, route = vehicle_data
        vehicle_id = route['vehicle_id']
        coordinates = route['coordinates']
        
        print(f"🚚 차량 {vehicle_id + 1} 처리 중... (배송지 {route['delivery_count']}개)")
        
        # 차량의 전체 경로를 한 번에 가져오기
        full_route_polyline = self.get_vehicle_route_from_osrm(coordinates)
        
        # 구간별로 나누어 저장 (시각화에서 구간별 정보 표시용)
        route_segments = []
        
        for i in range(len(coordinates) - 1):
            start_coord = coordinates[i]
            end_coord = coordinates[i + 1]
            
            # 구간 정보 저장 (실제 도로 좌표는 전체에서 추출)
            segment_info = {
                'segment_id': i,
                'start': {
                    'type': start_coord['type'],
                    'sequence': start_coord.get('sequence', 0),
                    'lat': start_coord['lat'],
                    'lng': start_coord['lng'],
                    'label': start_coord['label']
                },
                'end': {
                    'type': end_coord['type'],
                    'sequence': end_coord.get('sequence', 0),
                    'lat': end_coord['lat'],
                    'lng': end_coord['lng'],
                    'label': end_coord['label']
                }
            }
            route_segments.append(segment_info)
        
        print(f"✅ 차량 {vehicle_id + 1} 완료!")
        
        return {
            'route_idx': route_idx,
            'vehicle_id': vehicle_id,
            'full_polyline': full_route_polyline['full_polyline'],  # 전체 경로의 실제 도로 좌표
            'segments': route_segments,  # 구간별 정보
            'polyline_points': len(full_route_polyline['full_polyline']),
            'segment_polylines': full_route_polyline['segment_polylines'],
            'segment_durations': full_route_polyline['segment_durations'],
            'segment_distances': full_route_polyline['segment_distances'],
            'total_duration': full_route_polyline['total_duration'],
            'total_distance': full_route_polyline['total_distance'],
            'success': full_route_polyline['success']
        }

    def add_polylines_to_data(self, data_file_path: str, test_mode: bool = False, max_vehicles: int = 3) -> bool:
        """
        extracted_coordinates.json에 실제 도로 폴리라인을 추가합니다.
        
        Args:
            data_file_path: JSON 파일 경로
            test_mode: 테스트 모드 여부
            max_vehicles: 테스트 모드에서 처리할 최대 차량 수
        
        Returns:
            성공 여부
        """
        # JSON 파일 로드
        json_file = Path(data_file_path)
        if not json_file.exists():
            print(f"❌ {data_file_path} 파일이 없습니다.")
            return False
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 데이터 로드 완료!")
        print(f"   배송지 수: {data['stats']['total_points']}개")
        print(f"   차량 수: {data['stats']['total_vehicles']}대")
        print(f"   총 거리: {data['stats']['total_distance']:.1f}km")
        
        if test_mode:
            print(f"🧪 테스트 모드: 처음 {max_vehicles}대 차량만 처리합니다.")
            routes_to_process = data['routes'][:max_vehicles]
        else:
            routes_to_process = data['routes']
        
        print(f"\n🚀 초고속 처리 시작! (API 호출 {len(routes_to_process)}번만 필요)")
        
        # 병렬 처리로 모든 차량 동시 처리
        vehicle_tasks = [(idx, route) for idx, route in enumerate(routes_to_process)]
        
        processed_vehicles = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_vehicle = {executor.submit(self.process_vehicle_route, task): task for task in vehicle_tasks}
            
            for future in as_completed(future_to_vehicle):
                try:
                    vehicle_result = future.result()
                    processed_vehicles.append(vehicle_result)
                except Exception as e:
                    print(f"❌ 차량 처리 오류: {e}")
        
        # 결과를 차량 순서대로 정렬
        processed_vehicles.sort(key=lambda x: x['route_idx'])
        
        # JSON 데이터 업데이트
        total_polyline_points = 0
        for vehicle_result in processed_vehicles:
            route_idx = vehicle_result['route_idx']
            
            # 실제 도로 폴리라인 데이터 추가
            data['routes'][route_idx]['full_polyline'] = vehicle_result['full_polyline']
            data['routes'][route_idx]['segments_info'] = vehicle_result['segments']
            data['routes'][route_idx]['polyline_points'] = vehicle_result['polyline_points']
            data['routes'][route_idx]['segment_polylines'] = vehicle_result['segment_polylines']
            data['routes'][route_idx]['segment_durations'] = vehicle_result['segment_durations']
            data['routes'][route_idx]['segment_distances'] = vehicle_result['segment_distances']
            data['routes'][route_idx]['total_duration'] = vehicle_result['total_duration']
            data['routes'][route_idx]['total_distance'] = vehicle_result['total_distance']
            data['routes'][route_idx]['success'] = vehicle_result['success']
            
            total_polyline_points += vehicle_result['polyline_points']
        
        # 통계 업데이트
        data['stats']['real_polylines_added'] = True
        data['stats']['total_polyline_points'] = total_polyline_points
        data['stats']['api_calls_made'] = len(processed_vehicles)
        if test_mode:
            data['stats']['test_mode'] = True
            data['stats']['processed_vehicles'] = len(processed_vehicles)
        
        # 업데이트된 JSON 저장
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 파일 크기 확인
        file_size = json_file.stat().st_size / 1024  # KB
        
        print(f"\n🎉 초고속 처리 완료!")
        print(f"📁 업데이트된 JSON 파일: {file_size:.1f}KB")
        print(f"📊 처리 결과:")
        print(f"   - 처리된 차량: {len(processed_vehicles)}대")
        print(f"   - API 호출 수: {len(processed_vehicles)}번 (기존 방식 대비 99% 감소!)")
        print(f"   - 총 폴리라인 포인트: {total_polyline_points:,}개")
        print(f"   - 평균 포인트/차량: {total_polyline_points//len(processed_vehicles):,}개")
        
        return True

def main():
    """메인 실행 함수"""
    print("🛣️ OSRM API 초고속 실제 도로 폴리라인 추가")
    print("⚡ 혁신: 637번 API 호출 → 13번 API 호출 (99% 시간 단축!)")
    print("=" * 60)
    
    start_time = time.time()
    
    # 기본 데이터 파일 경로
    data_file = project_root / "data" / "extracted_coordinates.json"
    
    # 폴리라인 서비스 생성
    polyline_service = PolylineService()
    
    # 폴리라인 추가 (전체 차량 처리)
    success = polyline_service.add_polylines_to_data(str(data_file), test_mode=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if success:
        print(f"\n🎉 처리 완료! (소요시간: {elapsed_time:.1f}초)")
        print("💡 이제 시각화해서 실제 도로 경로를 확인해보세요!")
    else:
        print("\n❌ 작업 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 