import json
import random
from datetime import datetime, timedelta

def generate_suwon_points(num_points: int = 20) -> dict:
    """수원시 지역의 테스트 배송지점 데이터 생성"""
    points = []
    base_time = datetime.now()
    
    # 수원시 중심부 좌표
    suwon_center = (37.2636, 127.0286)
    
    # 주요 지역 리스트
    areas = [
        "팔달구 인계동",
        "영통구 영통동",
        "권선구 권선동",
        "장안구 정자동",
        "팔달구 매산동",
        "영통구 망포동",
    ]
    
    for i in range(num_points):
        lat = suwon_center[0] + (random.random() - 0.5) * 0.1
        lon = suwon_center[1] + (random.random() - 0.5) * 0.1
        
        points.append({
            "id": i,
            "latitude": lat,
            "longitude": lon,
            "address1": f"수원시 {random.choice(areas)}",
            "address2": f"{random.randint(1, 999)}번길 {random.randint(1, 100)}",
            "time_window": [
                (base_time + timedelta(hours=i//2)).isoformat(),
                (base_time + timedelta(hours=i//2 + 2)).isoformat()
            ],
            "service_time": random.randint(5, 15),
            "volume": round(random.uniform(0.1, 0.5), 2),
            "weight": round(random.uniform(5, 20), 1),
            "special_requirements": [],
            "priority": random.randint(1, 5)
        })
    
    return {"points": points}

if __name__ == "__main__":
    # 여러 크기의 테스트 데이터 생성
    sizes = [20, 50, 100]
    for size in sizes:
        data = generate_suwon_points(size)
        filename = f"suwon_points_{size}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"{filename} 생성 완료") 