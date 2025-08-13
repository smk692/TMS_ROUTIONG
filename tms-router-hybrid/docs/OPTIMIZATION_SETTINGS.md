# OR-Tools VRP 최적화 설정 가이드

## 🚨 타임아웃 문제 해결 완료
- **Subprocess 타임아웃**: 300초 → 600초 (10분)로 증가
- **VRP 솔버 타임아웃**: 120초 → 180초 (3분)로 증가

## 📊 권장 운영 설정

### 주문 규모별 설정

#### 소규모 (30-50개 주문)
```python
config = ORToolsVRPConfig(
    max_solve_time_seconds=60,
    use_clustering=False,  # 클러스터링 불필요
    unassigned_penalty=100000,
    distance_weight=1.0,
    vehicle_fixed_cost=5000
)
```

#### 중규모 (50-100개 주문)
```python
config = ORToolsVRPConfig(
    max_solve_time_seconds=120,
    use_clustering=True,
    min_cluster_size=8,
    max_cluster_size=35,
    epsilon=0.005,  # ~500m
    unassigned_penalty=100000,
    distance_weight=1.0,
    vehicle_fixed_cost=5000
)
```

#### 대규모 (100-200개 주문)
```python
config = ORToolsVRPConfig(
    max_solve_time_seconds=180,
    use_clustering=True,
    min_cluster_size=15,
    max_cluster_size=50,
    epsilon=0.008,  # ~800m
    unassigned_penalty=50000,  # 미배정 페널티 감소
    distance_weight=0.8,  # 거리 가중치 감소
    vehicle_fixed_cost=3000  # 차량 비용 감소
)
```

## 🔧 성능 최적화 전략

### 1. 차량 용량 부족 시
- **즉시 조치**: 차량 수 증가 (최소 주문수/15대)
- **장기 조치**: 권역별 분할 처리

### 2. 처리 시간 과다 시
- **Haversine 전용 모드** 사용 (이미 적용됨)
- **클러스터링 파라미터 조정**
- **VRP 솔버 시간 제한 조정**

### 3. 미배정 주문 발생 시
```python
# 페널티 조정으로 더 많은 주문 배정
unassigned_penalty=50000,  # 기본 100000에서 감소
vehicle_fixed_cost=3000,    # 기본 5000에서 감소
```

## 📈 성능 지표

| 주문 수 | 차량 수 | 실행 시간 | 배정률 | 품질 점수 |
|---------|---------|-----------|--------|-----------|
| 31개    | 2대     | ~30초     | 100%   | 0.971     |
| 50개    | 5대     | ~20초     | 100%   | 1.000     |
| 100개   | 9대     | ~63초     | 100%   | 0.983     |
| 200개   | 5대     | ~19분     | 53.5%  | 0.704     |

## ⚠️ 주의사항

1. **200개 이상 주문**: 반드시 충분한 차량 확보 (최소 15대)
2. **타임아웃 발생 시**: 주문을 분할하여 처리
3. **품질 저하 시**: 클러스터링 파라미터 미세 조정

## 🎯 최종 권장사항

- **100개 이하**: OR-Tools VRP 단독 사용
- **100-200개**: 충분한 차량 확보 후 사용
- **200개 이상**: 권역별 분할 또는 시간대별 분할 처리