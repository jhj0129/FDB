# 실험 0024 분류와 보행 최종 요약

## 로봇팔

- 5개 MLP portable 앙상블이 RGB와 세 축 크기로 세 클래스를 실제 실행 중 분류
- 고정 장면 3/3, 위치 변화 포함 9/9 분류·배치 성공
- 최대 최종 XY 오차 3.15mm, 모든 장면 로봇-테이블 접촉 및 관통 0
- 장면 내 최소 신뢰도 0.9998
- prototype 거리 및 신뢰도 gate로 애매한 분포 밖 물체 `unknown` 거부
- 남은 한계: 실제 카메라 영상, 가림, 조명, 처음 보는 일반 물체, 실패 후 재계획

## 휴머노이드

- 사람 보행의 반대 위상 다리·팔, 유각기 무릎 굽힘, 발목 보상, hip roll 무게 이동 적용
- 낙상과 이동거리만 보지 않고 발 접촉에서 단일 지지 비율과 지지 단계 전환을 직접 계측
- 최종 폐루프 영상: G1과 OP3 접촉 보행 성공, T1은 4초 직립하지만 단일 지지 0%,
  Berkeley 하체 모델은 기본 자세부터 약 0.45초에 낙상
- 개발/독립 강건성: G1 6/6 및 7/7 보행
- OP3: 방향별 운동량 보상 뒤 개발 4/6, 독립 6/7 보행, 모든 조건 직립
- 남은 한계: 외란 방향을 시뮬레이터에서 제공, 실제 IMU 추정 없음, capture step 없음

## 검증과 결과물

- 전체 회귀검사: `45 passed` (106.59초)
- 최종 통합 영상: `artifacts/fdb_sort_and_walk_final.mp4` (38.93초, 960×540, 30fps)
- 로봇팔 상세 영상: `artifacts/fdb_multi_object_sorting.mp4`
- 휴머노이드 상세 영상: `artifacts/fdb_human_gait.mp4`
- 보행 전체 지표: `artifacts/fdb_human_gait.json`
- 물체 조작 전체 지표: `artifacts/fdb_multi_object_sorting.json`

이 결과는 MuJoCo 연구 증거이며 실제 하드웨어 실행 승인이 아니다.
