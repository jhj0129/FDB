# 시연 결과물

`fdb_final_integrated_demo.mp4`는 네 형상 카메라 판단·Panda 삽입 4회와 사람 보행률
H1 20초·40걸음을 연결한 최신 최종본이다.

`fdb_multi_shape_insertion.mp4`는 정사각형·원·삼각형·직사각형을 차례로 인식하고,
크기와 회전을 판단해 각 수용구에 삽입하는 49초 물리 시연이다. 4/4 성공과 비의도
로봇-테이블 접촉 0회를 JSON에 기록했다.

`fdb_human_cadence_h1_walk.mp4`는 Unitree 공식 H1 정책에 FDB의 사람 보행률 위상 제어와
접촉 기반 걸음 판정을 결합한 자유 물리 시연이다. 20초 동안 분당 120걸음으로 좌우
40걸음을 수행한다. 함께 있는 JSON은 출처 revision, 가중치 SHA-256, 공중시간, 발 여유,
전진 착지, 몸통 이동과 직립 여부를 기록한다.

`fdb_unitree_h1_robustness.json`은 같은 정책의 횡방향 외란 경계다. 350N을 0.15초 가한
조건까지 보행 연속성을 복구하고, 400N 연속성 실패와 500N 낙상도 숨기지 않는다.

`fdb_shape_insertion.mp4`는 상단 카메라 픽셀에서 CNN이 정사각형 물체와 수용구,
삽입 가능성 및 상대 회전을 판단한 뒤 Panda가 집기·회전·삽입·해제를 수행하는 시연이다.
`fdb_shape_insertion_camera.png`는 원 카메라 영상, `_vision.png`는 신경망 입력이다.

`fdb_behavioral_walk.mp4`는 LAFAN1 사람 동작을 Unitree H1에 재표적화해 좌우 10걸음이
연속되는 새 보행 정의를 보여주는 로컬 검증 영상이다. LAFAN1 라이선스 때문에 Git에는
포함하지 않는다. 이 영상은 kinematic reference이며 학습 정책의 물리 rollout이 아니다.

`fdb_neural_gait_skill.mp4`는 같은 10걸음 구간을 69,338개 매개변수 위상 조건 MLP가
학습해 생성한 로컬 영상이다. 좌우 교대 10회와 실제 발 들림을 다시 계측해 통과했다.
여전히 kinematic 기술이며 동역학 균형 정책은 아니다.

`fdb_insertion_recovery.mp4`는 카메라 회전에 40도 오류를 넣은 실패 장면과, 독립 후보
시뮬레이션이 -20도 보정을 찾아 재시도에 성공한 장면을 연속해서 보여준다.

`fdb_sort_and_walk_final.mp4`는 아래 Panda 분류 시연 34.93초와 휴머노이드 비교 4초를
960×540으로 이어 붙인 최종 통합본이다.

`fdb_multi_object_sorting.mp4`는 Panda가 5개 MLP portable 앙상블로 세 물체를 분류해
각 색상 구역으로 옮기는 시연이다. JSON에는 3/3 성공, 신뢰도, 최종 위치 오차와 접촉
안전 지표가 있다.

`fdb_human_gait.mp4`는 네 휴머노이드에 반대 위상 다리, 유각기 무릎 굽힘, 발목 보상과
반대쪽 팔 흔들기를 적용한 4분할 비교다. G1과 OP3의 완주 및 T1과 Berkeley의 낙상을
모두 보존했다.

`fdb_final_pick_place.mp4`는 Panda가 빨간 상자를 집어 파란 목표로 옮기고 놓은 뒤
후퇴하는 접촉 안전 시연이다. 비의도 로봇 테이블 접촉 0과 단계 순서를 검증했다.

`fdb_v5_neural_prediction.mp4`는 빨간 실제 물체, 파란 목표 영역, 노란 신경망 예측
위치를 함께 보여준다. 각 영상 옆의 JSON에는 기계 판독 가능한 검증 지표가 있다.
