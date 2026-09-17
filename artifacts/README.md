# 시연 결과물

`fdb_shape_insertion.mp4`는 상단 카메라 픽셀에서 CNN이 정사각형 물체와 수용구,
삽입 가능성 및 상대 회전을 판단한 뒤 Panda가 집기·회전·삽입·해제를 수행하는 시연이다.
`fdb_shape_insertion_camera.png`는 원 카메라 영상, `_vision.png`는 신경망 입력이다.

`fdb_behavioral_walk.mp4`는 LAFAN1 사람 동작을 Unitree H1에 재표적화해 좌우 10걸음이
연속되는 새 보행 정의를 보여주는 로컬 검증 영상이다. LAFAN1 라이선스 때문에 Git에는
포함하지 않는다. 이 영상은 kinematic reference이며 학습 정책의 물리 rollout이 아니다.

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
