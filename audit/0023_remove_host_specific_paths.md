# 감사 0023 — 호스트 전용 경로 제거

H1 upstream과 DROK 저장소 기본 경로에 `/home/hgui`가 고정돼 있었다. 기본 위치는
`Path.home()`에서 계산하고 각각 `FDB_UNITREE_RL_GYM`, `FDB_DROK_ROOT` 환경변수로
덮어쓸 수 있게 변경했다. 기존 PC에서는 같은 위치를 가리키므로 동작은 바뀌지 않는다.
