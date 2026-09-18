# Jetson AGX Orin 64GB 이식성 감사

이번 감사는 코드와 자산만 대상으로 했다. Jetson 설치, TensorRT 변환, 실제 로봇 연결은
수행하지 않았다.

## 결론

중앙 Runtime의 순수 Python 경로는 ARM64에 옮길 준비가 됐다. 전체 저장소는 아직
준비 완료가 아니다. 가장 큰 차단 요소는 JAX/Flax 학습 경로, 외부 H1 정책 환경,
ARM64용 MuJoCo/OpenCV/ffmpeg 패키지 검증 부재다.

## 조사 결과

- Python: 프로젝트는 3.10 이상이다. 현재 시험은 3.12에서 통과한다. JetPack이 제공하는
  Python/CUDA 조합과 Torch wheel을 먼저 고정해야 한다.
- MuJoCo 3.13 / Menagerie 2026.9: headless renderer는 EGL로 사용할 수 있지만 Jetson에서
  wheel과 EGL context를 직접 확인해야 한다. 물리 계산은 CPU 경로를 유지할 수 있다.
- Torch: 학습 체크포인트와 portable NumPy 체크포인트가 함께 있어 추론은 Torch 없이도
  가능한 부분이 많다. 장치 자동 선택 공통 계층은 아직 없다.
- JAX/Flax/Optax: 형상 CNN과 DeepMimic에 사용한다. Jetson의 공식 설치 경로가 단순하지
  않으므로 첫 이식에서는 학습을 제외하고 portable 추론 형식으로 변환하는 작업이 필요하다.
- OpenCV: 카메라 전처리에 필요하다. GUI 호출은 발견되지 않았고 headless 동작이 가능하다.
- ffmpeg: 렌더 산출물에만 필요하며 실행 자체와 분리 가능하다. 시스템 ffmpeg가 없으면
  `imageio-ffmpeg`를 찾는 fallback이 있다.
- ROS2: 프로젝트 코드가 ROS2에 직접 결합되지는 않았다. 현재 PC의 ROS Jazzy pytest
  plugin 자동 로드가 `lark` 누락으로 시험을 방해하므로 CI는 plugin 자동 로드를 끈다.
- 외부 경로: 사용자명이 박힌 절대 경로를 제거했다. H1은 `FDB_UNITREE_RL_GYM`, DROK은
  `FDB_DROK_ROOT`로 덮어쓸 수 있고 기본값은 현재 사용자 홈 아래의 기존 위치다.
- 실제 카메라/로봇: 현재 Runtime에는 장치 드라이버가 없고 실제 안전 승인이 없다.
- 병렬 처리: 중앙 Runtime에는 multiprocessing 의존성이 없다. MemoryStore는 단일
  프로세스 기준이며 Jetson 서비스화 전에 파일 잠금 또는 DB가 필요하다.

## 자산 정책

현재 추적 중인 `artifacts/`는 약 16MB, 추적 중인 `models/`는 약 1.2MB다. 로컬 전용
학습 모델과 LAFAN1 파생물은 이미 ignore 대상이다. 앞으로 코드·설정·작은 평가 JSON·승격
memory·skill metadata만 Git에 두고 영상, raw frame, dataset, 큰 checkpoint는 로컬
artifact 디렉터리에 둔다. 기존 증거는 삭제하지 않았다.

## 이식 순서

1. JetPack/Python/Torch/MuJoCo 호환 행렬 고정
2. 외부 H1 및 DROK 저장소를 환경변수로 지정
3. 순수 Python Runtime fast test 실행
4. MuJoCo headless와 portable NumPy 모델 실행
5. 카메라 OpenCV 입력 연결 및 지연 측정
6. JAX 모델의 추론 전용 portable 형식 결정
7. 마지막에만 TensorRT 후보 선정
