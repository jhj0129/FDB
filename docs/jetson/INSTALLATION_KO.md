# Jetson Phase 3 설치와 재현

## 검증한 호스트

- NVIDIA Jetson AGX Orin Developer Kit, RAM 61 GiB (64 GB 제품군), aarch64
- Ubuntu 22.04.5, kernel 5.15.199-tegra, L4T 36.5.2
- CUDA toolkit 12.6.68, cuDNN 9.3.0.75, TensorRT 10.3.0.30
- JetPack apt 후보 6.2.3+b81. `nvidia-jetpack` 메타패키지는 설치되어 있지 않으므로
  apt 후보 버전을 설치된 메타패키지 버전으로 간주하지 않는다.
- Python 3.10.12, GCC/G++ 11.4.0, CMake 3.22.1
- 기존 전력 모드 MAXN (0), 변경하지 않음
- `/`는 INTEL 512 GB NVMe. 조사 시 가용 174 GiB, 가용 RAM 54 GiB.

## 저장소와 환경

2026-09-20에 `https://github.com/jhj0129/FDB`를 `/home/kudos/FDB`로 clone했다.
기존 경로는 없었고 최초 main은 `465ac48f706b0093893f5b44b549a90485260c83`, clean 상태였다.
Phase 2.5 기준과 일치하므로 reset하지 않았다.

호스트에 이미 PyTorch 2.8.0 (CUDA 12.6)이 있었다. 위치는
`~/.local/lib/python3.10/site-packages/torch`이며 Orin에서 1024x1024 CUDA 행렬 곱과
synchronize가 성공했다. wheel의 설치 원본 URL은 metadata에 없어 공급처를 단정하지 않는다.
검증된 기존 wheel을 재사용하기 위해 `.venv`를 `--system-site-packages`로 만들었다.
이 환경은 호스트 패키지를 읽을 수 있으므로 완전한 독립 환경은 아니다.
추가 설치와 버전 변경은 `.venv` 안에 한정했다.

초기화한 다른 Jetson에서는 먼저 **해당 L4T/CUDA/Python과 호환되는 CUDA torch**를 준비해야 한다.
설치 스크립트는 torch가 없거나 CUDA가 작동하지 않으면 중단하며 임의 wheel을 설치하지 않는다.
기존 wheel을 포함한 완전한 offline 재설치는 이번 작업으로 보장하지 않는다.
공식 설치 지침: <https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html>

```bash
git clone https://github.com/jhj0129/FDB ~/FDB
cd ~/FDB
bash scripts/setup_jetson.sh
bash scripts/run_jetson_headless.sh
```

`setup_jetson.sh`는 apt, JetPack, CUDA, driver, TensorRT, ROS 또는 전력 모드를 변경하지 않는다.
이미 존재하는 `.venv`를 삭제하지 않는다. `requirements-jetson.txt`는 이번 Python 3.10 경로의
직접 의존성을 고정한다. 실제 전체 설치 버전은 로컬 `artifacts/jetson_phase3/pip_freeze.txt`에 있다.

## 의존성 분류와 선택

| 분류 | Phase 3에서 사용 |
|---|---|
| Core | 기본 editable 설치, core 자체 외부 의존성 없음 |
| Physics | MuJoCo 3.13.0, Menagerie 2026.9.0의 시뮬레이션 Panda 모델 |
| Camera | NumPy 1.26.4, OpenCV 4.11.0.86, EGL RGB-D |
| Legacy neural vision | JAX/JAXLIB 0.6.2 CPU ARM64, Flax 0.10.6, Optax 0.2.5 |
| Neural performance | 기존 Torch 2.8.0, 기존 push dynamics checkpoint |
| Numeric/video | SciPy 1.15.3, imageio-ffmpeg 0.6.0 |
| Development | pytest 8.4.2, jsonschema 4.26.0 |
| External training/robot | LocoMuJoCo, DROK 외부 저장소, ROS2, RealSense 설치 제외 |

현재 pyproject의 `vision` extra는 JAX >=0.7, Flax >=0.12를 요구하며 Python 3.10 경로와
맞지 않는다. 따라서 `.[vision]` 전체를 설치하지 않았다. Python 3.10을 지원하는
[JAX 0.6.2](https://pypi.org/project/jax/0.6.2/)와
[Flax 0.10.6](https://pypi.org/project/flax/0.10.6/)를 고정했다.
JAX GPU 플러그인은 설치하지 않았으며 `JAX_PLATFORMS=cpu`를 사용한다.
Camera-only는 CNN을 호출하지 않지만 기존 환경 생성자가 CNN을 로드하므로 이 의존성이 필요하다.

호스트의 `PIP_CONSTRAINT=~/numpy-constraint.txt`에는 `numpy<2`가 설정되어 있었다.
이 설정을 존중하고 NumPy 1.26.4를 유지했다. `.[neural]`의 NumPy >=2 요구와 구분해
Phase 3에 필요한 패키지를 직접 설치하고 기본 프로젝트를 editable 설치했다.
`pip check`로 실제 활성 환경의 의존성 충돌이 없는 것을 확인했다.
pip는 dry-run/현대 metadata 지원을 위해 venv 안에서 22.0.2에서 25.3으로 고정했다.

## 실행

```bash
cd ~/FDB
export MUJOCO_GL=egl JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
.venv/bin/python -m pytest -q
bash scripts/run_jetson_headless.sh --goals triangle --observation-mode camera \
  --output-directory artifacts/jetson_phase3_new/smoke
bash scripts/run_jetson_headless.sh --observation-mode camera \
  --output-directory artifacts/jetson_phase3_new/persistent
bash scripts/run_jetson_headless.sh --observation-mode camera --inject-failures \
  --output-directory artifacts/jetson_phase3_new/recovery
.venv/bin/python scripts/jetson_reproduce.py \
  --output-directory artifacts/jetson_phase3_new/full
.venv/bin/python scripts/jetson_profile.py --mode camera --repeats 10 \
  --output-directory artifacts/jetson_phase3_new/profile
.venv/bin/python scripts/jetson_profile.py --mode neural --samples 500 \
  --output-directory artifacts/jetson_phase3_new/neural
```

재현 및 profiling 스크립트의 출력 경로는 새 경로여야 한다. 기존 증거를 덮어쓰지 않도록
이미 존재하면 오류로 중단한다. benchmark는 safety와 Episode 기록을 항상 유지한다.
`tegrastats`는 스크립트가 시작한 process만 종료한다.
성능 비교 시 다른 테스트/benchmark를 동시에 실행하지 않는다.

## 측정 해석

- 전체 재현은 baseline 6, position 20, yaw 20, target 16, combined 12,
  ablation 12, boundary seed 1/2 32로 총 118 tasks, 31 worlds다.
- Camera-only 신경망 호출은 0이다. `--mode neural`은 별도의 기존 push dynamics 모델에서
  같은 checkpoint/seed/입력을 사용한 CPU/CUDA 실행성 및 지연 비교다.
- `predict` 시간에는 NumPy 전처리, host/device 전송, GPU 완료 대기 및 결과 반환이 포함된다.
  resident-input inference는 입력 전송을 제외한 보조 수치다.
- camera profiler의 inclusive 시간은 중첩되어 합산할 수 없다. exclusive 시간은
  계측된 하위 호출 시간을 뺀 값이다. `total_task`의 exclusive에는 선택, 상태 변환,
  직렬화 및 미계측 Python overhead가 함께 들어간다.
- `physics`는 `mj_step`만, skill execution exclusive는 IK/제어/접촉 확인 등을 포함한다.
- 첫 run에는 프로세스 최초 모델 초기화와 cache 효과가 포함된다. OS page cache를 비우지 않는다.
- RSS는 해당 process, tegrastats RAM은 시스템 전체다. CUDA allocator 값은 통합 메모리의
  일부이며 별도 PC GPU VRAM으로 합산하지 않는다.
- Episode I/O는 기존 JSON 직렬화와 buffered 파일 쓰기 시간이다. fsync를 포함한 저장장치
  내구성 지연은 측정하지 않는다. Memory 검색은 각 world의 최대 3개 Episode 규모이며
  대용량 장기 기록이나 RealSense raw RGB-D 저장 부하로 일반화하지 않는다.
- 짧은 반복 실행으로 장시간 memory leak 부재나 모든 thermal throttling 부재를 증명하지 않는다.

## 문제 해결

- `StrEnum` import 실패: 기존 선언이 Python >=3.11 API를 사용했다. `FailureType`을
  문자열 Enum으로 바꾸어 Python 3.10에서도 JSON/문자열 계약을 유지했다.
- EGL 오류: 먼저 `MUJOCO_GL=egl`을 import 전에 설정하고 NVIDIA EGL을 확인한다.
  X11/GUI 추가 설치로 우회하지 않는다.
- pytest plugin 충돌: 기존 ROS 환경의 plugin 자동 로드를 피하도록
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`을 사용한다.
- LocoMuJoCo/DROK 의존 테스트: 외부 환경 미설치는 전체 suite 제한으로 보고한다.
  test 삭제나 성공 기대치 완화로 통과 처리하지 않는다.
- GitHub 인증: `gh auth login`은 사용자가 직접 수행한다. 인증 전에는 push할 수 없다.

RealSense는 `CameraSource.capture()`의 RGB/depth/timestamp/calibration 계약으로 연결할 수 있다.
다만 현재 환경 클래스가 MuJoCo 생성자와 제어를 소유하므로 실제 sensor 주입 wiring과
extrinsic/depth 검증은 후속 작업이다. `FDBRuntime`의 의사결정 구조를 재작성할 필요와는 구분한다.
