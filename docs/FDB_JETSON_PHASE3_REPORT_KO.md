# FDB Jetson Phase 3 보고서

측정일: 2026-09-20. 기준 소스: `465ac48f706b0093893f5b44b549a90485260c83`.
작업 위치: `/home/kudos/FDB`. 기존 x86 Phase 2.5 evidence는 변경하지 않았다.

## 환경과 설치

| 항목 | 직접 확인한 값 |
|---|---|
| Hardware | NVIDIA Jetson AGX Orin Developer Kit, RAM 61 GiB (64 GB 제품군) |
| Architecture | aarch64 |
| Ubuntu / kernel | 22.04.5 LTS / 5.15.199-tegra |
| L4T | R36.5.2 |
| JetPack | apt 후보 6.2.3+b81, 메타패키지 자체는 미설치 |
| CUDA toolkit | 12.6.68 |
| cuDNN / TensorRT | 9.3.0.75 / 10.3.0.30 |
| Python / GCC / CMake | 3.10.12 / 11.4.0 / 3.22.1 |
| PyTorch | 기존 2.8.0, CUDA build 12.6, device Orin |
| MuJoCo / Menagerie | 3.13.0 / 2026.9.0 |
| NumPy / SciPy / OpenCV | 1.26.4 / 1.15.3 / 4.11.0 |
| JAX / Flax | 0.6.2 CPU / 0.10.6 |
| 전력 모드 | 기존 MAXN (0), 변경하지 않음 |
| 저장장치 | `/`는 INTEL NVMe 512 GB, 초기 가용 174 GiB |
| 초기 메모리 | 사용 약 5.8 GiB, 가용 약 54 GiB |

`~/FDB`가 없음을 확인한 뒤 공개 GitHub 저장소를 clone했다. 최초 `main`은 지정한
Phase 2.5 커밋과 일치하고 clean이었다. Git·curl·wget·venv·compiler는 이미 존재했다.
apt 설치/upgrade, driver/CUDA/JetPack/TensorRT/ROS 변경은 하지 않았다.

기존 사용자 site의 Torch에서 CUDA 텐서 연산이 성공하여 `.venv`를
`python3 -m venv --system-site-packages .venv`로 생성했다. venv에서도
1024x1024 CUDA 행렬 곱과 synchronize를 검증했다. Torch wheel 설치 URL은 기록이 없어
공급처를 단정하지 않는다. 초기화된 Jetson의 Torch bootstrap은 여전히 별도 선행 조건이다.

호스트의 `numpy<2` constraint를 유지했다. 프로젝트의 일반 `vision` extra 대신
Python 3.10용 JAX/Flax를 고정 설치했고, MuJoCo/JAXLIB는 aarch64 wheel을 사용했다.
실제 로봇 의존성은 설치하지 않았다. 자세한 명령·버전·복구 절차는
[설치 문서](jetson/INSTALLATION_KO.md), [환경 JSON](jetson/environment.json),
`requirements-jetson.txt`와 `scripts/setup_jetson.sh`에 있다.
설치 스크립트를 실제 재실행하여 `pip check` 및 CUDA/MuJoCo EGL 검증을 통과했다.

## 플랫폼 수정

1. `FailureType`이 Python 3.11 전용 `StrEnum`에 의존해 Python 3.10에서 import되지 않았다.
   `str, Enum`과 동일 문자열 변환을 사용해 기존 실패 코드/JSON 계약을 유지했다.
2. 기존 Torch `NeuralDynamicsEnsemble`에 선택적 `device`를 추가했다. 기본 CPU를 유지하고,
   CUDA 사용 시 모델/입력을 이동한 뒤 결과를 CPU NumPy로 반환한다.
3. 설치·headless 실행·재현·profiling·telemetry·환경 기록 도구와 관련 검증을 추가했다.

Camera-only 인식, 후보 생성, 안전 제한, 물리 제어 알고리즘과 모델 가중치는 변경하지 않았다.

## CUDA와 MuJoCo

- system Python과 venv 양쪽에서 Torch CUDA 가용성과 실제 행렬 연산 PASS.
- 단순 MuJoCo 모델 생성, MjData, 10 physics steps PASS.
- EGL offscreen RGB/Depth 생성 PASS. X11/GUI 추가 설치 없음.
- 실제 FDB camera RGB: `512x512x3`, uint8, 값 2~255.
- 실제 FDB camera Depth: `512x512`, float32, 약 0.80982~0.92000 m.
- FDB import PASS. 실제 final camera PNG도 확인했다.

## 테스트와 미해결 회귀

| 테스트 | 결과 |
|---|---|
| 기존 전체 103개 | **95 passed, 4 failed, 4 skipped**, 517.08 s |
| CI fast 범위 + Python 3.10 검사 | 18 passed, 0.32 s |
| 추가 호환성·안전·CPU/CUDA 검사 | 11 passed, 3.69 s |
| segmentation 복구 실패 단독 재시험 | FAIL 재현, 107.63 s |

별도 실행 간 중복 테스트가 있어 통과 수를 단순 합산하지 않는다.
전체 원본 로그와 테스트별 분류는 [회귀 결과](jetson/regression_results.json)에 기록했다.

- 실패 3개: `loco_mujoco` 미설치. 이번 camera simulation과 별개인 외부 보행 학습 경로다.
- skip 4개: 기존 skip 조건에 의해 외부 DROK 저장소가 없어서 제외됐다.
- 실패 1개: legacy segmentation 모드의 persistent recovery. Triangle은 복구 성공했지만
  circle 삽입에서 ALIGNMENT_FAILURE가 반복되고 이후 rectangle도 timeout으로 끝난다.
  첫 circle 삽입 후 target 추정 오차는 23.07 mm였다. 플랫폼별 numerical/rendering 차이와
  기존 고정 calibration 경로의 기여를 분리하지 못했으므로 ARM 자체 원인으로 단정하지 않는다.
  테스트나 tolerance를 변경하지 않았다.

Camera-only 실패 주입에서는 triangle 9 actions/1 replan, circle 10 actions/1 replan,
rectangle 8 actions/0 replans로 **3/3 성공**했다. 이는 위 segmentation 실패와 다른 관측 경로다.
정렬 실패 → 분류 → recover_alignment, grasp 실패 → 재접근/재grasp → 성공을 Episode로 확인했다.

## Safety와 센서 경계

기존 safety gate를 켠 상태에서 모든 benchmark를 수행한다. 추가 테스트로 workspace,
잘못된 pose, table clearance, predicted collision, IK 예외, 비유한 trajectory,
joint/velocity/acceleration limit 거부를 확인했다. 물리 안전 수치는 evaluator의
`robot_table_contact_steps`를 읽는다. Camera-only agent trace에서 contact 필드가 빠졌다고
그것만으로 contact 0으로 판정하지 않는다.

Camera-only 결정 경로는 RGB-D, calibration, tracker와 action belief를 사용하고
segmentation ID/exact body pose를 agent 입력으로 사용하지 않는다. 평가용 정답은 별도로 유지한다.
카메라 충돌 감지는 force/torque sensor를 대체하지 않으며 실제 로봇 안전 검증은 아니다.

`CameraSource` protocol은 RGB/depth/timestamp/intrinsics/extrinsic 계약을 제공한다.
RealSense adapter를 연결할 때 runtime의 의사결정 구조를 재작성할 필요는 없지만,
현재 환경 생성자의 MuJoCo source wiring과 실제 calibration 검증은 후속 작업이다.
이번 단계에서 RealSense/실제 robot/ROS2 설치·연결은 하지 않았다.

## Cross-platform 재현

| Test | x86 Phase 2.5 | Jetson |
|---|---:|---:|
| 기존 전체 tests | 103 pass | 95 pass / 4 fail / 4 skip |
| Camera baseline | PASS | 3/3 PASS |
| Persistent 3-task | 3/3 | 3/3, reset count 1 |
| P20 boundary | PASS | 12/12 (4 shapes x 3 seeds) |
| Y40 boundary | PASS | 12/12 |
| T10 boundary | PASS | 12/12 |
| P10/Y20/T10 boundary | PASS | 12/12 |
| Camera recovery | PASS | 3/3, 2 replans |
| Retained full benchmark | 118/118 | 118/118 |
| Atomic actions | 944 | 944 |
| Persistent worlds preserved | 31/31 | 31/31 |
| Evaluator robot-table contact steps | 0 | 0 |

Jetson 전체 재현 wall time은 1070.25 s였다. 총수는 x86과 동일한 suite/seed/goal matrix다.
중간 교란 level은 seed 0, 최대 경계는 seed 0/1/2다. 118개 독립 분포 표본을 의미하지 않는다.
최초 일부 재현 구간에는 기존 test 실행이 겹쳤으므로 이 wall time으로 정밀 플랫폼 속도비를
주장하지 않는다. 아래 profiling은 다른 FDB benchmark가 끝난 후 단독 실행했다.

## Neural CPU/CUDA

기존 `models/v5/push_dynamics_ensemble.pt`를 동일한 가중치와 seed 0 입력으로 비교했다.
각 device 20회 warm-up 후 500회 측정, Torch CPU threads는 기본 12였다.
Camera-only Atomic World Model은 이 모델을 호출하지 않으므로 별도 component benchmark다.
Camera-only의 CPU/default 표기는 인식·규칙·물리·제어가 CPU라는 뜻이다. EGL 렌더링은
GPU를 사용하므로 전체를 CPU-only workload로 부르지 않는다. CUDA 모델을 억지로
의사결정 경로에 연결하거나 동일 파이프라인이 CUDA로 가속됐다고 주장하지 않는다.

| 항목 | CPU | CUDA |
|---|---:|---:|
| 모델 load | 14.36 ms | 142.13 ms |
| 최초 predict | 2.39 ms | 111.44 ms |
| 전체 predict 평균 | 1.024 ms | 3.771 ms |
| 전체 predict p50 | 1.004 ms | 3.730 ms |
| 전체 predict p95 | 1.120 ms | 5.852 ms |
| 전체 predict max | 2.553 ms | 8.848 ms |
| resident-input inference 평균 | 0.790 ms | 2.481 ms |

전체 predict에는 host/device transfer, synchronize, NumPy 전후 처리가 포함된다.
CUDA가 평균 약 3.68배 느렸다. CPU/CUDA 최종 위치 예측 최대 차이는
`9.54e-7 m`였고 허용 오차 검증을 통과했다. TensorRT 적용 전의 측정이다.
학습/모델 변경, torch compile, graph capture, batching 최적화는 하지 않았다.

CUDA allocation은 50회마다 측정한 10개 표본 모두 9,673,216 bytes,
peak 9,676,800 bytes였다. process RSS는 CPU 측정 후 약 401 MiB에서 CUDA 초기화 후
약 658 MiB로 증가했다. 이 수치를 별도 VRAM으로 시스템 RAM에 더하지 않는다.

## Component 성능과 병목

동일 seed 0의 persistent triangle/circle/rectangle을 10회 반복했다. 첫 3-task run은
33.64 s, warm 9회 평균 27.36 s였다. warm task당 평균은 9.014 s다.
아래 component 값은 warm run별 누적 **exclusive** 시간을 3으로 나눈 뒤 평균한 값이다.
렌더링에는 마지막 evidence frame도 포함되며 Total에는 runtime task 시간만 사용하므로
행들을 합산해 Total을 재구성하지 않는다.

| Component | Jetson CPU/default (task당) | Jetson CUDA |
|---|---:|---|
| Camera/render | 3.350 s, EGL GPU 사용 | 같은 EGL 경로 |
| Perception | 146.99 ms | CPU 경로 유지 |
| Tracking | 3.44 ms | CPU 경로 유지 |
| Atomic World Model (rules) | 0.155 ms | neural call 0, 비교 대상 없음 |
| Candidate generation | 0.606 ms | CPU 경로 유지 |
| Decision + Safety + runtime overhead | 1.141 s | CPU 경로 유지 |
| MuJoCo mj_step | 1.966 s | CPU physics 유지 |
| Skill execution (physics 제외) | 2.400 s | CPU IK/control 유지 |
| Evaluation | 3.38 ms | CPU 경로 유지 |
| Memory retrieval | 33.14 ms | CPU 경로 유지 |
| Episode I/O | 19.00 ms | 동일 저장 경로 |
| Total task | 9.014 s | 동일 task의 CUDA 가속 경로 없음 |

Decision 행은 safety exclusive, candidate generation, runtime 미계측 overhead의 합이다.
따라서 위 candidate 행과 중복되며 순수 `max()` 선택 연산 시간으로 해석하지 않는다.
독립 Torch component의 CPU/CUDA 비교는 앞 표에 따로 기록했다.

마지막 warm run의 **호출당** p50/p95는 render 153.83/159.78 ms,
perception 6.64/7.99 ms, tracking 0.155/0.178 ms, memory 4.01/9.09 ms,
Episode write 18.86/19.43 ms였다. 전체 10 run의 mean/p50/p95/max와 inclusive/exclusive
원값은 [측정 JSON](jetson/phase3_evidence.json)에 있다.

warm wall time 기준 비중은 render 36.73%, skill execution exclusive 26.32%,
physics 21.56%, safety exclusive 12.29%, perception 1.61%, Episode I/O 0.21%였다.
이번 workload의 가장 큰 측정 병목은 renderer다. `capture()`가 frame마다 Renderer를
생성/종료하므로 다음 최적화 후보는 renderer/context 재사용이며, lifecycle과 이미지 동일성을
검증한 뒤 적용해야 한다. 그다음은 IK/model 재생성과 controller 비용 조사다.
안전 검사 제거, TensorRT 선적용, Memory/Neural의 효과 과장은 하지 않는다.

## Resource와 장기 안정성

| Metric | 측정 결과 |
|---|---:|
| 초기 시스템 RAM 사용 | 약 5.8 GiB (첫 조사 시점) |
| profile 전 process RSS | 94.0 MiB (runtime model load 전) |
| 첫 run 후 process RSS | 894.8 MiB |
| 마지막 run 후 process RSS | 966.3 MiB |
| profile process peak RSS | 1154.8 MiB |
| profile 시스템 RAM peak | tegrastats 14576 MB |
| 전체 재현 중 시스템 RAM peak | 14823 MB (일부 tests 동시 실행) |
| profile GPU utilization | 평균 56.76%, 최대 74% |
| profile CPU utilization | active core 평균 32.07%, 최대 45.83% |
| profile 최고 온도 | CPU 51.718 C, GPU 46.750 C |
| 전체 재현 최고 온도 | CPU 57.343 C |
| profile CPU clock | 관측 최대 2201 MHz |
| profile GPU clock | 306~408 MHz |
| VDD_GPU_SOC | 평균 4.172 W, 최대 4.229 W |
| VDD_CPU_CV | 평균 5.096 W, 최대 6.530 W |
| VIN_SYS_5V0 | 평균 5.682 W, 최대 5.956 W |
| 로컬 raw artifacts | 약 64 MiB |
| venv | 약 583 MiB (공유 host Torch 제외) |
| 종료 후 NVMe 가용 | 약 173 GiB |

전력 rail을 임의로 합산해 board total이라고 부르지 않는다. 시스템 utilization/RAM/power에는
기존 GNOME, VS Code, RustDesk 등이 포함된다. 다른 FDB benchmark는 종료했지만 장비 전체가
독점 실험 환경인 것은 아니다. FDB 종료 후에도 GPU 사용률이 높게 나타났으므로 위 GPU 수치를
FDB 신경망이나 렌더러의 전용 사용률로 해석하지 않는다.

반복 실행은 총 279.90 s, 30/30 tasks, 10/10 worlds preserved, contact 0, crash 0이었다.
warm 3-task p50/p95/max는 27.345/27.588/27.690 s, 마지막 run은 첫 warm run보다 1.94% 느렸다.
관측한 CPU/GPU throttle-alert와 cooling state는 모두 0이며 이것이 모든 throttling 원인의
부재를 증명하지는 않는다. descriptor 수는 모든 run 후 18이었다.

RSS는 오르내렸지만 첫 run 대비 마지막에 약 71.5 MiB 높았다. 시스템 전체 RAM도 초기보다
높게 남았고 FDB process 종료 후 20초 관측에서 tegrastats 약 13.6 GB였다.
공유 메모리/driver cache/다른 process 영향과 실제 leak을 분리하지 못했으므로 **memory leak
없음으로 판정하지 않는다**. GPU allocator 측정은 앞의 짧은 Torch component 범위에 한정된다.
장시간 renderer/GPU resource 안정성은 추가 확인이 필요하다. nvmap debug 조회는 sudo 인증이
필요해 수행하지 못했으며, 임의 sudo/driver 변경/재부팅으로 상태를 지우지 않았다.

Episode I/O는 buffered JSON write다. fsync/disk durability 또는 실제 RGB-D raw recording의
부하를 측정한 것은 아니다. 이번 작은 Episode workload에서는 저장 I/O가 주 병목이 아니었다.
Memory는 동일 decision behavior를 유지하며 retrieval latency만 측정했다.

## Evidence와 GitHub 상태

- [환경 정보](jetson/environment.json)
- [작은 측정 요약](jetson/phase3_evidence.json)
- [회귀 분류](jetson/regression_results.json)
- [설치·실행·문제 해결](jetson/INSTALLATION_KO.md)
- 로컬 `artifacts/jetson_phase3/full/reproduction.json` 및 suite별 JSON/CSV
- 로컬 `artifacts/jetson_phase3/profile/`, `neural/`: profiling, tegrastats, clocks
- 로컬 `artifacts/jetson_phase3/segmentation_failure_episodes/`: 실패 원본 Episode

`.venv`, raw logs, 전체 benchmark Episode는 Git ignore 대상이다. 작은 코드/문서/요약만
반영 대상으로 검토했다. 최초 push dry-run은 인증 부족으로 실패했지만, 이후 사용자가
브라우저 인증을 완료하여 `jhj0129` 계정의 GitHub 접근을 확인했다. 커밋 작성자는
`jhj0129 <210678204+jhj0129@users.noreply.github.com>`을 저장소 로컬 설정으로 사용한다.
원격 main의 기준 커밋을 확인하고 일반 push를 사용하며 강제 push는 하지 않는다.

## Jetson Deployment

**FAIL (전체 완료 조건 기준).** CUDA, MuJoCo, Camera-only 핵심 배포와 118/118 재현은 PASS다.
그러나 기존 전체 회귀가 103 pass로 재현되지 않았고 legacy segmentation 복구 실패가 남았다.
장시간 메모리 증가 원인과 초기화 이후 Torch bootstrap의 완전한 재현성도 미해결이다.
따라서 핵심 경로 성공을 Phase 3 전체 완료로 대체하지 않는다.

## TensorRT

**DEFER.** Camera-only neural call 0, renderer 36.73%가 가장 큰 비용이다.
별도 기존 신경망도 CPU 1.024 ms 대비 CUDA 3.771 ms였으므로 현 단계에서 TensorRT
변환을 우선할 수치 근거가 없다.

## RealSense Phase 4

**NO-GO (이번 단계 전체 승인 기준).** CameraSource 계약과 simulation camera 실행성은
확인했지만 미해결 회귀와 resource 안정성 제한을 먼저 정리해야 한다.
그다음 실제 RGB-D calibration, depth noise/latency, 동일 외형 다중 물체 tracking을 검증한다.
이 판정은 실제 로봇 연결이나 자율 작동 승인을 포함하지 않는다.
