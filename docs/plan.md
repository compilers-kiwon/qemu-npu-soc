<!-- SPDX-FileCopyrightText: 2026 Kiwon Kwon -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# qemu-npu-soc 구현 계획

이 문서는 `docs/design.md`의 설계를 어떤 순서로 구현할지 정한다. 단계마다
만들 것과 완료 조건을 적는다. 완료 조건은 설계 11절의 검증 단계(V1~V6)로
확인한다. 구현하다가 설계를 바꿔야 하면 `docs/design.md`를 먼저 고친다.

---

## M0. 마일스톤

| 마일스톤 | 목표 | 완료 조건 |
|----------|------|-----------|
| M1 | 합성곱 하나를 모든 층에 관통시킨다 | 컴파일한 합성곱 모델이 kn1-evb에서 TFLite Micro와 비트 단위로 같다(V1~V6) |
| M2 | MNIST CNN | MNIST CNN이 kn1-evb에서 두 반올림 방식 모두 비트 단위로 같고(V6), 겹침이 있을 때와 없을 때의 사이클 추정을 보고하며, `kn1_defconfig`로 부팅한다 |
| M3 | ResNet-8 | MLPerf Tiny ResNet-8이 64 KiB SRAM에서 비트 단위로 같고, SRAM 크기 비교 결과를 문서로 남긴다 |
| M4 | 마무리 | 새로 받은 저장소에서 README의 절차만으로 빌드하고 데모를 돌릴 수 있다 |
| 그 뒤 | PCIe, 실제 저장 장치 | 설계 D3과 13절 |

M1은 설계 4절의 "얇은 조각 먼저" 원칙을 따른다. 모든 층을 합성곱 하나에
필요한 만큼만 만들고, 넓히는 일은 M2부터 한다. 이중 버퍼링(D31)은 M1부터
넣는다.

---

## M1. 얇은 조각: 합성곱 하나

### M1.1 저장소 뼈대

- `buildroot/`: Buildroot 2026.08.1 서브모듈(D16, D43).
- `br2-external/`: 외부 트리 뼈대(`external.desc`, `external.mk`,
  `Config.in`)와 `kn1_evb_defconfig`.
- `LICENSES/`: 쓰는 라이선스의 전문. 모든 파일에 SPDX 식별자를 둔다(D18).
- `README.md`: 프로젝트 소개와 빌드 방법의 첫 판(D19).
- GitHub Actions: push마다 V2~V4를 돌릴 작업의 뼈대(D42). 시험은 M1.4와
  M1.7에서 생기는 대로 붙인다.

완료 조건: Buildroot가 `kn1_evb_defconfig`를 읽어 설정을 만든다. CI 작업이
GitHub에서 한 번 성공한다.

### M1.2 부팅: NPU 없이 리눅스 셸까지

아래 순서로 한 층씩 올린다. 앞 단계가 되는 것을 확인한 뒤 다음 층을 더한다.

1. QEMU: SoC 장치 `kiwon-kn1`와 머신 `kn1-evb`(D20~D26), meson·Kconfig 연결
   패치, QEMU v11.1.3을 빌드하는 `br2-external` 호스트 패키지.
2. 디바이스 트리: `kn1.dtsi`와 `kn1-evb.dts`(D4). 이 단계에는 NPU 노드가
   없다.
3. 커널 직접 부팅: `-kernel`과 `-dtb`로 Linux 7.3.1(arm64 `defconfig`와 kn1
   조각, D35)을 띄워 Buildroot 셸까지 간다. PSCI는 QEMU 내장 기능이 맡는다.
4. TF-A: `plat/kiwon/kn1`(D34). BL1과 FIP를 부트 플래시에 넣고, BL31이
   PSCI를 맡는다.
5. U-Boot: 보드 `kiwon/kn1-evb`. 자체 DT로 동작하고, virtio 디스크의
   `extlinux.conf`로 커널과 `/boot/kn1-evb.dtb`를 올린다(D4, D27).

완료 조건: 펌웨어 체인 전체로 부팅해 다음을 확인한다.

- CPU 4개가 모두 켜진다(PSCI `CPU_ON`).
- `poweroff`로 QEMU가 끝나고, `reboot`로 다시 부팅한다(보안 GPIO, D23).
- RTC가 호스트 시각을 보여 주고, 워치독 장치가 보인다(D24).
- `-m`에 4 GiB가 아닌 값을 주면 머신이 거부한다(D26).
- 리눅스 DT 스키마 검사(`dtbs_check`)에서 벤더 접두사 `kiwon`이 어떻게
  처리되는지 확인하고, 필요하면 연결 패치를 더한다(설계 13절 3번).

### M1.3 NPU 명세 v0

`docs/npu-spec.md`에 설계 6.2절과 6.3절의 비트 단위 정의를 쓴다.

- 레지스터 필드, 리셋 값, 접근 규칙
- 명령 워드 인코딩: 명령 번호, 파라미터와 페이로드 필드
- 대기 명령의 의미와 동기화 규칙(설계 6.9절)
- 오류 코드와 `ERR_OFFSET`
- 성능 카운터가 세는 대상

명세는 6.3절의 명령 집합 전체를 정한다. 구현은 합성곱에 필요한 명령부터
한다.

완료 조건: 명세 v0가 검토를 거쳐 확정된다(설계 13절 2번).

### M1.4 공유 코어와 정답값 (V2, V3)

- `hw/npu-core/`(MIT, D33): 명령 해석기, SRAM, DMA, 합성곱 연산(두 반올림
  방식), 동기화 검사, 성능 카운터. 메모리 접근은 호출하는 쪽이 넘기는
  함수로 하므로 QEMU와 호스트 시험이 같은 코드를 쓴다.
- V2: 고정한 커밋의 TFLite Micro 참조 커널을 반올림 방식별로 두 번 빌드해
  합성곱 정답값을 만든다.
- V3: 시험용 명령열을 공유 코어에서 실행해 V2와 비트 단위로 비교한다.
  시험용 명령열은 `tests/`에 두는 작은 어셈블러로 만든다.

완료 조건: 커널 크기(1x1, 3x3), stride(1, 2), padding, 채널별 재양자화, 두
반올림 방식을 조합한 경우가 모두 V2와 같다. 대기를 빠뜨린 명령열은 동기화
오류로 멈춘다.

### M1.5 QEMU kn1-npu 장치 (V1)

- `hw/misc/kiwon-kn1-npu.c`: 레지스터, `CTRL.START`의 bottom half 실행(D32),
  시스템 주소 공간을 거치는 DMA, 인터럽트, soft reset, `sram-size` 속성.
  계산은 `hw/npu-core/`가 맡는다.
- `kn1.dtsi`에 NPU 노드와 고정 클록을 더한다(D37).
- V1 qtest: 레지스터 리셋 값, 작업 완료 인터럽트, 명령 하나하나, 오류
  경로(알 수 없는 명령, SRAM 범위 초과, `QSIZE` 초과, 동기화 오류), 성능
  카운터.

완료 조건: V1이 모두 통과한다.

### M1.6 knpu 드라이버 (V5)

- 바인딩 `kiwon,kn1-npu.yaml`(D37)과 스키마 검사(`dt_binding_check`).
- 트리 밖 모듈 `knpu`(D15): probe, 클록과 runtime PM, GEM DMA 버퍼, 32비트보다
  넓은 DMA 마스크, `drm_sched` 작업 큐, 인터럽트 처리, 타임아웃 뒤 soft
  reset.
- UAPI `knpu_accel.h`의 ioctl 9개(D36)와 명령열 검증기(설계 8.4절).
- V5: 게스트 안에서 도는 시험. 검증기 부정 시험(잘못된 명령열, 영역을
  벗어나는 DMA, 필요한 크기보다 작은 버퍼), 4 GiB 위의 버퍼, 타임아웃과
  복구.

완료 조건: V5가 모두 통과하고, 바인딩이 스키마 검사를 통과한다.

### M1.7 컴파일러 최소 경로 (V2~V4)

- `models/conv/`: 합성곱 하나(입력 14x14x8, 3x3 커널, 출력 16채널, `SAME`,
  stride 1, ReLU)를 완전 int8로 양자화한 `.tflite`와 그것을 만드는 스크립트.
  스크립트는 손으로 돌리고 결과 파일을 함께 둔다. TensorFlow는 시험 모델을
  만드는 데만 쓰고, 컴파일러 경로와 CI에는 넣지 않는다(D12).
- 생성기 `tflite2c`(설계 9.2절, 9.3절): `CONV_2D`를 C89 함수와
  `#pragma knpu`로, 모델을 `knpu_model`로 쓴다. gcc 시험용 `main` 파일도
  쓴다.
- lcc: `compiler/` CMake가 `v4.2-llvm.3`의 aarch64 mlir 드라이버를
  sha256으로 확인하며 내려받는다(설계 12.4절). lcc-opt는 M2에서 붙인다.
  M1.7의 시험 모델은 NPU 층 하나뿐이고, 그 층 함수는 `knpu-import`가 NPU
  연산으로 바꾼 뒤 지우므로 CPU 코드로 내릴 것이 남지 않는다.
- `knpu` 다이얼렉트(그래프 연산 `conv2d`와 명령 연산), 패스 `knpu-import`,
  `knpu-tile`(H 타일), `knpu-schedule`(SRAM과 arena 배치, 이중 버퍼링, D31),
  `knpu-insert-waits`, 그리고 `knpu-translate`와 `knpu-compile`(설계 9.4절).
- `docs/model-format.md`: `model.knpu` 형식 v0(D39, D40).
- V2: TFLite Micro 인터프리터를 반올림 방식별로 두 번 빌드해 `.tflite`를
  그대로 실행한다.
- V4: 패스마다 `lit`/`FileCheck` 시험(오류 경로 포함)과, 실제 lcc 출력을
  거치는 시험 하나.
- CI: push마다 도는 `compiler` 작업(MLIR 23 설치, 빌드, V4, V3).

완료 조건:

- 생성기가 쓴 C를 gcc로 돌린 결과가 두 반올림 방식 모두 V2와 같다.
- SRAM 64 KiB(타일 하나)와, 타일이 둘 이상 나오는 작은 SRAM으로 컴파일한
  합성곱이 공유 코어에서 두 반올림 방식 모두 V2와 비트 단위로 같다(V3).
- 명령열에 대기 명령이 들어 있고 동기화 오류 없이 실행된다. 대기 삽입을 끄고
  컴파일하면 동기화 오류로 멈춘다.
- `PERF_MACS`와 `PERF_DMA_WR_BYTES`가 타일 수와 관계없이 계산값과 같다.
- push CI의 `compiler` 작업이 한 번 성공한다.

### M1.8 런타임과 끝에서 끝까지 (V6)

- 시험 벡터 `models/conv/vectors/`: M1.7 컴파일러가 낸 모델 넷(반올림
  double·single × SRAM 64 KiB·4 KiB), 입력 다섯(무작위 셋, 전부 -128, 전부
  127), TFLite Micro 정답값 열. V3가 이 커밋본이 지금 생성물과 같은지
  검사하고, `--update-vectors`일 때 다시 쓴다.
- `runtime/`: Rust cargo 작업 공간. `libknpu`(UAPI 바인딩, `model.knpu`
  읽기와 검사, 영역 버퍼, arena 복사, 제출과 대기, perfmon)와 `knpu-infer`.
  외부 크레이트는 쓰지 않는다(D41, 설계 10절).
- 호스트 시험(`cargo test`): 모델 파싱과 형식 검사, UAPI 정의와 C 헤더
  대조, `knpu-infer` 인자.
- `br2-external`: cargo 패키지 `knpu-runtime`(`host-rust-bin`)과 벡터를
  넣는 `knpu-models`.
- V6(`tests/e2e/kn1_v6_test.py`): kn1-evb를 펌웨어 체인으로 부팅하고
  `knpu-infer`로 모델 넷과 입력 다섯을 돌려 정답값과 비교한다. 카운터
  (`PERF_MACS`, `PERF_DMA_WR_BYTES`, `PERF_OPS`)를 확인하고, 잘린 모델
  파일을 거부하는지 본다.
- CI: push마다 런타임 호스트 시험(`runtime` 작업)을 돌리고, 매일 밤과 수동
  실행 때 V1, V5 다음에 V6을 돌린다(D42).

완료 조건:

- V6이 통과한다. 모든 모델과 입력이 정답값과 같고, 카운터가 모델과 맞고,
  커널 경고가 없다.
- 런타임 호스트 시험과 V3(벡터 비교 포함)가 push CI에서 통과한다.
- 야간 CI 작업이 V6을 포함해 한 번 성공한다.

---

## M2. MNIST CNN

설계 6.6절의 MNIST CNN(conv3x3(8)-avgpool-conv3x3(16)-avgpool-FC64-FC10-softmax)을
모든 층에 관통시킨다. NPU에는 AVGPOOL과 1x1 합성곱으로 돌리는 FC를 더하고,
NPU에서 돌지 않는 softmax는 같은 C89를 lcc-opt로 컴파일한 aarch64 코드로
돌린다(D45). 여러 층의 스케줄(층 사이 활성값의 SRAM 상주, 다음 층 상수 미리
읽기)과 사이클 추정(D31), 최소 커널 설정(D35)도 여기서 한다.

완료 조건: MNIST CNN이 kn1-evb에서 두 반올림 방식 모두 TFLite Micro와 비트
단위로 같다(V6). 겹침(이중 버퍼링)이 있을 때와 없을 때의 사이클 추정을
보고한다. `kn1_defconfig`로 부팅한다.

### M2.1 MNIST 모델과 정답값 (V2)

- `models/mnist/make_model.py`: 설계 6.6절 구조(입력 28x28x1, conv3x3(8,
  VALID, ReLU), avgpool 2x2, conv3x3(16, VALID, ReLU), avgpool 2x2, Flatten,
  Dense 64(ReLU), Dense 10, Softmax)를 MNIST로 학습하고(고정 시드), 학습
  이미지로 calibration해 int8로 변환한다(TensorFlow 2.21.0). 시험 집합에서
  숫자마다 처음 나오는 이미지 하나씩을 모델 입력의 양자화로 바꿔 쓰고, float
  모델과 int8 모델의 시험 집합 정확도를 찍는다. 손으로 돌리고 결과를
  커밋한다(D12): `models/mnist/mnist.tflite`,
  `models/mnist/digits/digit-{0..9}.bin`(CC BY-SA 3.0, 설계 12.4절).
- V2: `tests/compiler/tflm_run.cc`가 `preserve_all_tensors`로 softmax의
  입력(logits)도 쓴다.
- `tests/compiler/test_mnist_model.py`: MNIST 시험의 틀. 시험 입력은 숫자
  이미지 열 장과 무작위 셋, 전부 -128, 전부 127의 열다섯이다.

완료 조건:

- `mnist.tflite`의 연산이 `CONV_2D`, `AVERAGE_POOL_2D`, `CONV_2D`,
  `AVERAGE_POOL_2D`, `RESHAPE`, `FULLY_CONNECTED` 둘, `SOFTMAX`이고, MAC 수와
  파라미터 수가 설계 6.6절과 같다.
- V2가 두 반올림 방식으로 시험 입력 열다섯의 최종 출력과 logits를 낸다. 두
  방식의 최종 출력이 다른 입력이 하나 이상 있다.

### M2.2 공유 코어의 `OP_AVGPOOL` (V1, V2, V3)

- `hw/npu-core/`: `OP_AVGPOOL`(명세 6.4절, 8.4절). 모양, SRAM 범위, 겹침
  검사와 동기화 규칙(명세 9절), 카운터(`PERF_OPS`에 1, `PERF_MACS`에 0).
  QEMU 장치는 공유 코어를 쓰므로 따로 고치지 않고, 드라이버 검증기는 이미
  `OP_AVGPOOL`을 받는다.
- V2: `tests/npu-core`의 TFLite Micro 정답값에 int8 `AveragePool`을 더한다
  (고정 커밋의 `pooling.h`).
- V3: 어셈블러로 만든 명령열로 창 크기, stride, padding(VALID, SAME), clamp,
  오류 코드(`BAD_PARAM`, `SRAM_RANGE`, `SYNC`)를 확인한다. `OP_AVGPOOL`이
  `UNKNOWN_CMD`로 멈추기를 기대하던 경우는 고친다.
- V1: kn1-npu qtest에 `OP_AVGPOOL` 한 경우와 `PERF_OPS`를 더한다.

완료 조건: `OP_AVGPOOL`의 모든 경우가 V2와 비트 단위로 같고(두 반올림 빌드의
정답값도 서로 같다), 오류 경우는 명세의 오류 코드와 `ERR_OFFSET`으로 멈춘다.
V1이 통과한다.

### M2.3 생성기 (V2)

- `tflite2c`에 `AVERAGE_POOL_2D`, `RESHAPE`, `FULLY_CONNECTED`, `SOFTMAX`와
  pragma `avgpool`, `fc`, `reshape`, `cpu`를 더한다(설계 9.2절, 9.3절). FC는
  채널별·텐서 단위 곱셈 계수와 bias가 없는 경우를 받는다. softmax는 TFLite
  Micro의 int8 `Softmax`와 고정소수점 함수를 C89로 옮기고, 매개변수를 반올림
  방식마다 계산한다.
- `--main` 파일이 logits도 쓴다. `--npu-only`는 CPU 층을 빼고 logits를 모델
  출력으로 하는 C를 쓴다.
- `tests/compiler/test_softmax.py`: MNIST의 logits는 softmax의 경로를 일부만
  지나므로, softmax 하나만 있는 모델을 TFLite Python 스키마의 객체 API로
  만들어 입력 스케일, beta, 행 길이를 바꿔 가며 TFLite Micro와 생성기 C를
  비교한다.

완료 조건:

- 생성기가 쓴 MNIST C를 `gcc -std=c89 -pedantic-errors`로 빌드해 시험 입력
  열다섯을 돌린 최종 출력과 logits가 두 반올림 방식 모두 V2와 같다.
- softmax 시험의 모든 경우에서 생성기 C의 출력이 두 반올림 방식 모두 TFLite
  Micro와 같다. softmax 이식의 비트 일치는 여기서 확인된다.
- 같은 C들을 gcc의 UBSan(`-fsanitize=undefined`)으로 돌려도 보고가 없다.
- lcc가 MNIST C를 경고 없이 받는다(`-A -Wf-pragma=knpu`).
- 합성곱 모델의 결과는 그대로다.

### M2.4 컴파일러: NPU 경로 (V3, V4)

- `knpu` 방언: 그래프 연산 `knpu.avgpool`, `knpu.cpu`, 명령 연산 `knpu.pool`,
  CPU 단계 `knpu.cpu_step`(설계 9.4절).
- `knpu-import`: `avgpool`, `fc`(1x1 `knpu.conv2d`), `reshape`, `cpu` 층과
  흐름 검사. `knpu-tile`: avgpool의 출력 H 타일. `knpu-schedule`: 층 사이
  활성값 SRAM 상주, 다음 층 상수 미리 읽기, CPU 단계, `overlap=false`.
  `knpu-insert-waits`: `knpu.pool`.
- 사이클 추정: `knpu-estimate`, `knpu-compile --report`와 `--overlap`(설계
  9.1절의 가정값). `knpu-translate`: `OP_AVGPOOL`.
- V4: 패스마다 lit 시험(새 연산과 오류 경로), 손으로 계산한 사이클 추정 사례,
  새 pragma를 실제 lcc 출력으로 거치는 시험 하나.
- V3: 생성기 `--npu-only`로 만든 MNIST 모델을 공유 코어에서 돌린다. CPU 층이
  있는 모델의 컴파일은 M2.5에서 한다.

완료 조건:

- `--npu-only` MNIST 모델을 SRAM 64 KiB·32 KiB와 두 반올림 방식으로 컴파일해
  공유 코어에서 돌린 logits가 시험 입력 열다섯 모두 V2와 비트 단위로 같다.
  FC를 1x1 합성곱으로 돌린 결과의 비트 일치(설계 13절 4번)가 여기서
  확인된다.
- `--overlap=false`로 컴파일한 모델도 같은 logits를 내고, 추정 사이클은
  겹침이 있을 때가 더 작다.
- 대기 삽입을 끄고 컴파일하면 동기화 오류로 멈춘다.
- `PERF_MACS`가 214,304이고, `PERF_DMA_*`와 `PERF_OPS`가 명령열에서 계산한
  값과 같다.
- 합성곱 모델의 V3(벡터 비교 포함)가 통과한다. 명령열이 바뀌면 벡터를 다시
  만든다.

### M2.5 컴파일러: CPU 경로 (V3, V4)

- `docs/model-format.md`: 형식 minor 1(`CPU_CODE`, `CPU_STEPS`, 읽는 쪽이
  검사할 것. 설계 9.5절).
- `compiler/` CMake: lcc_llvm `v4.2-llvm.3`의 소스 압축(sha256)을 받아
  `lcc-opt/`를 빌드하고, `mlir-translate`, `clang-23`, `ld.lld-23`을 찾는다.
- `knpu-compile`: CPU 모듈(CPU 층 함수와 그 전역, external 연결, `knpu.*`
  없음)을 bytecode로 써서 lcc-opt, `mlir-translate`, clang, ld.lld를 부르고
  공유 객체를 `knpu.cpu_code`로 넣는다. `knpu-translate`: minor 1 섹션.
- V4: CPU 모듈 나누기, 오류 경로(NPU 층 앞의 CPU 층 등), minor 1 섹션.
- V3: 전체 MNIST 모델의 NPU 부분을 공유 코어에서 돌리고, 그 logits로 CPU
  섹션을 qemu-aarch64 사용자 모드에서 돌린다. 시험 프로그램은 aarch64 교차
  gcc로 빌드한 작은 C 프로그램이다(`dlopen`, `CPU_STEPS`대로 복사와 호출).
- CI: `compiler` 작업에 `clang-23`, `lld-23`, `gcc-aarch64-linux-gnu`,
  `libc6-dev-arm64-cross`, `qemu-user`를 더한다.

완료 조건:

- 전체 MNIST 모델(SRAM 64 KiB·32 KiB, 두 반올림 방식)의 최종 출력이 시험
  입력 열다섯 모두 V2와 비트 단위로 같다(V3, qemu-aarch64).
- 공유 객체는 `CPU_STEPS`가 가리키는 함수와 배열을 내보내고, 텍스트
  재배치가 없다.
- 같은 입력을 두 번 컴파일한 모델 파일이 바이트 단위로 같다.
- push CI의 `compiler` 작업이 한 번 성공한다.

### M2.6 런타임 (호스트 시험)

- `libknpu`: 형식 minor 1의 파싱과 검사, `CPU_CODE`를 `memfd_create`로 만든
  파일에서 `dlopen`, NPU 작업 뒤 `CPU_STEPS` 실행(설계 10절). CPU 코드를
  올리는 메서드는 `unsafe`로 따로 둔다.
- 컴파일러: CPU 코드를 `ld.lld -Bsymbolic`으로 링크해, 런타임 프로세스의 같은
  이름 기호가 단계의 배열을 가로채지 못하게 한다(설계 9.4절, 형식 9절).
- `knpu-infer`: 분류 벡터 출력의 `top1=`.
- 호스트 시험: minor 1 파싱, 깨진 CPU 섹션 거부(ELF가 아님, 다른 기계, 끝나지
  않는 이름, 범위를 벗어난 단계 등), 호스트 `cc`로 만든 시험 공유 객체로
  올리기·기호 검사·단계 실행, `top1=` 출력.

완료 조건: 런타임 호스트 시험이 push CI의 `runtime` 작업에서 통과한다. minor
0 모델(합성곱 벡터)도 전처럼 읽는다. CPU 코드에 `DF_SYMBOLIC`이 있고, V3가
push CI의 `compiler` 작업에서 그대로 통과한다.

### M2.7 커널 설정 `kn1_defconfig` (부팅 시험, V5)

- `br2-external/board/kn1-evb/kn1_defconfig`: `allnoconfig`에 설계 8.1절 표의
  기호를 켠 설정의 `savedefconfig` 출력(주석 없음, 라이선스는 `REUSE.toml`).
  GEM DMA 헬퍼와 `drm_sched`는 `CONFIG_DRM_ACCEL_ARM_ETHOSU=m`으로, 타임아웃
  복구가 쓰는 `PM_SLEEP`은 `CONFIG_SUSPEND`로 켠다(설계 8.2절). Buildroot의
  init 스크립트가 쓰는 네트워크(`NET`, `UNIX`, `INET`)도 켠다.
- `kn1_evb_defconfig`가 `BR2_LINUX_KERNEL_USE_CUSTOM_CONFIG`로 이것을 쓴다.
- `knpu` 패키지의 빌드 전 검사에 `CONFIG_PM_SLEEP`을 더한다.
- 야간 CI: 빌드 뒤 `make br-linux-update-defconfig`와 `git diff --exit-code`로
  파일이 `savedefconfig` 형태인지 본다.
- README: 빌드 설명의 커널 설정과 CI 절의 야간 작업.

완료 조건: 이 설정으로 빌드한 커널에서 부팅 시험(M1.2의 확인 항목), V5,
합성곱 모델의 V6이 통과하고, 커널 경고가 없으며, init 스크립트가 모두 "OK"다.
야간 작업이 형태 검사를 포함해 한 번 성공한다.

### M2.8 끝에서 끝까지 (V6)

- 시험 벡터 `models/mnist/vectors/`: 모델 넷(CPU 코드까지 든 전체 모델,
  반올림 double·single × SRAM 64 KiB·32 KiB), 입력 다섯(무작위 셋, 전부 -128,
  전부 127. 숫자 이미지 열 장은 `models/mnist/digits/`), 시험 입력 열다섯의
  정답값(두 방식). 파일 이름에는 V3의 입력 이름을 쓴다(`random-2.bin`,
  `expected-double-digit-7.bin`). V3가 커밋본과 지금 생성물이 같은지 검사하고,
  `make update-vectors`(CMake 대상 `update-mnist-vectors`)로 다시 쓴다.
  LLVM 23의 판이 오르면 다시 만든다(설계 12.2절). 라이선스는 설계 12.4절대로
  모델·정답값과 입력을 나눠 선언한다.
- V3: 전체 모델의 `PERF_DMA_*`와 `PERF_OPS`도 명령열에서 계산한 값과
  비교한다.
- `knpu-models`가 MNIST 벡터와 숫자 이미지를 `/usr/share/knpu/mnist/`에
  넣는다.
- V6: `kn1_defconfig` 커널로 부팅해 MNIST 모델 넷 × 입력 열다섯을 돌린다.
  `top1`과 카운터의 기대값은 이 트리의 벡터에서 계산하므로, 먼저 게스트의
  파일이 이 트리의 벡터·숫자 이미지와 같은지 `md5sum`으로 본다. 다른 반올림
  방식의 정답값을 가려내는지 보고, 숫자 이미지 열 장의 `top1`을 찍는다.
  합성곱 시험은 그대로 둔다.
- 사이클 추정 보고: MNIST를 SRAM 64 KiB·32 KiB에서 겹침을 켜고 끄고 컴파일한
  추정치를 README에 표로 남긴다.
- README: M2 절(새 연산, CPU 경로와 lcc-opt, 사이클 추정, `kn1_defconfig`),
  현재 상태.

완료 조건:

- V6이 통과한다. 모든 모델과 입력이 정답값과 같고, `top1`이 정답값의
  argmax와 같고, `PERF_MACS`가 214,304이며 다른 카운터가 모델에서 계산한
  값과 같고, 커널 경고가 없다.
- 야간 CI가 V6을 포함해 한 번 성공한다.

---

## M3. ResNet-8

설계 6.6절의 ResNet-8(MLPerf Tiny 학습 스크립트의 구조, 필터 26개)을 모든 층에
관통시킨다. NPU에는 ADD(입력 스케일링 포함)를 더하고, 컴파일러는 잔차
연결(사슬이 아닌 층 그래프, 입력이 둘인 층)과 64 KiB에 들어가지 않는 층의
타일링(출력 H 타일과 출력 채널 타일, 2-D DMA, arena로 내보내기)을 다룬다.
stride 2 합성곱과 2-D DMA는 공유 코어에 이미 있다. 같은 모델을 32, 64,
128 KiB로 컴파일해 `PERF_DMA_*`와 사이클 추정을 비교한다(설계 6.6절).

완료 조건: MLPerf Tiny ResNet-8이 kn1-evb에서 두 반올림 방식 모두 TFLite
Micro와 비트 단위로 같다(V6). SRAM 크기 시연 결과를 문서로 남긴다.

### M3.1 ResNet-8 모델과 정답값 (V2)

- `models/resnet8/make_model.py`: MLPerf Tiny `keras_model.py`의
  `resnet_v1_eembc(conv_filters=26)`와 같은 층(입력 32x32x3, 합성곱 아홉, 잔차
  덧셈 셋, avgpool 8x8, FC 10, softmax, L2 1e-4)을 이 저장소의 코드로
  쓰고(업스트림 코드를 복사하지 않는다), CIFAR-10으로 학습해 int8로 바꾼다.
  학습은 업스트림 `train.py`의 방법을 따른다: 배치 32, Adam(학습률 0.001,
  epoch마다 × 0.99), 500 epoch, 회전 15도·이동 0.1·좌우 뒤집기 증강(Keras 3의
  `RandomRotation(15/360)`, `RandomTranslation(0.1, 0.1)`,
  `RandomFlip("horizontal")`), 픽셀 0~255를 그대로 입력. TensorFlow 2.21.0,
  고정 시드, `enable_op_determinism()`으로 CPU에서 돌린다. Keras가 CIFAR-10을
  받으며 SHA-256을 확인한다. 변환은 int8 입출력이고, 보정 이미지는 MNIST처럼
  학습 집합에서 고른다. 시험 집합 10,000장의 float, int8 정확도를 찍는다.
  손으로 돌리고 결과 `models/resnet8/resnet8.tflite`를 커밋한다(D12, 설계
  12.4절).
- `models/resnet8/make_photos.py`: CIFAR-10의 클래스마다 Wikimedia Commons의
  CC0-1.0 사진 한 장(물체가 가운데 있고 하나인 것)을 받아 정사각형으로 자르고
  32x32로 줄인 뒤, 모델 입력의 양자화로 바꾼 int8 바이트(3,072 B)를
  `models/resnet8/photos/photo-<클래스>.bin`에 쓴다. 사진마다 URL, 저작자,
  라이선스, 내려받은 원본의 sha256을 스크립트에 적고, 자르고 줄이는
  방법(필터)을 고정한다. 사진은 모델의 분류 결과를 보기 전에 정하고, 틀리게
  분류돼도 바꾸지 않는다. 손으로 돌리고 결과를 커밋한다(CC0-1.0,
  `LICENSES/CC0-1.0.txt`).
- `tests/compiler/test_resnet8_model.py`: ResNet-8 시험의 틀. 시험 입력은 사진
  열 장과 무작위 셋, 전부 -128, 전부 127의 열다섯이다. V2(`tflm_run`)가 두
  반올림 방식으로 최종 출력과 logits를 낸다.

완료 조건:

- `resnet8.tflite`의 연산이 `CONV_2D` 아홉, `ADD` 셋, `AVERAGE_POOL_2D`,
  `FULLY_CONNECTED`, `SOFTMAX`이고, MAC 수와 파라미터 수가 설계 6.6절과 같다.
  TF 2.21.0의 변환기는 Flatten을 `RESHAPE`로 남기지 않고, FC가 avgpool의
  1x1x104 출력을 바로 읽게 한다.
- V2가 두 반올림 방식으로 시험 입력 열다섯의 최종 출력과 logits를 낸다. 두
  방식의 최종 출력이 다른 입력이 하나 이상 있다.
- `reuse lint`가 통과한다(모델은 저장소 라이선스, 사진은 CC0-1.0).

### M3.2 공유 코어의 `OP_ADD` (V1, V2, V3)

- `hw/npu-core/`: `OP_ADD`와 그것만 쓰는 SET(`SET_IFM2_BASE`, `SET_IFM2_ZP`,
  `SET_IFM_SCALE`, `SET_IFM2_SCALE`, `SET_OFM_SCALE`. 명세 6.1, 6.4, 8.3절).
  모양, SRAM 범위(IFM, IFM2, OFM), 겹침(제자리 덧셈은 허용) 검사와 동기화
  규칙(명세 9절), 카운터(`PERF_OPS`에 1, `PERF_MACS`에 0). QEMU 장치는 공유
  코어를 쓰므로 따로 고치지 않고, 드라이버 검증기는 이미 `OP_ADD`를 받는다.
- V2: `tests/npu-core`의 TFLite Micro 정답값에 int8 `Add`를 더한다(고정 커밋의
  `add.h`).
- V3: 어셈블러로 만든 명령열로 여러 스케일과 영점, clamp, 제자리 덧셈, 두
  반올림 방식, 오류 코드(`BAD_PARAM`, `SRAM_RANGE`, `SYNC`)를 확인한다.
  `OP_ADD`와 그 SET이 `UNKNOWN_CMD`로 멈추기를 기대하던 경우는 고친다.
- V1: kn1-npu qtest에 `OP_ADD` 한 경우와 `PERF_OPS`를 더한다.

완료 조건: `OP_ADD`의 모든 경우가 두 반올림 방식 모두 V2와 비트 단위로
같고(TFLite Micro ADD의 반올림이 명세 8.3절과 같다는 것이 여기서 확인된다),
오류 경우는 명세의 오류 코드와 `ERR_OFFSET`으로 멈춘다. V1이 통과한다.

### M3.3 생성기 (V2)

- `tflite2c`에 `ADD`와 pragma `add`를 더한다(설계 9.2절, 9.3절). 매개변수는
  TFLite Micro의 int8 식(`add_common.cc`)대로 구하고, ReLU는 clamp로 쓴다.
  본문은 `AddFunc`(`add.h`)와 같은 C89 루프다.
- 층이 사슬이어야 한다는 검사를 "층은 앞에서 쓴 배열을 읽는다"로 바꾼다. 배열
  이름(`act0`, `actN`)은 그대로이므로 합성곱·MNIST 모델의 C는 바뀌지 않는다.
- `tests/compiler/test_resnet8_model.py`: 생성기가 쓴 C를 gcc로 돌린 결과를
  V2와 비교한다.

완료 조건:

- 생성기가 쓴 ResNet-8 C를 `gcc -std=c89 -pedantic-errors`로 빌드해 시험 입력
  열다섯을 돌린 최종 출력과 logits가 두 반올림 방식 모두 V2와 같다.
- 같은 C를 gcc의 UBSan(`-fsanitize=undefined`)으로 돌려도 보고가 없다.
- lcc가 ResNet-8 C를 경고 없이 받는다(`-A -Wf-pragma=knpu`).
- 합성곱·MNIST 모델의 결과는 그대로다.

### M3.4 컴파일러: ADD와 두 입력 층 (V3, V4)

- `knpu` 방언: 그래프 연산 `knpu.add`, 명령 연산 `knpu.elementwise`(설계
  9.4절).
- `knpu-import`: `add` 층과 그 검사(두 입력과 출력의 모양, 영점).
  `knpu-schedule`: 입력이 둘인 층. 잔차의 텐서는 지금의 수명 배치로 SRAM에
  남거나 arena를 거치고, 제자리 덧셈은 쓰지 않는다. `knpu-insert-waits`:
  `knpu.elementwise`의 SRAM 범위 셋.
- 사이클 추정: `knpu-estimate`의 ADD 길이(옵션 `add-elements-per-cycle`, 기본
  256)와 `knpu-compile`의 같은 옵션. `knpu-translate`: `OP_ADD`와 그 SET.
- V4: 패스마다 새 연산과 오류 경로의 lit 시험, `OP_ADD` 인코딩, 손으로 계산한
  ADD의 추정 사례, `add` pragma를 실제 lcc 출력으로 거치는 시험.
- V3: SRAM 128 KiB(모든 층이 타일 하나)로 컴파일한 ResNet-8의 NPU 부분을 공유
  코어에서, CPU 부분을 qemu-aarch64에서 돌린다. 공유 코어가 ResNet-8 작업
  하나를 도는 시간을 재서 QEMU 메인 루프를 얼마나 멈추는지 본다(D32).

완료 조건:

- SRAM 128 KiB, 두 반올림 방식으로 컴파일한 ResNet-8의 최종 출력이 시험 입력
  열다섯 모두 V2와 비트 단위로 같다. `--overlap=false`로 컴파일한 모델도 같고,
  대기 삽입을 끄면 동기화 오류로 멈춘다.
- `PERF_MACS`가 32,562,192이고, `PERF_DMA_*`와 `PERF_OPS`가 명령열에서 계산한
  값과 같다.
- 합성곱·MNIST 모델의 V3(벡터 비교 포함)가 그대로 통과한다.

### M3.5 컴파일러: 출력 채널 타일과 2-D DMA (V3, V4)

- `knpu-tile`: 한 행 타일로도 들어가지 않는 합성곱과 FC의 출력 채널
  타일(필요하면 출력 H 타일과 겹쳐 씀), ADD의 출력 H 타일(설계 9.4절).
- `knpu-schedule`: 채널 구간이 있는 타일. 채널 타일들이 SRAM의 같은 입력을
  읽고, 다음 타일의 상수를 미리 읽으며, 출력은 2-D DMA(행 길이 n, 횟수 H·W,
  DRAM 쪽 간격 C)로 arena에 쓴다.
- V4: 출력 채널 타일, ADD의 H 타일, 2-D 저장의 lit 시험.
- V3: SRAM 64 KiB(출력 H 타일 두 층, 출력 채널 타일 한 층)와 32 KiB(출력 H
  타일 일곱 층, 출력 채널 타일 두 층)로 컴파일한 ResNet-8.

완료 조건:

- SRAM 64 KiB·32 KiB, 두 반올림 방식, 겹침이 있을 때와 없을 때로 컴파일한
  ResNet-8의 최종 출력이 시험 입력 열다섯 모두 V2와 비트 단위로 같고, 카운터가
  명령열에서 계산한 값과 같다.
- 합성곱·MNIST 모델의 V3가 그대로 통과한다. 명령열이 바뀌면 벡터를 다시
  만든다.

### M3.6 SRAM 배치 개선과 SRAM 크기 시연 (V3, V4)

- `knpu-schedule`의 SRAM 배치(설계 9.4절):
  - 활성값을 SRAM에 두는 단계를 셋으로 한다: 마지막으로 읽는 층까지(지금의
    상주), 처음 읽는 층까지, arena를 거침. 둘째 단계에서 처음 읽는 층은
    활성값을 arena로 쓰는 DMA와 함께 SRAM에서 읽고(둘 다 읽기라 명세 9절의
    충돌이 아니다), 그 뒤 층은 arena에서 읽는다. 처음 읽는 층이 타일
    하나이면 뒤에서 타일 층이 읽어도 둘째 단계를 쓸 수 있다.
  - SRAM이 모자라 미리 읽기나 상주를 한 단계 거둘 때는, 거두면 들어가지 않은
    버퍼의 수명 동안 SRAM이 비는 선택 가운데 DMA로 더 옮길 바이트가 가장
    적은 것을 고른다(그런 선택이 없으면 지금처럼 가장 큰 것). 배치가
    들어가면 필요 없던 거둠을 비싼 것부터 되돌린다.
  - 한 층짜리 버퍼를 만든 순서로 놓아 들어가지 않으면 큰 것부터 다시 놓아
    본다. 만든 순서로 들어가면 그대로 둔다.
- V4: 처음 읽는 층까지 두는 경우, 거둘 선택을 고르고 되돌리는 경우, 큰
  것부터 다시 놓는 경우의 lit 시험.
- ResNet-8을 SRAM 32, 64, 128 KiB에서 겹침을 켜고 끄고 컴파일해,
  `knpu-compile --report`의 추정 사이클과 V3의 `PERF_DMA_RD_BYTES`,
  `PERF_DMA_WR_BYTES`, `PERF_OPS`를 README의 ResNet-8 절에 표로 남긴다. 어느
  층을 어떻게 나눴고 어느 활성값이 SRAM에 남았는지, `make_model.py`가 찍은
  시험 집합 정확도(float, int8)도 적는다.

완료 조건:

- 겹침이 있을 때 ResNet-8의 추정 사이클이 128 KiB와 64 KiB에서 고치기
  전(145,108, 160,165)보다 작고, 32 KiB와 겹침이 없을 때는 그대로다.
- V3(세 SRAM 크기, 두 반올림 방식, 겹침 유무)와 V4가 통과한다. 합성곱·MNIST
  모델의 명령열은 그대로다(벡터를 다시 만들지 않는다).
- README의 표가 `--report`와 V3에서 나온 값과 같다.

### M3.7 끝에서 끝까지 (V6)

- 시험 벡터 `models/resnet8/vectors/`: 모델 넷(반올림 double·single × SRAM
  64 KiB·32 KiB), 입력 다섯(무작위 셋, 전부 -128, 전부 127. 사진 열 장은
  `models/resnet8/photos/`), 시험 입력 열다섯의 정답값(두 방식). V3가 커밋본과
  지금 생성물이 같은지 검사하고, `make update-vectors`(CMake 대상
  `update-resnet8-vectors`)로 다시 쓴다. 라이선스는 설계 12.4절대로 선언한다.
- `knpu-models`가 ResNet-8 벡터와 사진을 `/usr/share/knpu/resnet8/`에 넣는다.
- V6: 64 KiB 장치에서 ResNet-8 모델 넷 × 입력 열다섯을 돌린다(128 KiB 모델은
  V3에서만 돈다). 먼저 게스트의 파일이 이 트리의 벡터·사진과 같은지
  `md5sum`으로 보고, 다른 반올림 방식의 정답값을 가려내는지 보고, 사진 열 장의
  `top1`을 찍는다. 합성곱·MNIST 시험은 그대로 둔다.
- README: M3 절(ADD와 잔차, 타일링, 사진의 분류 결과), 구성 표, 런타임 절,
  현재 상태.

완료 조건:

- V6이 통과한다. 모든 모델과 입력이 정답값과 같고, `top1`이 정답값의 argmax와
  같고, `PERF_MACS`가 32,562,192이며 다른 카운터가 모델에서 계산한 값과 같고,
  커널 경고가 없다.
- 야간 CI가 V6을 포함해 한 번 성공한다.

---

## M4. 마무리

M3.6에서 찾은 SRAM 배치 문제를 고치고, 드라이버 검증기의 부정 시험을 넓히고,
업스트림 형식 검사와 라이선스 점검(D18)을 CI에 넣는다. 끝으로 README를
정리하고, 새로 받은 저장소에서 README의 절차만으로 빌드와 데모가 되는지
확인한다.

완료 조건: 새로 받은 저장소에서 README의 절차만으로 빌드하고 ResNet-8 데모를
돌릴 수 있다. CI가 모두 통과한다.

### M4.1 SRAM 배치 규칙 선택 (V3, V4)

- `knpu-schedule`의 옵션 `placement`: 기본값 `cost`는 M3.6의 규칙이고,
  `largest`는 M3.6 이전(커밋 `484f090`)의 규칙이다. 이전 규칙은 활성값을
  마지막으로 읽는 층까지 두거나 arena를 거치게 하고, 한 층 안의 버퍼를 만든
  순서로만 놓으며, 들어가지 않으면 가장 큰 선택부터 거두고 되돌리지
  않는다(설계 9.4절).
- `knpu-compile`: tile 뒤의 모듈을 두 규칙으로 각각 스케줄하고 대기 삽입과
  추정까지 거쳐, 추정 사이클이 작은 쪽을 쓴다(같으면 `cost`). M3.6 규칙이 이전
  규칙보다 느린 모델이 있기 때문이다(4 KiB 합성 모델 810개 중 19개, 최대
  70사이클). CPU 코드는 고른 쪽에서 한 번만 만든다. `--placement=cost`나
  `--placement=largest`는 한 규칙으로 고정하고, `--report`는 다른 규칙의 총
  추정 사이클도 찍는다.
- V4: `placement=largest`의 lit 시험, 이전 규칙이 더 빠른 모델을
  `knpu-compile`이 그 규칙으로 컴파일하는 lcc 시험(`compiler/test/lcc/`).

완료 조건:

- `placement=largest`인 `knpu-opt`가 `484f090`의 `knpu-opt`와 같은 MLIR을
  낸다: M3.6에서 비교한 4 KiB 합성 모델 810개, 합성곱·MNIST·ResNet-8 모델의 V3
  경우(SRAM 크기, 겹침 유무)마다.
- `knpu-compile`이 쓰는 추정 사이클이 합성 모델 810개 모두 두 규칙 가운데 작은
  쪽이고, 어느 모델도 `484f090`보다 크지 않다.
- V3와 V4가 통과한다. 벡터 모델이 바뀌면 `make update-vectors`로 다시 만들고
  README의 수치를 고친다.

### M4.2 검증기 부정 시험 (V5)

- `tests/driver/knpu_test.c`:
  - 부정 사례마다 거부 이유를 확인한다. 시험 동안 `drm.debug`의 드라이버
    메시지(0x2)를 켜고, 그 규칙의 메시지와 명령 오프셋이 커널 로그에 나오는지
    본다. 끝나면 원래 값으로 되돌린다.
  - 끝에서 잘린 페이로드의 사례를 페이로드 검사에 닿게 고친다(지금은 `OP_STOP`
    뒤의 워드라는 규칙에 먼저 걸린다).
  - 설계 8.4절의 규칙 가운데 시험이 없던 경우를 더한다: 페이로드 비트가 선 알
    수 없는 명령, 앞 `OP_DMA` 뒤 바뀐 상태로 다시 `OP_DMA`, DST 쪽의 알 수
    없는 공간 종류와 영역 번호, 4 GiB를 넘는 영역 끝(큰 오프셋, 큰 STRIDE), 한
    영역에 `OP_DMA`가 여럿일 때의 최댓값, 마지막 영역 번호 7.
  - 무작위 변형: 유효한 명령열의 워드를 고정 시드로 바꾸거나 지우거나 끼운
    명령열 1,000개를 넣고, 시험 프로그램 안의 참조 검사기(설계 8.4절의 규칙)와
    판정(수락·거부)과 메시지(거부한 규칙과 명령 오프셋, 받은 `OP_DMA`의 영역
    범위)를 비교한다. 어긋나면 시드, 변형 번호, 명령열을 찍는다.
  - 출력 영역 표시: 타임아웃 시험이 붙잡아 둔 작업의 입력·출력 버퍼를
    dma-buf로 내보내, 읽는 쪽이 기다려야 하는 쓰기 펜스가 출력 버퍼에만 있는지
    본다. 이 표시는 펜스의 종류만 바꾸므로 다른 ioctl로는 보이지 않는다.
  - 커널 로그: 디버그 메시지가 로그 버퍼를 여러 번 덮으므로, 시험 프로그램이
    `/dev/kmsg`를 처음부터 읽어 모든 레코드에서 경고를 검사한다.

완료 조건:

- V5가 통과한다. 부정 사례마다 기대한 규칙의 메시지가 커널 로그에 나오고,
  무작위 변형 명령열 모두에서 커널과 참조 검사기의 판정과 메시지가 같다.
- 검증기의 규칙을 하나씩 일부러 끈 모듈로 돌리면 시험이 실패한다. 64비트
  오버플로 검사는 필드 폭으로 닿을 수 없으므로, 검사를 지운 모듈 대신 합을
  32비트로 셈하는 모듈로 확인한다.

### M4.3 업스트림 형식 검사 (CI)

- `tests/style/checkpatch.sh`: 빌드가 받는 tarball(QEMU 11.1.3, Linux 7.3.1,
  U-Boot v2026.10, TF-A v2.15.0)에서 `checkpatch.pl`과 그 업스트림의 설정만
  임시 디렉터리에 풀고, 저장소 파일을 업스트림 트리의 경로에 복사해 각
  업스트림의 규칙으로 검사한다. QEMU는 `hw/qemu/`, `hw/npu-core/`와 연결 패치,
  리눅스는 `linux/`(드라이버, UAPI 헤더, DT, 바인딩), U-Boot는
  `firmware/u-boot/`(U-Boot의 `.checkpatch.conf`), TF-A는
  `firmware/tf-a/`(리눅스 `checkpatch.pl`과 TF-A의 `.checkpatch.conf`)다.
  오류나 경고가 하나라도 나오면 실패하고, 허용하는 경고는 이유와 함께
  스크립트에 적는다: `hw/npu-core/`의 MIT 라이선스 경고(D33), U-Boot 보드의
  livetree API 경고 둘(업스트림 `qemu-arm` 보드와 같은 DRAM 설정 함수).
- TF-A 포트의 경고(쓰지 않는 매크로 인자)를 고친다.
- `make check-style`: tarball은 `buildroot/dl/`의 것을 쓰고, 없으면
  Buildroot로 받는다.
- push CI 작업 `checkpatch`: Buildroot의 내려받기 대상으로 tarball 넷을
  받아(같은 `.hash`로 확인) 캐시하고 `make check-style`을 부른다. 바인딩과
  DT의 스키마 검사는 M1.6부터 push CI의 dt-binding 작업이 돈다.

완료 조건:

- `make check-style`이 로컬과 push CI에서 통과한다.
- 허용 목록 밖의 경고(예: 긴 줄)를 일부러 넣으면 네 업스트림 각각에서
  실패한다.

### M4.4 라이선스 점검 (CI)

- `knpu`, `knpu-tests`, `knpu-runtime`, `knpu-models` 패키지: `legal-info`가
  버전이 없는 패키지를 건너뛰므로 `_VERSION`을 `custom`으로 둔다. rsync 뒤
  훅으로 쓰는 라이선스 전문을 빌드 디렉터리에 복사하고 `_LICENSE_FILES`로
  가리키며, 그 해시를 패키지의 `.hash`에 둔다(설계 12.4절). 리눅스와 TF-A의
  버전 고정 해시 파일에도 라이선스 파일의 해시를 더한다.
- 훅이 복사해 넣는 파일(U-Boot 보드와 defconfig, TF-A 플랫폼, QEMU의 kn1
  소스와 공유 코어, `knpu_cmd.h`)은 `legal-info`가 저장하지 않으므로,
  legal-info 뒤 훅이 저장된 소스 옆에 둔다.
- 야간 CI: 전체 빌드 뒤 `legal-info`를 돌려, Buildroot 자체 소스를 저장하지
  않았다는 것 밖의 경고나 오류가 나거나 `br2-external/`의 패키지가
  매니페스트에 없으면 실패한다. 예약 시각을 정각에서 옮긴다(GitHub는 정각의
  예약 실행을 늦출 수 있다).
- `tests/license/check-licenses.py`: `reuse lint --json` 보고서에서 파일마다의
  SPDX 라이선스(머리글과 `REUSE.toml`)를 읽어 설계 12.4절의 경로별 표와
  대조한다. 표에 적은 예외(MNIST 데이터는 CC-BY-SA-3.0, 사진은 CC0-1.0)만
  받는다. `make check-licenses`(build/의 venv에 reuse 6.3.0을 설치하고
  `reuse lint`에 이어 표 검사)와 push CI의 reuse-lint 작업이 부른다.

완료 조건:

- 전체 빌드 뒤 `legal-info`가 `br2-external/`의 패키지를 모두 매니페스트에
  담고 라이선스 전문과 복사해 넣은 파일을 저장하며, Buildroot 자체 소스를
  저장하지 않았다는 것 밖의 경고를 내지 않는다(로컬과 야간 CI).
- `make check-licenses`와 `reuse lint`가 로컬과 push CI에서 통과하고, 표와
  다른 라이선스를 일부러 넣으면 표 검사가 실패한다.

### M4.5 README와 새 저장소 확인

- `make demo`: 이미지를 사람 손 없이 부팅해 64 KiB SRAM용 ResNet-8 모델(반올림
  double)로 사진 열 장을 `knpu-infer`로 돌리고, 사진마다 top1의 클래스 이름과
  정답값 일치 여부를 찍은 뒤 끈다(설계 10절).
- README(D19): 소개 바로 뒤에 결과 요약(ResNet-8이 kn1-evb에서 TFLite Micro와
  비트 단위로 같음, 정확도, 사진 분류, 사이클 추정)과 빠른 시작(clone, `make`,
  `make demo`, `make test`)을 둔다. 빌드 절에 `hw/qemu/`나 `hw/npu-core/`를
  바꾼 뒤 QEMU를 다시 빌드하는 방법(`make br-host-qemu-kn1-dirclean` 뒤
  `make`)을, 데모 절에 `make demo`와 손으로 돌리는 절차를 적는다. M4.1~M4.4의
  변경을 해당 절에 반영하고 현재 상태를 고친다.
- 새 저장소 확인: GitHub에서 `git clone --recurse-submodules`로 새로 받은
  저장소에서 README의 순서대로 `make`, `make demo`, `make test`,
  `make check`를 돌린다. Buildroot의 내려받기도 새로 한다.

완료 조건:

- 새로 받은 저장소에서 README의 절차만으로 빌드하고, `make demo`가 사진 열
  장의 top1과 정답값 일치를 찍는다. `make test`와 `make check`가 통과한다.
- push CI와 야간 CI가 모두 통과한다.

---

## 그 뒤

- PCIe ECAM 호스트 브리지와 GICv3 ITS를 거치는 MSI(D3).
- virtio-mmio를 대신할 실제 저장 장치 컨트롤러(설계 13절 1번).

---

## 설계의 열린 항목을 다루는 단계

| 설계 13절 항목 | 다루는 단계 |
|----------------|-------------|
| 1. 실제 저장 장치 컨트롤러 | 그 뒤 |
| 2. 비트 단위 NPU 명세 | M1.3 |
| 3. 벤더 접두사 `kiwon`의 스키마 검사 | M1.2 |
| 4. FC를 1x1 합성곱으로 돌린 결과의 비트 일치 | M2.4 |
| 5. 최소 `kn1_defconfig`의 GEM DMA 헬퍼와 `drm_sched` | M2.7 |
| 6. CPU 연산의 lcc-opt 경로 | M2.5 |
