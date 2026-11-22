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
| M2 | MNIST CNN | MNIST CNN 전체가 두 반올림 방식 모두에서 비트 단위로 같다 |
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

- `runtime/`: Rust `libknpu` 크레이트(UAPI 바인딩, `model.knpu` 읽기, arena
  복사, 제출과 대기, perfmon)와 `knpu-infer`(D41).
- `br2-external`에 cargo 패키지를 더하고 `host-rust-bin`으로 빌드한다.
- V6: kn1-evb를 펌웨어 체인으로 부팅하고, `knpu-infer`로 합성곱 모델을 돌려
  V2와 비교하고, 카운터를 출력한다.
- CI: 매일 밤과 수동 실행 때 Buildroot 전체 빌드와 V1, V5, V6을 돌린다(D42).

완료 조건: V6이 통과하고, 야간 CI 작업이 한 번 성공한다.

---

## M2. MNIST CNN

- 연산: AVGPOOL, 1x1 합성곱으로 처리하는 FC, `RESHAPE`, CPU에서 도는
  softmax.
- 생성기와 pragma: `FULLY_CONNECTED`, `AVERAGE_POOL_2D`, `RESHAPE`,
  `SOFTMAX`를 C89로 쓰고, NPU 연산의 pragma 어휘를 더한다(설계 9.3절).
- CPU 연산: softmax 같은 CPU 연산을 lcc-opt로 최적화해 aarch64 코드로
  내린다(설계 13절 6번). lcc-opt를 받는 방법과 `model.knpu`의 CPU 섹션을
  정해 구현한다.
- FC를 1x1 합성곱으로 돌린 결과가 `FullyConnected`와 비트 단위로 같은지
  확인한다(설계 13절 4번).
- 컴파일러: 여러 층으로 된 그래프, 층 사이의 SRAM과 arena 배치, 사이클
  추정(D31).
- 커널 설정: kn1에 필요한 것만 남긴 `kn1_defconfig`(D35).

완료 조건: MNIST CNN이 kn1-evb에서 두 반올림 방식 모두 TFLite Micro와 비트
단위로 같다(V6). 이중 버퍼링이 있을 때와 없을 때의 사이클 추정을 보고한다.
`kn1_defconfig`로 부팅한다.

---

## M3. ResNet-8

- 연산: ADD(입력 스케일링 포함), stride 2 합성곱, 잔차 연결.
- 64 KiB에 들어가지 않는 층의 타일링: H 타일과 출력 채널 타일, 2-D DMA,
  arena로 내보내기.
- SRAM 크기 시연: 같은 모델을 32, 64, 128 KiB로 컴파일해 `PERF_DMA_*`와
  사이클 추정을 비교한다(설계 6.6절).

완료 조건: MLPerf Tiny ResNet-8이 kn1-evb에서 TFLite Micro와 비트 단위로
같다(V6). SRAM 크기 시연 결과를 문서로 남긴다.

---

## M4. 마무리

- README: 프로젝트 소개, 구조, 빌드와 실행 방법, 결과(D19).
- 검증기 부정 시험을 넓힌다.
- 업스트림 형식 검사: QEMU와 리눅스 코드의 `checkpatch.pl`, 바인딩과 DT
  스키마 검사.
- 라이선스와 SPDX 점검(D18).

완료 조건: 새로 받은 저장소에서 README의 절차만으로 빌드하고 ResNet-8
데모를 돌릴 수 있다. CI가 모두 통과한다.

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
| 4. FC를 1x1 합성곱으로 돌린 결과의 비트 일치 | M2 |
| 6. CPU 연산의 lcc-opt 경로 | M2 |
