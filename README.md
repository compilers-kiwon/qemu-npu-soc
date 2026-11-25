<!-- SPDX-FileCopyrightText: 2026 Kiwon Kwon -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# qemu-npu-soc

가상 SoC `kn1`과 보드 `kn1-evb`를 QEMU에 만들고, 그 위에 TF-A, U-Boot,
리눅스, NPU 드라이버, 모델 컴파일러, 런타임까지 한 저장소에서 올린다.
SoC에는 int8 합성곱 NPU `kn1-npu`가 있다. TFLite 모델을 C89로 옮겨 lcc와
MLIR로 컴파일해 이 NPU에서 돌리고, 결과를 TFLite Micro 참조 구현과 비트
단위로 비교한다.

- 설계: [docs/design.md](docs/design.md)
- 구현 순서: [docs/plan.md](docs/plan.md)

## 구성

설계 12.1절의 구성이다. 아직 없는 디렉터리는 해당 단계에서 생긴다.

| 경로 | 내용 |
|------|------|
| `docs/` | 설계 문서, 구현 계획, NPU 명세 |
| `hw/qemu/` | kn1 SoC 장치, kn1-evb 머신, kn1-npu 장치 모델, 연결 패치 |
| `hw/npu-core/` | QEMU와 호스트 시험이 함께 쓰는 NPU 명령 해석기 |
| `firmware/tf-a/` | TF-A 플랫폼 포트(`plat/kiwon/kn1`) |
| `firmware/u-boot/` | U-Boot 보드 포트, 연결 패치 |
| `linux/` | `knpu` 드라이버, SoC `.dtsi`와 보드 `.dts`, 바인딩 YAML |
| `compiler/` | 생성기 `tflite2c`(Python), MLIR 다이얼렉트·패스·도구(C++) |
| `runtime/` | `libknpu`와 `knpu-infer`(Rust) |
| `br2-external/` | Buildroot 외부 트리(defconfig, 커널 설정, 패키지, 패치) |
| `buildroot/` | Buildroot 서브모듈(고정) |
| `models/` | `.tflite` 모델과 만드는 스크립트, 컴파일한 시험 벡터 |
| `tests/` | 장치, 드라이버, 컴파일러, 끝-끝 시험 |

## 고정 버전

| 프로젝트 | 버전 | 빌드 주체 |
|----------|------|-----------|
| QEMU | v11.1.3 | `br2-external`의 Buildroot 호스트 패키지 |
| Linux | v7.3.1 | Buildroot |
| TF-A | v2.15.0 | Buildroot(git.trustedfirmware.org의 태그) |
| U-Boot | v2026.10 | Buildroot(source.denx.de의 태그) |
| Buildroot | 2026.08.1 | 서브모듈 |
| LLVM/MLIR | 23.x | 호스트의 apt.llvm.org 패키지 |
| Rust | 1.97.1(Buildroot 2026.08.1의 `host-rust-bin`) | Buildroot |

컴파일러와 그 시험이 빌드할 때 받는 것(모두 SHA-256 검사): lcc
`v4.2-llvm.3`의 aarch64 mlir 드라이버, 같은 태그의 lcc_llvm 소스(lcc-opt만
빌드한다), TensorFlow Lite Micro `fd2a85d`(소스와 Python 스키마),
FlatBuffers v25.9.23(C++ 소스). 생성기가 쓰는 PyPI `flatbuffers`도
25.9.23으로 고정한다.

## 요구 사항

Buildroot 매뉴얼이 요구하는 호스트 패키지가 필요하다: which, sed, make,
binutils, diffutils, gcc, g++, bash, patch, gzip, bzip2, perl, tar, cpio,
unzip, rsync, file, bc, findutils, awk, wget. Debian 계열에서는
`build-essential`이 컴파일러와 make를 함께 설치한다.

```bash
sudo apt-get install build-essential bc bzip2 cpio file git patch perl rsync unzip wget
```

`buildroot/utils/check-package`를 돌리려면 `python3-magic`과 `python3-flake8`도 필요하다.
컴파일러 시험(`make check-compiler`)은 venv를 만들므로 `python3-venv`도 필요하다.

## 빌드

```bash
git clone --recurse-submodules https://github.com/compilers-kiwon/qemu-npu-soc.git
cd qemu-npu-soc
make
```

저장소 맨 위의 `Makefile`이 빌드와 시험의 진입점이다. `make help`가 대상
목록을 보여 준다. `make`는 처음에 `kn1_evb_defconfig`을 읽어 `output/`을
만들고, Buildroot로 툴체인, TF-A 2.15.0(`plat/kiwon/kn1`), U-Boot 2026.10,
Linux 7.3.1(kn1에 필요한 것만 켠 `kn1_defconfig`), NPU 드라이버 모듈
`knpu`와 V5 시험 프로그램, Rust 런타임 `knpu-infer`와 시험 모델, 루트 파일
시스템, 디바이스 트리, 그리고 kn1-evb 머신을 넣은 QEMU 11.1.3을 모두
빌드한다. 결과물은 `output/images/`에
놓인다: `bl1.bin`과 `fip.bin`(BL2, BL31, U-Boot)을 합친 부트 플래시 이미지
`flash.bin`, 루트 파일 시스템 `rootfs.ext4`(`/boot/Image`,
`/boot/kn1-evb.dtb`, `/boot/extlinux/extlinux.conf` 포함)를 GPT 파티션 하나에
담은 부트 디스크 `disk.img`, `start-qemu.sh`. QEMU 실행 파일은
`output/host/bin/qemu-system-aarch64`다.

```bash
make run
```

리셋 벡터의 BL1이 플래시의 FIP에서 BL2와 BL31을 올리고, BL31이 PSCI를 맡은
채 U-Boot로 넘어간다. U-Boot(보드 `kiwon/kn1-evb`)는 자기 디바이스 트리로
동작하고, 디스크의 `extlinux.conf`를 읽어 커널과 `kn1-evb.dtb`를 올린다.
터미널에는 TF-A 콘솔(보안 UART1)과 U-Boot·리눅스
콘솔(UART0)이 함께 나온다. 키보드는 UART0에 연결돼 있고, Ctrl-a c로
바꾸며, Ctrl-a x로 QEMU를 끝낸다.

`root`로 로그인하면 셸이 뜬다. `poweroff`와 `reboot`는 PSCI를 거쳐 BL31이
보안 GPIO로 처리하며, 각각 QEMU 종료와 BL1부터의 재부팅으로 이어진다.

빌드한 이미지로 부팅 시험을 돌리면, 사람 손 없이 펌웨어 체인 전체로 부팅해
CPU 4개, DRAM 4 GiB, RTC, 워치독, `reboot`, `poweroff`, 4 GiB가 아닌 `-m`
거부를 확인한다. 콘솔 로그는 `output/test-logs/`에 남는다.

```bash
make test-boot
```

`make test`는 부팅 시험, V1, V5, V6을 차례로 돌린다.

TF-A 포트(`firmware/tf-a/`)나 U-Boot 보드 포트(`firmware/u-boot/`)를 고친
뒤에는 Buildroot가 소스 트리를 다시 풀어 복사하도록
`make br-arm-trusted-firmware-dirclean`이나 `make br-uboot-dirclean`을 먼저
돌린다. 이처럼 Buildroot의 대상은 `make br-<대상>`으로 부른다(예:
`make br-linux-rebuild`, `make br-menuconfig`).
U-Boot는 FIP 안에 들어가므로, U-Boot를 고쳤을 때도 TF-A를 다시 빌드한다.

WSL은 `PATH`에 공백이 든 Windows 경로를 붙이는데, Buildroot는 그런 `PATH`를
거부한다. `Makefile`은 Buildroot를 부를 때 그 항목들을 뺀다.

NPU 바인딩과 디바이스 트리는 리눅스 스키마로 검사한다. 호스트에
dt-schema(pip `dtschema`), yamllint, flex, bison이 있으면, 다음 스크립트가
커널 tarball을 임시 디렉터리에 풀어 바인딩의 `dt_binding_check`와
`kn1-evb.dtb`의 `CHECK_DTBS`를 돌린다. 전체 빌드가 받아 둔 tarball을 쓰면
된다.

```bash
tests/dt/check-dt.sh buildroot/dl/linux/linux-7.3.1.tar.xz
```

## NPU 공유 코어 시험

`hw/npu-core/`는 [NPU 명세](docs/npu-spec.md)대로 명령열을 실행하는 공유
코어이고, QEMU 장치와 호스트 도구가 함께 쓴다. 합성곱(`OP_CONV`),
덧셈(`OP_ADD`), 평균 풀링(`OP_AVGPOOL`), DMA, 대기 명령과 SET 명령까지 명세의
명령을 모두 구현했다.

호스트 시험에는 CMake, C/C++ 컴파일러, Python 3가 필요하다. 구성 단계에서
고정한 커밋의 TensorFlow Lite Micro 파일 12개를 내려받아 SHA-256을 검사한다.

```bash
make check-npu-core
```

- V2: TFLite Micro 참조 합성곱, 평균 풀링, 덧셈을 이중·단일 반올림으로 한
  번씩 빌드해 정답값을 만든다(`tflm_golden_double`, `tflm_golden_single`).
- V3: 어셈블러(`knpu_asm.py`)로 만든 명령열을 공유 코어(`knpu_run`)에서 실행해
  정답값과 바이트 단위로 비교한다. 1x1·3x3 커널, stride 1·2, VALID·SAME
  패딩, 두 반올림 방식, 채널별 재양자화, 2-D DMA, 덧셈의 입력 스케일링과
  제자리 덧셈을 다룬다. 명세의 오류 코드와 `ERR_OFFSET`, 대기 누락이 내는
  동기화 오류, 암묵 대기도 확인한다.

## NPU 장치 시험

QEMU의 `kiwon-kn1-npu` 장치(`hw/qemu/hw/misc/kiwon-kn1-npu.c`)는 레지스터,
인터럽트(GIC SPI 32), 작업 실행을 맡고, 명령열은 공유 코어가 실행한다.
`CTRL.START`는 bottom half를 예약하므로 완료는 언제나 MMIO 쓰기 뒤에 온다.
SRAM 크기는 `-global kiwon-kn1-npu.sram-size=32768`처럼 바꿀 수 있다.

V1은 전체 빌드가 남긴 QEMU 빌드 트리에서 돈다. kn1-npu qtest(레지스터,
인터럽트, DMA와 명령, 오류, 카운터, soft reset, SRAM 크기)와 QEMU의
`device-introspect-test`, `qom-test`를 돌린다.

```bash
make test-v1
```

## knpu 드라이버 시험

`linux/knpu/`는 DRM accel 드라이버(`/dev/accel/accel0`)이고, Buildroot가
트리 밖 모듈로 빌드한다. 사용자 공간 인터페이스는
`linux/include/uapi/drm/knpu_accel.h`의 ioctl 9개다. 명령열은 커널 버퍼로
복사한 뒤 검증하고(설계 8.4절), 제출할 때는 영역마다 붙인 버퍼가 명령열이
닿는 크기보다 작지 않은지 본다. 보드 DT가 NPU 전용 CMA 풀을 4 GiB 위에
두므로 NPU 버퍼의 주소는 모두 64비트다.

V5는 펌웨어 체인으로 부팅한 게스트에서 돈다. `knpu` 모듈을 올려 시험
프로그램 `knpu-test`(`tests/driver/knpu_test.c`)를 돌리고, 모듈을 내렸다가
다시 올려 한 번 더 돌린다. ioctl, 검증기와 제출 검사의 부정 시험, 4 GiB 위의
버퍼, 장치 오류와 타임아웃 뒤의 복구(debugfs `inject_hang`), 성능 카운터,
runtime PM을 확인한다.

```bash
make test-v5
```

## 모델 컴파일러

`compiler/`는 모델 컴파일러다(설계 9절). 생성기 `tflite2c`가 `.tflite`를
C89로 옮기고, 층 정보는 층 함수 첫 줄의 `#pragma knpu`에 적는다. lcc가 이
C를 MLIR로 내리면, `knpu-compile`이 MLIR 패스(가져오기, 타일링, 스케줄링,
대기 명령 삽입)를 거쳐 `model.knpu`([형식](docs/model-format.md))를 쓴다.
NPU에서는 합성곱, 평균 풀링, FC(1x1 합성곱으로)를 돌리고, softmax처럼 NPU에서
돌지 않는 층은 aarch64 코드로 만들어 모델 파일에 넣는다(아래 MNIST CNN 절).

apt.llvm.org의 LLVM 23 패키지(`libmlir-23-dev`, `mlir-23-tools`,
`llvm-23-dev`, `llvm-23-tools`, `clang-23`, `lld-23`), CMake, Ninja,
Python 3과 `python3-venv`가 필요하다. 시험에는 `gcc-aarch64-linux-gnu`,
`libc6-dev-arm64-cross`, `qemu-user`도 필요하다(모델의 CPU 코드를
qemu-aarch64에서 돌린다). `make check-compiler`가 `build/py-venv`에 venv를
만들어 PyPI `flatbuffers` 25.9.23을 받고, 구성 단계에서 lcc 릴리스
압축(x86_64에서 도는 aarch64용 mlir 드라이버), lcc-opt를 빌드할 lcc_llvm
소스 압축, TensorFlow Lite Micro와 FlatBuffers 소스, TFLite Python 스키마를
내려받아 SHA-256을 검사한다. lcc는 라이선스 때문에 저장소에 넣지 않는다(설계
12.4절).

```bash
make check-compiler
```

`make compiler`는 시험 없이 빌드만 한다. MLIR이 다른 곳에 있으면
`MLIR_DIR=`로 준다.

- V4: `compiler/test/`의 lit 시험. 방언 검증기, 패스마다의 결과와 오류,
  명세 12절 예시와 워드 단위로 같은 명령열, 실제 lcc 출력 하나를 다룬다.
- V2, V3: `tests/compiler/`. 시험 모델 `models/conv/conv.tflite`(14x14x8
  입력, 3x3 합성곱, 출력 16채널)를 두 반올림 방식으로 다룬다. TensorFlow
  Lite Micro 인터프리터가 정답값을 만들고, 생성기가 쓴 C를 gcc로 돌린
  결과와, SRAM 64 KiB(타일 하나)와 4 KiB(타일 여럿, 이중 버퍼링)로
  컴파일한 `model.knpu`를 공유 코어에서 돌린 결과를 비교한다. 대기 명령을
  빼고 컴파일하면 동기화 오류로 멈추는지도 본다.

모델 하나를 직접 컴파일하는 순서는 다음과 같다. `--round`는 대상의 반올림
방식이다.

```bash
L=build/compiler/deps/lcc-v4.2-llvm.3-mlir-aarch64-linux
build/py-venv/bin/python3 compiler/tflite2c/tflite2c.py \
    --schema build/compiler/deps/schema_py_generated.py --round double \
    models/conv/conv.tflite -o conv.c
LCCDIR=$L $L/lcc -Wf-pragma=knpu -S conv.c -o conv.mlir
build/compiler/bin/knpu-compile --sram-size=65536 conv.mlir -o conv.knpu
```

시험 모델은 `models/conv/make_model.py`가 TensorFlow 2.21.0으로 만든 것이다.
CI와 컴파일러에는 TensorFlow가 필요 없다.

## 런타임과 끝-끝 시험

`runtime/`은 Rust cargo 작업 공간이다(설계 10절). `libknpu`가 `model.knpu`를
읽어 검사하고, 영역 버퍼와 명령열을 만들어 `knpu` 드라이버로 실행한다.
`knpu-infer`는 모델 하나를 입력 하나로 돌려 성능 카운터를 찍고, 기대값과
비교한다. 외부 크레이트는 쓰지 않는다. Buildroot가 `knpu-infer`를 `/usr/bin`에,
시험 모델과 입력, 정답값을 `/usr/share/knpu/conv/`(합성곱,
`models/conv/vectors/`)와 `/usr/share/knpu/mnist/`(MNIST CNN,
`models/mnist/vectors/`와 숫자 이미지)에 넣는다.

부팅한 kn1-evb에서 4 KiB SRAM용으로 컴파일한 모델(타일 5개)을 돌리면 다음과
같다.

```
# modprobe knpu
# cd /usr/share/knpu/conv
# knpu-infer conv-double-4k.knpu input-0.bin --expect expected-double-0.bin
model=conv-double-4k.knpu format=0.0 sram_size=4096 command_bytes=520 model_ops=16
input_shape=1x14x14x8 input_scale=0.003921531 input_zero_point=-128 output_shape=1x14x14x16 output_scale=0.0072401874 output_zero_point=-128
device=/dev/accel/accel0 npu=0.0 device_sram_size=65536
macs=225792 dma_rd_bytes=3808 dma_wr_bytes=3136 ops=16
match=yes
```

호스트 시험(모델 파싱과 검사, UAPI 정의와 C 헤더 대조, 명령줄)은 Rust
1.97.1로 돈다. 전체 빌드를 했으면 Buildroot가 만든 cargo(`output/host/bin`)를
쓰고, 아니면 시스템의 cargo를 쓴다.

```bash
make check-runtime
```

V6은 펌웨어 체인으로 부팅한 게스트에서 합성곱 모델 넷(반올림 double·single ×
SRAM 64 KiB·4 KiB)을 입력 다섯으로, MNIST CNN 모델 넷(SRAM 64 KiB·32 KiB)을
입력 열다섯으로 돌려 TensorFlow Lite Micro의 출력과 비교하고, `top1`, 카운터,
잘린 모델의 거부를 확인한다. MNIST의 기대값은 저장소의 벡터에서 계산하므로,
먼저 이미지에 든 파일이 저장소의 것과 같은지 본다.

```bash
make test-v6
```

컴파일러 출력이 바뀌면 시험 벡터(`models/*/vectors/`)를 다시 만들어 커밋한다.
커밋본이 다르면 V3(`ctest`)가 실패한다. MNIST 모델의 CPU 코드는 LLVM 23 도구의
판에 따라 바이트가 달라지므로(`ld.lld`가 판 번호를 남긴다), apt.llvm.org의
판이 오를 때도 다시 만든다(설계 12.2절). 새 벡터를 이미지에 넣으려면
`knpu-models` 패키지와 이미지를 다시 만든다.

```bash
make update-vectors
make br-knpu-models-rebuild all
```

## MNIST CNN

M2는 MNIST 손글씨 숫자 분류 CNN(설계 6.6절)을 모든 층에 관통시킨다.
`models/mnist/make_model.py`가 TensorFlow 2.21.0으로 MNIST를 학습해 int8로
바꾼 `models/mnist/mnist.tflite`는 입력이 28x28x1이고 층은
conv3x3(8)-avgpool-conv3x3(16)-avgpool-FC64-FC10-softmax이며, MAC 214,304개와
파라미터 27,562개다. `models/mnist/digits/`의 숫자 이미지 열 장은 MNIST 시험
집합에서 숫자마다 처음 나오는 이미지다.

- NPU: 공유 코어에 평균 풀링(`OP_AVGPOOL`)을 더했고, FC는 1x1 합성곱으로
  돌린다. 모든 층이 타일 하나에 들어가므로 겹침은 층 사이에서 생긴다.
  컴파일러는 층 사이 활성값을 SRAM에 남기고, 다음 층의 상수를 앞 층을
  계산하는 동안 미리 읽는다.
- CPU: softmax는 생성기가 TFLite Micro의 int8 Softmax를 옮긴 C89 함수다.
  `knpu-compile`이 이 함수를 lcc_llvm의 lcc-opt(affine·linalg로 올려
  최적화)와 LLVM 23의 `mlir-translate`, clang, `ld.lld`로 aarch64 공유 객체로
  만들어 `model.knpu`의 CPU 섹션(형식 minor 1)에 넣는다. 런타임은 이 코드를
  `memfd_create`와 `dlopen`으로 올려 두고 NPU 작업 뒤 실행하므로,
  `knpu-infer`로는 실행 파일처럼 믿을 수 있는 모델 파일만 돌린다.
- 커널: `kn1_defconfig`(설계 8.1절)로 빌드한 커널에서 돈다.
- 검증: 두 반올림 방식 모두 시험 입력 열다섯(숫자 이미지 열 장, 무작위 셋,
  전부 -128, 전부 127)의 출력이 TFLite Micro와 비트 단위로 같다. 생성기의 C를
  gcc로 돌리고(V2), 컴파일한 모델의 NPU 부분을 공유 코어에서, CPU 부분을
  qemu-aarch64에서 돌리고(V3), kn1-evb에서 `knpu-infer`로 돌린다(V6). 숫자
  이미지 열 장은 모두 제 숫자로 분류된다.

부팅한 kn1-evb에서 32 KiB SRAM용으로 컴파일한 모델에 숫자 7 이미지를 넣으면
다음과 같다. `cpu_steps=3`은 NPU 작업 뒤 CPU가 하는 단계(logits 복사, softmax
호출, 출력 복사)이고, `dma_wr_bytes=10`은 층 사이 활성값이 SRAM에 남아 NPU가
logits 10바이트만 쓴다는 뜻이다.

```
# modprobe knpu
# cd /usr/share/knpu/mnist
# knpu-infer mnist-double-32k.knpu digit-7.bin --expect expected-double-digit-7.bin
model=mnist-double-32k.knpu format=0.1 sram_size=32768 command_bytes=556 model_ops=12 cpu_steps=3
input_shape=1x28x28x1 input_scale=0.003921569 input_zero_point=-128 output_shape=1x1x1x10 output_scale=0.00390625 output_zero_point=-128
device=/dev/accel/accel0 npu=0.0 device_sram_size=65536
macs=214304 dma_rd_bytes=29440 dma_wr_bytes=10 ops=12
top1=7
match=yes
```

### 사이클 추정

장치는 시간을 모델링하지 않으므로(설계 D1), 컴파일러가 대기 명령까지 들어간
명령열을 명세 9절의 큐 모델로 따라가 사이클을 추정한다. 가정값은 사이클당 MAC
256개, DMA 16바이트, 풀링 창 원소 256개이고, 실제 칩의 값이 아니라 스케줄을
비교하려는 값이다(설계 9.1절). MNIST CNN의 추정치는 다음과 같고, 두 반올림
방식이 같다.

| SRAM | 겹침 있음 | 겹침 없음(`--overlap=false`) |
|------|-----------|------------------------------|
| 64 KiB | 1,896 | 3,855 |
| 32 KiB | 2,025 | 3,855 |

겹침이 있으면 층 사이 활성값이 SRAM에 남아 DMA는 입력과 상수를 읽고 logits
10바이트만 쓰며, 다음 층의 상수를 미리 읽는다. 가장 긴 작업은 FC 400->64의
상수 26,368바이트를 읽는 DMA(1,648사이클)인데, 이 DMA가 64 KiB에서는
144사이클에, 32 KiB에서는 273사이클에 시작해 그 차이만큼 끝이 늦다. 겹침이
없으면 층마다 입력과 상수를 읽고 출력을 DRAM에 쓴 뒤 다음 층으로 가므로,
모든 층이 타일 하나에 들어가는 이 모델에서는 SRAM 크기와 관계없다.

층별 보고는 `--report`가 찍는다. 합성곱 모델과 같은 순서로 컴파일하고,
`--overlap=false`를 더하면 겹침 없는 스케줄의 보고가 나온다.

```bash
L=build/compiler/deps/lcc-v4.2-llvm.3-mlir-aarch64-linux
build/py-venv/bin/python3 compiler/tflite2c/tflite2c.py \
    --schema build/compiler/deps/schema_py_generated.py --round double \
    models/mnist/mnist.tflite -o mnist.c
LCCDIR=$L $L/lcc -Wf-pragma=knpu -S mnist.c -o mnist.mlir
build/compiler/bin/knpu-compile --sram-size=32768 --report mnist.mlir -o mnist.knpu
```

## CI

GitHub Actions에서는 push마다 defconfig, `check-package`, `reuse lint`,
NPU 공유 코어 시험(V2, V3), 모델 컴파일러 시험(V2~V4), 런타임 호스트 시험,
바인딩과 디바이스 트리의 스키마 검사를 돌리고(`ci.yml`), 매일 밤과 수동 실행
때 전체 빌드, `kn1_defconfig` 형태 검사, 부팅 시험, V1, V5, V6을
돌린다(`nightly.yml`). 두 워크플로는 위의 `make` 대상을 부른다.

## 현재 상태

| 마일스톤 | 목표 | 상태 |
|----------|------|------|
| M1 | 합성곱 하나를 모든 층에 관통시킨다 | 끝났다. M1.1 저장소 뼈대, M1.2 펌웨어 체인 부팅(QEMU 머신, 디바이스 트리, TF-A, U-Boot, 리눅스 셸), M1.3 NPU 명세 v0([docs/npu-spec.md](docs/npu-spec.md)), M1.4 NPU 공유 코어와 정답값 비교(V2, V3), M1.5 QEMU NPU 장치(V1), M1.6 `knpu` 드라이버(V5), M1.7 모델 컴파일러 최소 경로(V2~V4), M1.8 런타임과 끝-끝 시험(V6) |
| M2 | MNIST CNN | 끝났다. M2.1 MNIST 모델과 정답값(V2), M2.2 공유 코어의 `OP_AVGPOOL`(V1~V3), M2.3 생성기의 새 연산과 softmax(V2), M2.4 컴파일러 NPU 경로와 사이클 추정(V3, V4), M2.5 컴파일러 CPU 경로와 lcc-opt(V3, V4), M2.6 런타임의 CPU 단계와 top-1, M2.7 커널 설정 `kn1_defconfig`, M2.8 끝-끝 시험(V6)과 사이클 추정 보고 |
| M3 | MLPerf Tiny ResNet-8 | 시작 전 |
| M4 | 마무리: 새로 받은 저장소에서 README만으로 빌드와 데모 | 시작 전 |

## 라이선스

업스트림 트리에 들어가는 코드는 각 업스트림의 관례를 따르고, 그 밖의 것은
Apache-2.0 WITH LLVM-exception이다(설계 D18, 12.4절).

| 경로 | 라이선스 |
|------|----------|
| `hw/qemu/` | GPL-2.0-or-later |
| `hw/npu-core/` | MIT |
| `linux/` 드라이버 | GPL-2.0-only OR MIT |
| `linux/` UAPI 헤더 | MIT |
| `linux/` 바인딩 YAML | GPL-2.0-only OR BSD-2-Clause |
| `linux/` `.dtsi` / `.dts` | GPL-2.0-or-later OR MIT |
| `firmware/tf-a/` | BSD-3-Clause |
| `firmware/u-boot/` | GPL-2.0-or-later |
| `br2-external/` | GPL-2.0-or-later. 패치는 대상 프로젝트를 따른다 |
| 그 밖의 모든 것 | Apache-2.0 WITH LLVM-exception |

모든 파일에 SPDX 식별자를 두고, 전문은 `LICENSES/`에 둔다. 머리글을 둘 수
없는 파일은 `REUSE.toml`에 선언한다. 저장소는 [REUSE](https://reuse.software)
규격 3.3을 따르고, CI가 `reuse lint`로 검사한다. 라이선스 전문은 해당
라이선스를 쓰는 코드가 생길 때 추가한다. 빌드할 때 내려받는 lcc는 저장소에
넣지 않으며 lcc의 라이선스(`CPYRIGHT`)를 따른다(설계 12.4절).
