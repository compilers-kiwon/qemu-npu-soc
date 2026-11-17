<!-- SPDX-FileCopyrightText: 2026 Kiwon Kwon -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# qemu-npu-soc

가상 SoC `kn1`과 보드 `kn1-evb`를 QEMU에 만들고, 그 위에 TF-A, U-Boot,
리눅스, NPU 드라이버, 모델 컴파일러, 런타임까지 한 저장소에서 올린다.
SoC에는 int8 합성곱 NPU `kn1-npu`가 있다. TFLite 모델을 MLIR로 컴파일해
이 NPU에서 돌리고, 결과를 TFLite Micro 참조 구현과 비트 단위로 비교한다.

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
| `compiler/` | TFLite 임포터(Python), MLIR 다이얼렉트·패스·도구(C++) |
| `runtime/` | `libknpu`와 `knpu-infer`(Rust) |
| `br2-external/` | Buildroot 외부 트리(defconfig, 커널 설정, 패키지, 패치) |
| `buildroot/` | Buildroot 서브모듈(고정) |
| `models/` | `.tflite` 모델, 학습·변환 스크립트 |
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
| Rust | Buildroot 2026.08.1에 들어 있는 버전 | Buildroot |

## 요구 사항

Buildroot 매뉴얼이 요구하는 호스트 패키지가 필요하다: which, sed, make,
binutils, diffutils, gcc, g++, bash, patch, gzip, bzip2, perl, tar, cpio,
unzip, rsync, file, bc, findutils, awk, wget. Debian 계열에서는
`build-essential`이 컴파일러와 make를 함께 설치한다.

```bash
sudo apt-get install build-essential bc bzip2 cpio file git patch perl rsync unzip wget
```

`buildroot/utils/check-package`를 돌리려면 `python3-magic`과 `python3-flake8`도 필요하다.

## 빌드

```bash
git clone --recurse-submodules https://github.com/compilers-kiwon/qemu-npu-soc.git
cd qemu-npu-soc
make -C buildroot O="$PWD/output" BR2_EXTERNAL="$PWD/br2-external" kn1_evb_defconfig
make -C buildroot O="$PWD/output" BR2_EXTERNAL="$PWD/br2-external"
```

두 번째 `make`가 툴체인, TF-A 2.15.0(`plat/kiwon/kn1`), U-Boot 2026.10,
Linux 7.3.1(arm64 `defconfig`), 루트 파일 시스템, 디바이스 트리, 그리고
kn1-evb 머신을 넣은 QEMU 11.1.3을 모두 빌드한다. 결과물은 `output/images/`에
놓인다: `bl1.bin`과 `fip.bin`(BL2, BL31, U-Boot)을 합친 부트 플래시 이미지
`flash.bin`, 루트 파일 시스템 `rootfs.ext4`(`/boot/Image`,
`/boot/kn1-evb.dtb`, `/boot/extlinux/extlinux.conf` 포함)를 GPT 파티션 하나에
담은 부트 디스크 `disk.img`, `start-qemu.sh`. QEMU 실행 파일은
`output/host/bin/qemu-system-aarch64`다.

```bash
output/images/start-qemu.sh
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
python3 tests/boot/kn1_boot_test.py
```

TF-A 포트(`firmware/tf-a/`)나 U-Boot 보드 포트(`firmware/u-boot/`)를 고친
뒤에는 Buildroot가 소스 트리를 다시 풀어 복사하도록 `make …
arm-trusted-firmware-dirclean`이나 `make … uboot-dirclean`을 먼저 돌린다.
U-Boot는 FIP 안에 들어가므로, U-Boot를 고쳤을 때도 TF-A를 다시 빌드한다.

WSL에서는 Windows 경로가 섞인 `PATH`를 Buildroot가 거부하므로, `make` 앞에
`PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`을 붙인다.

디바이스 트리를 리눅스 바인딩 스키마로 검사하려면 호스트에 dt-schema(pip
`dtschema`)를 두고 커널 빌드 디렉터리에서 다음을 돌린다.

```bash
make -C output/build/linux-7.3.1 ARCH=arm64 CROSS_COMPILE=aarch64-buildroot-linux-gnu- CHECK_DTBS=y kiwon/kn1-evb.dtb
```

## NPU 공유 코어 시험

`hw/npu-core/`는 [NPU 명세](docs/npu-spec.md)대로 명령열을 실행하는 공유
코어이고, 나중에 QEMU 장치와 호스트 도구가 함께 쓴다. 지금은 합성곱(`OP_CONV`),
DMA, 대기 명령과 그에 필요한 SET 명령을 구현했다. `OP_ADD`와 `OP_AVGPOOL`은
아직 `UNKNOWN_CMD`로 멈춘다.

호스트 시험에는 CMake, C/C++ 컴파일러, Python 3가 필요하다. 구성 단계에서
고정한 커밋의 TensorFlow Lite Micro 파일 9개를 내려받아 SHA-256을 검사한다.

```bash
cmake -S tests/npu-core -B build/npu-core
cmake --build build/npu-core
ctest --test-dir build/npu-core --output-on-failure
```

- V2: TFLite Micro 참조 합성곱을 이중·단일 반올림으로 한 번씩 빌드해
  정답값을 만든다(`tflm_golden_double`, `tflm_golden_single`).
- V3: 어셈블러(`knpu_asm.py`)로 만든 명령열을 공유 코어(`knpu_run`)에서 실행해
  정답값과 바이트 단위로 비교한다. 1x1·3x3 커널, stride 1·2, VALID·SAME
  패딩, 두 반올림 방식, 채널별 재양자화, 2-D DMA를 다룬다. 명세의 오류
  코드와 `ERR_OFFSET`, 대기 누락이 내는 동기화 오류, 암묵 대기도 확인한다.

## NPU 장치 시험

QEMU의 `kiwon-kn1-npu` 장치(`hw/qemu/hw/misc/kiwon-kn1-npu.c`)는 레지스터,
인터럽트(GIC SPI 32), 작업 실행을 맡고, 명령열은 공유 코어가 실행한다.
`CTRL.START`는 bottom half를 예약하므로 완료는 언제나 MMIO 쓰기 뒤에 온다.
SRAM 크기는 `-global kiwon-kn1-npu.sram-size=32768`처럼 바꿀 수 있다.

V1은 전체 빌드가 남긴 QEMU 빌드 트리에서 돈다. kn1-npu qtest(레지스터,
인터럽트, DMA와 명령, 오류, 카운터, soft reset, SRAM 크기)와 QEMU의
`device-introspect-test`, `qom-test`를 돌린다.

```bash
tests/qemu/run-v1.sh
```

## CI

GitHub Actions에서는 push마다 defconfig, `check-package`, `reuse lint`,
NPU 공유 코어 시험(V2, V3)을 돌리고(`ci.yml`), 매일 밤과 수동 실행 때 전체
빌드, 부팅 시험, V1을 돌린다(`nightly.yml`).

## 현재 상태

| 마일스톤 | 목표 | 상태 |
|----------|------|------|
| M1 | 합성곱 하나를 모든 층에 관통시킨다 | 진행 중. M1.1 저장소 뼈대, M1.2 펌웨어 체인 부팅(QEMU 머신, 디바이스 트리, TF-A, U-Boot, 리눅스 셸), M1.3 NPU 명세 v0([docs/npu-spec.md](docs/npu-spec.md)), M1.4 NPU 공유 코어와 정답값 비교(V2, V3), M1.5 QEMU NPU 장치(V1)까지 끝났다 |
| M2 | MNIST CNN | 시작 전 |
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
라이선스를 쓰는 코드가 생길 때 추가한다.
