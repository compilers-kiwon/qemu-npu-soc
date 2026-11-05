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
| TF-A | v2.15.0 | Buildroot |
| U-Boot | v2026.10 | Buildroot |
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
make -C buildroot O="$PWD/output" BR2_EXTERNAL="$PWD/br2-external" host-qemu-kn1
```

두 번째 `make`는 QEMU 11.1.3을 받아 `hw/qemu/`의 kn1 SoC와 kn1-evb 머신을
넣고 빌드해 `output/host/bin/qemu-system-aarch64`에 설치한다. 머신은 다음과
같이 띄운다. 아직 부트 펌웨어와 커널이 없으므로 모니터에서 메모리 맵만 볼 수
있다.

```bash
output/host/bin/qemu-system-aarch64 -M kn1-evb -nographic -monitor stdio -S
```

```text
(qemu) info mtree
(qemu) quit
```

WSL에서는 Windows 경로가 섞인 `PATH`를 Buildroot가 거부하므로, `make` 앞에
`PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`을 붙인다.

전체 빌드와 부팅 절차는 리눅스 셸까지 부팅하는 단계(M1.2)가 끝나면 채운다.

## 현재 상태

| 마일스톤 | 목표 | 상태 |
|----------|------|------|
| M1 | 합성곱 하나를 모든 층에 관통시킨다 | 진행 중. M1.1 저장소 뼈대, M1.2의 QEMU 머신 `kn1-evb`까지 끝났다 |
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
