<!-- SPDX-FileCopyrightText: 2026 Kiwon Kwon -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# qemu-npu-soc 설계

---

## 1. 목적

가상의 Arm SoC와 NPU 형태의 가속기에 필요한 소프트웨어 전체를 한 저장소에서
만들고, 모두 QEMU에서 실행한다.

```
SoC 모델(QEMU) -> 부트 펌웨어(TF-A, U-Boot) -> 리눅스(디바이스 트리, DRM accel 드라이버)
               -> 모델 컴파일러(MLIR) -> 런타임과 데모 앱
```

성공 기준: 저장소를 5분 동안 훑어본 검토자가, 저자가 실리콘 바로 위부터
애플리케이션까지 혼자 시스템을 올릴 수 있다고 판단하는 것이다. "실리콘 바로
위"는 SoC를 레지스터 수준에서 다루는 일과, 새로 설계된 SoC에 커널을 올리는
일을 모두 뜻한다.

이 프로젝트는 NVIDIA의 NVDLA Virtual Platform에서 영감을 받았다. NVDLA VP는
QEMU ARMv8 `virt` 보드 위에 SystemC 가속기 모델, 리눅스 드라이버, 컴파일러를
쌓았고, 그 저장소들의 push는 2018년에서 2022년 사이에 멈췄다. 이 프로젝트는
세 가지가 다르다.

1. QEMU 네이티브 장치 모델: 업스트림 QEMU 장치와 같은 방식으로 짠다(SystemC
   없음).
2. 현재 도구: 2026년 리눅스 커널, Buildroot, MLIR 기반 모델 컴파일러.
3. SoC 정의부터 컴파일러까지 모든 층을 한 저장소에서 처음부터 만든다.

### 범위 밖

- RTL 설계와 실물 하드웨어
- 직접 만든 컴파일러로 리눅스나 U-Boot를 빌드하는 일
- 사이클 정확도나 타이밍 정확도를 갖춘 모델링(D1 참고)

---

## 2. 결정 기록

| ID | 주제 | 결정 | 주된 이유 |
|----|------|------|-----------|
| D1 | 타이밍 충실도 | 기능 모델에 결정적인 성능 카운터(MAC 수, DMA로 읽고 쓴 바이트 수, 실행한 명령 수)를 더한다. 시간은 모델링하지 않는다. 작업 완료는 인터럽트로 알린다. | 카운터는 컴파일러 결정(타일링, 재사용)의 효과를 명령열에만 의존하는 수치로 보여 준다. QEMU 자체도 명령어 타이밍을 모델링하지 않는다(`docs/devel/tcg-icount.rst`). |
| D2 | NPU 연결 | SoC 버스의 MMIO 플랫폼 장치로 두고, 디바이스 트리에 기술하며, 인터럽트는 GIC SPI로 낸다. | 메인라인의 Arm SoC NPU가 이렇게 연결된다(`drivers/accel/ethosu`, `drivers/accel/rocket`). "새 SoC에 커널을 올린다"는 목표와 맞는다. |
| D3 | PCIe | 범용 ECAM 호스트 브리지(QEMU `TYPE_GPEX_HOST`, 리눅스 `pci-host-ecam-generic`). 주소 창은 지금 예약하고, 브리지는 첫 끝-끝 데모 뒤에 넣는다. MSI는 GICv3 ITS를 거친다. | 첫 경로에 필요 없고 NPU 설계에 영향을 주지 않는다. |
| D4 | 디바이스 트리 | 손으로 쓴 SoC `.dtsi`, 보드 `.dts`, NPU 바인딩 YAML. U-Boot는 보드 `.dts`로 자기 제어용 DT를 빌드하고, 리눅스는 extlinux를 거쳐 `/boot`에서 DTB를 읽는다. QEMU는 펌웨어 없이 커널을 바로 띄울 때만 DTB(`-dtb`)를 올린다. | SoC 벤더가 만드는 것과 같은 산출물이다(예: `rk3588-base.dtsi`, `rockchip,rk3588-rknn-core.yaml`). 자체 DT는 U-Boot의 기본 방식(`OF_SEPARATE`)이고, 앞 단계에서 DT를 받는 Arm 플랫폼은 9개뿐이다(`arch/arm/Kconfig`의 `OF_HAS_PRIOR_STAGE`). |
| D5 | 데모 모델 | 처음 완성할 모델은 작은 MNIST CNN이고, 최종 목표는 MLPerf Tiny 이미지 분류(ResNet-8, CIFAR-10)다. NPU 연산은 ResNet-8을 기준으로 정한다. | 합성곱과 잔차 덧셈이 레이아웃, 타일링, SRAM 할당, 재양자화를 모두 거치게 한다. MLPerf Tiny는 업계 기준점이다. |
| D6 | 데이터형 | TFLite 양자화 규격을 따르는 int8. | MLPerf Tiny 참조 구현이 TFLite Micro이므로, 그 참조 커널을 비트 단위 정답으로 쓸 수 있다. |
| D7 | NPU 연산 | 층 단위 연산 CONV(완전 연결층은 1x1 합성곱이 된다), ADD, AVGPOOL. 각 연산은 출력에서 bias, 재양자화, ReLU clamp를 적용한다. softmax는 CPU에서 돈다. | Ethos-U(`NPU_OP_CONV`, `NPU_OP_POOL`, `NPU_OP_ELEMENTWISE`)와 TFLite Micro 참조 커널과 같은 단위다. |
| D8 | 온칩 메모리 | NPU 안의 전용 SRAM 스크래치패드로, NPU DMA로만 채운다. 크기는 QEMU 장치 속성이고 기본값은 64 KiB이며, 읽기 전용 레지스터로 알린다. 드라이버는 이 값을 사용자 공간에 전달한다. | 64 KiB에서 MNIST CNN은 타일링이 필요 없고, ResNet-8은 14개 층 중 3개에서 타일링이 필요하다(6.6절). Ethos-U도 SRAM 크기를 사용자 공간에 알린다(`npu_info.sram_size`). |
| D9 | 명령 형식 | Ethos-U 방식의 가변 길이 명령열로, 32비트 워드로 이루어진다. 상태를 설정하는 SET 명령, 연산을 수행하는 OP 명령, 비동기 연산을 위한 대기 명령(D31)이 있다. 커널은 명령열을 커널 버퍼로 복사하고 상태를 추적하며 검증한다. | 메인라인의 두 방식 중 IOMMU 없이 동작하는 쪽이다. Ethos-U는 커널에서 검증하고 `iommus` 속성이 없다. Rocket은 명령을 해석하지 않고 `iommus`를 요구한다. |
| D10 | 주소 지정 | 연산 명령은 SRAM 오프셋만 쓴다. DRAM은 DMA 명령으로만 접근하고 (영역, 오프셋)으로 지정한다. 작업 하나에 버퍼를 최대 8개 영역에 붙이고, 드라이버가 그 DMA 주소를 영역 베이스 레지스터에 쓴다. | 명령열을 고칠 필요가 없어 컴파일한 모델을 그대로 재사용할 수 있다. Ethos-U와 같은 방식이다(`NPU_REG_BASEP(region)`). |
| D11 | 재양자화 반올림 | 명령마다 이중 반올림(TFLite Micro 기본값)과 단일 반올림(`TFLITE_SINGLE_ROUNDING=1`) 중에서 고른다. 시험은 두 참조 빌드에 대해 모두 돈다. | Ethos-U도 연산마다 반올림 방식을 고른다(Mesa `ETHOSU_ROUNDING_DOUBLE/TRUNCATE/NATURAL`). |
| D12 | 모델 가져오기 | `.tflite`가 원본이다. Python 생성기 `tflite2c`가 층마다 C89 함수 하나(본문은 TFLite Micro 참조 커널과 같은 루프)를 쓰고, 층 정보는 그 함수 첫 줄의 `#pragma knpu`에 적는다. lcc의 mlir 드라이버가 이 C를 MLIR로 내리며 pragma를 함수 속성으로 넘긴다. 컴파일러 경로에 TensorFlow는 없다. | 같은 C를 gcc로 돌려 생성기를 정답값과 비교할 수 있다. C에서는 재양자화 같은 뜻이 산술로만 남으므로, 뜻은 pragma로 넘기고 `knpu-opt`는 산술을 알아보지 않는다. Arm Vela와 Apache TVM도 `.tflite`를 직접 파싱한다. |
| D13 | 부트 체인 | 처음부터 TF-A(BL1, BL2, BL31)와 U-Boot(BL33)를 쓴다. SoC에는 EL3, 보안 부트 플래시, 보안 RAM이 있고, PSCI는 BL31이 제공한다. | Cortex-A SoC의 표준 Arm 부트 체인이다. EL3 펌웨어를 올리면 QEMU 내장 PSCI는 꺼진다(`hw/arm/virt.c`). |
| D14 | 인터럽트 컨트롤러 | GICv3. | Cortex-A72 코어를 쓰는 메인라인 arm64 SoC `.dtsi` 26개 중 18개가 GICv3를 쓴다. ITS는 D3의 MSI 경로다. |
| D15 | 커널 드라이버 프레임워크 | DRM accel 서브시스템, GEM DMA 헬퍼(물리적으로 연속인 버퍼, IOMMU 없음), 작업 큐와 타임아웃은 `drm_sched`. 참조 구현은 `drivers/accel/ethosu`다. | 메인라인 NPU 드라이버는 모두 DRM accel을 쓴다. Ethos-U가 D9, D10에 가장 가깝다. |
| D16 | 저장소 구조 | Buildroot 외부 트리(`BR2_EXTERNAL`)를 둔 단일 저장소. 우리 코드는 일반 소스 파일로 두고, 업스트림 트리에 등록하는 부분(QEMU meson/Kconfig, U-Boot Kconfig)만 작은 연결 패치로 한다. Buildroot는 고정한 서브모듈이고, 나머지 업스트림 프로젝트는 Buildroot가 고정 버전으로 받아 온다. | 모든 층을 한곳에서 읽을 수 있다. |
| D17 | LLVM/MLIR | apt.llvm.org의 MLIR 23.x(`libmlir-23-dev`, `mlir-23-tools`). | 최신 릴리스 계열(23.1.5)이고, lcc의 mlir 드라이버도 LLVM 23(`/usr/lib/llvm-23/bin`)을 쓴다. 방언 형태는 릴리스마다 바뀌므로(TOSA의 `rounding_mode`가 21.1.8에서는 문자열, 22.1.8과 23.1.5에서는 enum) 메이저 버전을 고정한다. |
| D18 | 라이선스 | 업스트림 트리 안에서는 각 업스트림의 관례를 따른다(12.4절). 그 밖의 것(컴파일러, 런타임, 데모, 시험, 도구)은 Apache-2.0 WITH LLVM-exception이고, 공유 장치 코어(D33)만 예외다. 모든 파일에 SPDX 식별자를 두고, 전문은 `LICENSES/`에 둔다. | 업스트림 제출 가능성을 남긴다. 컴파일러가 링크하는 MLIR도 같은 라이선스다. |
| D19 | 언어 | README와 문서, 커밋 메시지는 한국어로 쓴다. 코드 주석은 영어로 쓴다. 식별자, SPDX, DT compatible 문자열, Kconfig 이름 같은 코드 자체는 영어다. | 주석은 QEMU, 리눅스, U-Boot, TF-A의 관행을 따른다. |
| D20 | 이름 | 벤더 접두사 `kiwon`. SoC `kn1`(compatible `"kiwon,kn1"`, QEMU SoC 장치 `kiwon-kn1`). 보드 `kn1-evb`(compatible `"kiwon,kn1-evb"`, QEMU 머신 `kn1-evb`). NPU compatible `"kiwon,kn1-npu"`. 리눅스 드라이버 `knpu`. | QEMU 머신은 SoC를 장치로 품은 보드다(D25). |
| D21 | CPU | EL3와 EL2를 갖춘 Cortex-A72 4개, 클러스터 하나. 개수는 고정한다(QEMU `min_cpus = max_cpus = default_cpus = 4`). | A72에서 흔한 구성이다(LS1046A, Armada 7040/8040, CN9130, TI J742S2, QEMU 모델의 BCM2711). 실제 SoC용 QEMU 머신은 개수를 고정하고, 정적 DT에는 CPU가 나열된다. 보조 코어가 BL31의 PSCI `CPU_ON`을 거치게 한다. |
| D22 | 메모리 맵과 인터럽트 맵 | kn1이 QEMU `virt`와 공유하는 블록은 모두 `virt`의 주소와 SPI를 쓴다. NPU는 비어 있는 `0x0B00_0000`(SPI 32)에 두고, kn1에 필요 없는 `virt` 블록은 뺀다. | 검증된, 겹치지 않는 배치다. TF-A와 U-Boot의 QEMU 코드를 같은 주소로 읽을 수 있다. |
| D23 | 리셋과 전원 끄기 | 보안 PL061 GPIO(`0x090B_0000`)의 0번 선은 전원 끄기, 1번 선은 리셋이다. 보드에서 두 선은 QEMU `gpio-pwr` 장치로 간다. BL31이 PSCI `SYSTEM_OFF`와 `SYSTEM_RESET`에 쓴다. | `virt`와 TF-A QEMU 플랫폼의 연결과 같다. 이 GPIO가 없으면 TF-A의 QEMU 코드는 전원 끄기를 semihosting으로 하고 리셋에서 panic한다(`qemu_pm.c`). Rockchip의 TF-A도 GPIO로 전원을 끈다. |
| D24 | 추가 주변장치 | PL031 RTC(`0x0901_0000`, SPI 2)와 SBSA 범용 워치독(`0x0F00_0000`, SPI 10)을 `virt`의 주소에 둔다. 비보안 플래시 뱅크는 두지 않는다. | 실제 SoC에 있는 부품이고, QEMU 모델과 리눅스 드라이버가 이미 있다. |
| D25 | 보드와 QEMU 구조 | 보드는 `kn1-evb`다. QEMU 코드는 SoC 장치(`kiwon-kn1`: CPU, GIC, UART, RTC, 워치독, GPIO, 보안 RAM, NPU)와 보드 머신(`kn1-evb`: DRAM, 부트 플래시, virtio-mmio 저장 장치, `gpio-pwr`)으로 나눈다. | QEMU가 실제 SoC 머신을 이렇게 만든다(`imx8mp-evk`는 `fsl-imx8mp`를, `raspi4b`는 `bcm2838`을 품는다). 이 나눔은 `kn1.dtsi`와 `kn1-evb.dts`의 나눔과 같다. `-evb`는 QEMU Arm 머신 10개와 리눅스 arm64 보드 파일 24개가 쓴다. |
| D26 | DRAM | `0x4000_0000`부터 4 GiB(`0x1_3FFF_FFFF`까지)로 고정한다. 머신은 다른 `-m` 값을 거부한다. | 보드의 메모리는 고정이고, DT를 생성하지 않으므로 D4에 필요하다. QEMU `raspi` 머신도 다른 크기를 거부한다. 4 GiB를 넘는 메모리로 NPU의 64비트 주소와 드라이버의 DMA 마스크를 시험한다. |
| D27 | 저장 장치 | 먼저 virtio-mmio 블록 장치를 쓰고, 나중에 실제 컨트롤러로 바꾼다(13절). | U-Boot와 리눅스가 제약 없이 지원한다. QEMU의 PL181 모델에는 DMA가 없고, SD 카드 모델은 이미지 크기를 제한한다. |
| D28 | 데이터 레이아웃 | TFLite 레이아웃: 활성값은 NHWC int8, 합성곱 가중치는 OHWI int8, bias는 출력 채널마다 int32, 재양자화 표는 채널별로 둔다. | 생성기가 상수를 전치 없이 C 배열로 옮기고, TFLite Micro 참조 커널도 같은 방식으로 인덱싱한다. |
| D29 | DMA 형태 | 1-D와 2-D 전송. 간격과 횟수는 DRAM 쪽에만 있고, SRAM 쪽은 연속이다. | 계획한 타일링(H 타일, 출력 채널 타일, 열 타일)에 충분하고, 양쪽 모두에 부호 있는 간격을 두는 Ethos-U의 3-D 전송보다 범위 검사가 단순하다. |
| D30 | DRAM 범위 | 장치는 DRAM 범위를 검사하지 않고 커널 검증기가 검사한다. 레지스터 오프셋 `0x140`~`0x1FF`는 영역 크기 레지스터를 위해 비워 둔다. | Ethos-U에는 영역 베이스 레지스터만 있고, 버퍼 크기는 드라이버가 제출할 때 검사한다(`ethosu_job.c:490`). |
| D31 | 이중 버퍼링 | DMA와 연산은 비동기이고, Ethos-U처럼 명시적인 대기 명령으로 순서를 맞춘다. 모델은 여전히 명령을 하나씩 실행하지만, 아직 기다리지 않은 연산이 쓰는 SRAM을 건드리는 명령이 오면 오류로 멈춘다. 컴파일러는 타일을 이중 버퍼링하고 추정 사이클 수를 보고한다. | 스케줄링과 SRAM 할당은 NPU 컴파일러의 핵심이다. 오류 규칙(Ethos-U에서 가져오지 않은 자체 규칙)은 대기 누락을 결정적으로 드러낸다. 추정치는 Vela의 `npu_performance.py`처럼, 시간 모델 없이 이득을 보여 준다(D1). |
| D32 | QEMU 실행 방식 | `CTRL.START`가 명령열을 실행할 bottom half를 예약한다. 큰 모델이 메인 루프를 멈추게 할 때만 작업 스레드를 쓴다. | 모든 일이 BQL 아래에서 돌므로 잠금을 더할 필요가 없다. `edu.c`처럼 작업 스레드를 쓰면 레지스터 접근과 soft reset에 대비한 잠금이 필요하다. |
| D33 | 공유 장치 코어 라이선스 | QEMU 장치와 호스트 도구가 함께 쓰는 명령 해석기는 MIT. | QEMU `checkpatch.pl`은 MIT를 받아들인다(커밋 메시지에 이유를 적으라는 경고를 낸다). MIT 코드는 Apache-2.0 도구에도 링크할 수 있지만, Apache-2.0은 GPLv2와 호환되지 않는다(FSF 라이선스 목록). |
| D34 | TF-A 플랫폼 | `plat/kiwon/kn1`의 `PLAT=kn1`. `plat/qemu/qemu`를 본뜨되 선택 기능은 뺀다. 소스는 `firmware/tf-a/`에 두고 빌드할 때 TF-A 트리로 복사한다. | `TF_PLATFORM_ROOT`는 도구 빌드용이고(`plat_helpers.mk:18`), 트리 밖 플랫폼을 위한 것이 아니다. |
| D35 | 커널 설정 | 첫 부팅까지는 arm64 `defconfig`에 kn1 조각을 더해 쓰고, 그 뒤 최소 `kn1_defconfig`를 저장소에 둔다. | v7.3 arm64 `defconfig`는 SoC 플랫폼 52개와 옵션 약 1,900개를 켠다. 최소 설정은 kn1에 무엇이 필요한지 정확히 보여 준다. |
| D36 | UAPI | Ethos-U 방식: 조회, 버퍼 생성, mmap 오프셋, 명령열 생성, 제출, 버퍼 대기(`drm_gem_dma_resv_wait()`처럼 기다리고, 실패한 작업은 `-EIO`로 알린다), perfmon 생성·해제·읽기. | 같은 종류의 장치를 다루는 메인라인 드라이버와 모양이 같다. |
| D37 | 바인딩의 클록 | `clocks`는 필수다. `kn1.dtsi`가 고정 클록을 제공하고, 드라이버는 runtime PM으로 이를 켠다. | 메인라인 NPU 바인딩 두 개가 모두 `clocks`를 요구한다(`arm,ethos.yaml`, `rockchip,rk3588-rknn-core.yaml`). |
| D38 | 컴파일러 구조 | MLIR standalone 예제의 구성을 따르고 모두 C++로 짠다. 도구는 `knpu-opt`(패스), `knpu-translate`(`knpu` 다이얼렉트에서 `model.knpu`로), `knpu-compile`(lcc가 낸 `.mlir`에서 `model.knpu`까지)이다. Python은 생성기 `tflite2c`에만 쓴다(D12). | MLIR의 트리 밖 예제(`mlir/examples/standalone`)와 같은 구성이다. |
| D39 | 모델 파일 | 자체 리틀 엔디언 컨테이너 `model.knpu`. | 런타임이 다른 라이브러리 없이 읽는다. |
| D40 | 영역 역할 | Vela 관례를 따른다. 영역 0은 상수, 영역 1은 arena(입력, 출력, SRAM에서 내보낸 활성값)이고, 영역 2~7은 쓰지 않는다. | Vela와 같은 관례다(`BasePointerIndex`: 0 가중치, 1 스크래치 arena). |
| D41 | 런타임 | Rust로 `libknpu` 크레이트와 `knpu-infer`를 만든다. Buildroot의 cargo 패키지 방식과 미리 빌드된 Rust 툴체인(`host-rust-bin`)으로 빌드한다. | 메모리 안전성. Buildroot에서 glibc를 쓰는 aarch64는 Rust tier 1 대상이다. |
| D42 | CI | GitHub Actions. push마다 V2~V4를 돌리고, 매일 밤과 수동 실행 때는 Buildroot 전체 빌드와 V1, V5, V6을 돌린다. | 공개 저장소에서 Actions는 무료이고, 작업 하나는 6시간까지 돌 수 있으며, 캐시는 저장소당 10 GB다. |
| D43 | 업스트림 버전 | 최신 안정판을 고정한다: QEMU v11.1.3, Linux v7.3.1, TF-A v2.15.0, U-Boot v2026.10, Buildroot 2026.08.1. | 현재 LTS 커널(6.18)에는 가장 가까운 참조인 Ethos-U 드라이버가 없다. 버전을 올리는 일은 따로 결정한다. |
| D44 | 대상 툴체인 | Buildroot 내부 툴체인(GCC, glibc). | 툴체인은 Buildroot로 구성해 소스에서 재현한다. 직접 만든 컴파일러로 커널을 빌드하는 일은 범위 밖이다(1절). |

결정마다 근거(업스트림 스냅숏과 파일 위치)는 14절에 있다.

---

## 3. 시스템 개요

```
 빌드 호스트 (x86_64)                         게스트: QEMU 안의 kn1-evb 보드 (AArch64)
 ---------------------------------------      ------------------------------------------
 model.tflite
     |  tflite2c (Python)
     v
 model.c (C89, #pragma knpu)
     |  lcc -S (mlir 드라이버)
     v
 model.mlir (MLIR 23)
     |  knpu-compile (C++, MLIR 패스)
     v                                        knpu-infer (데모 앱, Rust)
 model.knpu ---- rootfs에 복사 ---------------->  |  libknpu (런타임, Rust)
 (명령열, 상수, 메타데이터)                       |  /dev/accel/accel0에 ioctl
                                                  v
                                              knpu.ko (DRM accel 드라이버)
                                                  |  명령열 검증,
                                                  |  영역 베이스, QBASE, START 기록
                                                  v
                                              kn1-npu (QEMU 장치 모델)
                                                  DMA <-> DRAM, SRAM 스크래치패드, IRQ

 부팅: 리셋 -> BL1 (부트 플래시) -> BL2 -> BL31 (PSCI) -> U-Boot (내장 DT)
       -> extlinux: /boot/Image + /boot/kn1-evb.dtb -> Linux
```

---

## 4. 원칙

- **처음부터 비트 단위 일치.** 모든 NPU 연산은 두 반올림 방식 모두에서
  TFLite Micro 참조 커널을 비트 단위로 재현해야 한다.
- **얇은 조각 먼저.** 첫 마일스톤은 어느 층을 넓히기 전에 합성곱 하나를 모든
  층(컴파일러, 런타임, 드라이버, 장치, 부트 체인)에 관통시킨다. 자세한 계획은
  별도 문서에 둔다.
- **자리부터 잡고 나중에 넣기.** 첫 경로에 없는 부분(PCIe, ITS, NPU 영역 크기
  레지스터)은 주소 공간과 인터럽트 번호를 지금 잡아 두고 나중에 구현한다.
- **신뢰 경계에서 검증.** 명령열은 사용자 공간이 만들고, 커널이 검사한 뒤에야
  장치가 실행할 수 있다.

---

## 5. SoC kn1과 보드 kn1-evb

### 5.1 구성 요소

| 구성 요소 | 속한 곳 | 모델 |
|-----------|---------|------|
| CPU | SoC | Cortex-A72 4개, EL3와 EL2, 클러스터 하나(D21) |
| 인터럽트 컨트롤러 | SoC | GICv3 distributor와 redistributor, ITS는 예약 |
| 타이머 | SoC | Arm generic timer |
| 보안 RAM | SoC | TF-A BL2와 BL31용 온칩 RAM 16 MiB |
| 콘솔 | SoC | PL011 UART0(비보안), PL011 UART1(보안, TF-A용) |
| RTC | SoC | PL031(D24) |
| 워치독 | SoC | SBSA 범용 워치독(D24) |
| GPIO | SoC | PL061, 보안, 리셋과 전원 끄기 선(D23) |
| NPU | SoC | kn1-npu, MMIO + SPI |
| PCIe | SoC | ECAM 호스트 브리지, 주소 창 예약(D3) |
| DRAM | 보드 | 4 GiB(D26) |
| 부트 플래시 | 보드 | CFI 병렬 플래시 64 MiB, 보안, BL1과 FIP를 담음 |
| 저장 장치 | 보드 | virtio-mmio 블록 장치(D27) |
| 전원 컨트롤러 | 보드 | GPIO 선이 구동하는 QEMU `gpio-pwr`(D23) |

### 5.2 메모리 맵

| 시작 주소 | 크기 | 블록 | 월드 |
|-----------|------|------|------|
| `0x0000_0000` | 64 MiB | 부트 플래시(리셋 벡터의 BL1, FIP) | 보안 |
| `0x0800_0000` | 64 KiB | GICv3 distributor | — |
| `0x0808_0000` | 128 KiB | GICv3 ITS(예약) | — |
| `0x080A_0000` | 512 KiB | GICv3 redistributor(128 KiB x 4) | — |
| `0x0900_0000` | 4 KiB | UART0(PL011) | 비보안 |
| `0x0901_0000` | 4 KiB | RTC(PL031) | 비보안 |
| `0x0904_0000` | 4 KiB | UART1(PL011) | 보안 |
| `0x090B_0000` | 4 KiB | GPIO(PL061), 리셋과 전원 끄기 선 | 보안 |
| `0x0A00_0000` | 512 B x 4 | virtio-mmio 트랜스포트 | 비보안 |
| `0x0B00_0000` | 64 KiB | kn1-npu 레지스터 | 비보안 |
| `0x0E00_0000` | 16 MiB | 보안 RAM | 보안 |
| `0x0F00_0000` | 4 KiB | 워치독 refresh 프레임(SBSA) | 비보안 |
| `0x0F00_1000` | 4 KiB | 워치독 control 프레임(SBSA) | 비보안 |
| `0x1000_0000` | 0x2EFF_0000 | PCIe MMIO 창(예약) | 비보안 |
| `0x3EFF_0000` | 64 KiB | PCIe I/O 창(예약) | 비보안 |
| `0x3F00_0000` | 16 MiB | PCIe ECAM(예약) | 비보안 |
| `0x4000_0000` | 4 GiB | DRAM(`0x1_3FFF_FFFF`까지) | 비보안 |

kn1이 QEMU `virt`와 공유하는 블록은 모두 `virt`의 주소에 있고, NPU는
`virt`가 비워 둔 자리를 쓴다(D22).

### 5.3 인터럽트 맵

| 출처 | SPI | INTID |
|------|-----|-------|
| UART0 | 1 | 33 |
| RTC | 2 | 34 |
| UART1(보안) | 8 | 40 |
| PCIe INTA~INTD(예약) | 3~6 | 35~38 |
| 워치독 WS0 | 10 | 42 |
| virtio-mmio 0~3 | 16~19 | 48~51 |
| kn1-npu | 32 | 64 |

generic timer의 PPI는 아키텍처가 정한 INTID를 쓴다(보안 물리 29, 비보안 물리
30, 가상 27, 하이퍼바이저 26). 보안 UART의 인터럽트는 SPI 8에 연결하지만 TF-A는 폴링한다(리눅스 `arm,pl011` 바인딩이 `interrupts`를 요구한다). 보안 GPIO의 인터럽트는
연결하지 않는다. TF-A가 출력선만 쓰기 때문이다. `virt`에서는 인터럽트 표에
이 GPIO의 항목이 없어서 SPI 0에 연결된다. 워치독의 두 번째 단계(WS1)는
QEMU의 워치독 동작으로 머신을 리셋한다.

### 5.4 리셋과 전원 끄기

BL31은 GPIO 0번 선으로 PSCI `SYSTEM_OFF`를, 1번 선으로 `SYSTEM_RESET`을
구현한다(D23). 보드에서 두 선은 QEMU `gpio-pwr` 장치로 가고, 이 장치가
QEMU에 종료나 리셋을 요청한다(`hw/gpio/gpio_pwr.c`). `gpio-pwr`는 실제
보드의 전원 컨트롤러(PMIC)를 대신한다.

### 5.5 디바이스 트리

- `linux/dts/kiwon/kn1.dtsi`: SoC(CPU, GIC, 타이머, UART, RTC, 워치독, NPU와
  그 고정 클록, `method = "smc"`인 PSCI). 보안 전용 블록에는
  `status = "disabled"`와 `secure-status = "okay"`를 단다. `virt`가 보안
  GPIO에 하는 방식과 같다.
- `linux/dts/kiwon/kn1-evb.dts`: 보드(compatible, 4 GiB 메모리, `chosen`,
  virtio-mmio 저장 장치, NPU 전용 CMA 풀). 풀은 DRAM의 마지막 64 MiB
  (`0x1_3C00_0000`)에 둔 `reusable` `shared-dma-pool`이고, NPU 노드의
  `memory-region`이 가리킨다. 그래서 NPU 버퍼는 모두 4 GiB 위에 놓인다(D26).
  arm64는 기본 CMA 영역을 4 GiB 아래에 두므로(`arch/arm64/mm/init.c`), 풀이
  없으면 64비트 주소가 쓰이지 않는다.
- `linux/bindings/npu/kiwon,kn1-npu.yaml`: NPU 바인딩(8.5절).
- U-Boot는 `kn1-evb.dts`로 자기 제어용 DT를 빌드한다(`OF_SEPARATE`.
  Buildroot가 `BR2_TARGET_UBOOT_CUSTOM_DTS_PATH`로 소스를 복사한다).
- 리눅스 DTB는 Buildroot가 빌드해(`BR2_LINUX_KERNEL_CUSTOM_DTS_DIR`로
  `linux/dts/`를 커널의 `arch/arm64/boot/dts/`에 덮어씀. `_PATH` 변수는
  커널 6.12부터 deprecated) `/boot/kn1-evb.dtb`로 설치하고, U-Boot가 `extlinux.conf`의 `fdt` 줄로
  리눅스에 넘긴다.
- 펌웨어 없이 커널을 바로 띄우는 길(`-kernel`과 `-dtb`)은 빠른 시험용으로
  남긴다. 이때 QEMU가 DTB를 올리고 `/memory`와 `/chosen`을 고쳐 쓴다
  (`hw/arm/boot.c`의 `arm_load_dtb()`). 머신은 QEMU 내장 PSCI를 SMC 방식으로
  켠다. `virt`도 EL2가 있고 보안 펌웨어가 없는 게스트에는 이렇게 한다.
- 벤더 접두사 `kiwon`은 리눅스 `vendor-prefixes.yaml`에 등록돼 있지 않다.
  v7.3.1의 `dtbs_check`는 이를 오류로 보지 않고, 루트 노드의 compatible에
  맞는 보드 스키마가 없다는 경고만 낸다(2026-11-06 확인). 연결 패치는 두지
  않는다.
- `tests/dt/check-dt.sh`가 커널 tarball을 풀어 NPU 바인딩의
  `dt_binding_check`와 `kn1-evb.dtb`의 `CHECK_DTBS`를 돌린다. 위의 루트
  compatible 경고만 허용하고, push마다 CI가 돌린다.

---

## 6. NPU: kn1-npu

### 6.1 프로그래밍 모델

**작업**은 명령열 하나와 영역 연결로 이루어진다. 장치에는 영역이 8개 있고,
컴파일한 모델은 영역 0과 1을 쓴다(D40). 드라이버는 다음 순서로 작업을
시작한다.

1. 연결한 버퍼마다 DMA 주소를 `REGION_BASE(n)`에 쓴다.
2. 명령열의 주소와 크기를 `QBASE`와 `QSIZE`에 쓴다.
3. `CTRL.START`를 쓴다.

장치는 `OP_STOP`까지 명령을 차례로 읽은 뒤 `STATUS.DONE`을 세우고 인터럽트를
올린다. `OP_DMA`와 연산 명령(`OP_CONV`, `OP_ADD`, `OP_AVGPOOL`)은 정의상
비동기다. 그 결과는 뒤에 오는 `OP_DMA_WAIT`나 `OP_KERNEL_WAIT` 다음에만 쓸 수
있다(D31). 모델은 명령 하나를 끝까지 실행한 뒤에 다음 명령을 읽고, 대기
규칙을 검사한다(6.9절). 오류가 나면 실행을 멈추고 `STATUS.ERROR`를 세운 뒤
인터럽트를 올린다.

### 6.2 레지스터

레지스터는 모두 32비트이고, 64비트 값은 `_LO`/`_HI`로 나눈다. 아래 표는
윤곽이고, 비트 단위 정의(필드, 리셋 값, 접근 규칙)는
[npu-spec.md](npu-spec.md)에 있다.

| 오프셋 | 이름 | 접근 | 내용 |
|--------|------|------|------|
| `0x000` | `ID` | RO | `0x4B4E5055`("KNPU") |
| `0x004` | `VERSION` | RO | 프로그래밍 모델 major/minor |
| `0x008` | `CONFIG` | RO | 영역 수(8), 지원하는 반올림 방식 |
| `0x00C` | `SRAM_SIZE` | RO | 스크래치패드 크기, 바이트 단위(D8) |
| `0x010` | `CTRL` | WO | `START`, `SOFT_RESET` |
| `0x014` | `STATUS` | RO | `BUSY`, `DONE`, `ERROR`, 오류 코드 |
| `0x018` | `ERR_OFFSET` | RO | 실패한 명령 워드의 바이트 오프셋 |
| `0x020` | `IRQ_STATUS` | RW1C | `DONE`, `ERROR` |
| `0x024` | `IRQ_ENABLE` | RW | `DONE`, `ERROR` |
| `0x030` | `QBASE_LO/HI` | RW | 명령열 주소 |
| `0x038` | `QSIZE` | RW | 명령열 크기, 바이트 단위 |
| `0x100 + 8n` | `REGION_BASE_LO/HI(n)` | RW | n = 0..7 |
| `0x140`~`0x1FF` | — | — | 영역 크기 레지스터용 예약(D30) |
| `0x200` | `PERF_CTRL` | RW | `ENABLE`, `CLEAR` |
| `0x208` | `PERF_MACS` | RO | 64비트 |
| `0x210` | `PERF_DMA_RD_BYTES` | RO | 64비트, DRAM에서 SRAM으로 |
| `0x218` | `PERF_DMA_WR_BYTES` | RO | 64비트, SRAM에서 DRAM으로 |
| `0x220` | `PERF_OPS` | RO | 64비트, 실행한 OP 명령 수 |

### 6.3 명령열

명령열은 32비트 워드로 이루어진다(D9). SET 명령은 장치 상태를 바꾸고, OP
명령은 현재 상태로 동작하며, 대기 명령은 비동기 연산의 순서를 맞춘다.
상태는 OP 명령을 지나도 유지된다.

워드 형식은 Ethos-U와 같은 배치다. 리눅스 드라이버는 상위 16비트를
파라미터로 읽고, 비트 14를 "페이로드 워드가 뒤따름"으로 본다
(`ethosu_gem.c:416-418`).

```
 31              16 15 14 13               0
+------------------+--+--+------------------+
|     param16      | 0| L|      opcode      |
+------------------+--+--+------------------+
 L = 1: 32비트 페이로드 워드 하나가 뒤따른다(주소, 크기, 곱셈 계수)
```

명령 집합(윤곽):

| 묶음 | 명령 |
|------|------|
| 제어 | `OP_NOP`, `OP_STOP` |
| DMA | `SET_DMA_SRC` / `SET_DMA_DST`(공간: SRAM 또는 영역 n, 오프셋), `SET_DMA_LEN`, `SET_DMA_STRIDE` / `SET_DMA_COUNT`(2-D), `OP_DMA`(비동기 시작) |
| 동기화 | `OP_DMA_WAIT`, `OP_KERNEL_WAIT`(파라미터: 남아 있어도 되는 작업 수, Ethos-U와 같음) |
| 피처맵 | `SET_IFM_BASE`, `SET_IFM_SHAPE`(H, W, C), `SET_IFM_ZP`, `SET_IFM2_BASE`, `SET_IFM2_ZP`, `SET_OFM_BASE`, `SET_OFM_SHAPE`, `SET_OFM_ZP` |
| 커널 | `SET_WEIGHT_BASE`, `SET_BIAS_BASE`, `SET_KERNEL`(kh, kw), `SET_STRIDE`, `SET_PAD`(위, 왼쪽, 아래, 오른쪽) |
| 재양자화 | `SET_RQ_TABLE`(SRAM에 있는 채널별 곱셈 계수와 시프트), `SET_IFM_SCALE` / `SET_IFM2_SCALE` / `SET_OFM_SCALE`(ADD의 입력과 출력 곱셈 계수, 시프트), `SET_CLAMP`(최소, 최대), `SET_ROUND_MODE`(이중, 단일) |
| 연산 | `OP_CONV`, `OP_ADD`, `OP_AVGPOOL`(비동기) |

연산 명령의 `*_BASE` 값은 모두 SRAM 오프셋이다(D10). 비트 단위
인코딩(명령 번호, 필드 위치, 대기 의미, 오류 코드)은
[npu-spec.md](npu-spec.md)에 있다.

### 6.4 연산 의미

각 연산은 TFLite Micro 참조 커널과 비트 단위로 같아야 한다
(`tensorflow/lite/micro/kernels/internal/reference/integer_ops/`).

| OP | 참조 | 비고 |
|----|------|------|
| `OP_CONV` | `ConvPerChannel`(`conv.h`) | int32 누산기, `filter * (input + input_offset)`, bias, 채널별 재양자화, 출력 오프셋, clamp. |
| FC로 쓰는 `OP_CONV` | `FullyConnected`(`fully_connected.h`) | 1x1xC 입력에 1x1 커널. 텐서 단위 곱셈 계수는 같은 값으로 채운 표가 된다(동등성은 시험으로 확인할 예정). |
| `OP_ADD` | `Add`(`add.h`) | 입력 오프셋, 입력 스케일링, 출력 재양자화, clamp. |
| `OP_AVGPOOL` | `AveragePool`(`pooling.h`) | 참조 구현의 반올림을 따르는 int8 평균. |

ReLU는 별도 연산이 아니다. 참조 커널처럼, 값을 만드는 연산의 clamp
범위로 표현한다.

반올림(D11): `SET_ROUND_MODE`는 `kernels/internal/common.cc`에 있는
`MultiplyByQuantizedMultiplier`의 두 구현 중 하나를 고른다. 이중 반올림은
`SaturatingRoundingDoublingHighMul` 다음에 `RoundingDivideByPOT`를 하고,
단일 반올림은 곱셈, 덧셈, 시프트를 한 번씩 한다.

### 6.5 데이터 레이아웃

| 텐서 | 레이아웃 | 형 |
|------|----------|----|
| 활성값 | NHWC, N = 1 | int8 |
| 합성곱 가중치 | OHWI(TFLite 레이아웃) | int8, zero point 0 |
| bias | 출력 채널마다 | int32 |
| 재양자화 표 | 출력 채널마다 곱셈 계수와 시프트 | int32, int32 |

TFLite 레이아웃을 그대로 쓰므로 생성기는 상수를 전치 없이 C 배열로
옮긴다(D28).

### 6.6 SRAM 스크래치패드

- NPU 전용이다. 크기는 QEMU 속성 `sram-size`로 정하고(기본 64 KiB, D8)
  `SRAM_SIZE`로 알린다.
- DRAM과 SRAM 사이의 데이터는 DMA만 옮기고, 연산 명령은 SRAM만 읽고 쓴다.
- 장치는 모든 SRAM 접근을 `SRAM_SIZE`와 비교하고, 넘으면 오류를 낸다.

아래 표는 이중 버퍼링 없이 층 전체를 SRAM에 올릴 때의 작업 집합(int8 입력
+ int8 출력 + int8 가중치 + int32 bias)을 층 모양에서 계산한 값이다.

| 모델 | MAC | 파라미터 | 가장 큰 층 | 32 / 64 / 128 KiB를 넘는 층 |
|------|-----|----------|------------|-----------------------------|
| MNIST CNN, 예시: conv3x3(8)-avgpool-conv3x3(16)-avgpool-FC64-FC10 | 214,304 | 27,562 | 25.7 KiB(FC 400->64) | 6개 중 0 / 0 / 0 |
| ResNet-8, 필터 26개(MLPerf Tiny 학습 스크립트) | 32,562,192 | 203,746 | 108.5 KiB(conv3x3 104->104) | 14개 중 9 / 3 / 0 |

이중 버퍼링이 없으면 기본 크기에서 ResNet-8의 세 층에 타일링이 필요하다
(add 26ch 78.0 KiB, conv3x3 52->104 s2 67.4 KiB, conv3x3 104->104
108.5 KiB). 이중 버퍼링(D31)은 미리 읽는 타일마다 두 벌을 두므로, 같은 SRAM
크기에서 표보다 많은 층을 타일링하게 된다. 같은 모델을 32, 64, 128 KiB로
컴파일해 `PERF_DMA_*`와 컴파일러의 사이클 추정을 비교하는 시연을 계획한다.

### 6.7 DMA

- 영역(DRAM)과 SRAM 사이를 어느 방향으로든 전송한다.
- 1-D와 2-D 전송이다. 간격과 횟수는 DRAM 쪽에 있고 SRAM 쪽은 연속이다(D29).
  H 방향 타일(NHWC에서 연속), 출력 채널 타일(OHWI에서 연속), 열 타일에
  충분하다.
- `OP_DMA`는 전송을 시작만 한다. 완료는 `OP_DMA_WAIT`로 확인한다(6.3절).
- 접근은 시스템 주소 공간을 거치는 비보안 트랜잭션이다.

### 6.8 성능 카운터

`PERF_MACS`, `PERF_DMA_RD_BYTES`, `PERF_DMA_WR_BYTES`, `PERF_OPS`(D1).
값은 명령열에만 의존하므로 시험에서 단정할 수 있다. 사용자 공간은 작업별
값을 perfmon 객체로 읽는다(8.3절).

### 6.9 오류

다음 경우 장치는 멈추고, 오류 코드와 `ERR_OFFSET`을 기록한 뒤 인터럽트를
올린다: 알 수 없는 명령, 빠진 페이로드 워드, `QSIZE`를 넘어가는 명령열, 범위를
벗어난 SRAM 접근, DMA 트랜잭션 실패, 동기화 오류.

동기화 규칙(D31): 두 연산의 SRAM 범위가 겹치고 그중 하나라도 쓰기이면 두
연산은 충돌한다. DMA는 그것을 기다리는 `OP_DMA_WAIT`가 나올 때까지, 연산은
그것을 기다리는 `OP_KERNEL_WAIT`가 나올 때까지 진행 중으로 남는다. 진행 중인
연산과 충돌하는 명령은 동기화 오류다. 이 규칙은 Ethos-U에서 가져온 것이
아니라 자체 설계다. 모델은 명령마다 끝까지 실행한 뒤 다음으로 넘어가므로,
이 규칙이 없으면 대기 누락이 드러나지 않는다.

DRAM 범위는 장치가 아니라 커널 검증기가 지킨다(8.4절, D30).

### 6.10 QEMU 구현

- `hw/qemu/`에는 SoC 장치 `hw/arm/kiwon-kn1.c`(`kiwon-kn1`), 보드
  `hw/arm/kn1-evb.c`(머신 `kn1-evb`), NPU `hw/misc/kiwon-kn1-npu.c`가 있다.
  `edu`를 등록하는 방식(`hw/misc/meson.build`)처럼 작은 meson·Kconfig 연결
  패치로 등록한다.
- 명령 해석기, 동기화 검사, 연산 코드는 독립된 코어 `hw/npu-core/`(MIT,
  D33)에 두고, QEMU 장치와 호스트 시험(11절)이 함께 컴파일한다.
- `CTRL.START`는 명령열 전체를 실행한 뒤 인터럽트를 올리는 bottom half를
  예약한다. 그래서 완료는 항상 MMIO 쓰기와 비동기다(D32). 큰 모델이 메인
  루프를 멈추게 할 때만 작업 스레드를 쓴다.
- 출발점: `hw/misc/edu.c`(레지스터, IRQ, DMA)와 그 명세 `docs/specs/edu.rst`.

---

## 7. 부트 펌웨어

### 7.1 TF-A

- 플랫폼 `kn1`(`PLAT=kn1`)은 `plat/kiwon/kn1`에 둔다. TF-A의
  `plat/qemu/qemu`를 본뜨되 선택 기능(OP-TEE, RMM, SDEI, 측정 부트, 신뢰
  부트)은 뺀다. `plat/qemu/qemu`는 `plat/qemu/common`까지 합쳐, 이 기능들을
  포함해 5,548줄이다(D34).
- BL1은 부트 플래시에서 실행한다. BL2와 BL31은 `plat/qemu`처럼(`SEC_SRAM`)
  보안 RAM의 앞쪽 1 MiB에서 실행한다. BL33은 U-Boot다. `plat/qemu`가 보안
  OS용으로 잡아 두는 나머지 보안 RAM은 쓰지 않는다.
- GICv3 드라이버를 쓴다. QEMU 플랫폼은 `QEMU_USE_GIC_DRIVER`의 기본값으로
  GICv2를 고르지만, kn1은 GICv3 드라이버를 쓴다.
- BL31이 PSCI를 제공한다: CPU 켜기와 끄기, 보안 GPIO를 통한 시스템 끄기와
  리셋(D23).
- 소스는 `firmware/tf-a/plat/kiwon/kn1/`에 두고, Buildroot가 TF-A 트리를 푼
  뒤 그 안으로 복사해서 빌드한다. `TF_PLATFORM_ROOT`는 쓰지 않는다.
  `plat_helpers.mk`가 이것을 도구를 직접 빌드할 때 쓰는 변수로 설명하기
  때문이다.

- 부트 플래시 배치는 `plat/qemu`와 같다: BL1은 0번지(리셋 벡터)에, FIP는
  256 KiB 오프셋에 둔다. U-Boot(BL33)는 DRAM 시작에서 512 MiB 떨어진
  곳(`0x6000_0000`)에 올린다.
- TF-A의 부트·런타임·크래시 콘솔은 보안 UART1이다. `start-qemu.sh`는
  UART0과 UART1을 QEMU의 chardev mux로 한 터미널에 묶는다.
- Cortex-A72는 CCI 뒤에 있는 Armv8.0-A 코어라 `HW_ASSISTED_COHERENCY=0`,
  `USE_COHERENT_MEM=1`로 빌드한다(Arm FVP의 A72 설정과 같다. `plat/qemu`는
  반대로 1과 0을 쓴다).
- BL2는 디바이스 트리를 고치지 않는다. kn1의 DT는 정적이고 PSCI 노드와
  `enable-method`가 이미 들어 있다(`plat/qemu`는 QEMU가 만든 DT에 PSCI
  노드를 더한다).

### 7.2 U-Boot

- 보드 `kiwon/kn1-evb`. Kconfig 연결 패치로 등록한다(`arch/arm/Kconfig`가
  보드마다 Kconfig를 읽어 들인다).
- 제어용 DT는 `kn1-evb.dts`로 빌드해 U-Boot에 넣는다(U-Boot 기본값인
  `OF_SEPARATE`, D4). 보드는 DRAM 크기를 이 DT에서 읽는다.
- 부트 디스크의 `extlinux.conf`로 표준 부팅을 한다. 그 `fdt` 줄이 리눅스용
  `/boot/kn1-evb.dtb`를 올린다(`boot/pxe_utils.c`의 scenario 1).
- 소스는 `firmware/u-boot/`에 두고, Buildroot가 U-Boot 트리를 푼 뒤 보드
  디렉터리와 `kn1-evb_defconfig`를 복사하고 연결 패치를 적용한다. DT
  Makefile은 고치지 않는다. `scripts/Makefile.dts`가
  `CONFIG_DEFAULT_DEVICE_TREE`를 빌드 목록에 넣기 때문이다. 보드 헤더
  (`include/configs/`)도 두지 않는다. `CFG_SYS_SDRAM_BASE`가 없으면 U-Boot는
  DT에서 읽은 첫 DRAM 뱅크를 쓴다.
- DRAM은 `fdtdec_setup_mem_size_base_lowest()`로 읽는다.
  `fdtdec_setup_mem_size_base()`는 `/memory` 경로로 찾은 첫 노드를 상태와
  상관없이 읽는데, kn1 DT에서는 꺼져 있는 보안 RAM 노드가 먼저 나온다.
- TF-A가 BL33을 올리는 `0x6000_0000`에 링크하고(`TEXT_BASE`), 그 아래를
  재배치 전 스택으로 쓴다.
- kn1에 있는 장치에 필요한 기능만 켠다: PL011, virtio-mmio 블록 장치, GPT와
  ext4, bootstd의 extlinux, PSCI 리셋과 끄기. arm64의 기본값인 UEFI 로더는
  그대로 둔다.

### 7.3 저장 장치와 루트 파일 시스템

virtio-mmio 블록 장치에 GPT 디스크 이미지(`disk.img`)를 두고, 그 유일한
ext4 파티션에 `/boot/Image`, `/boot/kn1-evb.dtb`, `extlinux.conf`, Buildroot
루트 파일 시스템을 담는다(D27). U-Boot 표준 부팅의 extlinux 방식은 파티션
테이블이 있는 디스크만 읽으므로(`boot/bootmeth_extlinux.c`), 파티션이 없는
ext4 이미지를 그대로 쓰지 않고 Buildroot의 genimage로 GPT 디스크를 만든다.
나중에 실제 SoC 저장 장치 컨트롤러로 바꾼다(13절). 후보 중 하나인
QEMU의 PL181 모델에는 DMA가 없고, QEMU의 SD 카드 모델은 2 GiB 이하 이미지의
크기가 2의 거듭제곱이어야 한다(`hw/sd/sd.c`).

---

## 8. 리눅스

### 8.1 커널

Linux 7.3.1(D43)을 Buildroot로 빌드한다. 설정(D35)은 첫 부팅까지 arm64
`defconfig`에 kn1 조각을 더해 쓰고, 그 뒤에는 최소 `kn1_defconfig`를
`br2-external/`에 둔다. kn1에 필요한 것은 GICv3, generic timer, PSCI,
PL011, PL031, SBSA 워치독, virtio-mmio 블록, 고정 클록, DRM accel(`knpu`
모듈용)이다.

`CONFIG_DRM_ACCEL`은 `knpu` 패키지가 Buildroot의 커널 설정 수정
(`KNPU_LINUX_CONFIG_FIXUPS`)으로 켠다. 드라이버가 쓰는 GEM DMA 헬퍼와
`drm_sched`는 프롬프트가 없는 심볼이라 다른 드라이버가 select해야 켜진다.
지금은 arm64 `defconfig`의 드라이버들이 켠다. 패키지는 빌드 전에 세 옵션이
모두 켜졌는지 확인한다(13절).

### 8.2 드라이버 `knpu`

`linux/knpu/`. 구조는 `drivers/accel/ethosu`를 따른다(`drv`, `gem`, `job`,
`perfmon`).

- DRM accel 드라이버(`DRIVER_COMPUTE_ACCEL`, `/dev/accel/accel0`)다.
  Buildroot의 커널 모듈 방식으로 트리 밖 모듈로 빌드한다(D15). 명령 번호는
  QEMU 장치와 같은 `hw/npu-core/knpu_cmd.h`를 빌드할 때 복사해 쓴다.
- probe: `"kiwon,kn1-npu"`로 `of_match_table`을 맞추고, `ID`, `VERSION`,
  `CONFIG`, `SRAM_SIZE`를 읽는다. `ID`, `VERSION.MAJOR`, 영역 수 중 하나라도
  다르면 거부한다(명세 3.2절).
- 버퍼: GEM DMA 헬퍼(연속 메모리. SoC에 IOMMU가 없다). 장치의 주소
  레지스터가 64비트이므로 DMA 마스크도 64비트다(D26). DT의 `memory-region`이
  있으면 그 CMA 풀에서 할당한다(5.5절). 마스크가 좁으면 `dma-direct`가 풀
  대신 4 GiB 아래의 페이지를 조용히 내주므로, V5가 장치에 쓴 주소를 확인한다.
- 작업: 하드웨어 큐 하나에 `drm_sched`(한 번에 한 작업). 영역 버퍼마다 작업의
  fence를 단다(DMA가 쓰는 영역은 쓰기, 나머지는 읽기).
- 오류: 장치가 오류로 멈추면 오류 코드와 `ERR_OFFSET`을 커널 로그에 남기고
  그 작업의 fence를 `-EIO`로 끝낸다. 장치는 이미 멈춰 있으므로 리셋하지 않고
  다음 작업을 받는다.
- 타임아웃(500 ms): runtime PM으로 장치를 껐다 켜서 soft reset하고, 그 작업은
  `-ECANCELED`로 끝난다. 완료 인터럽트가 올라와 있거나 처리 중인 작업은 멈춘
  것으로 보지 않는다. QEMU는 작업을 메인 루프에서 한 번에 실행하므로, 긴
  작업 뒤에는 게스트 시계가 한꺼번에 흐를 수 있기 때문이다.
- 클록: DT에서 받아 runtime PM(autosuspend 50 ms)으로 켜고 끈다. resume은
  soft reset 뒤 `IRQ_ENABLE`을 다시 쓴다. `ethosu_drv.c`와 같은 방식이다(D37).
- debugfs(`/sys/kernel/debug/accel/<장치>/`): `inject_hang`에 1을 쓰면 다음
  작업의 `START`를 쓰지 않아 타임아웃을 일으킨다(V5). `last_submit`은 마지막
  작업에 쓴 `QBASE`, `QSIZE`, 영역 베이스를 보여 준다.

### 8.3 UAPI

헤더 `linux/include/uapi/drm/knpu_accel.h`는 `include/uapi/drm/ethosu_accel.h`를
본뜨고(D36), Ethos-U와 Rocket의 UAPI 헤더처럼 MIT 라이선스다. Buildroot는
이를 staging의 `drm/knpu_accel.h`로 설치한다.

| ioctl | 용도 |
|-------|------|
| `KNPU_QUERY` | `ID`, `VERSION`, `CONFIG` 레지스터와 SRAM 크기. 영역 수와 반올림 방식은 `CONFIG`에서 읽는다 |
| `KNPU_BO_CREATE` | 버퍼를 할당하고 GEM 핸들을 돌려준다 |
| `KNPU_BO_MMAP_OFFSET` | 버퍼의 mmap 오프셋. 명령열 버퍼는 매핑할 수 없다 |
| `KNPU_BO_WAIT` | 버퍼를 쓰는 작업이 끝나기를 기다린다. 절대 시각 타임아웃이 지나면 `-ETIME`, 그 버퍼를 마지막으로 쓴 작업이 실패했으면 `-EIO` |
| `KNPU_CMDSTREAM_CREATE` | 사용자 메모리의 명령열을 커널 소유 버퍼로 복사하고 검증한다 |
| `KNPU_SUBMIT` | 명령열 핸들, 영역마다 버퍼 핸들, 선택적인 perfmon |
| `KNPU_PERFMON_CREATE` | 성능 카운터 객체를 만든다 |
| `KNPU_PERFMON_DESTROY` | 그 객체를 없앤다 |
| `KNPU_PERFMON_GET_VALUES` | 그 객체로 제출한 작업들이 누적한 카운터 4개(명세 11절)를 읽는다 |

Ethos-U와 다른 점:

- `SUBMIT`은 작업 배열 대신 작업 하나를 받고, SRAM 크기 인자가 없다. SRAM이
  NPU 안에 있기 때문이다.
- perfmon은 고정 카운터 4개를 모으므로 이벤트를 고르는 인자가 없고,
  `PERFMON_SET_GLOBAL`도 두지 않는다.
- `BO_WAIT`는 기다리기만 하지 않고 실패한 작업을 `-EIO`로 알린다.
  `drm_gem_dma_resv_wait()`는 fence의 오류를 보지 않는다. 기다린 뒤에는 예약
  객체를 잠그고 fence를 본다. 잠그지 않은 순회는 signal된 fence를 건너뛰기
  때문이다(`dma-resv.c`). fence context마다 가장 최근 것만 본다.

### 8.4 명령열 검증기

`ethosu_gem_cmdstream_copy_and_validate()`처럼 먼저 복사한 뒤 커널 안의
사본을 검증한다. 그래서 검사가 끝난 명령열을 사용자 공간이 바꿀 수 없다.

- 알 수 없는 명령(비트 15가 1인 워드 포함)과 잘린 페이로드를 거부한다.
  명세 v0의 명령은 M1의 공유 코어가 아직 실행하지 않는 것까지 모두 받는다.
- 마지막 워드는 `OP_STOP`이어야 한다(Ethos-U와 같다). 그 뒤의 워드는 실행되지
  않기 때문이다.
- DMA 상태(`DMA_SRC`, `DMA_DST`, `DMA_LEN`, `DMA_COUNT`, `DMA_STRIDE`)를
  기본값부터 추적한다. 공간 종류나 영역 번호를 해석할 수 없거나, 두 쪽이
  모두 SRAM이거나 모두 영역인 `OP_DMA`는 거부한다.
- `OP_DMA`마다 영역 쪽이 `오프셋 + (COUNT−1)·STRIDE + LEN`까지 닿는다고 보고,
  64비트 오버플로를 검사하며, 영역마다 그 최댓값을 기록한다(Ethos-U의
  `dma_length()`처럼). DMA가 쓰는 영역은 출력 영역으로 표시한다.
- `OP_DMA`의 SRAM 쪽 범위(`LEN·COUNT` 바이트)를 `SRAM_SIZE`와 비교한다. 연산
  명령의 SRAM 범위와 필드 값은 장치가 검사한다(`SRAM_RANGE`, `BAD_PARAM`).
- 제출할 때, 명령열이 쓰는 영역마다 버퍼가 있어야 하고 기록한 크기보다 작지
  않아야 한다(작으면 `-EOVERFLOW`). 쓰지 않는 영역에 붙인 버퍼, 두 영역에 붙인
  같은 버퍼, 영역에 붙인 명령열 버퍼는 거부한다.

대기 누락은 붙인 버퍼 밖의 메모리에 닿을 수 없으므로 검증기가 다룰 일이
아니다. 장치가 보고한다(6.9절).

### 8.5 디바이스 트리 바인딩

`linux/bindings/npu/kiwon,kn1-npu.yaml`은 `arm,ethos.yaml`처럼 `compatible`,
`reg`, `interrupts`, `clocks`를 필수로 둔다(D37). `memory-region`은 선택이고,
보드가 NPU 전용 CMA 풀을 줄 때 쓴다(5.5절). `kn1.dtsi`는 클록을
`fixed-clock` 노드로 제공한다. 시간을 모델링하지 않으므로(D1) 주파수 값은
설명용일 뿐이다.

---

## 9. 모델 컴파일러

### 9.1 파이프라인

```
model.tflite
  -> tflite2c           (Python. 층마다 C89 함수와 #pragma knpu, 9.2절)
  -> model.c            (C89. gcc로 빌드하면 TFLite Micro와 같은 결과)
  -> lcc -S             (mlir 드라이버, -Wf-pragma=knpu. pragma는 함수 속성으로)
  -> knpu-import        (pragma가 붙은 층 함수를 knpu 그래프 연산으로,
                         상수 전역을 텐서 상수로)
  -> 타일링             (이중 버퍼링을 고려한 H 타일과 출력 채널 타일)
  -> SRAM 할당          (정적, 수명 기반, 미리 읽는 타일마다 버퍼 두 개)
  -> 스케줄링           (현재 타일을 계산하는 동안 다음 타일을 DMA로 읽음)
  -> 대기 명령 삽입     (명세 9절의 큐 규칙에서 충돌하는 곳에만)
  -> 사이클 추정        (DMA와 연산의 해석적 모델)
  -> knpu-translate     (명령열 + 상수 -> model.knpu)
```

사이클 추정은 측정값이 아니라 모델이다. 장치가 시간을 모델링하지 않기
때문이다(D1). Vela의 `npu_performance.py`처럼 스케줄을 비교하는 데 쓴다.
이중 버퍼링이 있을 때와 없을 때, 그리고 SRAM 크기가 다를 때를 비교한다.

softmax처럼 NPU에서 돌지 않는 연산은 같은 C89를 lcc-opt로 최적화해 aarch64
코드로 내린다. M2에서 구현한다(13절 6번).

### 9.2 생성기 `tflite2c`

`.tflite`를 TFLite Micro의 Python 스키마(`schema_py_generated.py`, flatc가
만든 것)와 PyPI `flatbuffers`로 읽는다. 두 데모 모델의 연산자 `CONV_2D`,
`FULLY_CONNECTED`, `ADD`, `AVERAGE_POOL_2D`, `RESHAPE`, `SOFTMAX`를
다룬다(D12).

- 층마다 매개변수가 없는 `static void` 함수 하나를 쓴다. 본문은 TFLite Micro
  참조 커널과 같은 C89 루프이고, 첫 줄은 그 층의 `#pragma knpu`다(9.3절).
- 상수(가중치, bias, 곱셈 계수, 시프트)는 초기값이 있는 `static const`
  배열로, 활성값은 `const`가 아닌 파일 범위 배열로 쓴다. 원소 순서는 TFLite
  그대로다(D28).
- 곱셈 계수와 시프트는 TFLite Micro의 식(`PopulateConvolutionQuantizationParams`,
  `QuantizeMultiplier`)을 반올림 방식마다 그대로 옮겨 계산한다. 시프트가
  30을 넘으면 단일 반올림 빌드만 값을 포화시키기 때문이다.
- 시험용으로 입력을 읽어 모델 함수를 부르고 출력을 쓰는 `main` 파일을 따로
  쓴다. 같은 C를 `gcc -std=c89 -pedantic-errors`로 빌드한 결과는 V2와 같아야
  한다.

### 9.3 C89 규약과 `#pragma knpu`

생성기는 lcc가 받는 C89만 쓴다. 층 함수와 모델 함수는 다음 모양이다.

```c
static void layer1(void)
{
#pragma knpu conv2d in=act0 out=act1 w=L1_W bias=L1_B mult=L1_M shift=L1_S in_shape=(14,14,8) out_shape=(14,14,16) kernel=(3,3) stride=(1,1) pad=(1,1,1,1) in_zp=-128 out_zp=-5 clamp=(-5,127)
	int oy, ox, oc, ky, kx, ic;
	/* TFLite Micro ConvPerChannel과 같은 루프 */
}

void knpu_model(void)
{
#pragma knpu model in=act0 out=act1 in_scale=ACT0_SCALE in_zp=-128 out_scale=ACT1_SCALE out_zp=-5 round="double"
	layer1();
}
```

- **문법:** `#pragma knpu <연산> <키>=<값> ...`. 값은 그 자리에서 보이는 파일
  범위 객체의 이름, 정수(10진이나 16진, 부호 가능), 정수 목록 `(a,b,...)`,
  문자열 중 하나다. 실수는 쓰지 않고, 매크로는 펼쳐지지 않는다.
- **자리:** 함수 본문을 여는 `{` 다음 줄, 지역 선언 앞. 한 함수에 하나.
- **lcc가 하는 일:** `-Wf-pragma=knpu`로 켜면 그 함수에 속성 `knpu.op`(문자열)와
  `knpu.args`(딕셔너리)를 단다. 이름은 심볼 참조, 정수는 `i64`, 목록은
  `array<i64: ...>`, 문자열은 문자열 속성이 된다. 가리킨 객체는 쓰인 것으로
  세므로 본문에서 쓰지 않는 상수 배열도 남는다. lcc는 `knpu`의 뜻을 모르고,
  연산과 키의 뜻은 이 저장소가 정한다.
- **gcc:** 모르는 pragma를 무시하므로(`-Wno-unknown-pragmas`) 같은 파일로
  정답값을 확인한다.
- **형:** int8은 `signed char`, int32는 `int`, 64비트 중간값은 `long`이다(LP64.
  aarch64와 x86_64). 실수 스케일은 `static const float` 전역에 두고 이름으로
  가리킨다.
- **바이트:** lcc는 초기값이 있는 전역을 원소형 없는 바이트 문자열(대상
  엔디언)로 낸다. `knpu-import`는 키가 정한 원소형으로 이를 다시 읽는다.
- **파일:** 줄 끝은 LF만 쓰고, 빈 `#pragma` 줄은 쓰지 않는다.

| 연산 | 키 | 값 |
|------|----|----|
| `conv2d` | `in`, `out` | 입력, 출력 활성값(`signed char`, NHWC) |
| | `w` | 가중치(`signed char`, OHWI) |
| | `bias`, `mult`, `shift` | 출력 채널마다 bias, 곱셈 계수, 시프트(`int`) |
| | `in_shape`, `out_shape` | (H, W, C) |
| | `kernel`, `stride` | (KH, KW), (SH, SW) |
| | `pad` | (위, 왼쪽, 아래, 오른쪽). `SET_PAD`와 같은 순서 |
| | `in_zp`, `out_zp` | 입력, 출력 영점 |
| | `clamp` | (최소, 최대) |
| `model` | `in`, `out` | 모델 입력, 출력 배열 |
| | `in_scale`, `out_scale` | `static const float` 전역 |
| | `in_zp`, `out_zp` | 모델 입력, 출력 영점 |
| | `round` | `"double"` 또는 `"single"`(D11) |

- `model`은 파일에 하나이고, 본문은 층 함수 호출을 차례로 둔다. 반올림 방식은
  TFLite Micro에서 빌드 전체의 설정이므로 층이 아니라 모델에 둔다.
- `knpu-import`는 키가 빠지거나 남는지, 배열 크기가 모양 × 원소 크기와
  같은지, 명세 6.4절의 모양 조건, 모델 입출력의 영점이 층의 영점과 같은지를
  검사한다.
- 곱셈 계수와 시프트를 따로 둔 것은 TFLite Micro의 채널별 배열과 같게 하려는
  것이다. 이를 SRAM의 재양자화 표(채널마다 8바이트)로 엮는 일은 컴파일러가
  한다.
- 다른 연산(`add`, `avgpool` 등)과 그 키는 M2에서 더한다.

### 9.4 구현

MLIR의 `mlir/examples/standalone`과 같은 구성을 가진 트리 밖 MLIR
프로젝트다(D38). CMake에서 apt.llvm.org 패키지(`/usr/lib/llvm-23`)를 대상으로
`find_package(MLIR REQUIRED CONFIG)`를 쓰고, `include/`, `lib/`, `tools/`,
`test/` 디렉터리를 둔다.

방언 `knpu` 하나에 두 층의 연산을 둔다.

- 그래프 연산(`knpu.conv2d`. 나중에 `add`, `avgpool`): 텐서 값을 받고 낸다.
  `knpu-import`가 만든 `func.func @knpu_model(%in: tensor<...>) -> tensor<...>`
  안에 놓이고, 상수는 `arith.constant dense<...>`다. 검증기는 명세 6.4절의
  모양 조건을 본다.
- 명령 연산(`knpu.dma`, `knpu.conv`, `knpu.dma_wait`, `knpu.kernel_wait`,
  `knpu.stop`): 명세의 OP·대기 명령과 거의 1:1이고, SET 명령의 값은 연산의
  속성이다. 검증기는 SRAM 범위와 정렬을 본다.

| 패스 | 하는 일 |
|------|---------|
| `knpu-import` | pragma가 붙은 함수를 그래프 연산으로 바꾼다. 상수 전역은 원소형을 되살려 텐서 상수로 만들고, 재양자화 표는 채널마다 (곱셈 계수, 시프트)로 엮는다. 바꾼 층 함수, `knpu.*` 속성, 더 쓰이지 않는 전역을 지운다. |
| `knpu-tile` | `sram-size`와 이중 버퍼링을 고려해 타일 크기(출력 H, 출력 채널)를 정하고, 층을 타일로 나눈다(`tensor.extract_slice`, 타일별 pad, `tensor.insert_slice`). 상수는 SRAM에 상주한다. |
| `knpu-schedule` | arena 배치(입력, 출력)와 SRAM 배치(상수, 입출력 버퍼 두 벌씩)를 정하고 명령 연산으로 낮춘다. 다음 타일의 입력 DMA를 현재 타일의 연산 앞에 둔다. |
| `knpu-insert-waits` | 명세 9절의 큐 규칙(큐 깊이 2, 암묵 대기, 반대쪽 큐와의 SRAM 겹침)을 따라가며 충돌하는 곳에만 대기 명령을 넣는다. Vela의 대기 계산과 같은 방식이다. |

도구는 다음과 같다.

- `knpu-opt`: 패스를 실행한다. `lit`와 `FileCheck` 시험이 쓴다.
- `knpu-translate`: 명령 연산에서 `model.knpu`를 낸다. 바뀐 SET 값만 명령으로
  쓰고, 명령 번호는 `hw/npu-core/knpu_cmd.h`(MIT)를 그대로 include한다.
- `knpu-compile`: lcc가 낸 `.mlir`을 받아 패스와 번역을 한 번에 실행한다.

lcc는 저장소에 넣지 않는다(12.4절). `compiler/` CMake가 고정 태그의 릴리스
압축을 sha256으로 확인하며 내려받는다. 게스트가 AArch64이므로 aarch64용 mlir
드라이버(x86_64에서 돈다)를 쓴다. 생성기 다음 단계 중 이 저장소의 것은 모두
C++다.

### 9.5 모델 파일 `model.knpu`

리틀 엔디언 컨테이너다(D39). v0의 구성은 다음과 같고, 비트 단위 정의는
[model-format.md](model-format.md)에 있다.

| 부분 | 내용 |
|------|------|
| 헤더(32 B) | 매직 `"KNPU"`, 형식 버전, 대상 SRAM 크기, 명세 `VERSION.MAJOR`, 섹션 수, 섹션 표 위치 |
| 섹션 표 | 섹션마다 종류, 오프셋, 크기 |
| `CMDS` | 명령열. 크기가 곧 `QSIZE`다 |
| `REGIONS` | 영역마다 번호, 역할, 크기, 내용 섹션 |
| `CONST` | 영역 0의 내용(가중치, bias, 재양자화 표) |
| `TENSORS` | 입출력마다 방향, 영역, 오프셋, 형, 모양(NHWC), 스케일, 영점 |

섹션은 16바이트 경계에 둔다. 장치의 `SRAM_SIZE`가 대상 SRAM 크기보다 작으면
런타임은 모델을 거부한다. CPU 후처리 연산의 섹션은 M2에서 minor 버전을 올려
더한다.

영역 역할은 Vela를 따른다(D40). 0은 상수, 1은 arena(파일에 적힌 오프셋에
입력, 출력, SRAM에서 내보낸 활성값을 둔다), 2~7은 쓰지 않는다. 런타임은
제출 전에 입력을 arena로 복사하고, 대기 뒤에 출력을 꺼낸다.

---

## 10. 런타임과 데모 앱

- `libknpu`(Rust 크레이트): 장치를 열고 조회하며, 버퍼를 할당하고 매핑하고,
  `model.knpu`를 읽고, 영역을 붙이고, 입력을 arena로 복사하고, 제출하고
  기다리며, perfmon 카운터를 읽고, CPU 후처리(softmax, argmax)를 한다.
- `knpu-infer`(Rust 바이너리): 모델과 입력 이미지를 읽어 top-1 클래스와 성능
  카운터를 출력하고, 출력을 기대값과 비교한다.
- UAPI 구조체는 Rust의 `#[repr(C)]` 타입으로 옮긴다. bindgen으로 생성할지
  손으로 쓸지는 구현할 때 정한다.
- Buildroot의 cargo 패키지 방식으로 빌드한다. 이 방식은 내려받는 단계에서
  크레이트 의존성을 vendoring한다(`package/pkg-cargo.mk`). Rust 툴체인은
  미리 빌드된 것(`host-rust-bin`)을 쓴다(D41).

---

## 11. 검증

| 단계 | 내용 | 위치 |
|------|------|------|
| V1 | 장치 모델: 레지스터, 인터럽트, 명령 하나하나, 동기화 오류를 포함한 오류 경로 | QEMU qtest |
| V2 | 정답값: 두 번 빌드한 TFLite Micro 참조 커널(이중·단일 반올림)로 층별, 끝-끝 정답을 만든다 | 호스트 |
| V3 | 컴파일한 명령열을 공유 장치 코어에서 실행해 V2와 비트 단위로 비교 | 호스트 |
| V4 | 컴파일러 패스 | `lit` / `FileCheck` |
| V5 | 드라이버: 검증기 부정 시험(잘못된 명령열과 범위를 벗어난 명령열은 거부돼야 함), 4 GiB 위의 버퍼, 장치 오류와 타임아웃 뒤 복구, perfmon, runtime PM, 모듈 재적재 | 게스트 |
| V6 | 끝-끝: kn1-evb를 부팅해 MNIST CNN과 ResNet-8을 돌리고, V2와 비교하고, 카운터를 보고 | QEMU |

CI(D42, GitHub Actions): push마다 V2~V4를 돌리고, 매일 밤과 수동 실행 때
Buildroot 전체 빌드와 V1, V5, V6을 돌린다. 바인딩과 DT의 스키마 검사
(`tests/dt/check-dt.sh`)도 push마다 돈다.
야간 작업은 지금 전체 빌드, 부팅 스모크 시험(`tests/boot/`, M1.2 완료
조건), V1(`tests/qemu/run-v1.sh`), V5(`tests/driver/kn1_v5_test.py`)를
돌리고, V6은 생기면 더한다.

---

## 12. 빌드와 저장소

### 12.1 디렉터리 구성

```
qemu-npu-soc/
├── docs/            설계 문서, NPU 명세
├── hw/qemu/         kn1 SoC 장치, kn1-evb 머신, kn1-npu 장치 모델, 연결 패치
├── hw/npu-core/     QEMU와 호스트 시험이 함께 쓰는 NPU 명령 해석기(MIT)
├── firmware/tf-a/   TF-A 플랫폼 포트(plat/kiwon/kn1)
├── firmware/u-boot/ U-Boot 보드 포트, 연결 패치
├── linux/           knpu 드라이버(트리 밖 모듈), SoC .dtsi / 보드 .dts, 바인딩 YAML
├── compiler/        생성기 tflite2c(Python), MLIR 다이얼렉트·패스·도구(C++)
├── runtime/         libknpu와 knpu-infer(Rust)
├── br2-external/    Buildroot 외부 트리(defconfig, 커널 설정, 패키지, 패치)
├── buildroot/       Buildroot 서브모듈(고정)
├── models/          .tflite 모델, 학습·변환 스크립트
└── tests/           장치, 드라이버, 컴파일러, 끝-끝 시험
```

### 12.2 고정 버전 (2026-11-05 기준 최신 안정판)

| 프로젝트 | 버전 | 빌드 주체 |
|----------|------|-----------|
| QEMU | v11.1.3 | `br2-external`의 Buildroot 호스트 패키지 |
| Linux | v7.3.1(7.3.y) | Buildroot |
| TF-A | v2.15.0 | Buildroot(`BR2_TARGET_ARM_TRUSTED_FIRMWARE_CUSTOM_GIT` 또는 tarball) |
| U-Boot | v2026.10 | Buildroot |
| Buildroot | 2026.08.1 | 서브모듈 |
| LLVM/MLIR | 23.x(D17) | 호스트의 apt.llvm.org 패키지 |
| Rust | Buildroot 2026.08.1에 들어 있는 버전(`host-rust-bin`) | Buildroot |

고정할 때의 지원 기간: Buildroot 2026.08.x는 2026년 12월까지다(LTS 계열은
2025.02.x로 2028년 3월까지). Linux 7.3.y는 stable 계열이고, 현재 LTS는
6.18.y다. 버전을 올리는 일은 따로 결정한다(D43).

대상 툴체인: Buildroot 내부 툴체인(GCC, glibc)(D44).

컴파일러와 그 시험이 빌드할 때 받는 것(압축과 파일은 모두 sha256으로
확인): lcc `v4.2-llvm.3`의 aarch64 mlir 드라이버, TFLite Micro
`fd2a85d`(소스와 Python 스키마), flatbuffers v25.9.23(C++ 소스). 생성기가
쓰는 PyPI `flatbuffers`도 25.9.23으로 고정한다.

### 12.3 연결 패치

연결 패치(glue patch)는 되도록 작게, 등록에 필요한 것만 둔다.

- QEMU: SoC 장치, 머신, NPU의 meson과 Kconfig 항목
- U-Boot: 보드 Kconfig 항목
- TF-A: 없음. 플랫폼 디렉터리를 트리로 복사한다(D34).
- 리눅스(필요할 때만): `vendor-prefixes.yaml`의 `kiwon` 항목

### 12.4 라이선스

| 경로 | 라이선스 | 이유 |
|------|----------|------|
| `hw/qemu/` | GPL-2.0-or-later | QEMU는 2013년 7월부터 몇몇 디렉터리에서만 GPL-2.0-only를 받는다(`LICENSE`). |
| `hw/npu-core/` | MIT | QEMU(GPLv2)에 컴파일돼 들어가고, Apache-2.0 호스트 도구에도 링크된다(D33). |
| `linux/` 드라이버 | GPL-2.0-only OR MIT | `drivers/accel/ethosu`와 같다. |
| `linux/` UAPI 헤더 | MIT | Ethos-U와 Rocket의 UAPI 헤더와 같다. |
| `linux/` 바인딩 YAML | GPL-2.0-only OR BSD-2-Clause | `Documentation/devicetree/bindings/submitting-patches.rst`가 권장하는 태그다. |
| `linux/` `.dtsi` / `.dts` | GPL-2.0-or-later OR MIT | `rk3588-base.dtsi`, `imx93.dtsi`와 같다. |
| `firmware/tf-a/` | BSD-3-Clause | TF-A에 기여하려면 필요하다. |
| `firmware/u-boot/` | GPL-2.0-or-later | U-Boot 관례다. |
| `br2-external/` | GPL-2.0-or-later. 패치는 대상 프로젝트를 따른다 | Buildroot `COPYING`. |
| 그 밖의 모든 것 | Apache-2.0 WITH LLVM-exception | D18. |

외부에서 가져온 입력은 원래 라이선스를 유지한다: MLPerf Tiny
모델(Apache-2.0), TFLite `schema.fbs`와 빌드할 때 받는 TFLite Micro
소스(Apache-2.0), flatbuffers(Apache-2.0).

lcc는 저장소에 넣지 않고 빌드할 때 내려받는다(9.4절). lcc의
라이선스(`CPYRIGHT`)는 그 파일과 함께 재배포하는 것은 허락하지만, lcc나
lcc가 가치의 큰 부분을 차지하는 제품을 파는 것은 금지한다. 이 저장소의 다른
부분과 조건이 다르므로 저장소의 파일과 섞지 않는다.

---

## 13. 열린 항목

1. virtio-mmio를 대신할 실제 저장 장치 컨트롤러(D27).
2. 확정됨(2026-11-17): 비트 단위 NPU 명세 `docs/npu-spec.md` v0. 명령 번호,
   필드 배치, 대기 의미, 오류 코드를 정했다.
3. 확인됨(2026-11-06): 리눅스 v7.3.1의 `dtbs_check`는 등록되지 않은 벤더
   접두사 `kiwon`을 오류로 보지 않는다. 루트 노드 compatible에 대한 "스키마
   없음" 경고만 남으며, 보드 바인딩 YAML을 둘지는 열어 둔다(5.5절).
4. 코드로 확인함(2026-11-16): 가중치 zero point가 0이면 TFLite Micro의
   `FullyConnected`는 1x1 `ConvPerChannel`과 산술이 같다(npu-spec.md
   8.2절). 비트 일치 시험은 M2에서 한다(6.4절).
5. 최소 `kn1_defconfig`로 갈 때(D35): `knpu`가 쓰는 GEM DMA 헬퍼와
   `drm_sched`는 프롬프트가 없는 심볼이다. 이를 select하는 드라이버 하나를
   켜 두거나(예: `CONFIG_DRM_ACCEL_ARM_ETHOSU=m`) 리눅스 연결 패치를 둬야
   한다(8.1절).
6. 확정됨(2026-11-22): softmax처럼 NPU에서 돌지 않는 연산은 같은 C89를
   lcc_llvm의 lcc-opt(Apache-2.0 WITH LLVM-exception)로 affine·linalg로 올려
   최적화한 뒤 aarch64 코드로 내린다. M2에서 구현하며, lcc-opt를 받는
   방법(소스 빌드, 고정 커밋)과 `model.knpu`의 CPU 섹션 형식은 그때 정한다.
   lcc-opt의 `lcc-globals-to-memref`는 함수 속성 안에서 가리키는 전역을
   memref로 바꾸지 않으므로, lcc-opt에는 `knpu-import`가 층 함수와
   `knpu.*` 속성을 지운 모듈을 넘긴다(9.4절).

---

## 14. 근거 (결정에 쓴 업스트림 스냅숏)

| 프로젝트 | 스냅숏 | 쓰인 곳 |
|----------|--------|---------|
| QEMU | master `f29ccc2`(2026-11-03) | D1 `docs/devel/tcg-icount.rst`; `hw/misc/edu.c`(DMA 타이머 191행, 스레드 325·388행, `bql_lock` 364행, PCI 439행); `hw/arm/virt.c`(`base_memmap` 174-224, `a15irqmap` 262-274, MMIO 연결 1100-1115, ITS 1120, GICv2m 1143, `create_rtc` 1564, `create_secure_gpio_pwr` 1637-1668, `create_gpio_devices` 1670, 워치독 DT 2101, PCIe 2151-2282, `msi-map` 2238, PSCI conduit 2967-2973); `hw/arm/boot.c`(`have_dtb` 276, `/memory` 재작성 530-575, 펌웨어용 DTB를 RAM 시작에 둠 1125, `arm_load_dtb` 호출 1297); `hw/gpio/gpio_pwr.c`; `hw/watchdog/sbsa_gwdt.c`(WS0/WS1 232-252); `hw/arm/imx8mp-evk.c:87`; `hw/arm/raspi.c`(코어 수 고정 327, RAM 크기 검사 258-262); `hw/arm/bcm2838.c:262-263`; 실제 SoC 머신의 CPU 수(`min_cpus`/`max_cpus`); `-evb`/`-evk` 머신 이름; `hw/sd/pl181.c`, `hw/sd/sd.c`(이미지 크기 3125-3150); `include/hw/intc/arm_gicv3_common.h:46`; `hw/core/machine.c:1119`(`dumpdtb`); `include/hw/intc/arm_gic_common.h:33`의 `GIC_NCPU 8`; `scripts/checkpatch.pl:1434-1466`; `LICENSE` |
| Linux | mainline `05441d6`(7.4-rc1, 2026-11-03) | `drivers/accel/*`의 버스 종류; Ethos-U 명령 해석(`ethosu_gem.c:385`, 405-426, `NPU_SET_*` case 72개), DMA 모드와 `dma_length()`(161-181, 652-678), 영역 베이스와 시작(`ethosu_job.c:50-56`, 69-73, 402, 436-453), 제출 때 크기 검사(`ethosu_job.c:490`), BO 대기(`ethosu_drv.c:92`), 클록과 runtime PM(`ethosu_drv.c:262-278`, 362), SRAM 크기(`ethosu_drv.c:284`, 288), 레지스터(`ethosu_device.h`); UAPI 헤더 `ethosu_accel.h`, `rocket_accel.h`; Rocket 레지스터 명령(`rocket_job.c:25-27`)과 `iommus` 요구; MMUv2가 없을 때만 하는 etnaviv 검증(`etnaviv_gem_submit.c:544-545`); NPU 바인딩(필수 속성)과 바인딩 라이선스 정책; RK3588 NPU와 PCIe 노드; Cortex-A72 SoC `.dtsi` 파일(26개 중 GICv3 18개, 코어 수와 MPIDR 값); 보드 `.dts` 이름 접미사; `pci-host-generic.c:56`; DRM accel 문서 |
| Linux 릴리스 | kernel.org `releases.json`(2026-11-05 조회), GitHub `torvalds/linux` 태그 | stable 7.3.1, longterm 6.18.60; v6.18(`ethosu` 없음)과 v6.19의 `drivers/accel`; v6.12와 v6.18의 `include/drm/drm_gem.h`에 있는 `drm_gem_dma_resv_wait()`; v7.3 arm64 `defconfig`(`=y` 887개, `=m` 1,024개, `CONFIG_ARCH_*` 52개); v7.3 `drivers/accel/ethosu/ethosu_gem.c`의 명령 해석과 `OP_STOP` 위치 검사(행 번호는 `docs/npu-spec.md` 13절) |
| Linux v7.3.1 | Buildroot가 받은 tarball(2026-11-17) | `drivers/accel/ethosu`(드라이버 구조, `ethosu_job.c`의 타임아웃·IRQ·제출 검사, `ethosu_perfmon.c`, UAPI의 ioctl 10개); `drm_sched_start()`가 끝나지 않은 작업을 `-ECANCELED`로 끝냄(`sched_main.c:499`)과 `DRM_GPU_SCHED_STAT_NO_HANG`(366-367); 의존성 fence의 오류를 다음 작업에 넘기지 않음(오류를 다는 곳은 guilty 엔티티뿐, `sched_entity.c:547-549`); `drm_gem_dma_resv_wait()`(`drm_gem.c:903-925`); 잠그지 않은 fence 순회가 signal된 fence를 건너뜀(`dma-resv.c:402`); arm64 CMA 위치(`arch/arm64/mm/init.c:323`); `dma-direct`의 대신 할당(`kernel/dma/direct.c:127-148`); 드라이버 해제 순서(`drivers/base/dd.c`)와 `devm_pm_runtime_set_active_enabled()`(`runtime.c:1597-1632`); accel debugfs 위치(`drm_debugfs.c:400-407`); `ivpu`의 debugfs `force_recovery`; `arm,ethos.yaml`; arm64 `defconfig`의 `CONFIG_DRM_ACCEL` 꺼짐과 `DRM_GEM_DMA_HELPER`·`DRM_SCHED`=m |
| U-Boot | master `ab4dc5a`(2026-11-02), 릴리스 v2026.10 | `qemu-arm.c:147-150`; `dts/Kconfig`(`OF_SEPARATE`, `OF_BOARD`, `OF_HAS_PRIOR_STAGE`); `arch/arm/Kconfig`(보드 Kconfig 2482행, `OF_HAS_PRIOR_STAGE`를 고르는 플랫폼); `boot/pxe_utils.c:645-662`; `lib/fdtdec.c:1779`; `drivers/mmc/arm_pl180_mmci.c:601-606`; `drivers/virtio/`; `doc/board/emulation/qemu-arm.rst:61-88`의 TF-A 흐름; `Licenses/README` |
| TF-A | master `2fbca70`(2026-11-03), 릴리스 v2.15.0 | `plat/qemu`(`platform_def.h`: 클러스터 20-41, 보안 메모리 80-90, 보안 GPIO 92-95; `qemu_pm.c:217-253`; `qemu_bl2_setup.c`의 DT 수정과 `TRANSFER_LIST`); `platform.mk:17`, 159-160; `plat_helpers.mk:18-39`; `drivers/arm/pl061/pl061_gpio.c`; Rockchip `params_setup.c`(`poweroff_gpio`)와 `rk3399/drivers/soc/soc.c:342`; `imx8m_psci_common.c:163`; BSD-3-Clause 기여 규칙 |
| Buildroot | master `f03de2f`(2026-11-02), 릴리스 2026.08.1 | `BR2_LINUX_KERNEL_CUSTOM_DTS_PATH`(`linux/Config.in:502`)와 설정 방식(228-267); `BR2_TARGET_UBOOT_CUSTOM_DTS_PATH`(`boot/uboot/Config.in:824`); Bootlin 외부 툴체인(`toolchain-external-bootlin`); Rust(`package/rustc/Config.in.host`, `package/pkg-cargo.mk`); `BR2_GLOBAL_PATCH_DIR`; 호스트 QEMU; TF-A 옵션; 커널 모듈 방식; `BR2_EXTERNAL` 매뉴얼 |
| Buildroot 릴리스 | buildroot.org 내려받기 페이지(2026-11-05 조회) | 2026.08.1 안정판(지원 종료 2026년 12월); 2025.02.19 LTS(지원 종료 2028년 3월) |
| TFLite Micro | `fd2a85d`(2026-11-04) | 참조 커널(`conv.h:54-131`); 반올림 구현(`common.cc:31-113`); `schema.fbs`; int8 `ConvPerChannel`, `FullyConnected`, `AddFunc`, `AveragePool`과 고정소수점 보조 함수(행 번호는 `docs/npu-spec.md` 13절); CMake 빌드(`CMakeLists.txt`, `cmake/generated/tflm_sources.cmake`)와 flatbuffers 고정(`MODULE.bazel`의 v25.9.23); Python 스키마(`tensorflow/lite/micro/python/schema_py_generated.py`); 합성곱 양자화 파라미터와 clamp(`tensorflow/lite/micro/kernels/kernel_util.cc:60-90`, 369-445), `QuantizeMultiplier`(`tensorflow/lite/micro/kernels/internal/quantization_util.cc:54-104`) |
| PyPI | `flatbuffers` 25.9.23(2026-11-22 조회) | 생성기의 `.tflite` 읽기. Apache-2.0 |
| lcc_llvm | 태그 `v4.2-llvm.3`(커밋 `3619447`, 2026-11-22) | `#pragma`의 문법, 자리, 속성 모양(README의 "#pragma를 MLIR 함수 속성으로" 절, `src/pragma.c`, `tst/pragma/`); mlir 드라이버의 LLVM 23(README "필요한 것"); 초기값이 있는 전역의 바이트 문자열 출력과 `long`의 i64(aarch64 mlir 드라이버로 내려 확인); lcc-opt가 함수 속성 안에서 가리키는 전역을 바꾸지 않음(`lcc-opt/README.md:263`, `lcc-opt/test/pragma-attrs.mlir`); `CPYRIGHT`; 릴리스 압축의 sha256 |
| LiteRT | 양자화 규격(developers.google.com/edge/litert/models/quantization_spec, 2026-11-05 조회) | D6 |
| MLPerf Tiny | `4addd0f`(2026-07-13) | ResNet-8 정의(`keras_model.py`, 35행의 "200k params"), int8 `.tflite` 파일, Apache-2.0 |
| llvm-project | main `da1fdef`(24.0.0, 2026-11-05), 태그 `llvmorg-22.1.8`, `llvmorg-23.1.5` | `tosa.rescale`(`TosaOps.td:2651`), 반올림 enum(`TosaOpBase.td:397-403`); `ConvertConv2DToImg2Col.cpp`; 설치된 21.1.8은 문자열 속성을 씀(`TosaTypesBase.td:256-258`); `mlir/examples/standalone`(`CMakeLists.txt`, `standalone-opt`, `standalone-translate`) |
| apt.llvm.org | noble, `libmlir-23-dev` 23.1.6 스냅숏(2026-11-16 빌드, `5b10d2d`) | D17; lcc 출력을 `mlir-opt`로 읽어 확인 |
| Mesa | main `3df8fc4`(2026-11-05) | Ethos-U 반올림 방식(`ethosu_ml.h:98-100`); Teflon 장치 목록(`tfl_device.c:1649-1671`) |
| Arm Vela | `4e52413`(2026-11-04) | 자체 `tflite_reader.py`; 의존성 `flatbuffers`, `numpy`; `BasePointerIndex`(`high_level_command_to_npu_op.py:80-83`); NHCWB16 사용; `.tflite`와 `.npz` 출력(`vela.py:95-100`, `tflite_writer.py:198`); 대기 명령(`register_command_stream_generator.py:213-216`, 641-646); 명령 번호 공간(`ethos_u55_regs.py:574-673`); 진행 중 작업 수와 암묵 대기(`architecture_features.py:282-290`, 350, `register_command_stream_util.py:354-395`); 사이클 추정(`npu_performance.py`) |
| Apache TVM | `cbb1e12`(2026-11-03) | `relax/frontend/tflite`가 `tflite` 패키지로 flatbuffer를 파싱 |
| FSF | 라이선스 목록(gnu.org/licenses/license-list.html, 2026-11-05 조회) | Apache-2.0은 GPLv3와 호환되고 GPLv2와는 호환되지 않음; Expat(MIT)은 GPL과 호환 |
| GitHub Actions | docs.github.com 과금·제한 페이지(2026-11-05 조회) | 공개 저장소의 표준 러너는 무료; 작업당 6시간; 저장소당 캐시 10 GB |

6.6절의 층 크기와 MAC 수는 층 정의에서 계산했다(MLPerf Tiny `keras_model.py`.
MNIST CNN은 예시 구조다).
