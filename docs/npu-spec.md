<!-- SPDX-FileCopyrightText: 2026 Kiwon Kwon -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# kn1-npu 프로그래밍 명세 v0

설계 6절([design.md](design.md))의 윤곽을 비트 단위로 정한다. QEMU 장치
모델(`hw/misc/kiwon-kn1-npu.c`), 공유 장치 코어(`hw/npu-core/`), 커널
드라이버와 명령열 검증기(`knpu`), 컴파일러(`knpu-translate`)가 모두 이
문서를 따른다. 이 문서와 구현이 다르면 구현이 틀린 것이다.

명세 v0은 `VERSION` 레지스터 값 0.0에 해당한다.

## 1. 표기

- 수는 리틀 엔디언이다. 명령 워드, SRAM 안의 int32 값, 레지스터가 모두
  그렇다.
- `[m:n]`은 비트 m부터 n까지의 필드다. 비트 0이 최하위다.
- int8, int16, int32는 2의 보수 부호 있는 정수다. "wrap"은 결과의 하위
  32비트만 남긴다는 뜻이다.
- `/`는 0 쪽으로 버리는 정수 나눗셈이고, `>>`는 산술 오른쪽 시프트다.
- 예약 비트는 읽으면 0이다. 레지스터의 예약 비트에 쓴 값은 무시하고, 명령
  워드의 예약 비트가 0이 아니면 오류다.

## 2. 작업 흐름

**작업**은 명령열 하나와 영역 연결로 이루어진다(설계 6.1절). 드라이버는 다음
순서로 작업을 시작한다.

1. 연결한 버퍼마다 DMA 주소를 `REGION_BASE(n)`에 쓴다.
2. 명령열의 주소와 크기를 `QBASE`와 `QSIZE`에 쓴다.
3. `CTRL.START`에 1을 쓴다.

`START`를 쓰면 `STATUS`는 `BUSY`만 1인 값이 되고, `ERR_OFFSET`은 0이 된다.
장치는 SET 상태(5절)를 기본값으로 되돌린 뒤 `QBASE`부터 명령을 차례로
실행한다. 앞 작업의 SET 상태는 다음 작업으로 넘어가지 않는다.

작업은 다음 둘 중 하나로 끝난다.

- `OP_STOP`을 실행하면 진행 중인 작업을 모두 끝낸 뒤 `STATUS`를 `DONE`으로
  바꾸고 `IRQ_STATUS.DONE`을 세운다.
- 오류가 나면 그 자리에서 멈추고, `STATUS`를 `ERROR`와 오류 코드로,
  `ERR_OFFSET`을 실패한 명령의 위치로 바꾼 뒤 `IRQ_STATUS.ERROR`를
  세운다(10절).

완료는 언제나 `START`를 쓴 MMIO 접근이 끝난 뒤에 일어난다(설계 D32). 작업이
`BUSY`인 동안 CPU가 명령열이나 영역 버퍼를 바꾸면 결과는 정의하지 않는다.

## 3. 레지스터

레지스터 창은 64 KiB이고 레지스터는 모두 32비트다. 4바이트 정렬 32비트
접근만 받는다. 다른 크기나 정렬의 접근은 버스 오류로 거부한다. 아래 표에
없는 오프셋은 읽으면 0이고 쓰면 무시한다. RO 레지스터에 쓴 값은 무시한다.

| 오프셋 | 이름 | 접근 | 리셋 값 |
|--------|------|------|---------|
| `0x000` | `ID` | RO | `0x4B4E_5055` |
| `0x004` | `VERSION` | RO | `0x0000_0000` |
| `0x008` | `CONFIG` | RO | `0x0022_0308` |
| `0x00C` | `SRAM_SIZE` | RO | SRAM 크기(기본 `0x0001_0000`) |
| `0x010` | `CTRL` | WO | — |
| `0x014` | `STATUS` | RO | `0x0000_0000` |
| `0x018` | `ERR_OFFSET` | RO | `0x0000_0000` |
| `0x020` | `IRQ_STATUS` | RW1C | `0x0000_0000` |
| `0x024` | `IRQ_ENABLE` | RW | `0x0000_0000` |
| `0x030` | `QBASE_LO` | RW | `0x0000_0000` |
| `0x034` | `QBASE_HI` | RW | `0x0000_0000` |
| `0x038` | `QSIZE` | RW | `0x0000_0000` |
| `0x100 + 8n` | `REGION_BASE_LO(n)`, n = 0..7 | RW | `0x0000_0000` |
| `0x104 + 8n` | `REGION_BASE_HI(n)`, n = 0..7 | RW | `0x0000_0000` |
| `0x140`~`0x1FF` | 영역 크기 레지스터용 예약(설계 D30) | — | — |
| `0x200` | `PERF_CTRL` | RW | `0x0000_0000` |
| `0x208` / `0x20C` | `PERF_MACS_LO` / `_HI` | RO | 0 |
| `0x210` / `0x214` | `PERF_DMA_RD_BYTES_LO` / `_HI` | RO | 0 |
| `0x218` / `0x21C` | `PERF_DMA_WR_BYTES_LO` / `_HI` | RO | 0 |
| `0x220` / `0x224` | `PERF_OPS_LO` / `_HI` | RO | 0 |

`BUSY`인 동안에는 `QBASE_*`, `QSIZE`, `REGION_BASE_*`, `PERF_CTRL`에 쓴 값을
무시한다. `IRQ_STATUS`와 `IRQ_ENABLE`은 언제든 쓸 수 있다.

### 3.1 `ID`

ASCII "KNPU"(`0x4B` `0x4E` `0x50` `0x55`)를 상위 바이트부터 담은 값이다.

### 3.2 `VERSION`

| 비트 | 필드 | 값 |
|------|------|----|
| `[31:16]` | `MAJOR` | 0 |
| `[15:0]` | `MINOR` | 0 |

`MAJOR`가 다르면 프로그래밍 모델이 호환되지 않는다. 드라이버는 아는
`MAJOR`가 아니면 probe를 거부한다.

### 3.3 `CONFIG`

| 비트 | 필드 | 값 | 뜻 |
|------|------|----|----|
| `[3:0]` | `NUM_REGIONS` | 8 | 영역 수 |
| `[8]` | `ROUND_DOUBLE` | 1 | 이중 반올림 지원 |
| `[9]` | `ROUND_SINGLE` | 1 | 단일 반올림 지원 |
| `[19:16]` | `DMA_DEPTH` | 2 | 진행 중일 수 있는 DMA 수(9절) |
| `[23:20]` | `KERNEL_DEPTH` | 2 | 진행 중일 수 있는 연산 수(9절) |

### 3.4 `SRAM_SIZE`

스크래치패드 크기(바이트)다. QEMU 속성 `sram-size`로 정하며, 4 KiB의
배수이고 4 KiB 이상 16 MiB 이하다. 기본값은 64 KiB다(설계 D8).

### 3.5 `CTRL`

| 비트 | 필드 | 동작 |
|------|------|------|
| `[0]` | `START` | 1을 쓰면 작업을 시작한다. `BUSY`이면 무시한다. |
| `[1]` | `SOFT_RESET` | 1을 쓰면 장치를 리셋한다. |

읽으면 0이다. 두 비트를 함께 쓰면 `SOFT_RESET`만 일어난다.

`SOFT_RESET`은 진행 중인 작업을 완료 인터럽트 없이 버리고, `SRAM_SIZE`를
뺀 모든 레지스터를 리셋 값으로 되돌린다. 인터럽트 선도 내려간다. SRAM의
내용은 정의하지 않는다.

### 3.6 `STATUS`

| 비트 | 필드 | 뜻 |
|------|------|----|
| `[0]` | `BUSY` | 작업 실행 중 |
| `[1]` | `DONE` | 마지막 작업이 `OP_STOP`으로 끝났다 |
| `[2]` | `ERROR` | 마지막 작업이 오류로 끝났다 |
| `[15:8]` | `ERR_CODE` | 오류 코드(10절). `ERROR`가 아니면 0 |

`BUSY`, `DONE`, `ERROR` 중 많아야 하나가 1이다. `START`가 이 레지스터를
`BUSY`로 바꾼다.

### 3.7 `ERR_OFFSET`

마지막 작업이 오류로 끝났을 때, 실패한 명령의 첫 워드가 `QBASE`에서 몇
바이트 떨어져 있는지를 담는다(10절). 그 밖에는 0이다.

### 3.8 `IRQ_STATUS`, `IRQ_ENABLE`

| 비트 | 필드 |
|------|------|
| `[0]` | `DONE` |
| `[1]` | `ERROR` |

작업이 끝나면 장치는 `IRQ_ENABLE`과 관계없이 `IRQ_STATUS`의 해당 비트를
세운다. `IRQ_STATUS`는 1을 쓴 비트만 지워진다. 인터럽트 선(SPI 32)은
`IRQ_STATUS & IRQ_ENABLE`이 0이 아닌 동안 올라가 있는 레벨 신호다.

### 3.9 `QBASE`, `QSIZE`

`QBASE`(`_HI:_LO`, 64비트)는 명령열의 시스템 주소이고, `QSIZE`는 그
크기(바이트)다. `START` 때 `QBASE`가 4의 배수가 아니거나, `QSIZE`가 0이거나
4의 배수가 아니면 명령을 하나도 실행하지 않고 오류 `BAD_QUEUE`로 끝난다.

### 3.10 `REGION_BASE(n)`

영역 n의 시스템 주소(64비트)다. DMA 명령이 영역 n의 오프셋 `off`를 가리키면
장치는 `REGION_BASE(n) + off`(64비트 wrap)에 접근한다. 장치는 영역 크기를
검사하지 않는다(설계 D30).

### 3.11 `PERF_CTRL`과 성능 카운터

| 비트 | 필드 | 동작 |
|------|------|------|
| `[0]` | `ENABLE` | 1이면 작업이 실행한 명령을 카운터에 더한다. |
| `[1]` | `CLEAR` | 1을 쓰면 네 카운터를 0으로 만든다. 읽으면 0이다. |

`ENABLE`과 `CLEAR`를 함께 쓰면 카운터를 지운 뒤 `ENABLE`을 정한다. 카운터는
64비트이고 2^64에서 wrap한다. `_LO`와 `_HI`는 `BUSY`가 아닐 때 읽어야 같은
값의 두 반쪽이다. 세는 대상은 11절에 있다.

## 4. 명령 워드 형식

명령열은 32비트 워드로 이루어진다. 명령 하나는 워드 하나이거나, 워드 하나와
페이로드 워드 하나다.

```
 31              16 15 14 13               0
+------------------+--+--+------------------+
|      PARAM       | 0| L|      OPCODE      |
+------------------+--+--+------------------+
 CMD = 워드[15:0]   L = 워드[14]: 1이면 페이로드 워드 하나가 뒤따른다
```

- 명령은 `CMD`(워드의 하위 16비트)로 식별한다. `L`은 `CMD`의 일부라서
  명령마다 정해져 있다. 페이로드가 없는 명령은 `0x0xxx`, 있는 명령은
  `0x4xxx`다. Ethos-U와 같은 배치다.
- 6절의 표에 없는 `CMD`는 오류 `UNKNOWN_CMD`다. 비트 15가 1인 워드도
  여기에 해당한다.
- `L = 1`인 명령의 페이로드 워드가 `QSIZE` 밖에 있으면 오류 `TRUNCATED`다.
- 쓰지 않는 `PARAM`은 0이어야 한다. 아니면 오류 `BAD_PARAM`이다.

## 5. SET 상태

SET 명령은 아래 상태를 바꾸고, OP 명령은 현재 상태로 동작한다. 상태는 OP
명령을 지나도 유지된다. `START` 때 모든 상태는 기본값으로 돌아간다.

| 상태 | 형 | 기본값 | 바꾸는 명령 |
|------|----|--------|-------------|
| `IFM_BASE`, `IFM2_BASE`, `OFM_BASE` | SRAM 오프셋 | 0 | `SET_IFM_BASE`, `SET_IFM2_BASE`, `SET_OFM_BASE` |
| `WEIGHT_BASE`, `BIAS_BASE`, `RQ_TABLE` | SRAM 오프셋 | 0 | `SET_WEIGHT_BASE`, `SET_BIAS_BASE`, `SET_RQ_TABLE` |
| `IFM_SHAPE`, `OFM_SHAPE` | (H, W, C) | (0, 0, 0), 설정 안 됨 | `SET_IFM_SHAPE`, `SET_OFM_SHAPE` |
| `IFM_ZP`, `IFM2_ZP`, `OFM_ZP` | int8 | 0 | `SET_IFM_ZP`, `SET_IFM2_ZP`, `SET_OFM_ZP` |
| `KERNEL` | (KH, KW) | (1, 1) | `SET_KERNEL` |
| `STRIDE` | (SH, SW) | (1, 1) | `SET_STRIDE` |
| `PAD` | (위, 왼쪽, 아래, 오른쪽) | (0, 0, 0, 0) | `SET_PAD` |
| `CLAMP` | (최소, 최대), int8 | (-128, 127) | `SET_CLAMP` |
| `ROUND_MODE` | `DOUBLE` 또는 `SINGLE` | `DOUBLE` | `SET_ROUND_MODE` |
| `IFM_SCALE`, `IFM2_SCALE`, `OFM_SCALE` | (곱셈 계수, 시프트) | (0, 0) | `SET_IFM_SCALE`, `SET_IFM2_SCALE`, `SET_OFM_SCALE` |
| `DMA_SRC`, `DMA_DST` | (공간, 오프셋) | (SRAM, 0) | `SET_DMA_SRC`, `SET_DMA_DST` |
| `DMA_LEN` | 바이트 | 0 | `SET_DMA_LEN` |
| `DMA_COUNT` | 행 수 | 1 | `SET_DMA_COUNT` |
| `DMA_STRIDE` | 바이트 | 0 | `SET_DMA_STRIDE` |

SET 명령은 자기 필드만으로 판단할 수 있는 값(범위, 예약 비트, 정렬)을
그 자리에서 검사하고, 틀리면 오류 `BAD_PARAM`으로 멈춘다. 여러 상태를 함께
봐야 하는 검사는 OP 명령이 한다.

## 6. 명령

### 6.1 명령 번호 표

| `CMD` | 이름 | `PARAM` | 페이로드 |
|-------|------|---------|----------|
| `0x0000` | `OP_STOP` | 0 | — |
| `0x0001` | `OP_NOP` | 0 | — |
| `0x0002` | `OP_CONV` | 0 | — |
| `0x0003` | `OP_ADD` | 0 | — |
| `0x0004` | `OP_AVGPOOL` | 0 | — |
| `0x0010` | `OP_DMA` | 0 | — |
| `0x0011` | `OP_DMA_WAIT` | `[3:0]` n | — |
| `0x0012` | `OP_KERNEL_WAIT` | `[3:0]` n | — |
| `0x0100` | `SET_IFM_ZP` | int16, -128..127 | — |
| `0x0101` | `SET_IFM2_ZP` | int16, -128..127 | — |
| `0x0102` | `SET_OFM_ZP` | int16, -128..127 | — |
| `0x0103` | `SET_KERNEL` | `[15:8]` KH, `[7:0]` KW, 각 1 이상 | — |
| `0x0104` | `SET_STRIDE` | `[15:8]` SH, `[7:0]` SW, 각 1 이상 | — |
| `0x0105` | `SET_PAD` | `[15:12]` 오른쪽, `[11:8]` 아래, `[7:4]` 왼쪽, `[3:0]` 위 | — |
| `0x0106` | `SET_CLAMP` | `[15:8]` 최대, `[7:0]` 최소, 각 int8, 최소 ≤ 최대 | — |
| `0x0107` | `SET_ROUND_MODE` | 0 = `DOUBLE`, 1 = `SINGLE` | — |
| `0x0108` | `SET_DMA_COUNT` | 행 수, 1 이상 | — |
| `0x4000` | `SET_IFM_BASE` | 0 | SRAM 오프셋 |
| `0x4001` | `SET_IFM2_BASE` | 0 | SRAM 오프셋 |
| `0x4002` | `SET_OFM_BASE` | 0 | SRAM 오프셋 |
| `0x4003` | `SET_WEIGHT_BASE` | 0 | SRAM 오프셋 |
| `0x4004` | `SET_BIAS_BASE` | 0 | SRAM 오프셋, 4의 배수 |
| `0x4005` | `SET_RQ_TABLE` | 0 | SRAM 오프셋, 4의 배수 |
| `0x4006` | `SET_IFM_SHAPE` | C, 1 이상 | `[31:16]` H, `[15:0]` W, 각 1 이상 |
| `0x4007` | `SET_OFM_SHAPE` | C, 1 이상 | `[31:16]` H, `[15:0]` W, 각 1 이상 |
| `0x4008` | `SET_IFM_SCALE` | 시프트, int16, -31..0 | 곱셈 계수, 0..2^31-1 |
| `0x4009` | `SET_IFM2_SCALE` | 시프트, int16, -31..0 | 곱셈 계수, 0..2^31-1 |
| `0x400A` | `SET_OFM_SCALE` | 시프트, int16, -31..0 | 곱셈 계수, 0..2^31-1 |
| `0x4010` | `SET_DMA_SRC` | 공간 | 오프셋 |
| `0x4011` | `SET_DMA_DST` | 공간 | 오프셋 |
| `0x4012` | `SET_DMA_LEN` | 0 | 행 길이(바이트), 1 이상 |
| `0x4013` | `SET_DMA_STRIDE` | 0 | DRAM 쪽 행 간격(바이트) |

DMA의 **공간** `PARAM`은 `[15:8]`이 종류(0 = 영역, 1 = SRAM)이고 `[7:0]`이
영역 번호(0..7)다. SRAM이면 영역 번호는 0이어야 한다. 표의 범위를 벗어난
값은 모두 오류 `BAD_PARAM`이다.

### 6.2 제어

- `OP_NOP`은 아무것도 하지 않는다.
- `OP_STOP`은 진행 중인 DMA와 연산을 모두 끝내고 작업을 마친다(2절). 그
  뒤의 워드는 읽지 않는다. 명령열의 끝(`QSIZE`)에 `OP_STOP` 없이 닿으면
  오류 `NO_STOP`이다.

### 6.3 DMA

`OP_DMA`는 `DMA_SRC`와 `DMA_DST` 사이에 `DMA_COUNT`개의 행을 옮긴다. 행 하나는
`DMA_LEN` 바이트다(설계 D29).

- `DMA_SRC`와 `DMA_DST` 중 정확히 하나가 SRAM이어야 한다. 아니면 오류
  `BAD_PARAM`이다.
- SRAM 쪽은 연속이다. SRAM 오프셋 `s`에서 `DMA_LEN × DMA_COUNT` 바이트를
  차지한다.
- 영역 쪽에서 행 r(0부터)은 `REGION_BASE(n) + off + r × DMA_STRIDE`에서
  시작한다. `DMA_STRIDE`는 부호 없는 값이고, `DMA_COUNT`가 1이면 쓰이지
  않는다.
- 행은 0번부터 차례로 옮긴다. 영역 쪽 행이 겹치면 SRAM에서 영역으로 쓸 때
  나중 행이 앞 행을 덮는다.
- 영역 접근은 시스템 주소 공간을 거치는 비보안 트랜잭션이다. 트랜잭션이
  실패하면 오류 `BUS`다. 그때까지 옮긴 바이트는 정의하지 않는다.
- SRAM 범위 `[s, s + DMA_LEN × DMA_COUNT)`가 `[0, SRAM_SIZE)` 안에 있지
  않으면 오류 `SRAM_RANGE`다. 계산은 오버플로 없이 한다.

DMA는 SRAM 범위를, DRAM에서 SRAM으로 옮기면 쓰고 SRAM에서 DRAM으로 옮기면
읽는다(9절의 충돌 판정).

### 6.4 연산

연산 명령 `OP_CONV`, `OP_ADD`, `OP_AVGPOOL`은 SRAM만 읽고 쓴다(설계 D10). 각
연산은 TFLite Micro 참조 커널과 비트 단위로 같다(8절). 연산 명령은 다음
순서로 검사하며, 처음 걸린 오류로 멈춘다.

1. **모양**(오류 `BAD_PARAM`): 아래 조건을 모두 만족해야 한다.
2. **SRAM 범위**(오류 `SRAM_RANGE`): 읽고 쓰는 모든 범위가
   `[0, SRAM_SIZE)` 안에 있어야 한다.
3. **겹침**(오류 `BAD_PARAM`): OFM 범위가 그 연산이 읽는 범위와 겹치면 안
   된다. 단, `OP_ADD`에서 `OFM_BASE`가 `IFM_BASE`나 `IFM2_BASE`와 정확히
   같으면 허용한다(제자리 덧셈).
4. **재양자화 값**(오류 `BAD_PARAM`, `OP_CONV`만): 재양자화 표의 모든
   곱셈 계수가 0..2^31-1이고 모든 시프트가 -31..30이어야 한다.
5. **동기화**(오류 `SYNC`): 9절.

`IFM_SHAPE` = (IH, IW, IC), `OFM_SHAPE` = (OH, OW, OC), `KERNEL` = (KH, KW),
`STRIDE` = (SH, SW), `PAD` = (PT, PL, PB, PR)이라 하자.

**창 연산의 모양 조건**(`OP_CONV`, `OP_AVGPOOL`)

- 두 모양이 설정돼 있다.
- PT < KH, PB < KH, PL < KW, PR < KW.
- IH + PT + PB ≥ KH, IW + PL + PR ≥ KW.
- OH = (IH + PT + PB − KH) / SH + 1, OW = (IW + PL + PR − KW) / SW + 1.

TFLite의 `SAME`과 `VALID` 패딩은 모두 마지막 식을 정확히 만족한다. 패딩이
커널보다 작으므로 모든 출력 위치의 창에는 입력 원소가 하나 이상 있다.

| 명령 | 추가 모양 조건 | 읽는 SRAM | 쓰는 SRAM |
|------|----------------|-----------|-----------|
| `OP_CONV` | — | IFM: `IH·IW·IC`, 가중치: `OC·KH·KW·IC`, bias: `4·OC`, 재양자화 표: `8·OC` | OFM: `OH·OW·OC` |
| `OP_AVGPOOL` | OC = IC | IFM: `IH·IW·IC` | OFM: `OH·OW·OC` |
| `OP_ADD` | `IFM_SHAPE` = `OFM_SHAPE` | IFM, IFM2: 각 `IH·IW·IC` | OFM: `OH·OW·OC` |

각 범위는 해당 `*_BASE`에서 시작한다. `OP_ADD`는 `KERNEL`, `STRIDE`, `PAD`를
쓰지 않는다. IFM2의 모양은 IFM과 같다(브로드캐스트 없음).

### 6.5 대기

- `OP_DMA_WAIT n`은 진행 중인 DMA가 n개 이하가 될 때까지 가장 오래된 DMA부터
  끝낸다.
- `OP_KERNEL_WAIT n`은 진행 중인 연산에 대해 같은 일을 한다.

n이 깊이(`CONFIG`) 이상이면 아무 일도 하지 않는다. Ethos-U의
`NPU_OP_DMA_WAIT`, `NPU_OP_KERNEL_WAIT`와 같은 의미다. 다만 채널 필드는
없다.

## 7. 데이터 레이아웃

설계 6.5절과 같다.

| 데이터 | SRAM 배치 |
|--------|-----------|
| 활성값 (H, W, C) | NHWC(N = 1), int8. (y, x, c)의 오프셋은 `(y·W + x)·C + c` |
| 가중치 (OC, KH, KW, IC) | OHWI, int8. (o, ky, kx, i)의 오프셋은 `((o·KH + ky)·KW + kx)·IC + i` |
| bias | 출력 채널 c마다 int32 하나, 오프셋 `4c` |
| 재양자화 표 | 출력 채널 c마다 8바이트. 오프셋 `8c`에 곱셈 계수(int32), `8c + 4`에 시프트(int32) |

## 8. 연산 의미

모든 연산은 TFLite Micro 참조 커널(`tensorflow/lite/micro/kernels/internal/`)과
비트 단위로 같다. 아래 의사 코드에서 누산과 덧셈은 int32 wrap이다.

### 8.1 재양자화 `rescale(x, m, s)`

`ROUND_MODE`에 따라 `MultiplyByQuantizedMultiplier`(`common.cc`)의 두 구현 중
하나를 쓴다(설계 D11). x는 int32, m은 0..2^31-1, s는 -31..30이다.

`DOUBLE`(이중 반올림, TFLite Micro 기본):

```
ls = max(s, 0); rs = max(-s, 0)
x1 = x << ls                                  # int32 wrap
p  = int64(x1) * m
nudge = (p >= 0) ? 2^30 : 1 - 2^30
h  = (p + nudge) / 2^31                       # 0 쪽으로 버림
mask = 2^rs - 1
thr  = (mask >> 1) + (h < 0 ? 1 : 0)
r  = (h >> rs) + ((h & mask) > thr ? 1 : 0)
```

`h`는 `SaturatingRoundingDoublingHighMul`이고(m ≥ 0이라 포화가 일어나지
않는다), 마지막 두 줄은 `RoundingDivideByPOT`이다.

`SINGLE`(단일 반올림, `TFLITE_SINGLE_ROUNDING=1`):

```
t = 31 - s                                    # 1..62
r = (int64(x) * m + 2^(t-1)) >> t             # 결과의 하위 32비트
```

### 8.2 `OP_CONV`

`ConvPerChannel`(`reference/integer_ops/conv.h`)과 같다. dilation은 1이고
그룹은 하나다.

```
for oy, ox, oc:
    acc = 0
    for ky in 0..KH-1, kx in 0..KW-1:
        iy = oy·SH - PT + ky;  ix = ox·SW - PL + kx
        if iy, ix가 입력 밖이면: 건너뜀          # 패딩은 계산에서 빠진다
        for ic:
            acc += W[oc, ky, kx, ic] * (I[iy, ix, ic] - IFM_ZP)
    acc += B[oc]
    acc = rescale(acc, M[oc], S[oc]) + OFM_ZP
    O[oy, ox, oc] = clamp(acc, CLAMP.최소, CLAMP.최대)
```

가중치의 zero point는 0이다(설계 6.5절). 완전 연결층은 1x1x C 입력에 1x1
커널을 쓰고 모든 채널에 같은 곱셈 계수와 시프트를 둔다. 가중치 zero point가
0이면 `FullyConnected`(`reference/integer_ops/fully_connected.h`)와 산술이
같다.

### 8.3 `OP_ADD`

`AddFunc`(`reference/integer_ops/add.h`)와 같다. int8에서 TFLite Micro가 쓰는
`left_shift` 20을 고정값으로 쓴다(`add_common.cc`).

```
for 각 원소 i:
    a = (I1[i] - IFM_ZP)  << 20
    b = (I2[i] - IFM2_ZP) << 20
    a = rescale(a, IFM_SCALE.m,  IFM_SCALE.s)
    b = rescale(b, IFM2_SCALE.m, IFM2_SCALE.s)
    o = rescale(a + b, OFM_SCALE.m, OFM_SCALE.s) + OFM_ZP
    O[i] = clamp(o, CLAMP.최소, CLAMP.최대)
```

세 `rescale`은 모두 `ROUND_MODE`를 따른다. 시프트가 0 이하이므로 TFLite
Micro의 `MultiplyByQuantizedMultiplierSmallerThanOneExp`와 같다. 제자리
덧셈에서도 원소마다 두 입력을 먼저 읽고 출력을 쓰므로 결과가 같다.

### 8.4 `OP_AVGPOOL`

int8 `AveragePool`(`reference/integer_ops/pooling.h`)과 같다. 입력과 출력의
양자화 파라미터가 같다고 보므로 zero point, 스케일, `ROUND_MODE`를 쓰지
않는다.

```
for oy, ox, c:
    sum = 0; n = 0
    for ky in 0..KH-1, kx in 0..KW-1:
        iy = oy·SH - PT + ky;  ix = ox·SW - PL + kx
        if iy, ix가 입력 안이면: sum += I[iy, ix, c]; n += 1
    avg = (sum > 0) ? (sum + n/2) / n : (sum - n/2) / n
    O[oy, ox, c] = clamp(avg, CLAMP.최소, CLAMP.최대)
```

n은 언제나 1 이상이다(6.4절). 반올림은 0에서 먼 쪽으로 반올림하는 방식이다.

## 9. 비동기 실행과 동기화

`OP_DMA`와 연산 명령은 정의상 비동기다(설계 D31). 장치에는 DMA 큐와 연산
큐가 하나씩 있고, 각 큐에는 진행 중인 작업이 `DMA_DEPTH`(2)개,
`KERNEL_DEPTH`(2)개까지 있을 수 있다. 큐의 원소는 그 작업이 읽고 쓰는 SRAM
범위를 기억한다.

모델은 명령을 하나씩 끝까지 실행한다. 비동기 명령 X를 실행할 때는 다음 순서를
따른다.

1. X가 들어갈 큐가 차 있으면 그 큐에서 가장 오래된 작업을 끝낸다(암묵
   대기). 실제 장치라면 여기서 멈춰 기다린다. Ethos-U와 Vela도 이렇게
   동작한다.
2. X를 **다른** 큐에서 진행 중인 모든 작업과 비교한다. SRAM 범위가 겹치고
   둘 중 하나라도 쓰기이면 충돌이고, 오류 `SYNC`다.
3. X의 효과를 SRAM과 메모리에 반영하고 X를 큐 끝에 넣는다.

같은 큐의 작업은 들어간 순서대로 끝나므로 서로 충돌하지 않는다. 작업이 큐에서
빠지는 경우는 세 가지다: 대기 명령(6.5절), 암묵 대기, `OP_STOP`. 이 규칙은
Ethos-U에서 가져온 것이 아니라 자체 규칙이다. 모델이 명령을 순서대로 끝까지
실행하므로, 이 규칙이 없으면 대기를 빠뜨려도 드러나지 않는다.

DRAM 쪽의 순서는 장치가 따지지 않는다. DRAM을 읽고 쓰는 것은 DMA뿐이고, DMA
큐는 순서대로 끝나기 때문이다.

## 10. 오류

오류가 나면 장치는 멈추고, 오류 코드와 `ERR_OFFSET`을 기록한 뒤 인터럽트를
올린다. 앞선 명령의 효과는 남아 있다. 실패한 명령의 효과는 `BUS` 오류에서만
일부 남을 수 있다.

| 코드 | 이름 | 조건 | `ERR_OFFSET` |
|------|------|------|--------------|
| `0x00` | `NONE` | 오류 없음 | 0 |
| `0x01` | `UNKNOWN_CMD` | 6.1절의 표에 없는 `CMD` | 그 명령 |
| `0x02` | `TRUNCATED` | 페이로드 워드가 `QSIZE` 밖에 있다 | 그 명령 |
| `0x03` | `NO_STOP` | `OP_STOP` 없이 `QSIZE`에 닿았다 | `QSIZE` |
| `0x04` | `BAD_QUEUE` | `QBASE`나 `QSIZE`가 잘못됐다(3.9절) | 0 |
| `0x05` | `BAD_PARAM` | 필드 값, 모양, 겹침, 재양자화 값 | 그 명령 |
| `0x06` | `SRAM_RANGE` | SRAM 범위를 벗어난 접근 | 그 명령 |
| `0x07` | `SYNC` | 진행 중인 작업과의 충돌(9절) | 그 명령 |
| `0x08` | `BUS` | 명령 워드나 DMA의 버스 트랜잭션 실패 | 그 명령 |

`ERR_OFFSET`의 "그 명령"은 실패한 명령의 첫 워드가 `QBASE`에서 떨어진 바이트
수다. 명령 하나에서 오류는 다음 순서로 검사한다: 명령 워드 읽기(`BUS`),
`UNKNOWN_CMD`, `TRUNCATED`와 페이로드 읽기(`BUS`), 명령별 검사(6절).

DRAM 범위는 장치가 아니라 커널 검증기가 지킨다(설계 8.4절, D30).

## 11. 성능 카운터

`PERF_CTRL.ENABLE`이 1인 동안 실행을 마친 명령만 센다. 실패한 명령은 세지
않는다. 값은 명령열에만 의존한다(설계 D1).

| 카운터 | 더하는 값 |
|--------|-----------|
| `PERF_MACS` | `OP_CONV`마다 `OH·OW·OC·KH·KW·IC`(명목 MAC. 패딩 자리도 센다). `OP_ADD`와 `OP_AVGPOOL`은 0 |
| `PERF_DMA_RD_BYTES` | DRAM에서 SRAM으로 옮긴 `OP_DMA`마다 `DMA_LEN·DMA_COUNT` |
| `PERF_DMA_WR_BYTES` | SRAM에서 DRAM으로 옮긴 `OP_DMA`마다 `DMA_LEN·DMA_COUNT` |
| `PERF_OPS` | `OP_CONV`, `OP_ADD`, `OP_AVGPOOL`, `OP_DMA`마다 1 |

명목 MAC은 층 모양만으로 정해지므로 설계 6.6절의 표와 컴파일러의 추정치와
바로 비교할 수 있다.

## 12. 예시

8x8x1 입력에 3x3 합성곱(출력 채널 4, `SAME` 패딩, stride 1)을 하는 명령열이다.

- 영역 0(상수)은 모두 `0x70`바이트로, 오프셋 0에 가중치 36바이트(`0x40`까지
  채움), `0x40`에 bias 16바이트, `0x50`에 재양자화 표 32바이트가 있다.
- 영역 1(arena)은 오프셋 0에 입력 64바이트가 있고, 출력 256바이트를
  `0x40`에 쓴다.
- SRAM 배치: 입력 `0x000`, 상수 `0x100`부터(가중치 `0x100`, bias `0x140`,
  표 `0x150`), 출력 `0x200`.

| 오프셋 | 워드 | 페이로드 | 명령 |
|--------|------|----------|------|
| `0x00` | `0x0000_4010` | `0x0000_0000` | `SET_DMA_SRC` 영역 0, 0 |
| `0x08` | `0x0100_4011` | `0x0000_0100` | `SET_DMA_DST` SRAM, `0x100` |
| `0x10` | `0x0000_4012` | `0x0000_0070` | `SET_DMA_LEN` `0x70` |
| `0x18` | `0x0000_0010` | | `OP_DMA` |
| `0x1C` | `0x0001_4010` | `0x0000_0000` | `SET_DMA_SRC` 영역 1, 0 |
| `0x24` | `0x0100_4011` | `0x0000_0000` | `SET_DMA_DST` SRAM, 0 |
| `0x2C` | `0x0000_4012` | `0x0000_0040` | `SET_DMA_LEN` 64 |
| `0x34` | `0x0000_0010` | | `OP_DMA` |
| `0x38` | `0x0000_4000` | `0x0000_0000` | `SET_IFM_BASE` 0 |
| `0x40` | `0x0001_4006` | `0x0008_0008` | `SET_IFM_SHAPE` 8, 8, 1 |
| `0x48` | `0xFF80_0100` | | `SET_IFM_ZP` -128 |
| `0x4C` | `0x0000_4003` | `0x0000_0100` | `SET_WEIGHT_BASE` `0x100` |
| `0x54` | `0x0000_4004` | `0x0000_0140` | `SET_BIAS_BASE` `0x140` |
| `0x5C` | `0x0000_4005` | `0x0000_0150` | `SET_RQ_TABLE` `0x150` |
| `0x64` | `0x0303_0103` | | `SET_KERNEL` 3, 3 |
| `0x68` | `0x1111_0105` | | `SET_PAD` 1, 1, 1, 1 |
| `0x6C` | `0x0000_4002` | `0x0000_0200` | `SET_OFM_BASE` `0x200` |
| `0x74` | `0x0004_4007` | `0x0008_0008` | `SET_OFM_SHAPE` 8, 8, 4 |
| `0x7C` | `0x0000_0011` | | `OP_DMA_WAIT` 0 |
| `0x80` | `0x0000_0002` | | `OP_CONV` |
| `0x84` | `0x0100_4010` | `0x0000_0200` | `SET_DMA_SRC` SRAM, `0x200` |
| `0x8C` | `0x0001_4011` | `0x0000_0040` | `SET_DMA_DST` 영역 1, `0x40` |
| `0x94` | `0x0000_4012` | `0x0000_0100` | `SET_DMA_LEN` 256 |
| `0x9C` | `0x0000_0012` | | `OP_KERNEL_WAIT` 0 |
| `0xA0` | `0x0000_0010` | | `OP_DMA` |
| `0xA4` | `0x0000_0000` | | `OP_STOP` |

`QSIZE`는 `0xA8`이다. 작업이 끝나면 성능 카운터는 `PERF_MACS` 2,304
(8·8·4·3·3·1), `PERF_DMA_RD_BYTES` 176, `PERF_DMA_WR_BYTES` 256, `PERF_OPS` 4가
된다.

대기를 빠뜨리면 다음처럼 된다.

- `0x7C`의 `OP_DMA_WAIT`를 빼면 `OP_CONV`가 두 DMA가 쓰는 SRAM을 읽는다.
  그래서 `SYNC` 오류가 나고, 이때 `ERR_OFFSET`은 `0x7C`다.
- `0x9C`의 `OP_KERNEL_WAIT`를 빼면 마지막 `OP_DMA`가 `OP_CONV`가 쓰는 출력을
  읽어 `SYNC` 오류가 난다.

## 13. 근거

| 출처 | 스냅숏 | 쓰인 곳 |
|------|--------|---------|
| TFLite Micro | `fd2a85d`(2026-11-04), `tensorflow/lite/micro/kernels/` 아래 | 8.1절 `internal/common.cc:31-113`, `internal/fixedpoint.h:328-367`, `internal/common.h:343-394`; 8.2절 `internal/reference/integer_ops/conv.h:27-136`, `fully_connected.h:137-181`; 8.3절 `integer_ops/add.h:105-125`, `add_common.cc:47`; 8.4절 `integer_ops/pooling.h:27-85` |
| Arm Vela | `4e52413`(2026-11-04), `ethosu/vela/` 아래 | 4절 명령 번호 공간(`ethos_u55_regs/ethos_u55_regs.py:574-673`, `register_command_stream_generator.py:116-120`); 6.5절 대기 파라미터(`register_command_stream_generator.py:213-216`); 9절 큐 깊이와 암묵 대기(`architecture_features.py:282-290`, 350, `register_command_stream_util.py:354-395`) |
| Linux | v7.3, `drivers/accel/ethosu/` | 4절 명령 해석(`ethosu_gem.c:403-430`: 하위 16비트로 명령 식별, 비트 14 페이로드); `OP_STOP`이 마지막 워드여야 함(430-433, 697). Ethos-U 검증기는 모르는 명령을 통과시키지만(692), kn1은 장치와 검증기 모두 거부한다 |
