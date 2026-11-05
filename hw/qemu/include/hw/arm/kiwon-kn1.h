/*
 * Kiwon kn1 SoC
 *
 * Copyright (c) 2026 Kiwon Kwon
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef HW_ARM_KIWON_KN1_H
#define HW_ARM_KIWON_KN1_H

#include "exec/hwaddr.h"
#include "hw/core/sysbus.h"
#include "hw/char/pl011.h"
#include "hw/intc/arm_gicv3_common.h"
#include "hw/rtc/pl031.h"
#include "hw/watchdog/sbsa_gwdt.h"
#include "qemu/units.h"
#include "qom/object.h"
#include "target/arm/cpu.h"

#define TYPE_KIWON_KN1 "kiwon-kn1"
OBJECT_DECLARE_SIMPLE_TYPE(KiwonKn1State, KIWON_KN1)

enum {
    KN1_NUM_CPUS = 4,
    KN1_NUM_UARTS = 2,
    KN1_NUM_IRQS = 256,         /* SPIs */
    KN1_NUM_VIRTIO_TRANSPORTS = 4,
};

/* Memory map; blocks shared with the QEMU "virt" machine keep its addresses */
enum KiwonKn1MemMap {
    KN1_BOOT_FLASH,             /* board: CFI flash, secure, reset vector */
    KN1_GIC_DIST,
    KN1_GIC_ITS,                /* reserved */
    KN1_GIC_REDIST,
    KN1_UART0,                  /* non-secure console */
    KN1_RTC,
    KN1_UART1,                  /* secure, TF-A console */
    KN1_SECURE_GPIO,            /* power-off and reset lines */
    KN1_VIRTIO_MMIO,            /* board: 4 transports */
    KN1_NPU,                    /* reserved until the NPU device is added */
    KN1_SECURE_RAM,
    KN1_GWDT_REFRESH,
    KN1_GWDT_CONTROL,
    KN1_PCIE_MMIO,              /* reserved */
    KN1_PCIE_PIO,               /* reserved */
    KN1_PCIE_ECAM,              /* reserved */
    KN1_DRAM,                   /* board */
};

extern const MemMapEntry kiwon_kn1_memmap[];

/* Interrupt map: SPI numbers (INTID = 32 + SPI) */
enum KiwonKn1Irq {
    KN1_IRQ_UART0 = 1,
    KN1_IRQ_RTC = 2,
    KN1_IRQ_PCIE_INTA = 3,      /* INTA..INTD: 3..6, reserved */
    KN1_IRQ_GWDT_WS0 = 10,
    KN1_IRQ_VIRTIO_MMIO = 16,   /* 16..19 */
    KN1_IRQ_NPU = 32,
};

/* Secure PL061 lines wired to the board power controller */
enum {
    KN1_GPIO_POWEROFF = 0,
    KN1_GPIO_RESET = 1,
};

#define KN1_VIRTIO_MMIO_SIZE    0x200
#define KN1_DRAM_SIZE           (4 * GiB)
#define KN1_CNTFRQ_HZ           62500000    /* QEMU's Cortex-A72 default */

struct KiwonKn1State {
    SysBusDevice parent_obj;

    ARMCPU cpu[KN1_NUM_CPUS];
    GICv3State gic;
    PL011State uart[KN1_NUM_UARTS];
    PL031State rtc;
    SBSA_GWDTState wdt;
    DeviceState *gpio;

    /* Secure view of the bus: system memory plus the secure-only blocks */
    MemoryRegion secure_sysmem;
    MemoryRegion secure_ram;
};

/* GIC input for the given SPI number, for board-level devices */
qemu_irq kiwon_kn1_spi(KiwonKn1State *s, int spi);

#endif
