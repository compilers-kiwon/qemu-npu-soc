/*
 * Kiwon kn1 SoC: 4x Cortex-A72, GICv3, PL011 UARTs, PL031 RTC,
 * SBSA generic watchdog, secure PL061 GPIO, secure on-chip RAM and the
 * kn1-npu convolution NPU.
 *
 * Copyright (c) 2026 Kiwon Kwon
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qobject/qlist.h"
#include "hw/arm/bsa.h"
#include "hw/arm/kiwon-kn1.h"
#include "hw/core/boards.h"
#include "hw/core/qdev-properties.h"
#include "hw/misc/unimp.h"
#include "system/address-spaces.h"
#include "system/system.h"
#include "target/arm/cpu-qom.h"
#include "target/arm/gtimer.h"

const MemMapEntry kiwon_kn1_memmap[] = {
    [KN1_BOOT_FLASH] =      { 0x00000000, 64 * MiB },
    [KN1_GIC_DIST] =        { 0x08000000, 0x10000 },
    [KN1_GIC_ITS] =         { 0x08080000, 0x20000 },
    [KN1_GIC_REDIST] =      { 0x080A0000, KN1_NUM_CPUS * GICV3_REDIST_SIZE },
    [KN1_UART0] =           { 0x09000000, 0x1000 },
    [KN1_RTC] =             { 0x09010000, 0x1000 },
    [KN1_UART1] =           { 0x09040000, 0x1000 },
    [KN1_SECURE_GPIO] =     { 0x090B0000, 0x1000 },
    [KN1_VIRTIO_MMIO] =     { 0x0A000000, KN1_NUM_VIRTIO_TRANSPORTS *
                                          KN1_VIRTIO_MMIO_SIZE },
    [KN1_NPU] =             { 0x0B000000, 0x10000 },
    [KN1_SECURE_RAM] =      { 0x0E000000, 16 * MiB },
    [KN1_GWDT_REFRESH] =    { 0x0F000000, 0x1000 },
    [KN1_GWDT_CONTROL] =    { 0x0F001000, 0x1000 },
    [KN1_PCIE_MMIO] =       { 0x10000000, 0x2EFF0000 },
    [KN1_PCIE_PIO] =        { 0x3EFF0000, 0x10000 },
    [KN1_PCIE_ECAM] =       { 0x3F000000, 16 * MiB },
    [KN1_DRAM] =            { 0x40000000, KN1_DRAM_SIZE },
};

qemu_irq kiwon_kn1_spi(KiwonKn1State *s, int spi)
{
    assert(spi >= 0 && spi < KN1_NUM_IRQS);
    return qdev_get_gpio_in(DEVICE(&s->gic), spi);
}

static void kiwon_kn1_init(Object *obj)
{
    KiwonKn1State *s = KIWON_KN1(obj);
    int i;

    object_initialize_child(obj, "gic", &s->gic, gicv3_class_name());

    for (i = 0; i < KN1_NUM_UARTS; i++) {
        g_autofree char *name = g_strdup_printf("uart%d", i);
        object_initialize_child(obj, name, &s->uart[i], TYPE_PL011);
    }

    object_initialize_child(obj, "rtc", &s->rtc, TYPE_PL031);
    object_initialize_child(obj, "wdt", &s->wdt, TYPE_WDT_SBSA);
    object_initialize_child(obj, "npu", &s->npu, TYPE_KIWON_KN1_NPU);
}

static bool kiwon_kn1_realize_cpus(KiwonKn1State *s, MachineState *ms,
                                   MemoryRegion *sysmem, Error **errp)
{
    const char *cpu_type = ms->cpu_type ?: ARM_CPU_TYPE_NAME("cortex-a72");
    int i;

    if (ms->smp.cpus != KN1_NUM_CPUS) {
        error_setg(errp, "%s: exactly %d CPUs are supported (%d requested)",
                   TYPE_KIWON_KN1, KN1_NUM_CPUS, ms->smp.cpus);
        return false;
    }

    for (i = 0; i < KN1_NUM_CPUS; i++) {
        g_autofree char *name = g_strdup_printf("cpu%d", i);
        Object *cpuobj;

        object_initialize_child(OBJECT(s), name, &s->cpu[i], cpu_type);
        cpuobj = OBJECT(&s->cpu[i]);

        /* One cluster: mp-affinity defaults to Aff0 = i */
        if (object_property_find(cpuobj, "reset-cbar")) {
            object_property_set_int(cpuobj, "reset-cbar",
                                    kiwon_kn1_memmap[KN1_GIC_DIST].base,
                                    &error_abort);
        }
        object_property_set_int(cpuobj, "cntfrq", KN1_CNTFRQ_HZ,
                                &error_abort);
        /* EL3 and EL2 stay at the CPU model's defaults: both present */
        object_property_set_link(cpuobj, "memory", OBJECT(sysmem),
                                 &error_abort);
        object_property_set_link(cpuobj, "secure-memory",
                                 OBJECT(&s->secure_sysmem), &error_abort);

        if (!qdev_realize(DEVICE(&s->cpu[i]), NULL, errp)) {
            return false;
        }
    }
    return true;
}

static bool kiwon_kn1_realize_gic(KiwonKn1State *s, Error **errp)
{
    DeviceState *gicdev = DEVICE(&s->gic);
    SysBusDevice *gicsbd = SYS_BUS_DEVICE(&s->gic);
    QList *redist_region_count;
    int i;

    qdev_prop_set_uint32(gicdev, "revision", 3);
    qdev_prop_set_uint32(gicdev, "num-cpu", KN1_NUM_CPUS);
    qdev_prop_set_uint32(gicdev, "num-irq", KN1_NUM_IRQS + GIC_INTERNAL);
    qdev_prop_set_bit(gicdev, "has-security-extensions", true);
    redist_region_count = qlist_new();
    qlist_append_int(redist_region_count, KN1_NUM_CPUS);
    qdev_prop_set_array(gicdev, "redist-region-count", redist_region_count);

    if (!sysbus_realize(gicsbd, errp)) {
        return false;
    }
    sysbus_mmio_map(gicsbd, 0, kiwon_kn1_memmap[KN1_GIC_DIST].base);
    sysbus_mmio_map(gicsbd, 1, kiwon_kn1_memmap[KN1_GIC_REDIST].base);

    /*
     * Wire each CPU's generic timer and maintenance outputs to its PPIs,
     * and the GIC's per-CPU outputs to the CPU interrupt inputs.
     */
    for (i = 0; i < KN1_NUM_CPUS; i++) {
        DeviceState *cpudev = DEVICE(&s->cpu[i]);
        int intidbase = KN1_NUM_IRQS + i * GIC_INTERNAL;
        static const int timer_irq[] = {
            [GTIMER_PHYS] = ARCH_TIMER_NS_EL1_IRQ,
            [GTIMER_VIRT] = ARCH_TIMER_VIRT_IRQ,
            [GTIMER_HYP] = ARCH_TIMER_NS_EL2_IRQ,
            [GTIMER_SEC] = ARCH_TIMER_S_EL1_IRQ,
            [GTIMER_HYPVIRT] = ARCH_TIMER_NS_EL2_VIRT_IRQ,
            [GTIMER_S_EL2_PHYS] = ARCH_TIMER_S_EL2_IRQ,
            [GTIMER_S_EL2_VIRT] = ARCH_TIMER_S_EL2_VIRT_IRQ,
        };
        int j;

        for (j = 0; j < ARRAY_SIZE(timer_irq); j++) {
            qdev_connect_gpio_out(cpudev, j,
                                  qdev_get_gpio_in(gicdev,
                                                   intidbase + timer_irq[j]));
        }
        qdev_connect_gpio_out_named(cpudev, "gicv3-maintenance-interrupt", 0,
                                    qdev_get_gpio_in(gicdev, intidbase +
                                                     ARCH_GIC_MAINT_IRQ));
        qdev_connect_gpio_out_named(cpudev, "pmu-interrupt", 0,
                                    qdev_get_gpio_in(gicdev, intidbase +
                                                     VIRTUAL_PMU_IRQ));

        sysbus_connect_irq(gicsbd, i,
                           qdev_get_gpio_in(cpudev, ARM_CPU_IRQ));
        sysbus_connect_irq(gicsbd, i + KN1_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_FIQ));
        sysbus_connect_irq(gicsbd, i + 2 * KN1_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_VIRQ));
        sysbus_connect_irq(gicsbd, i + 3 * KN1_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_VFIQ));
        sysbus_connect_irq(gicsbd, i + 4 * KN1_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_NMI));
        sysbus_connect_irq(gicsbd, i + 5 * KN1_NUM_CPUS,
                           qdev_get_gpio_in(cpudev, ARM_CPU_VINMI));
    }
    return true;
}

static void kiwon_kn1_realize(DeviceState *dev, Error **errp)
{
    KiwonKn1State *s = KIWON_KN1(dev);
    MachineState *ms = MACHINE(qdev_get_machine());
    MemoryRegion *sysmem = get_system_memory();
    SysBusDevice *sbd;
    int i;

    /*
     * The secure view is the non-secure bus at low priority with the
     * secure-only blocks on top, as the "virt" machine does.
     */
    memory_region_init(&s->secure_sysmem, OBJECT(s), "kn1.secure-memory",
                       UINT64_MAX);
    memory_region_add_subregion_overlap(&s->secure_sysmem, 0, sysmem, -1);

    if (!memory_region_init_ram(&s->secure_ram, OBJECT(s), "kn1.secure-ram",
                                kiwon_kn1_memmap[KN1_SECURE_RAM].size, errp)) {
        return;
    }
    memory_region_add_subregion(&s->secure_sysmem,
                                kiwon_kn1_memmap[KN1_SECURE_RAM].base,
                                &s->secure_ram);

    if (!kiwon_kn1_realize_cpus(s, ms, sysmem, errp)) {
        return;
    }
    if (!kiwon_kn1_realize_gic(s, errp)) {
        return;
    }

    /* UART0: non-secure console */
    sbd = SYS_BUS_DEVICE(&s->uart[0]);
    qdev_prop_set_chr(DEVICE(sbd), "chardev", serial_hd(0));
    if (!sysbus_realize(sbd, errp)) {
        return;
    }
    sysbus_mmio_map(sbd, 0, kiwon_kn1_memmap[KN1_UART0].base);
    sysbus_connect_irq(sbd, 0, kiwon_kn1_spi(s, KN1_IRQ_UART0));

    /* UART1: secure console; the firmware polls it but the line exists */
    sbd = SYS_BUS_DEVICE(&s->uart[1]);
    qdev_prop_set_chr(DEVICE(sbd), "chardev", serial_hd(1));
    if (!sysbus_realize(sbd, errp)) {
        return;
    }
    memory_region_add_subregion(&s->secure_sysmem,
                                kiwon_kn1_memmap[KN1_UART1].base,
                                sysbus_mmio_get_region(sbd, 0));
    sysbus_connect_irq(sbd, 0, kiwon_kn1_spi(s, KN1_IRQ_UART1));

    /* RTC */
    sbd = SYS_BUS_DEVICE(&s->rtc);
    if (!sysbus_realize(sbd, errp)) {
        return;
    }
    sysbus_mmio_map(sbd, 0, kiwon_kn1_memmap[KN1_RTC].base);
    sysbus_connect_irq(sbd, 0, kiwon_kn1_spi(s, KN1_IRQ_RTC));

    /* SBSA generic watchdog, counting at the generic timer frequency */
    sbd = SYS_BUS_DEVICE(&s->wdt);
    qdev_prop_set_uint64(DEVICE(sbd), "clock-frequency", KN1_CNTFRQ_HZ);
    if (!sysbus_realize(sbd, errp)) {
        return;
    }
    sysbus_mmio_map(sbd, 0, kiwon_kn1_memmap[KN1_GWDT_REFRESH].base);
    sysbus_mmio_map(sbd, 1, kiwon_kn1_memmap[KN1_GWDT_CONTROL].base);
    sysbus_connect_irq(sbd, 0, kiwon_kn1_spi(s, KN1_IRQ_GWDT_WS0));

    /*
     * Secure GPIO: output lines only (power-off, reset), driven by the
     * firmware; its interrupt is not connected.
     */
    s->gpio = qdev_new("pl061");
    qdev_prop_set_uint8(s->gpio, "pullups", 0);
    qdev_prop_set_uint8(s->gpio, "pulldowns", 0xff);
    object_property_add_child(OBJECT(s), "gpio", OBJECT(s->gpio));
    sbd = SYS_BUS_DEVICE(s->gpio);
    if (!sysbus_realize_and_unref(sbd, errp)) {
        return;
    }
    memory_region_add_subregion(&s->secure_sysmem,
                                kiwon_kn1_memmap[KN1_SECURE_GPIO].base,
                                sysbus_mmio_get_region(sbd, 0));

    /* NPU: masters the non-secure bus for its DMA and command fetch */
    sbd = SYS_BUS_DEVICE(&s->npu);
    object_property_set_link(OBJECT(sbd), "memory", OBJECT(sysmem),
                             &error_abort);
    if (!sysbus_realize(sbd, errp)) {
        return;
    }
    sysbus_mmio_map(sbd, 0, kiwon_kn1_memmap[KN1_NPU].base);
    sysbus_connect_irq(sbd, 0, kiwon_kn1_spi(s, KN1_IRQ_NPU));

    /* Reserved windows: log accesses instead of faulting */
    create_unimplemented_device("kn1.gic-its",
                                kiwon_kn1_memmap[KN1_GIC_ITS].base,
                                kiwon_kn1_memmap[KN1_GIC_ITS].size);
    for (i = KN1_PCIE_MMIO; i <= KN1_PCIE_ECAM; i++) {
        static const char *const names[] = {
            [KN1_PCIE_MMIO] = "kn1.pcie-mmio",
            [KN1_PCIE_PIO] = "kn1.pcie-pio",
            [KN1_PCIE_ECAM] = "kn1.pcie-ecam",
        };
        create_unimplemented_device(names[i], kiwon_kn1_memmap[i].base,
                                    kiwon_kn1_memmap[i].size);
    }
}

static void kiwon_kn1_class_init(ObjectClass *oc, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);

    dc->realize = kiwon_kn1_realize;
    dc->desc = "Kiwon kn1 SoC";
    /* The SoC is created by the board, not with -device */
    dc->user_creatable = false;
}

static const TypeInfo kiwon_kn1_types[] = {
    {
        .name = TYPE_KIWON_KN1,
        .parent = TYPE_SYS_BUS_DEVICE,
        .instance_size = sizeof(KiwonKn1State),
        .instance_init = kiwon_kn1_init,
        .class_init = kiwon_kn1_class_init,
    },
};

DEFINE_TYPES(kiwon_kn1_types)
