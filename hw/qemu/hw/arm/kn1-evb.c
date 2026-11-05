/*
 * Kiwon kn1-evb board: kn1 SoC, 4 GiB DRAM, 64 MiB secure boot flash,
 * virtio-mmio transports and a GPIO-driven power controller.
 *
 * Copyright (c) 2026 Kiwon Kwon
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/cutils.h"
#include "qemu/datadir.h"
#include "qemu/error-report.h"
#include "hw/arm/boot.h"
#include "hw/arm/kiwon-kn1.h"
#include "hw/arm/machines-qom.h"
#include "hw/block/flash.h"
#include "hw/core/boards.h"
#include "hw/core/loader.h"
#include "hw/core/qdev-properties.h"
#include "hw/virtio/virtio-mmio.h"
#include "system/address-spaces.h"
#include "system/blockdev.h"
#include "system/qtest.h"

#define TYPE_KN1_EVB_MACHINE MACHINE_TYPE_NAME("kn1-evb")
OBJECT_DECLARE_SIMPLE_TYPE(Kn1EvbMachineState, KN1_EVB_MACHINE)

#define KN1_EVB_FLASH_SECTOR_SIZE (256 * KiB)

struct Kn1EvbMachineState {
    MachineState parent_obj;

    KiwonKn1State soc;
    PFlashCFI01 *flash;
    struct arm_boot_info boot_info;
};

/*
 * The flash exists from instance_init so that "-drive if=pflash" and
 * "-machine pflash0=" can refer to it; it is sized and mapped at init.
 */
static void kn1_evb_flash_create(Kn1EvbMachineState *s)
{
    DeviceState *dev = qdev_new(TYPE_PFLASH_CFI01);

    qdev_prop_set_uint64(dev, "sector-length", KN1_EVB_FLASH_SECTOR_SIZE);
    qdev_prop_set_uint8(dev, "width", 4);
    qdev_prop_set_uint8(dev, "device-width", 2);
    qdev_prop_set_bit(dev, "big-endian", false);
    qdev_prop_set_uint16(dev, "id0", 0x89);
    qdev_prop_set_uint16(dev, "id1", 0x18);
    qdev_prop_set_uint16(dev, "id2", 0x00);
    qdev_prop_set_uint16(dev, "id3", 0x00);
    qdev_prop_set_string(dev, "name", "kn1-evb.flash0");
    object_property_add_child(OBJECT(s), "kn1-evb.flash0", OBJECT(dev));
    object_property_add_alias(OBJECT(s), "pflash0", OBJECT(dev), "drive");
    s->flash = PFLASH_CFI01(dev);
}

/* Map the boot flash at the reset vector of the secure view; load -bios */
static bool kn1_evb_firmware_init(Kn1EvbMachineState *s)
{
    MachineState *machine = MACHINE(s);
    const MemMapEntry *entry = &kiwon_kn1_memmap[KN1_BOOT_FLASH];
    DeviceState *dev = DEVICE(s->flash);
    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);
    BlockBackend *blk;

    pflash_cfi01_legacy_drive(s->flash, drive_get(IF_PFLASH, 0, 0));

    assert(QEMU_IS_ALIGNED(entry->size, KN1_EVB_FLASH_SECTOR_SIZE));
    qdev_prop_set_uint32(dev, "num-blocks",
                         entry->size / KN1_EVB_FLASH_SECTOR_SIZE);
    sysbus_realize_and_unref(sbd, &error_fatal);
    memory_region_add_subregion(&s->soc.secure_sysmem, entry->base,
                                sysbus_mmio_get_region(sbd, 0));

    blk = pflash_cfi01_get_blk(s->flash);
    if (machine->firmware) {
        g_autofree char *fname = NULL;

        if (blk) {
            error_report("The boot flash may be given with -bios or with "
                         "-drive if=pflash, but not both");
            exit(1);
        }
        fname = qemu_find_file(QEMU_FILE_TYPE_BIOS, machine->firmware);
        if (!fname) {
            error_report("Could not find ROM image '%s'", machine->firmware);
            exit(1);
        }
        if (load_image_mr(fname, sysbus_mmio_get_region(sbd, 0)) < 0) {
            error_report("Could not load ROM image '%s'", machine->firmware);
            exit(1);
        }
    }

    return blk || machine->firmware;
}

static void kn1_evb_init(MachineState *machine)
{
    Kn1EvbMachineState *s = KN1_EVB_MACHINE(machine);
    const MemMapEntry *virtio = &kiwon_kn1_memmap[KN1_VIRTIO_MMIO];
    DeviceState *pwr;
    bool firmware_loaded;
    int i;

    /* The board's DRAM is soldered on: refuse any other -m value */
    if (machine->ram_size != KN1_DRAM_SIZE) {
        g_autofree char *size_str = size_to_str(KN1_DRAM_SIZE);

        error_report("kn1-evb: RAM size is fixed at %s", size_str);
        exit(1);
    }

    object_initialize_child(OBJECT(machine), "soc", &s->soc, TYPE_KIWON_KN1);
    sysbus_realize_and_unref(SYS_BUS_DEVICE(&s->soc), &error_fatal);

    memory_region_add_subregion(get_system_memory(),
                                kiwon_kn1_memmap[KN1_DRAM].base, machine->ram);

    firmware_loaded = kn1_evb_firmware_init(s);

    for (i = 0; i < KN1_NUM_VIRTIO_TRANSPORTS; i++) {
        sysbus_create_simple(TYPE_VIRTIO_MMIO,
                             virtio->base + i * KN1_VIRTIO_MMIO_SIZE,
                             kiwon_kn1_spi(&s->soc, KN1_IRQ_VIRTIO_MMIO + i));
    }

    /* Power controller: the secure GPIO lines shut down or reset QEMU */
    pwr = sysbus_create_simple("gpio-pwr", -1, NULL);
    qdev_connect_gpio_out(s->soc.gpio, KN1_GPIO_POWEROFF,
                          qdev_get_gpio_in_named(pwr, "shutdown", 0));
    qdev_connect_gpio_out(s->soc.gpio, KN1_GPIO_RESET,
                          qdev_get_gpio_in_named(pwr, "reset", 0));

    /*
     * With EL3 firmware in the flash, BL31 implements PSCI and all CPUs
     * start at the reset vector. Without it, a kernel given with -kernel
     * boots directly at EL2 with QEMU's PSCI behind SMC.
     */
    s->boot_info = (struct arm_boot_info) {
        .loader_start = kiwon_kn1_memmap[KN1_DRAM].base,
        .ram_size = machine->ram_size,
        .board_id = -1,
        .firmware_loaded = firmware_loaded,
        .psci_conduit = firmware_loaded ? QEMU_PSCI_CONDUIT_DISABLED
                                        : QEMU_PSCI_CONDUIT_SMC,
    };

    if (!qtest_enabled()) {
        arm_load_kernel(&s->soc.cpu[0], machine, &s->boot_info);
    }
}

static void kn1_evb_instance_init(Object *obj)
{
    kn1_evb_flash_create(KN1_EVB_MACHINE(obj));
}

static void kn1_evb_instance_finalize(Object *obj)
{
    Kn1EvbMachineState *s = KN1_EVB_MACHINE(obj);

    if (s->flash && !qdev_is_realized(DEVICE(s->flash))) {
        object_unref(OBJECT(s->flash));
    }
}

static void kn1_evb_class_init(ObjectClass *oc, const void *data)
{
    MachineClass *mc = MACHINE_CLASS(oc);
    static const char *const valid_cpu_types[] = {
        ARM_CPU_TYPE_NAME("cortex-a72"),
        NULL,
    };

    mc->desc = "Kiwon kn1-evb board (kn1 SoC: 4x Cortex-A72, GICv3)";
    mc->init = kn1_evb_init;
    mc->min_cpus = KN1_NUM_CPUS;
    mc->max_cpus = KN1_NUM_CPUS;
    mc->default_cpus = KN1_NUM_CPUS;
    mc->default_cpu_type = ARM_CPU_TYPE_NAME("cortex-a72");
    mc->valid_cpu_types = valid_cpu_types;
    mc->default_ram_id = "kn1-evb.ram";
    mc->default_ram_size = KN1_DRAM_SIZE;
    mc->block_default_type = IF_VIRTIO;
    mc->no_cdrom = 1;
    mc->no_floppy = 1;
    mc->no_parallel = 1;
}

static const TypeInfo kn1_evb_types[] = {
    {
        .name = TYPE_KN1_EVB_MACHINE,
        .parent = TYPE_MACHINE,
        .instance_size = sizeof(Kn1EvbMachineState),
        .instance_init = kn1_evb_instance_init,
        .instance_finalize = kn1_evb_instance_finalize,
        .class_init = kn1_evb_class_init,
        .interfaces = aarch64_machine_interfaces,
    },
};

DEFINE_TYPES(kn1_evb_types)
