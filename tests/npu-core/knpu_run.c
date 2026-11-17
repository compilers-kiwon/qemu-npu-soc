/*
 * Host runner for the kn1-npu shared core
 *
 * Builds a sparse system memory out of --mem blocks, runs one job on
 * hw/npu-core and prints the result. An access outside every block is a
 * bus error, which lets tests reach KNPU_ERR_BUS.
 *
 *   knpu_run [--sram-size BYTES] --queue ADDR:SIZE
 *            [--region N:ADDR]... [--mem ADDR:SIZE[:FILE]]...
 *            [--dump ADDR:SIZE:FILE]... [--sram-in FILE] [--sram-out FILE]
 *
 * Output: err_code=0xNN err_offset=0xNNNN macs=N dma_rd_bytes=N
 *         dma_wr_bytes=N ops=N
 *
 * SPDX-FileCopyrightText: 2026 Kiwon Kwon
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "knpu_core.h"

#define MAX_BLOCKS 32

typedef struct Block {
    uint64_t addr, size;
    uint8_t *data;
} Block;

static Block blocks[MAX_BLOCKS];
static int nblocks;

static void die(const char *msg, const char *arg)
{
    fprintf(stderr, "knpu_run: %s%s%s\n", msg, arg ? ": " : "",
            arg ? arg : "");
    exit(2);
}

static uint64_t num(const char *s)
{
    char *end;
    uint64_t v = strtoull(s, &end, 0);
    if (*s == '\0' || *end != '\0') {
        die("bad number", s);
    }
    return v;
}

/* Split "a:b:c" in place; returns the number of fields */
static int split(char *s, char **f, int max)
{
    int n = 0;
    f[n++] = s;
    for (char *p = s; *p && n < max; p++) {
        if (*p == ':') {
            *p = '\0';
            f[n++] = p + 1;
        }
    }
    return n;
}

static void load_file(const char *path, uint8_t *buf, uint64_t size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        die("cannot open", path);
    }
    size_t got = fread(buf, 1, size, f);
    if (ferror(f) || fgetc(f) != EOF || got > size) {
        die("file larger than its block or unreadable", path);
    }
    fclose(f);
}

static void save_file(const char *path, const uint8_t *buf, uint64_t size)
{
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(buf, 1, size, f) != size || fclose(f)) {
        die("cannot write", path);
    }
}

static uint8_t *find(uint64_t addr, uint64_t len)
{
    for (int i = 0; i < nblocks; i++) {
        Block *b = &blocks[i];
        if (addr >= b->addr && addr - b->addr <= b->size &&
            len <= b->size - (addr - b->addr)) {
            return b->data + (addr - b->addr);
        }
    }
    return NULL;
}

static bool mem_read(void *opaque, uint64_t addr, void *buf, uint64_t len)
{
    uint8_t *p = find(addr, len);
    (void)opaque;
    if (!p) {
        return false;
    }
    memcpy(buf, p, len);
    return true;
}

static bool mem_write(void *opaque, uint64_t addr, const void *buf,
                      uint64_t len)
{
    uint8_t *p = find(addr, len);
    (void)opaque;
    if (!p) {
        return false;
    }
    memcpy(p, buf, len);
    return true;
}

int main(int argc, char **argv)
{
    static const KnpuMemOps ops = { mem_read, mem_write };
    KnpuJob job = { 0 };
    KnpuResult res;
    uint64_t sram_size = 64 * 1024;
    const char *sram_in = NULL, *sram_out = NULL;
    char *dumps[MAX_BLOCKS];
    int ndumps = 0;
    bool have_queue = false;

    for (int i = 1; i < argc; i++) {
        char *f[3];
        const char *opt = argv[i];
        if (i + 1 >= argc) {
            die("missing value for", opt);
        }
        char *val = argv[++i];

        if (!strcmp(opt, "--sram-size")) {
            sram_size = num(val);
        } else if (!strcmp(opt, "--queue")) {
            if (split(val, f, 2) != 2) {
                die("expected ADDR:SIZE", opt);
            }
            job.qbase = num(f[0]);
            job.qsize = (uint32_t)num(f[1]);
            have_queue = true;
        } else if (!strcmp(opt, "--region")) {
            if (split(val, f, 2) != 2 || num(f[0]) >= KNPU_NUM_REGIONS) {
                die("expected N:ADDR", opt);
            }
            job.region_base[num(f[0])] = num(f[1]);
        } else if (!strcmp(opt, "--mem")) {
            int n = split(val, f, 3);
            if (n < 2 || nblocks == MAX_BLOCKS) {
                die("expected ADDR:SIZE[:FILE]", opt);
            }
            Block *b = &blocks[nblocks++];
            b->addr = num(f[0]);
            b->size = num(f[1]);
            b->data = calloc(1, b->size ? b->size : 1);
            if (!b->data) {
                die("out of memory", NULL);
            }
            if (n == 3) {
                load_file(f[2], b->data, b->size);
            }
        } else if (!strcmp(opt, "--dump")) {
            if (ndumps == MAX_BLOCKS) {
                die("too many", opt);
            }
            dumps[ndumps++] = val;
        } else if (!strcmp(opt, "--sram-in")) {
            sram_in = val;
        } else if (!strcmp(opt, "--sram-out")) {
            sram_out = val;
        } else {
            die("unknown option", opt);
        }
    }
    if (!have_queue) {
        die("--queue is required", NULL);
    }

    uint8_t *sram = calloc(1, sram_size);
    if (!sram) {
        die("out of memory", NULL);
    }
    if (sram_in) {
        load_file(sram_in, sram, sram_size);
    }

    knpu_core_run(sram, (uint32_t)sram_size, &ops, NULL, &job, &res);

    for (int i = 0; i < ndumps; i++) {
        char *f[3];
        if (split(dumps[i], f, 3) != 3) {
            die("expected ADDR:SIZE:FILE", "--dump");
        }
        uint64_t size = num(f[1]);
        uint8_t *p = find(num(f[0]), size);
        if (!p) {
            die("dump range is not inside one --mem block", f[0]);
        }
        save_file(f[2], p, size);
    }
    if (sram_out) {
        save_file(sram_out, sram, sram_size);
    }

    printf("err_code=0x%02x err_offset=0x%" PRIx32 " macs=%" PRIu64
           " dma_rd_bytes=%" PRIu64 " dma_wr_bytes=%" PRIu64
           " ops=%" PRIu64 "\n",
           res.err_code, res.err_offset, res.perf.macs,
           res.perf.dma_rd_bytes, res.perf.dma_wr_bytes, res.perf.ops);
    return 0;
}
