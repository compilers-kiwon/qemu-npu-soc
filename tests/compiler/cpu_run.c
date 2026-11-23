/*
 * SPDX-FileCopyrightText: 2026 Kiwon Kwon
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * cpu_run: the CPU steps of a model.knpu on an arena, for V3 of the CPU
 * code under qemu-aarch64 user mode (test_mnist_model.py). It loads the
 * CPU_CODE section the way the runtime does (memfd_create, dlopen), checks
 * the symbols of CPU_STEPS (docs/model-format.md 11, rule 7) and runs the
 * steps on the bytes of the arena region, which knpu_run wrote after the
 * NPU job (docs/model-format.md 12).
 *
 *   cpu_run MODEL ARENA_IN ARENA_OUT
 *
 * model_file.py checks the format first; this program only keeps its own
 * reads and writes inside the file and the arena.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

enum { CPU_CODE = 5, CPU_STEPS = 6 };
enum { COPY_IN = 0, CALL = 1, COPY_OUT = 2 };

static void fail(const char *what, const char *detail)
{
	fprintf(stderr, "cpu_run: %s%s%s\n", what, detail ? ": " : "",
		detail ? detail : "");
	exit(1);
}

static uint32_t u32(const unsigned char *p)
{
	return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24;
}

static unsigned char *read_file(const char *path, size_t *size)
{
	FILE *f = fopen(path, "rb");
	unsigned char *data;
	long n;

	if (!f || fseek(f, 0, SEEK_END) || (n = ftell(f)) < 0 ||
	    fseek(f, 0, SEEK_SET))
		fail("cannot read", path);
	data = malloc(n ? n : 1);
	if (!data || fread(data, 1, n, f) != (size_t)n)
		fail("cannot read", path);
	fclose(f);
	*size = n;
	return data;
}

int main(int argc, char **argv)
{
	unsigned char *model, *arena;
	const unsigned char *code = NULL, *steps = NULL, *names;
	size_t model_size, arena_size, code_size = 0, steps_size = 0;
	size_t names_size, count, i;
	unsigned table, sections;
	int region = -1;
	char path[64];
	void **symbols, *lib;
	int fd;
	FILE *out;

	if (argc != 4) {
		fprintf(stderr, "usage: cpu_run MODEL ARENA_IN ARENA_OUT\n");
		return 2;
	}
	model = read_file(argv[1], &model_size);
	arena = read_file(argv[2], &arena_size);

	/* The two CPU sections (docs/model-format.md 2, 4) */
	if (model_size < 32)
		fail("model too small", argv[1]);
	sections = model[14] | model[15] << 8;
	table = u32(model + 16);
	if (table > model_size || sections > (model_size - table) / 16)
		fail("section table outside the file", argv[1]);
	for (i = 0; i < sections; i++) {
		const unsigned char *e = model + table + 16 * i;
		uint32_t offset = u32(e + 4), size = u32(e + 8);

		if (offset > model_size || size > model_size - offset)
			fail("section outside the file", argv[1]);
		if (u32(e) == CPU_CODE)
			code = model + offset, code_size = size;
		else if (u32(e) == CPU_STEPS)
			steps = model + offset, steps_size = size;
	}
	if (!code || !steps || steps_size < 16)
		fail("no CPU sections", argv[1]);
	count = u32(steps);
	if (count > (steps_size - 16) / 16)
		fail("CPU_STEPS too small", argv[1]);
	names = steps + 16 + 16 * count;
	names_size = steps_size - 16 - 16 * count;
	if (!names_size || names[names_size - 1])
		fail("CPU_STEPS names do not end with NUL", argv[1]);

	/* The shared object, from a memfd as the runtime loads it */
	fd = memfd_create("knpu-cpu", MFD_CLOEXEC);
	if (fd < 0 || write(fd, code, code_size) != (ssize_t)code_size)
		fail("cannot write the CPU code to a memfd", NULL);
	snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
	lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (!lib)
		fail("dlopen", dlerror());

	/* Every symbol first: a function to call or an object of the step's
	 * size (rule 7), and copies inside the arena */
	symbols = calloc(count, sizeof(*symbols));
	if (!symbols)
		fail("out of memory", NULL);
	for (i = 0; i < count; i++) {
		const unsigned char *e = steps + 16 + 16 * i;
		uint32_t name = u32(e + 4), offset = u32(e + 8);
		uint32_t size = u32(e + 12);
		const char *sym;
		const ElfW(Sym) *entry = NULL;
		Dl_info info;

		if (name >= names_size)
			fail("step name outside the name table", argv[1]);
		sym = (const char *)names + name;
		symbols[i] = dlsym(lib, sym);
		if (!symbols[i] ||
		    !dladdr1(symbols[i], &info, (void **)&entry,
			     RTLD_DL_SYMENT) || !entry)
			fail("no such symbol", sym);
		if (e[0] == CALL) {
			if (ELF64_ST_TYPE(entry->st_info) != STT_FUNC)
				fail("not a function", sym);
			continue;
		}
		if (e[0] != COPY_IN && e[0] != COPY_OUT)
			fail("unknown step kind", sym);
		if (ELF64_ST_TYPE(entry->st_info) != STT_OBJECT ||
		    entry->st_size != size)
			fail("not an object of the step's size", sym);
		if (region >= 0 && e[1] != region)
			fail("copies in more than one region", sym);
		region = e[1];
		if (offset > arena_size || size > arena_size - offset)
			fail("copy outside the arena", sym);
	}

	/* The steps, in order */
	for (i = 0; i < count; i++) {
		const unsigned char *e = steps + 16 + 16 * i;
		unsigned char *at = arena + u32(e + 8), *array = symbols[i];
		uint32_t size = u32(e + 12), j;
		void (*fn)(void);

		switch (e[0]) {
		case COPY_IN:
			for (j = 0; j < size; j++)
				array[j] = at[j];
			break;
		case CALL:
			*(void **)&fn = symbols[i];
			fn();
			break;
		default:
			for (j = 0; j < size; j++)
				at[j] = array[j];
		}
	}

	out = fopen(argv[3], "wb");
	if (!out || fwrite(arena, 1, arena_size, out) != arena_size ||
	    fclose(out))
		fail("cannot write", argv[3]);
	dlclose(lib);
	close(fd);
	return 0;
}
