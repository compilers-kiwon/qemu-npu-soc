# SPDX-FileCopyrightText: 2026 Kiwon Kwon
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "KNPU"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir", ".c"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.knpu_obj_root, "test")
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in"]

llvm_config.with_system_environment(["HOME", "TMP", "TEMP"])
llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [os.path.join(config.knpu_obj_root, "bin"), config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    ["knpu-opt", "knpu-translate", "knpu-compile"], tool_dirs)

# lcc's aarch64 mlir driver (docs/design.md 9.4)
config.substitutions.append(
    ("%lcc", f"env LCCDIR={config.lcc_dir} {config.lcc_dir}/lcc"))
