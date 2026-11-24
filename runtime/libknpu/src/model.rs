// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! model.knpu v0 (docs/model-format.md): parsing and the checks of its
//! section 11 that need neither the device nor the loaded CPU code.
//! [`Model::check_device`] does those of the device, and the CPU code
//! loader (cpu.rs) rule 7.

use crate::{DeviceInfo, Error, Result};

const MAGIC: &[u8; 4] = b"KNPU";
const HEADER_SIZE: usize = 32;
const TABLE_ENTRY: usize = 16;
const REGION_ENTRY: usize = 16;
const TENSOR_ENTRY: usize = 48;
const NO_CONTENT: u32 = 0xFFFF_FFFF;

const KIND_CMDS: u32 = 1;
const KIND_REGIONS: u32 = 2;
const KIND_CONST: u32 = 3;
const KIND_TENSORS: u32 = 4;
const KIND_CPU_CODE: u32 = 5;
const KIND_CPU_STEPS: u32 = 6;

/// The ELF header of CPU_CODE (docs/model-format.md 9).
const ELF_HEADER_SIZE: usize = 64;
const ET_DYN: u16 = 3;
const EM_AARCH64: u16 = 183;

/// CPU_STEPS (docs/model-format.md 10).
const STEPS_HEADER: usize = 16;
const STEP_ENTRY: usize = 16;

/// Command numbers the runtime looks at (docs/npu-spec.md 6.1).
const CMD_MASK: u32 = 0xFFFF;
const CMD_PAYLOAD: u32 = 0x4000;
const OP_STOP: u32 = 0x0000;
const OPS: [u32; 4] = [0x0002, 0x0003, 0x0004, 0x0010]; // CONV ADD AVGPOOL DMA

/// Role of a region (docs/design.md D40).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionRole {
    Const,
    Arena,
}

/// One entry of the region table.
#[derive(Debug, Clone)]
pub struct Region {
    pub index: u32,
    pub role: RegionRole,
    pub size: u32,
    /// Contents of a constant region.
    pub content: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Input,
    Output,
}

/// One model input or output (int8).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub direction: Direction,
    pub region: u32,
    pub offset: u32,
    pub size: u32,
    pub shape: Vec<u32>,
    pub scale: f32,
    pub zero_point: i32,
}

/// What a CPU step does (docs/model-format.md 10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    /// Copies `size` bytes at `offset` of the region into the array.
    CopyIn,
    /// Calls the function, `void f(void)`.
    Call,
    /// Copies the array to `offset` of the region.
    CopyOut,
}

/// One entry of CPU_STEPS. A call has region, offset and size 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuStep {
    pub kind: StepKind,
    /// The arena region of a copy.
    pub region: u32,
    /// The symbol in the CPU code, a C identifier.
    pub name: String,
    pub offset: u32,
    pub size: u32,
}

/// A checked model.knpu file.
#[derive(Debug, Clone)]
pub struct Model {
    pub format_minor: u16,
    pub sram_size: u32,
    pub spec_major: u16,
    /// The command stream, as it goes to the driver.
    pub cmds: Vec<u8>,
    pub regions: Vec<Region>,
    pub tensors: Vec<Tensor>,
    /// The CPU code, an AArch64 shared object, if the model has CPU
    /// sections (format minor 1).
    pub cpu_code: Option<Vec<u8>>,
    /// The CPU steps, in order; empty without CPU sections.
    pub cpu_steps: Vec<CpuStep>,
}

fn bad(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

fn check(cond: bool, msg: &str) -> Result<()> {
    if cond { Ok(()) } else { Err(bad(msg)) }
}

fn u16_at(d: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([d[at], d[at + 1]])
}

fn u32_at(d: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(d[at..at + 4].try_into().unwrap())
}

/// The ELF header of CPU_CODE (docs/model-format.md 9).
fn check_elf(code: &[u8]) -> Result<()> {
    check(
        code.len() >= ELF_HEADER_SIZE && code.starts_with(b"\x7fELF"),
        "CPU_CODE is not an ELF file",
    )?;
    check(code[4] == 2 && code[5] == 1, "CPU_CODE is not 64-bit little-endian ELF")?;
    check(u16_at(code, 16) == ET_DYN, "CPU_CODE is not a shared object")?;
    check(u16_at(code, 18) == EM_AARCH64, "CPU_CODE is not for AArch64")
}

fn is_c_identifier(name: &[u8]) -> bool {
    name.first().is_some_and(|c| !c.is_ascii_digit())
        && name.iter().all(|&c| c.is_ascii_alphanumeric() || c == b'_')
}

/// The steps of CPU_STEPS (docs/model-format.md 10), whose copies must be
/// inside arena regions.
fn parse_steps(body: &[u8], regions: &[Region]) -> Result<Vec<CpuStep>> {
    check(body.len() >= STEPS_HEADER, "CPU_STEPS is smaller than its header")?;
    let count = u32_at(body, 0) as usize;
    check(body[4..STEPS_HEADER].iter().all(|&b| b == 0), "reserved CPU_STEPS bytes are not 0")?;
    check(count >= 1, "no CPU steps")?;
    let names_at = STEPS_HEADER + STEP_ENTRY * count;
    check(body.len() > names_at, "CPU_STEPS has no name table")?;
    let names = &body[names_at..];
    check(names.last() == Some(&0), "the name table does not end with NUL")?;
    let mut steps = Vec::new();
    for i in 0..count {
        let e = &body[STEPS_HEADER + STEP_ENTRY * i..][..STEP_ENTRY];
        let kind = match e[0] {
            0 => StepKind::CopyIn,
            1 => StepKind::Call,
            2 => StepKind::CopyOut,
            k => return Err(bad(format!("step {i}: bad kind {k}"))),
        };
        let (region, name, offset, size) =
            (u32::from(e[1]), u32_at(e, 4) as usize, u32_at(e, 8), u32_at(e, 12));
        check(u16_at(e, 2) == 0, &format!("step {i}: reserved field is not 0"))?;
        check(name < names.len(), &format!("step {i}: name outside the name table"))?;
        // The table ends with NUL, so every name does
        let len = names[name..].iter().position(|&b| b == 0).unwrap();
        let symbol = &names[name..name + len];
        check(
            is_c_identifier(symbol),
            &format!(
                "step {i}: name {:?} is not a C identifier",
                String::from_utf8_lossy(symbol)
            ),
        )?;
        if kind == StepKind::Call {
            check(
                region == 0 && offset == 0 && size == 0,
                &format!("step {i}: a call has region, offset and size 0"),
            )?;
        } else {
            let r = regions.iter().find(|r| r.index == region);
            check(
                r.is_some_and(|r| r.role == RegionRole::Arena),
                &format!("step {i}: not in an arena region"),
            )?;
            check(
                size >= 1 && u64::from(offset) + u64::from(size) <= u64::from(r.unwrap().size),
                &format!("step {i}: copy outside its region"),
            )?;
        }
        steps.push(CpuStep {
            kind,
            region,
            // ASCII, as a C identifier
            name: String::from_utf8(symbol.to_vec()).unwrap(),
            offset,
            size,
        });
    }
    Ok(steps)
}

impl Model {
    /// Parses model.knpu bytes and checks the rules of docs/model-format.md
    /// 11 that depend neither on the device nor on the loaded CPU code.
    pub fn parse(data: &[u8]) -> Result<Model> {
        check(data.len() > HEADER_SIZE, "the file is not larger than the header")?;
        check(&data[0..4] == MAGIC, "bad magic (not a model.knpu file)")?;
        let major = u16_at(data, 4);
        check(major == 0, &format!("unknown format_major {major}"))?;
        check(
            data[20..32].iter().all(|&b| b == 0),
            "reserved header bytes are not 0",
        )?;
        let sram_size = u32_at(data, 8);
        check(
            sram_size % 4096 == 0 && (4096..=16 << 20).contains(&sram_size),
            &format!("sram_size {sram_size} out of range"),
        )?;
        let nsec = u16_at(data, 14) as usize;
        let table = u32_at(data, 16) as usize;
        check(nsec >= 1, "no sections")?;
        check(table % 16 == 0, "the section table is not 16-byte aligned")?;
        check(
            table + TABLE_ENTRY * nsec <= data.len(),
            "the section table is outside the file",
        )?;

        // Section table and layout (format 2 and 4)
        let mut spans = vec![
            (0, HEADER_SIZE, "the header".to_string()),
            (table, table + TABLE_ENTRY * nsec, "the section table".to_string()),
        ];
        let mut sections: Vec<(u32, &[u8])> = Vec::new();
        for i in 0..nsec {
            let at = table + TABLE_ENTRY * i;
            let (kind, off, size) = (
                u32_at(data, at),
                u32_at(data, at + 4) as usize,
                u32_at(data, at + 8) as usize,
            );
            check(u32_at(data, at + 12) == 0, &format!("section {i}: reserved field is not 0"))?;
            check(
                (KIND_CMDS..=KIND_CPU_STEPS).contains(&kind),
                &format!("section {i}: unknown kind {kind}"),
            )?;
            check(off % 16 == 0, &format!("section {i}: offset not 16-byte aligned"))?;
            check(off + size <= data.len(), &format!("section {i}: outside the file"))?;
            spans.push((off, off + size, format!("section {i}")));
            sections.push((kind, &data[off..off + size]));
        }
        spans.sort();
        for w in spans.windows(2) {
            let ((_, end0, name0), (begin1, _, name1)) = (&w[0], &w[1]);
            check(*end0 <= *begin1, &format!("{name0} overlaps {name1}"))?;
            check(
                data[*end0..*begin1].iter().all(|&b| b == 0),
                &format!("the gap after {name0} is not 0"),
            )?;
        }
        check(
            spans.last().unwrap().1 == data.len(),
            "the file does not end at the last section",
        )?;
        let count = |k| sections.iter().filter(|(kind, _)| *kind == k).count();
        check(
            count(KIND_CMDS) == 1 && count(KIND_REGIONS) == 1 && count(KIND_TENSORS) == 1,
            "need exactly one CMDS, REGIONS and TENSORS section",
        )?;
        // CPU_CODE and CPU_STEPS come together (format 4), with minor 1
        // (format 11, rule 6)
        let cpu = count(KIND_CPU_CODE);
        check(
            cpu == count(KIND_CPU_STEPS) && cpu <= 1,
            "need both or none of CPU_CODE and CPU_STEPS, at most one each",
        )?;
        let format_minor = u16_at(data, 6);
        check(cpu == 0 || format_minor >= 1, "CPU sections need format_minor 1")?;
        let body = |k| sections.iter().find(|(kind, _)| *kind == k).unwrap().1;

        // CMDS (format 5)
        let cmds = body(KIND_CMDS);
        check(
            cmds.len() >= 4 && cmds.len() % 4 == 0,
            "CMDS size must be a positive multiple of 4",
        )?;
        check(u32_at(cmds, cmds.len() - 4) == OP_STOP, "CMDS must end with OP_STOP")?;

        // REGIONS (format 6)
        let table_body = body(KIND_REGIONS);
        check(table_body.len() % REGION_ENTRY == 0, "REGIONS size is not a multiple of 16")?;
        let mut regions = Vec::new();
        let mut used_const = Vec::new();
        for at in (0..table_body.len()).step_by(REGION_ENTRY) {
            let (index, role, size, content) = (
                u32_at(table_body, at),
                u32_at(table_body, at + 4),
                u32_at(table_body, at + 8),
                u32_at(table_body, at + 12),
            );
            check(index <= 7, &format!("region index {index} out of range"))?;
            if let Some(last) = regions.last() {
                let last: &Region = last;
                check(index > last.index, "regions are not in increasing index order")?;
            }
            check(size >= 1, &format!("region {index}: size 0"))?;
            match role {
                0 => {
                    let c = content as usize;
                    check(
                        c < nsec && sections[c].0 == KIND_CONST,
                        &format!("region {index}: content is not a CONST section"),
                    )?;
                    let blob = sections[c].1;
                    check(
                        blob.len() == size as usize,
                        &format!("region {index}: size differs from its CONST section"),
                    )?;
                    used_const.push(c);
                    regions.push(Region {
                        index,
                        role: RegionRole::Const,
                        size,
                        content: Some(blob.to_vec()),
                    });
                }
                1 => {
                    check(
                        content == NO_CONTENT,
                        &format!("region {index}: an arena has no content"),
                    )?;
                    regions.push(Region { index, role: RegionRole::Arena, size, content: None });
                }
                _ => return Err(bad(format!("region {index}: bad role {role}"))),
            }
        }
        used_const.sort();
        let const_sections: Vec<usize> =
            (0..nsec).filter(|&i| sections[i].0 == KIND_CONST).collect();
        check(
            used_const == const_sections,
            "every CONST section must belong to exactly one region",
        )?;

        // TENSORS (format 8)
        let tbody = body(KIND_TENSORS);
        check(tbody.len() % TENSOR_ENTRY == 0, "TENSORS size is not a multiple of 48")?;
        let mut tensors = Vec::new();
        let mut seen_output = false;
        for at in (0..tbody.len()).step_by(TENSOR_ENTRY) {
            let e = &tbody[at..at + TENSOR_ENTRY];
            let (direction, ttype, rank) = (e[0], e[1], e[2] as usize);
            check(
                e[3] == 0 && e[40..48].iter().all(|&b| b == 0),
                "reserved tensor bytes are not 0",
            )?;
            check(direction <= 1, "bad tensor direction")?;
            check(direction == 1 || !seen_output, "inputs must come before outputs")?;
            seen_output |= direction == 1;
            check(ttype == 0, "the tensor type must be int8")?;
            check((1..=4).contains(&rank), "tensor rank out of range")?;
            let dims: Vec<u32> = (0..4).map(|i| u32_at(e, 16 + 4 * i)).collect();
            check(
                dims[..rank].iter().all(|&d| d >= 1) && dims[rank..].iter().all(|&d| d == 0),
                "bad tensor shape",
            )?;
            let (region, offset, size) = (u32_at(e, 4), u32_at(e, 8), u32_at(e, 12));
            let n = dims[..rank].iter().try_fold(1u64, |a, &d| a.checked_mul(d as u64));
            check(n == Some(size as u64), "the tensor size differs from its shape")?;
            let scale = f32::from_le_bytes(e[32..36].try_into().unwrap());
            let zero_point = i32::from_le_bytes(e[36..40].try_into().unwrap());
            check(scale > 0.0 && scale.is_finite(), "bad tensor scale")?;
            check((-128..=127).contains(&zero_point), "tensor zero point out of range")?;
            let r = regions.iter().find(|r| r.index == region);
            check(
                r.is_some_and(|r| r.role == RegionRole::Arena),
                "the tensor is not in an arena region",
            )?;
            check(
                offset as u64 + size as u64 <= r.unwrap().size as u64,
                "the tensor is outside its region",
            )?;
            tensors.push(Tensor {
                direction: if direction == 0 { Direction::Input } else { Direction::Output },
                region,
                offset,
                size,
                shape: dims[..rank].to_vec(),
                scale,
                zero_point,
            });
        }
        for a in tensors.iter().filter(|t| t.direction == Direction::Input) {
            for b in tensors.iter().filter(|t| t.direction == Direction::Output) {
                check(
                    a.region != b.region
                        || a.offset + a.size <= b.offset
                        || b.offset + b.size <= a.offset,
                    "an input overlaps an output",
                )?;
            }
        }

        // CPU_CODE and CPU_STEPS (format 9 and 10)
        let (cpu_code, cpu_steps) = if cpu == 1 {
            let code = body(KIND_CPU_CODE);
            check_elf(code)?;
            (Some(code.to_vec()), parse_steps(body(KIND_CPU_STEPS), &regions)?)
        } else {
            (None, Vec::new())
        };

        Ok(Model {
            format_minor,
            sram_size,
            spec_major: u16_at(data, 12),
            cmds: cmds.to_vec(),
            regions,
            tensors,
            cpu_code,
            cpu_steps,
        })
    }

    /// The checks of docs/model-format.md 11 that need the device: the
    /// specification version, the SRAM size and the number of regions.
    pub fn check_device(&self, info: &DeviceInfo) -> Result<()> {
        if u32::from(self.spec_major) != info.version_major {
            return Err(Error::Mismatch(format!(
                "the model is for NPU specification {}, the device implements {}",
                self.spec_major, info.version_major
            )));
        }
        if self.sram_size > info.sram_size {
            return Err(Error::Mismatch(format!(
                "the model needs {} bytes of SRAM, the device has {}",
                self.sram_size, info.sram_size
            )));
        }
        if let Some(r) = self.regions.iter().find(|r| r.index >= info.num_regions) {
            return Err(Error::Mismatch(format!(
                "the model uses region {}, the device has {}",
                r.index, info.num_regions
            )));
        }
        Ok(())
    }

    /// The one input the runtime supports (v0 compilers write one).
    pub fn input(&self) -> Result<&Tensor> {
        self.single(Direction::Input)
    }

    /// The one output the runtime supports.
    pub fn output(&self) -> Result<&Tensor> {
        self.single(Direction::Output)
    }

    fn single(&self, dir: Direction) -> Result<&Tensor> {
        let mut it = self.tensors.iter().filter(|t| t.direction == dir);
        match (it.next(), it.next()) {
            (Some(t), None) => Ok(t),
            _ => Err(Error::Mismatch(format!(
                "the runtime supports models with one input and one output ({dir:?})"
            ))),
        }
    }

    /// Number of OP commands that PERF_OPS counts (OP_CONV, OP_ADD,
    /// OP_AVGPOOL, OP_DMA), walking over payload words.
    pub fn op_count(&self) -> usize {
        let words: Vec<u32> = self
            .cmds
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        let mut n = 0;
        let mut i = 0;
        while i < words.len() {
            let cmd = words[i] & CMD_MASK;
            if OPS.contains(&cmd) {
                n += 1;
            }
            i += if cmd & CMD_PAYLOAD != 0 { 2 } else { 1 };
        }
        n
    }
}
