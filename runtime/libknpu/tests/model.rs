// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Model::parse on the committed test models (models/conv/vectors/), on
//! copies broken one rule at a time (docs/model-format.md 11), and on
//! format minor 1 files made from them by adding CPU sections.

use std::path::PathBuf;

use knpu::{CpuStep, DeviceInfo, Direction, Error, Model, RegionRole, StepKind};

fn vector(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/conv/vectors")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn u32_at(d: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(d[at..at + 4].try_into().unwrap())
}

fn put32(d: &mut [u8], at: usize, v: u32) {
    d[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// Offset of the section of a kind, from the section table.
fn section(d: &[u8], kind: u32) -> (usize, usize) {
    let (n, table) = (u16::from_le_bytes([d[14], d[15]]) as usize, u32_at(d, 16) as usize);
    (0..n)
        .map(|i| table + 16 * i)
        .find(|&at| u32_at(d, at) == kind)
        .map(|at| (u32_at(d, at + 4) as usize, u32_at(d, at + 8) as usize))
        .unwrap()
}

/// The sections of a file in table order, as (kind, contents).
fn sections(d: &[u8]) -> Vec<(u32, Vec<u8>)> {
    let (n, table) = (u16::from_le_bytes([d[14], d[15]]) as usize, u32_at(d, 16) as usize);
    (0..n)
        .map(|i| table + 16 * i)
        .map(|at| {
            let (off, size) = (u32_at(d, at + 4) as usize, u32_at(d, at + 8) as usize);
            (u32_at(d, at), d[off..off + size].to_vec())
        })
        .collect()
}

/// A file with the header of `like`, format_minor `minor` and these
/// sections: the table after the header, then the sections, 16-byte
/// aligned.
fn build(like: &[u8], minor: u16, sections: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut d = like[..32].to_vec();
    d[6..8].copy_from_slice(&minor.to_le_bytes());
    d[14..16].copy_from_slice(&(sections.len() as u16).to_le_bytes());
    put32(&mut d, 16, 32);
    d.resize(32 + 16 * sections.len(), 0);
    for (i, (kind, body)) in sections.iter().enumerate() {
        d.resize(d.len().next_multiple_of(16), 0);
        let (at, off) = (32 + 16 * i, d.len() as u32);
        put32(&mut d, at, *kind);
        put32(&mut d, at + 4, off);
        put32(&mut d, at + 8, body.len() as u32);
        d.extend_from_slice(body);
    }
    d
}

/// A CPU_CODE that Model::parse takes, which looks at the ELF header only:
/// 64-bit, little-endian, ET_DYN, EM_AARCH64.
fn elf_header() -> Vec<u8> {
    let mut e = vec![0; 64];
    e[..4].copy_from_slice(b"\x7fELF");
    e[4..7].copy_from_slice(&[2, 1, 1]); // ELFCLASS64, ELFDATA2LSB, EV_CURRENT
    e[16..18].copy_from_slice(&3u16.to_le_bytes()); // ET_DYN
    e[18..20].copy_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
    e
}

/// CPU steps as (kind, region, name, offset, size).
type Steps<'a> = [(u8, u8, &'a str, u32, u32)];

/// The steps of the MNIST model's softmax layer on the conv model's arena
/// (region 1, 4704 bytes): ten bytes at 0 in, the result to 1568. In
/// CPU_STEPS the entries are at 16, 32 and 48 and the names at 64.
const STEPS: [(u8, u8, &str, u32, u32); 3] =
    [(0, 1, "act7", 0, 10), (1, 0, "layer8", 0, 0), (2, 1, "act8", 1568, 10)];

/// A CPU_STEPS section: the header, the entries and the name table.
fn step_table(steps: &Steps) -> Vec<u8> {
    let mut d = vec![0; 16];
    put32(&mut d, 0, steps.len() as u32);
    let mut names = Vec::new();
    for &(kind, region, name, offset, size) in steps {
        d.extend_from_slice(&[kind, region, 0, 0]);
        d.extend_from_slice(&(names.len() as u32).to_le_bytes());
        d.extend_from_slice(&offset.to_le_bytes());
        d.extend_from_slice(&size.to_le_bytes());
        names.extend_from_slice(name.as_bytes());
        names.push(0);
    }
    d.extend_from_slice(&names);
    d
}

/// The conv test model with CPU sections, as format minor 1.
fn with_cpu(code: Vec<u8>, steps: Vec<u8>) -> Vec<u8> {
    let conv = vector("conv-double-64k.knpu");
    let mut s = sections(&conv);
    s.extend([(5, code), (6, steps)]);
    build(&conv, 1, &s)
}

fn info(sram_size: u32) -> DeviceInfo {
    DeviceInfo {
        version_major: 0,
        version_minor: 0,
        num_regions: 8,
        round_double: true,
        round_single: true,
        sram_size,
    }
}

#[test]
fn parses_the_test_models() {
    for (name, sram, ops) in [
        ("conv-double-64k.knpu", 65536, 4),
        ("conv-single-64k.knpu", 65536, 4),
        ("conv-double-4k.knpu", 4096, 16),
        ("conv-single-4k.knpu", 4096, 16),
    ] {
        let m = Model::parse(&vector(name)).unwrap();
        assert_eq!(m.sram_size, sram, "{name}");
        assert_eq!(m.spec_major, 0);
        assert_eq!(m.op_count(), ops, "{name}: DMA and CONV commands");
        assert_eq!(m.regions.len(), 2);
        assert_eq!(m.regions[0].role, RegionRole::Const);
        assert_eq!(m.regions[0].size, 1344);
        assert_eq!(m.regions[1].role, RegionRole::Arena);
        let (i, o) = (m.input().unwrap(), m.output().unwrap());
        assert_eq!((i.direction, o.direction), (Direction::Input, Direction::Output));
        assert_eq!(i.shape, [1, 14, 14, 8]);
        assert_eq!(o.shape, [1, 14, 14, 16]);
        assert_eq!((i.size, o.size), (1568, 3136));
        assert_eq!((i.zero_point, o.zero_point), (-128, -128));
        m.check_device(&info(65536)).unwrap();
    }
}

#[test]
fn checks_the_device() {
    let m = Model::parse(&vector("conv-double-64k.knpu")).unwrap();
    assert!(matches!(m.check_device(&info(32768)), Err(Error::Mismatch(_))));
    let mut newer = info(65536);
    newer.version_major = 1;
    assert!(matches!(m.check_device(&newer), Err(Error::Mismatch(_))));
    let mut few = info(65536);
    few.num_regions = 1;
    assert!(matches!(m.check_device(&few), Err(Error::Mismatch(_))));
    // A model for a smaller SRAM runs on a larger one
    let small = Model::parse(&vector("conv-double-4k.knpu")).unwrap();
    small.check_device(&info(65536)).unwrap();
}

#[test]
fn rejects_broken_files() {
    let good = vector("conv-double-64k.knpu");
    let (cmds, cmds_len) = section(&good, 1);
    let (regions, _) = section(&good, 2);
    let (tensors, _) = section(&good, 4);
    type Edit = Box<dyn Fn(&mut Vec<u8>)>;
    let cases: Vec<(&str, Edit)> = vec![
        ("magic", Box::new(|d| d[0] = b'X')),
        ("format_major", Box::new(|d| d[4] = 1)),
        ("reserved header bytes", Box::new(|d| d[24] = 1)),
        ("sram_size", Box::new(|d| put32(d, 8, 4097))),
        ("truncated", Box::new(|d| d.truncate(100))),
        ("trailing", Box::new(|d| d.extend_from_slice(&[0; 16]))),
        ("misaligned section", Box::new(|d| {
            let off = u32_at(d, 36) + 4;
            put32(d, 36, off)
        })),
        ("unknown kind", Box::new(|d| put32(d, 32, 9))),
        ("no OP_STOP", Box::new(move |d| put32(d, cmds + cmds_len - 4, 0x0001))),
        ("region size", Box::new(move |d| put32(d, regions + 8, 1000))),
        ("arena content", Box::new(move |d| put32(d, regions + 16 + 12, 0))),
        ("tensor outside", Box::new(move |d| put32(d, tensors + 8, 4000))),
        ("tensor type", Box::new(move |d| d[tensors + 1] = 1)),
        ("tensor size", Box::new(move |d| put32(d, tensors + 12, 1569))),
        ("zero point", Box::new(move |d| put32(d, tensors + 36, 200))),
        ("scale", Box::new(move |d| put32(d, tensors + 32, 0))),
        ("output before input", Box::new(move |d| {
            d[tensors] = 1;
            d[tensors + 48] = 0;
        })),
    ];
    Model::parse(&good).unwrap();
    for (what, edit) in cases {
        let mut d = good.clone();
        edit(&mut d);
        match Model::parse(&d) {
            Err(Error::Model(msg)) => println!("{what}: {msg}"),
            other => panic!("{what}: accepted or wrong error: {other:?}"),
        }
    }
}

#[test]
fn parses_cpu_sections() {
    let m = Model::parse(&with_cpu(elf_header(), step_table(&STEPS))).unwrap();
    assert_eq!(m.format_minor, 1);
    assert_eq!(m.cpu_code, Some(elf_header()));
    let step = |kind, region, name: &str, offset, size| CpuStep {
        kind,
        region,
        name: name.to_string(),
        offset,
        size,
    };
    assert_eq!(
        m.cpu_steps,
        [
            step(StepKind::CopyIn, 1, "act7", 0, 10),
            step(StepKind::Call, 0, "layer8", 0, 0),
            step(StepKind::CopyOut, 1, "act8", 1568, 10),
        ]
    );
    // A copy may end at the end of its region
    let last = [(0, 1, "_x9", 4694, 10), (1, 0, "f", 0, 0)];
    Model::parse(&with_cpu(elf_header(), step_table(&last))).unwrap();

    // The test models are minor 0 without CPU sections, and a later minor
    // without new sections still reads (format 13)
    let conv = Model::parse(&vector("conv-double-64k.knpu")).unwrap();
    assert_eq!((conv.format_minor, conv.cpu_code, conv.cpu_steps.len()), (0, None, 0));
    let mut later = vector("conv-double-64k.knpu");
    later[6] = 1;
    assert_eq!(Model::parse(&later).unwrap().format_minor, 1);
}

#[test]
fn rejects_broken_cpu_sections() {
    let conv = vector("conv-double-64k.knpu");
    let elf = |at: usize, bytes: &[u8]| {
        let mut e = elf_header();
        e[at..at + bytes.len()].copy_from_slice(bytes);
        e
    };
    let steps = |edit: &dyn Fn(&mut Vec<u8>)| {
        let mut s = step_table(&STEPS);
        edit(&mut s);
        with_cpu(elf_header(), s)
    };
    let named = |name: &str| with_cpu(elf_header(), step_table(&[(0, 1, name, 0, 10)]));
    let added = |extra: &[(u32, Vec<u8>)]| {
        let mut s = sections(&conv);
        s.extend_from_slice(extra);
        build(&conv, 1, &s)
    };
    let (code, table) = (elf_header(), step_table(&STEPS));
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("minor 0", {
            let mut d = with_cpu(elf_header(), step_table(&STEPS));
            d[6] = 0;
            d
        }),
        ("CPU_CODE alone", added(&[(5, code.clone())])),
        ("CPU_STEPS alone", added(&[(6, table.clone())])),
        ("two of each", added(&[(5, code.clone()), (6, table.clone()), (5, code), (6, table)])),
        ("not ELF", with_cpu(elf(1, b"ELG"), step_table(&STEPS))),
        ("32-bit", with_cpu(elf(4, &[1]), step_table(&STEPS))),
        ("big-endian", with_cpu(elf(5, &[2]), step_table(&STEPS))),
        ("ET_EXEC", with_cpu(elf(16, &[2]), step_table(&STEPS))),
        ("x86_64", with_cpu(elf(18, &[62]), step_table(&STEPS))),
        ("short ELF header", with_cpu(elf_header()[..63].to_vec(), step_table(&STEPS))),
        ("short CPU_STEPS", with_cpu(elf_header(), step_table(&STEPS)[..15].to_vec())),
        ("no steps", with_cpu(elf_header(), step_table(&[]))),
        ("reserved CPU_STEPS bytes", steps(&|s| s[15] = 1)),
        ("steps past the end", steps(&|s| put32(s, 0, 1000))),
        ("no name table", steps(&|s| s.truncate(64))),
        ("names without NUL", steps(&|s| *s.last_mut().unwrap() = b'x')),
        ("kind 3", steps(&|s| s[16] = 3)),
        ("reserved step field", steps(&|s| s[19] = 1)),
        ("name outside the table", steps(&|s| put32(s, 20, 17))),
        ("empty name", steps(&|s| put32(s, 20, 4))),
        ("name starting with a digit", named("7act")),
        ("name with a dash", named("act-7")),
        ("call with a region", steps(&|s| s[33] = 1)),
        ("call with an offset", steps(&|s| put32(s, 40, 4))),
        ("call with a size", steps(&|s| put32(s, 44, 10))),
        ("no such region", steps(&|s| s[17] = 5)),
        ("constant region", steps(&|s| s[17] = 0)),
        ("copy of 0 bytes", steps(&|s| put32(s, 28, 0))),
        ("copy past the region", steps(&|s| put32(s, 56, 4695))),
        ("copy past 4 GiB", steps(&|s| put32(s, 56, u32::MAX - 4))),
    ];
    Model::parse(&with_cpu(elf_header(), step_table(&STEPS))).unwrap();
    for (what, d) in cases {
        match Model::parse(&d) {
            Err(Error::Model(msg)) => println!("{what}: {msg}"),
            other => panic!("{what}: accepted or wrong error: {other:?}"),
        }
    }
}
