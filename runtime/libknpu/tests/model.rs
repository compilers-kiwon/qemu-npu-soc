// SPDX-FileCopyrightText: 2026 Kiwon Kwon
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! Model::parse on the committed test models (models/conv/vectors/) and on
//! copies broken one rule at a time (docs/model-format.md 9).

use std::path::PathBuf;

use knpu::{DeviceInfo, Direction, Error, Model, RegionRole};

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
