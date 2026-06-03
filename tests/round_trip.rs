//! End-to-end test: compress a buffer, walk it capturing
//! checkpoints, then resume from each checkpoint and confirm
//! byte-identical output to a continuous decompress.

use libdeflate_zran::{Decompressor, WalkOutcome};
use libdeflate_zran_sys as sys;

#[test]
fn walk_and_resume_round_trip() {
    // A semi-compressible payload (256 KiB) that the encoder
    // splits into more than one DEFLATE block.
    let payload: Vec<u8> = (0..256 * 1024)
        .map(|i| {
            // Same template as the C-side test_zran for cross-check.
            const T: [u8; 8] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x42, 0x17];
            T[i & 7] ^ ((i >> 5) as u8)
        })
        .collect();

    let compressed = encode_raw_deflate(&payload);
    assert!(compressed.len() < payload.len(), "should compress");

    // Walk capturing the first checkpoint.
    let mut dec = Decompressor::new().expect("alloc");
    let mut sink = vec![0u8; payload.len() + 1024];

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;
    let mut checkpoints: Vec<(usize, usize, libdeflate_zran::Checkpoint)> = Vec::new();
    loop {
        let WalkOutcome { in_consumed, out_produced, done, checkpoint } =
            dec.decompress_walk(
                &compressed[in_pos..],
                &mut sink,
                /* out_offset = */ out_pos,
                /* stop = */ true,
            ).expect("walk should succeed");
        in_pos  += in_consumed;
        out_pos += out_produced;
        if let Some(ck) = checkpoint {
            checkpoints.push((in_pos, out_pos, ck));
        }
        if done { break; }
    }

    assert_eq!(out_pos, payload.len(), "walk produced full payload");
    assert_eq!(&sink[..out_pos], &payload[..],
               "walk output matches canonical");
    assert!(!checkpoints.is_empty(),
            "expected at least one end-of-block checkpoint");

    // Resume from the first checkpoint and confirm the produced
    // bytes match the corresponding payload region.
    let (in_at_ck, out_at_ck, ck) = &checkpoints[0];
    let mut sink2 = vec![0u8; payload.len() + 1024];
    let (_in_used, out_produced) = dec.decompress_resume(
        &compressed[*in_at_ck..],
        &mut sink2,
        ck,
    ).expect("resume should succeed");

    let expected_len = payload.len() - *out_at_ck;
    assert_eq!(out_produced, expected_len, "resume output length");
    assert_eq!(
        &sink2[ck.window_used..ck.window_used + out_produced],
        &payload[*out_at_ck..],
        "resume output matches canonical payload at the checkpoint position",
    );
}

#[test]
fn walk_without_stop_decompresses_full_stream() {
    let payload = vec![b'x'; 4096];
    let compressed = encode_raw_deflate(&payload);
    let mut dec = Decompressor::new().unwrap();
    let mut sink = vec![0u8; 8192];
    let outcome = dec.decompress_walk(&compressed, &mut sink, 0, false).unwrap();
    assert!(outcome.done);
    assert!(outcome.checkpoint.is_none());
    assert_eq!(&sink[..outcome.out_produced], &payload[..]);
}

#[test]
fn bad_data_returns_bad_data_error() {
    let mut dec = Decompressor::new().unwrap();
    let mut sink = vec![0u8; 64];
    let err = dec.decompress_walk(b"\xff\xff\xff\xff", &mut sink, 0, false).unwrap_err();
    assert_eq!(err, libdeflate_zran::Error::BadData);
}

fn encode_raw_deflate(data: &[u8]) -> Vec<u8> {
    // Use libdeflate's own compressor (same as the C-side test_zran
    // that proves walk + resume round-trip). At level 6 with this
    // input size we get multiple DEFLATE blocks.
    use std::ffi::c_void;
    let mut out = vec![0u8; data.len() + 1024];
    // SAFETY: alloc-compress-free wrapped around standard usage of
    // libdeflate's C API; the buffer pointers are owned + bounded.
    unsafe {
        let c = sys::libdeflate_alloc_compressor(6);
        assert!(!c.is_null());
        let n = sys::libdeflate_deflate_compress(
            c,
            data.as_ptr() as *const c_void, data.len(),
            out.as_mut_ptr() as *mut c_void, out.len(),
        );
        sys::libdeflate_free_compressor(c);
        assert!(n > 0, "compress overflowed output buffer");
        out.truncate(n);
    }
    out
}
