# libdeflate-zran

Safe Rust wrapper around a patched
[`libdeflate`](https://github.com/ebiggers/libdeflate) C library
with **zran-style resume and walk** entrypoints for random access
into large gzip / DEFLATE streams.

Built on
[`libdeflate-zran-sys`](https://github.com/maxsimov/libdeflate-zran-sys)
(the raw FFI half of the pair).

## What's different from upstream libdeflate

Upstream libdeflate is one-shot, no streaming, no preset dictionary,
no resume primitive. This crate exposes a patched libdeflate fork
that adds the minimum sufficient API for zran-style random access:

- `Decompressor::decompress_walk(stop_at_block_end: true)` — stops
  at each non-final DEFLATE end-of-block boundary, returning a
  `Checkpoint { bitbuf, bitsleft, window }` value.
- `Decompressor::decompress_resume(&checkpoint)` — resumes from
  the captured state plus the 32 KiB sliding window.

The default decompression path is byte-identical to upstream — the
new entrypoints are opt-in and the inner SIMD hot loop is unchanged.

See the
[fork](https://github.com/maxsimov/libdeflate/tree/zran-resume-and-walk)
for the C-level patches.

## License

MIT.
