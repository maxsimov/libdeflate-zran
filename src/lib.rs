//! Safe Rust API around a patched `libdeflate` with zran-style
//! resume and walk entrypoints.
//!
//! Use [`Decompressor`] to allocate a libdeflate decompressor; call
//! [`Decompressor::decompress_walk`] to drive index-build (stopping
//! at end-of-block boundaries to capture [`Checkpoint`]s); call
//! [`Decompressor::decompress_resume`] to start random-access reads
//! from a captured checkpoint.
//!
//! ```no_run
//! use libdeflate_zran::Decompressor;
//!
//! let mut dec = Decompressor::new()?;
//! let compressed: &[u8] = &[];     // raw DEFLATE bytes
//! let mut sink = vec![0u8; 1 << 20];
//! let mut out_pos = 0usize;
//!
//! // Walk capturing checkpoints at every end-of-block, threading
//! // the growing buffer's write offset.
//! let outcome = dec.decompress_walk(compressed, &mut sink, out_pos, true)?;
//! out_pos += outcome.out_produced;
//! if let Some(ck) = outcome.checkpoint {
//!     // Persist ck.bitbuf / ck.bitsleft / ck.window as part of
//!     // your zran index entry.
//! }
//!
//! // Later, given a target uncompressed byte position:
//! // let (consumed, produced) = dec.decompress_resume(
//! //     &compressed[ckpt_in..], &mut buf, &ck)?;
//! # Ok::<(), libdeflate_zran::Error>(())
//! ```

#![deny(missing_docs)]

use std::os::raw::{c_int, c_void};

use libdeflate_zran_sys as sys;

/// Maximum size of the DEFLATE sliding window: 32 KiB. This is also
/// the maximum size of a [`Checkpoint`]'s dictionary.
pub const WINDOW_SIZE: usize = 32 * 1024;

/// Result type used by every fallible method on [`Decompressor`].
pub type Result<T> = std::result::Result<T, Error>;

/// Errors surfaced by the decompressor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The compressed input is corrupt or not a valid DEFLATE stream.
    BadData,
    /// The output buffer wasn't big enough — either for the
    /// requested decompression itself or for the resume path's
    /// dictionary prefix.
    InsufficientSpace,
    /// The decompressed output was shorter than the caller required.
    /// Not raised by the `_walk` / `_resume` shapes; kept for
    /// symmetry with upstream libdeflate.
    ShortOutput,
    /// Out of memory (libdeflate's allocator returned NULL).
    OutOfMemory,
    /// A precondition was violated.
    InvalidInput(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadData             => f.write_str("libdeflate: bad data"),
            Self::InsufficientSpace   => f.write_str("libdeflate: insufficient output space"),
            Self::ShortOutput         => f.write_str("libdeflate: short output"),
            Self::OutOfMemory         => f.write_str("libdeflate: out of memory"),
            Self::InvalidInput(s)     => write!(f, "libdeflate: invalid input: {s}"),
        }
    }
}

impl std::error::Error for Error {}

/// Captured state at a DEFLATE end-of-block boundary. Sufficient to
/// resume decompression from this position.
///
/// The window is a fixed [`WINDOW_SIZE`]-byte array. For checkpoints
/// near the start of a stream where fewer than [`WINDOW_SIZE`]
/// bytes have been decoded, the unused prefix is zero-filled and
/// [`Checkpoint::window_used`] records the actual count.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Live bit-buffer value at the boundary. Opaque to callers —
    /// just feed back to [`Decompressor::decompress_resume`].
    pub bitbuf:   u64,
    /// Count of valid bits in `bitbuf` (0..=7).
    pub bitsleft: u32,
    /// 32 KiB sliding window: the last [`WINDOW_SIZE`] bytes of
    /// uncompressed output that preceded this checkpoint. Valid
    /// bytes occupy the END of this buffer; the prefix is
    /// zero-filled when fewer bytes are available.
    pub window:      Box<[u8; WINDOW_SIZE]>,
    /// Number of valid bytes at the END of `window`.
    pub window_used: usize,
}

/// Returned by [`Decompressor::decompress_walk`].
#[derive(Debug, Clone)]
pub struct WalkOutcome {
    /// Bytes consumed from the input slice.
    pub in_consumed:  usize,
    /// Bytes written to the output slice.
    pub out_produced: usize,
    /// `true` if the decoder reached the final block of the stream.
    pub done:         bool,
    /// `Some` if the decoder stopped at a non-final end-of-block
    /// boundary because `stop_at_block_end` was `true`. Use this
    /// for zran index build.
    pub checkpoint:   Option<Checkpoint>,
}

/// Owned libdeflate decompressor. [`Drop`] frees the underlying C
/// allocation.
///
/// Stateful across [`Decompressor::decompress_walk`] calls: a walk
/// that returns at an end-of-block boundary leaves leftover bits
/// in the decompressor; the next walk call automatically picks
/// them up. Use [`Decompressor::reset`] to clear that state when
/// starting a fresh stream (or, equivalently, [`Decompressor::new`]
/// to allocate a fresh one).
pub struct Decompressor {
    raw: *mut sys::libdeflate_decompressor,
    /// Leftover bits from the previous decompress_walk return at a
    /// non-final block boundary. Seeded into the next walk call so
    /// the loop continues correctly.
    pending_bitbuf:   u64,
    pending_bitsleft: u32,
}

// SAFETY: libdeflate documents that a decompressor carries no
// thread-local state and is "not safe to use by multiple threads
// concurrently" — so Send is correct, Sync is not.
unsafe impl Send for Decompressor {}

impl Decompressor {
    /// Allocate a fresh decompressor.
    ///
    /// # Errors
    /// Returns [`Error::OutOfMemory`] if the underlying allocator
    /// returns NULL.
    pub fn new() -> Result<Self> {
        // SAFETY: the C function takes no arguments and returns
        // NULL on OOM. We check before constructing the wrapper.
        let raw = unsafe { sys::libdeflate_alloc_decompressor() };
        if raw.is_null() {
            return Err(Error::OutOfMemory);
        }
        Ok(Self { raw, pending_bitbuf: 0, pending_bitsleft: 0 })
    }

    /// Clear any pending bit-buffer state from a prior
    /// [`Decompressor::decompress_walk`] call. Useful when reusing
    /// the same decompressor across unrelated streams.
    pub fn reset(&mut self) {
        self.pending_bitbuf   = 0;
        self.pending_bitsleft = 0;
    }


    /// Walk a raw-DEFLATE stream.
    ///
    /// With `stop_at_block_end = true`, the call returns when an
    /// end-of-block boundary is reached (`WalkOutcome::checkpoint`
    /// is `Some`) or when the stream's final block has been
    /// decoded (`WalkOutcome::done = true`). With
    /// `stop_at_block_end = false`, the call decompresses to
    /// end-of-stream or runs out of output space.
    ///
    /// # Errors
    /// - [`Error::BadData`] for corrupt input.
    /// - [`Error::InsufficientSpace`] if the output buffer isn't
    ///   large enough.
    /// Walk a raw-DEFLATE stream into the **growing output buffer**
    /// `output`. Decoded bytes are appended starting at
    /// `output[out_offset..]`. Back-references reach into
    /// `output[..out_offset]` (the prior output), so a single
    /// growing buffer across many walk calls forms one contiguous
    /// decompression session.
    ///
    /// With `stop_at_block_end = true` the call returns when an
    /// end-of-block boundary is reached (with a captured
    /// [`Checkpoint`]) or when the stream's final block has been
    /// decoded.
    pub fn decompress_walk(
        &mut self,
        input:      &[u8],
        output:     &mut [u8],
        out_offset: usize,
        stop_at_block_end: bool,
    ) -> Result<WalkOutcome> {
        if out_offset > output.len() {
            return Err(Error::InsufficientSpace);
        }
        let mut bitbuf:        u64   = 0;
        let mut bitsleft:      u32   = 0;
        let mut in_consumed:   usize = 0;
        let mut out_produced:  usize = 0;

        // SAFETY: slice pointers are valid for the duration of the
        // call; the C function honours the explicit (in_nbytes,
        // out_nbytes_avail) bounds; the four out-pointers are stack
        // locals we own. pending_bitbuf / pending_bitsleft are seeded
        // from the prior walk call so consecutive calls form a
        // coherent decompression session.
        let result = unsafe {
            sys::libdeflate_deflate_decompress_walk(
                self.raw,
                input.as_ptr() as *const c_void, input.len(),
                output.as_mut_ptr() as *mut c_void, output.len(),
                self.pending_bitbuf,
                self.pending_bitsleft,
                out_offset,
                if stop_at_block_end { 1 } else { 0 } as c_int,
                &mut bitbuf  as *mut u64,
                &mut bitsleft as *mut u32,
                &mut in_consumed  as *mut usize,
                &mut out_produced as *mut usize,
            )
        };

        match result {
            sys::libdeflate_result::LIBDEFLATE_SUCCESS => {
                self.reset();
                Ok(WalkOutcome {
                    in_consumed, out_produced, done: true, checkpoint: None,
                })
            }
            sys::libdeflate_result::LIBDEFLATE_BLOCK_END => {
                self.pending_bitbuf   = bitbuf;
                self.pending_bitsleft = bitsleft;
                let total_output_end = out_offset + out_produced;
                Ok(WalkOutcome {
                    in_consumed,
                    out_produced,
                    done: false,
                    checkpoint: Some(snapshot_window(
                        &output[..total_output_end], bitbuf, bitsleft,
                    )),
                })
            }
            sys::libdeflate_result::LIBDEFLATE_BAD_DATA           => Err(Error::BadData),
            sys::libdeflate_result::LIBDEFLATE_INSUFFICIENT_SPACE => Err(Error::InsufficientSpace),
            sys::libdeflate_result::LIBDEFLATE_SHORT_OUTPUT       => Err(Error::ShortOutput),
        }
    }

    /// Resume decompression from a captured [`Checkpoint`].
    ///
    /// The first `checkpoint.window_used` bytes of `output` are
    /// overwritten with a copy of the dictionary window (this is
    /// libdeflate's mechanism for priming the back-reference
    /// space). The actual decompressed bytes start at
    /// `output[checkpoint.window_used..]` and the returned tuple's
    /// `out_produced` counts only those.
    ///
    /// # Errors
    /// - [`Error::BadData`] for corrupt input.
    /// - [`Error::InsufficientSpace`] if the output buffer is
    ///   smaller than `checkpoint.window_used`.
    /// - [`Error::InvalidInput`] if `checkpoint.window_used`
    ///   exceeds [`WINDOW_SIZE`].
    pub fn decompress_resume(
        &mut self,
        input:      &[u8],
        output:     &mut [u8],
        checkpoint: &Checkpoint,
    ) -> Result<(usize, usize)> {
        if checkpoint.window_used > WINDOW_SIZE {
            return Err(Error::InvalidInput("checkpoint.window_used > WINDOW_SIZE"));
        }
        if output.len() < checkpoint.window_used {
            return Err(Error::InsufficientSpace);
        }
        // Resume starts a fresh decompression session — clear any
        // pending walk state.
        self.reset();

        let dict_slice =
            &checkpoint.window[WINDOW_SIZE - checkpoint.window_used..];

        let mut in_consumed:  usize = 0;
        let mut out_produced: usize = 0;

        // SAFETY: same shape as decompress_walk; dict_slice is
        // borrowed from `checkpoint` for the duration of the call.
        let result = unsafe {
            sys::libdeflate_deflate_decompress_resume(
                self.raw,
                input.as_ptr() as *const c_void, input.len(),
                output.as_mut_ptr() as *mut c_void, output.len(),
                checkpoint.bitbuf,
                checkpoint.bitsleft,
                dict_slice.as_ptr() as *const c_void,
                dict_slice.len(),
                &mut in_consumed  as *mut usize,
                &mut out_produced as *mut usize,
            )
        };

        match result {
            sys::libdeflate_result::LIBDEFLATE_SUCCESS            => Ok((in_consumed, out_produced)),
            sys::libdeflate_result::LIBDEFLATE_BAD_DATA           => Err(Error::BadData),
            sys::libdeflate_result::LIBDEFLATE_INSUFFICIENT_SPACE => Err(Error::InsufficientSpace),
            sys::libdeflate_result::LIBDEFLATE_SHORT_OUTPUT       => Err(Error::ShortOutput),
            sys::libdeflate_result::LIBDEFLATE_BLOCK_END          =>
                Err(Error::InvalidInput("unexpected BLOCK_END from resume path")),
        }
    }
}

impl Drop for Decompressor {
    fn drop(&mut self) {
        // SAFETY: self.raw was returned by libdeflate_alloc_decompressor
        // (non-null guaranteed by `new()`); pairs 1:1 with this free.
        unsafe { sys::libdeflate_free_decompressor(self.raw); }
    }
}

/// Build a [`Checkpoint`] from the captured `(bitbuf, bitsleft)`
/// plus the last [`WINDOW_SIZE`] bytes of the produced output.
fn snapshot_window(produced: &[u8], bitbuf: u64, bitsleft: u32) -> Checkpoint {
    let mut window = Box::new([0u8; WINDOW_SIZE]);
    let take = produced.len().min(WINDOW_SIZE);
    let dst_start = WINDOW_SIZE - take;
    window[dst_start..].copy_from_slice(&produced[produced.len() - take..]);
    Checkpoint {
        bitbuf, bitsleft,
        window,
        window_used: take,
    }
}
