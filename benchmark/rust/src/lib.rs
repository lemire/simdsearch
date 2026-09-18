//! C ABI around the Rust substring searchers the benchmark compares against:
//! the `memchr` crate's `memmem::find` (one-shot) and prebuilt `Finder`, and
//! the standard library's `str::find`. Every function reports the byte offset
//! of the first occurrence, `-1` for no match, and `-2` when the searcher
//! cannot take the input (`str::find` on bytes that are not UTF-8).
//!
//! An empty needle matches at 0, as every other searcher in the benchmark
//! reports it; `memmem::find` agrees, `str::find` does too.
use std::os::raw::c_char;
use std::slice;

#[inline]
unsafe fn bytes<'a>(p: *const c_char, n: usize) -> &'a [u8] {
    if n == 0 { &[] } else { slice::from_raw_parts(p as *const u8, n) }
}

fn report(r: Option<usize>) -> isize {
    match r { Some(i) => i as isize, None => -1 }
}

/// `memchr::memmem::find`: the searcher is chosen and built on every call.
#[no_mangle]
pub unsafe extern "C" fn simdsearch_rust_memchr_find(hay: *const c_char, n: usize,
                                                     needle: *const c_char, m: usize) -> isize {
    report(memchr::memmem::find(bytes(hay, n), bytes(needle, m)))
}

/// `memchr::memmem::Finder`, built once per needle and reused.
#[no_mangle]
pub unsafe extern "C" fn simdsearch_rust_finder_new(needle: *const c_char, m: usize) -> *mut memchr::memmem::Finder<'static> {
    Box::into_raw(Box::new(memchr::memmem::Finder::new(bytes(needle, m)).into_owned()))
}

#[no_mangle]
pub unsafe extern "C" fn simdsearch_rust_finder_find(f: *const memchr::memmem::Finder<'static>,
                                                     hay: *const c_char, n: usize) -> isize {
    report((*f).find(bytes(hay, n)))
}

#[no_mangle]
pub unsafe extern "C" fn simdsearch_rust_finder_free(f: *mut memchr::memmem::Finder<'static>) {
    if !f.is_null() { drop(Box::from_raw(f)); }
}

/// `str::find` with a `&str` pattern (Two-Way in the standard library). Both
/// inputs must be UTF-8; otherwise -2.
#[no_mangle]
pub unsafe extern "C" fn simdsearch_rust_std_find(hay: *const c_char, n: usize,
                                                  needle: *const c_char, m: usize) -> isize {
    let (h, p) = match (std::str::from_utf8(bytes(hay, n)), std::str::from_utf8(bytes(needle, m))) {
        (Ok(h), Ok(p)) => (h, p),
        _ => return -2,
    };
    report(h.find(p))
}

/// The crate version, for the results' provenance line.
#[no_mangle]
pub extern "C" fn simdsearch_rust_memchr_version() -> *const c_char {
    concat!(env!("SIMDSEARCH_MEMCHR_VERSION"), "\0").as_ptr() as *const c_char
}
