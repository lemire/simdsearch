// Record the memchr crate version the library was built with, so the
// benchmark can print it alongside the C library version.
fn main() {
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock")).unwrap_or_default();
    let mut ver = "unknown".to_string();
    let mut in_memchr = false;
    for line in lock.lines() {
        if line.trim() == "name = \"memchr\"" { in_memchr = true; continue; }
        if in_memchr && line.trim_start().starts_with("version = ") {
            ver = line.split('"').nth(1).unwrap_or("unknown").to_string(); break;
        }
        if line.trim().is_empty() { in_memchr = false; }
    }
    println!("cargo:rustc-env=SIMDSEARCH_MEMCHR_VERSION={}", ver);
    println!("cargo:rerun-if-changed=Cargo.lock");
}
