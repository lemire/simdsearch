use simdsearch_compare::Searcher;

struct Memmem;

impl Searcher for Memmem {
    const NAME: &'static str = "Rust memchr::memmem::find";
    type State = ();
    fn prepare(_: &[u8]) {}
    fn find((): &(), haystack: &[u8], needle: &[u8]) -> Option<usize> {
        memchr::memmem::find(haystack, needle)
    }
}

fn main() -> anyhow::Result<()> {
    simdsearch_compare::main::<Memmem>()
}
