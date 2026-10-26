use simdsearch_compare::Searcher;

struct NeedleHammer;

impl Searcher for NeedleHammer {
    const NAME: &'static str = "NeedleHammer (Rust)";
    type State = ();
    fn prepare(_: &[u8]) {}
    fn find((): &(), haystack: &[u8], needle: &[u8]) -> Option<usize> {
        simdsearch::needlehammer::find(haystack, needle)
    }
}

fn main() -> anyhow::Result<()> {
    simdsearch_compare::main::<NeedleHammer>()
}
