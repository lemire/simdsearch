use simdsearch_compare::Searcher;

struct TwoWay;

impl Searcher for TwoWay {
    const NAME: &'static str = "two-way (Rust)";
    type State = ();
    fn prepare(_: &[u8]) {}
    fn find((): &(), haystack: &[u8], needle: &[u8]) -> Option<usize> {
        simdsearch::twoway::find(haystack, needle)
    }
}

fn main() -> anyhow::Result<()> {
    simdsearch_compare::main::<TwoWay>()
}
