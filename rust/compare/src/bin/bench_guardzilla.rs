use simdsearch_compare::Searcher;

struct Guardzilla;

impl Searcher for Guardzilla {
    const NAME: &'static str = "Guardzilla (Rust)";
    type State = ();
    fn prepare(_: &[u8]) {}
    fn find((): &(), haystack: &[u8], needle: &[u8]) -> Option<usize> {
        simdsearch::guardzilla::find(haystack, needle)
    }
}

fn main() -> anyhow::Result<()> {
    simdsearch_compare::main::<Guardzilla>()
}
