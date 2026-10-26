use memchr::memmem::Finder;
use simdsearch_compare::Searcher;

struct Prebuilt;

impl Searcher for Prebuilt {
    const NAME: &'static str = "Rust memchr::memmem::Finder (prebuilt)";
    type State = Finder<'static>;
    fn prepare(needle: &[u8]) -> Finder<'static> {
        Finder::new(needle).into_owned()
    }
    fn find(finder: &Finder<'static>, haystack: &[u8], _: &[u8]) -> Option<usize> {
        finder.find(haystack)
    }
}

fn main() -> anyhow::Result<()> {
    simdsearch_compare::main::<Prebuilt>()
}
