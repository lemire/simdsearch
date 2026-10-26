mod common;

use simdsearch::needlehammer::{self, Settings};

macro_rules! settings {
    ($($test:ident: $name:ident { $($key:ident: $ty:ty = $value:expr),* })*) => {$(
        struct $name;
        impl Settings for $name {
            $(const $key: $ty = $value;)*
        }
        #[test]
        fn $test() {
            common::suite(needlehammer::find_with::<$name>);
        }
    )*};
}

settings! {
    eager: Eager {
        WINDOW: usize = 2, LONG_WINDOW: usize = 3, BAND: usize = 1, TIMED: bool = false
    }
    eager_anchors2: EagerAnchors2 {
        WINDOW: usize = 2, LONG_WINDOW: usize = 3, BAND: usize = 1, TIMED: bool = false,
        ANCHORS: usize = 2
    }
    eager_anchors5: EagerAnchors5 {
        WINDOW: usize = 2, LONG_WINDOW: usize = 3, BAND: usize = 1, TIMED: bool = false,
        ANCHORS: usize = 5
    }
    prologue0: Prologue0 { PROLOGUE: usize = 0 }
    prologue1: Prologue1 { PROLOGUE: usize = 1 }
    eager_prologue0: EagerPrologue0 {
        WINDOW: usize = 2, LONG_WINDOW: usize = 3, BAND: usize = 1, TIMED: bool = false,
        PROLOGUE: usize = 0
    }
    eager_prologue1: EagerPrologue1 {
        WINDOW: usize = 2, LONG_WINDOW: usize = 3, BAND: usize = 1, TIMED: bool = false,
        PROLOGUE: usize = 1
    }
    eager_budget1: EagerBudget1 {
        WINDOW: usize = 2, LONG_WINDOW: usize = 3, BAND: usize = 1, TIMED: bool = false,
        BUDGET: usize = 1
    }
}

#[test]
fn default_settings() {
    common::suite(needlehammer::find);
}

#[test]
fn dna_switches_to_second_scan() {
    let mut rng = common::Rng::new(7);
    let hay: Vec<u8> = (0..1 << 20).map(|_| b"ACGT"[rng.below(4)]).collect();
    let mut checker = common::Checker::new(needlehammer::find);
    for m in [4, 8, 16, 32, 256] {
        for _ in 0..20 {
            let at = rng.below(hay.len() - m + 1);
            let mut needle = hay[at..at + m].to_vec();
            checker.check(&hay, &needle, "dna-present");
            needle.reverse();
            needle[0] = b'N';
            checker.check(&hay, &needle, "dna-absent");
        }
    }
    assert_eq!(checker.fails, 0);
}
