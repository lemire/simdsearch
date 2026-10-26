mod common;

use simdsearch::guardzilla::{self, Settings};

macro_rules! settings {
    ($($test:ident: $name:ident { $($key:ident: $ty:ty = $value:expr),* })*) => {$(
        struct $name;
        impl Settings for $name {
            $(const $key: $ty = $value;)*
        }
        #[test]
        fn $test() {
            common::suite(guardzilla::find_with::<$name>);
        }
    )*};
}

settings! {
    anchors1: Anchors1 { ANCHORS: usize = 1 }
    anchors2: Anchors2 { ANCHORS: usize = 2 }
    anchors4: Anchors4 { ANCHORS: usize = 4 }
    anchors8: Anchors8 { ANCHORS: usize = 8 }
    noalign: NoAlign { ALIGN: bool = false }
    budget1: Budget1 { BUDGET: usize = 1 }
}

#[test]
fn default_settings() {
    common::suite(guardzilla::find);
}

#[test]
fn twoway_alone() {
    common::suite(simdsearch::twoway::find);
}
