use crate::simd::{BLOCK, Block};

pub const MAX: usize = 8;

pub type Offsets = [usize; MAX];

pub type Lanes<'a> = [&'a [Block]; MAX];

pub const fn stringzilla(needle: &[u8]) -> [usize; 3] {
    let s = needle;
    let len = s.len();
    let (mut first, mut mid, mut last) = (0, len / 2, len - 1);
    if len > 3 && (s[first] == s[mid] || s[first] == s[last] || s[mid] == s[last]) {
        while s[mid] == s[first] && mid + 1 < last {
            mid += 1;
        }
        while (s[last] == s[mid] || s[last] == s[first]) && last > mid + 1 {
            last -= 1;
        }
    }
    if len > 8 {
        let (mut vfirst, mut vmid) = (first, mid);
        while (s[vmid] > 191 || s[vmid] == s[last]) && vmid + 1 < last {
            vmid += 1;
        }
        if s[vmid] < 191 {
            mid = vmid;
        } else {
            vmid = mid;
        }
        while (s[vfirst] > 191 || s[vfirst] == s[vmid] || s[vfirst] == s[last]) && vfirst + 1 < vmid
        {
            vfirst += 1;
        }
        if s[vfirst] < 191 {
            first = vfirst;
        }
    }
    [first, mid, last]
}

pub fn spaced(len: usize, count: usize) -> Offsets {
    let k = count.min(len);
    core::array::from_fn(|j| {
        if k == 1 {
            len - 1
        } else {
            j.min(k - 1) * (len - 1) / (k - 1)
        }
    })
}

pub fn select(needle: &[u8], count: usize) -> Offsets {
    if count == 3 && needle.len() >= 3 {
        let [first, mid, last] = stringzilla(needle);
        [first, mid, last, last, last, last, last, last]
    } else {
        spaced(needle.len(), count)
    }
}

pub fn lanes<'a>(
    hay: &'a [u8],
    start: usize,
    offsets: &Offsets,
    count: usize,
    blocks: usize,
) -> Lanes<'a> {
    core::array::from_fn(|j| {
        if j < count {
            let from = start + offsets[j];
            hay[from..from + BLOCK * blocks].as_chunks().0
        } else {
            &[]
        }
    })
}
