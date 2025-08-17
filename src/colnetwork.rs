//! Coloured network data structure encapsulating adjacency, colours, buckets and position mapping.
//! Provides O(1) random edge operations with swap–pop buckets.

use rand::Rng;

/// Upper-triangle index for (u,v) with u < v.
#[inline]
pub fn tri_index(u: u32, v: u32) -> usize {
    debug_assert!(u < v);
    (u as u64 + (v as u64) * ((v as u64) - 1) / 2) as usize
}

/// Bucket kind from (present?, same_colour?). Order kept (C0,C1,D0,D1).
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BucketKind {
    C0 = 0, // concordant absent
    C1 = 1, // concordant present
    D0 = 2, // discordant absent
    D1 = 3, // discordant present
}
#[inline]
pub fn bidx(b: BucketKind) -> usize { b as usize }

#[inline]
pub fn which_bucket(present: bool, same_colour: bool) -> BucketKind {
    match (present, same_colour) {
        (false, true)  => BucketKind::C0,
        (true,  true)  => BucketKind::C1,
        (false, false) => BucketKind::D0,
        (true,  false) => BucketKind::D1,
    }
}

/// Dense bucket of canonical edges with swap–pop deletion.
#[derive(Default)]
pub struct Bucket {
    pub(crate) a: Vec<(u32, u32)>,
}
impl Bucket {
    #[inline]
    pub fn len(&self) -> usize { self.a.len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.a.is_empty() }
    #[inline]
    pub fn push(&mut self, u: u32, v: u32) -> usize {
        debug_assert!(u < v);
        let idx = self.a.len();
        self.a.push((u, v));
        idx
    }
    #[inline]
    pub fn pop_at(&mut self, i: usize) -> (u32, u32) {
        let last = self.a.len() - 1;
        if i != last { self.a.swap(i, last); }
        self.a.pop().unwrap()
    }
    #[inline]
    pub fn remove_random<R: Rng>(&mut self, rng: &mut R) -> Option<(u32, u32, usize)> {
        if self.a.is_empty() { return None; }
        let i = rng.random_range(0..self.a.len());
        let (u, v) = self.pop_at(i);
        Some((u, v, i))
    }
    #[inline]
    pub fn pick_random<R: Rng>(&self, rng: &mut R) -> Option<(u32, u32)> {
        if self.a.is_empty() { return None; }
        let i = rng.random_range(0..self.a.len());
        Some(self.a[i])
    }
}

/// Coloured Network encapsulating adjacency, colour vector, bucket classification and position table.
pub struct ColNetwork {
    pub(crate) n: u32,
    adj: Vec<u8>,        // n x n symmetric (0/1)
    colour: Vec<u8>,     // length n (0/1)
    pos: Vec<u32>,       // upper-triangle mapping -> index within its bucket
    buckets: [Bucket; 4],
    present_edges: usize,
    ones_count: usize,
}

impl ColNetwork {
    /// Build from supplied adjacency (square n x n) and colour vector (length n).
    pub fn new(adj: Vec<u8>, colour: Vec<u8>) -> Self {
        let n = (colour.len()) as u32;
        assert_eq!(adj.len(), (n as usize) * (n as usize), "adjacency must be n*n");
        let mut buckets: [Bucket; 4] = [
            Bucket::default(),
            Bucket::default(),
            Bucket::default(),
            Bucket::default(),
        ];
        let mut pos: Vec<u32> = vec![0; (n as usize * (n as usize - 1)) / 2];
        let mut present_edges = 0usize;
        for u in 0..n {
            for v in (u + 1)..n {
                let ui = u as usize;
                let vi = v as usize;
                let present = adj[ui * n as usize + vi] != 0;
                let same = colour[ui] == colour[vi];
                if present { present_edges += 1; }
                let b = which_bucket(present, same);
                let idx = buckets[bidx(b)].push(u, v);
                pos[tri_index(u, v)] = idx as u32;
            }
        }
        let ones_count = colour.iter().map(|&c| c as usize).sum();
        Self {
            n,
            adj,
            colour,
            pos,
            buckets,
            present_edges,
            ones_count,
        }
    }

    /// n accessor (currently unused externally but kept for future expansion)
    #[allow(dead_code)]
    #[inline]
    pub fn n(&self) -> usize { self.n as usize }
    #[inline]
    pub fn adj(&self) -> &[u8] { &self.adj }
    #[inline]
    pub fn colour(&self) -> &[u8] { &self.colour }
    #[inline]
    pub fn present_edges(&self) -> usize { self.present_edges }
    #[inline]
    pub fn ones_count(&self) -> usize { self.ones_count }

    /// Random edge from bucket index (0..3).
    pub fn pick_random<R: Rng>(
        &self,
        bucket_index: usize,
        rng: &mut R,
    ) -> Option<(u32, u32)> {
        self.buckets[bucket_index].pick_random(rng)
    }

    /// Move a random edge from one bucket to another, updating adjacency & present edge count if presence status changes.
    pub fn move_edge<R: Rng>(
        &mut self,
        from_bucket: usize,
        to_bucket: usize,
        rng: &mut R,
    ) -> Option<(u32, u32)> {
        if from_bucket == to_bucket { return None; }
        let (u, v, i_removed) = self.buckets[from_bucket].remove_random(rng)?;
        if let Some(&(su, sv)) = self.buckets[from_bucket].a.get(i_removed) {
            self.pos[tri_index(su, sv)] = i_removed as u32;
        }
        let was_present = matches!(from_bucket, 1 | 3);
        let will_present = matches!(to_bucket, 1 | 3);
        if was_present != will_present {
            self.set_edge(u as usize, v as usize, will_present);
            if will_present {
                self.present_edges += 1;
            } else {
                self.present_edges -= 1;
            }
        }
        let idx = self.buckets[to_bucket].push(u, v);
        self.pos[tri_index(u, v)] = idx as u32;
        Some((u, v))
    }

    /// Flip colour of vertex u and reclassify all incident edges.
    pub fn flip_colour(&mut self, u: u32) {
        let n = self.n as usize;
        let uidx = u as usize;
        let old = self.colour[uidx];
        let new = 1 - old;
        let mut partners: Vec<u32> = Vec::with_capacity(n - 1);
        for w in 0..self.n {
            if w == u { continue; }
            let (a, b) = if u < w { (u, w) } else { (w, u) };
            let k = tri_index(a, b);
            let i = self.pos[k] as usize;
            let present = self.adj[a as usize * n + b as usize] != 0;
            let same = self.colour[a as usize] == self.colour[b as usize];
            let b_from = which_bucket(present, same);
            let bucket_vec = &mut self.buckets[bidx(b_from)];
            let (ru, rv) = bucket_vec.pop_at(i);
            debug_assert_eq!((ru, rv), (a, b));
            if i < bucket_vec.len() {
                let (su, sv) = bucket_vec.a[i];
                self.pos[tri_index(su, sv)] = i as u32;
            }
            partners.push(w);
        }
        self.colour[uidx] = new;
        if old == 0 { self.ones_count += 1; } else { self.ones_count -= 1; }
        for &w in &partners {
            let (a, b) = if u < w { (u, w) } else { (w, u) };
            let present = self.adj[a as usize * n + b as usize] != 0;
            let same = self.colour[a as usize] == self.colour[b as usize];
            let b_new = which_bucket(present, same);
            let idx = self.buckets[bidx(b_new)].push(a, b);
            self.pos[tri_index(a, b)] = idx as u32;
        }
    }

    /// Set adjacency (symmetric)
    fn set_edge(&mut self, u: usize, v: usize, present: bool) {
        let val = if present { 1 } else { 0 };
        let n = self.n as usize;
        self.adj[u * n + v] = val;
        self.adj[v * n + u] = val;
    }

    #[inline]
    pub fn bucket_len(&self, i: usize) -> usize { self.buckets[i].len() }
    #[inline]
    pub fn bucket_is_empty(&self, i: usize) -> bool { self.buckets[i].is_empty() }
}
