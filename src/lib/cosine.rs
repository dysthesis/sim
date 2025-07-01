use bumpalo::{
    Bump,
    collections::{CollectIn, Vec},
    vec,
};
use core::{hash::Hash, primitive::f64};
use hashbrown::HashMap;
use micromath::{F32, F32Ext};

use crate::tf_idf::BumpHashMap;

type Score = F32;
/// The cosine similarity of a list of vectors.
/// Given some `cosine: Cosine`, the similarity of vertices i and j is cosine[i][j] == cosine[j][i]
pub struct Cosine<'bump>(Vec<'bump, Vec<'bump, Score>>);
impl<'bump> Cosine<'bump> {
    pub fn from<T>(value: &[BumpHashMap<'_, 'bump>], alloc: &'bump Bump) -> Self
    where
        T: Eq + Hash + Sync,
    {
        /// Dot product of two sparse vectors.
        fn dot<U>(a: &BumpHashMap<'_, '_>, b: &BumpHashMap<'_, '_>) -> Score
        where
            U: Eq,
        {
            // iterate over the smaller map
            let (short, long) = if a.len() < b.len() { (a, b) } else { (b, a) };
            short
                .iter()
                .filter_map(|(t, &w)| long.get(t).map(|&w2| w * w2))
                .sum()
        }
        let norms: Vec<Score> = value
            .iter()
            .map(|v| v.values().map(|&x| x * x).sum::<Score>().sqrt())
            .collect_in(alloc);

        let len = value.len();
        let mut sim = vec![in &alloc; vec![in & alloc;0 as Score; len]; len];

        // fill upper triangle
        sim.iter_mut().enumerate().for_each(|(i, row)| {
            for j in i..len {
                let denom = norms[i] * norms[j];
                row[j] = if denom == 0.0 {
                    0.0.into()
                } else {
                    dot(&value[i], &value[j]) / denom
                };
            }
        });

        //  mirror to lower triangle (single-threaded; O(n²/2))
        for i in 0..len {
            for j in 0..i {
                sim[i][j] = sim[j][i];
            }
        }

        Cosine(sim)
    }
}

impl<'bump> Cosine<'bump> {
    pub fn get(&self) -> &Vec<Vec<f64>> {
        &self.0
    }
}
