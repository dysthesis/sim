use std::{collections::HashMap, hash::Hash};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

type Score = f64;
/// The cosine similarity of a list of vectors.
/// Given some `cosine: Cosine`, the similarity of vertices i and j is cosine[i][j] == cosine[j][i]
pub struct Cosine(Vec<Vec<f64>>);
impl<T> From<&[HashMap<T, Score>]> for Cosine
where
    T: Eq + Hash + Sync,
{
    fn from(value: &[HashMap<T, Score>]) -> Self {
        /// Dot product of two sparse vectors.
        fn dot<U>(a: &HashMap<U, Score>, b: &HashMap<U, Score>) -> f64
        where
            U: Eq + Hash,
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
            .collect();
        let len = value.len();
        let mut sim = vec![vec![0 as Score; len]; len];

        // fill upper triangle
        sim.par_iter_mut() // each row is &mut [f64]
            .enumerate()
            .for_each(|(i, row)| {
                for j in i..len {
                    let denom = norms[i] * norms[j];
                    row[j] = if denom == 0.0 {
                        0.0
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

impl Cosine {
    pub fn get(&self) -> &Vec<Vec<f64>> {
        &self.0
    }
}
