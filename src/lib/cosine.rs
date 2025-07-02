use std::{collections::HashMap, hash::Hash};

type Score = f64;

/// Computes the cosine similarity between two sparse vectors.
pub fn cosine_similarity_sparse<T: Eq + Hash>(
    a: &HashMap<T, Score>,
    b: &HashMap<T, Score>,
) -> Score {
    fn dot<U: Eq + Hash>(v1: &HashMap<U, Score>, v2: &HashMap<U, Score>) -> Score {
        let (short, long) = if v1.len() < v2.len() {
            (v1, v2)
        } else {
            (v2, v1)
        };
        short
            .iter()
            .filter_map(|(key, &val1)| long.get(key).map(|&val2| val1 * val2))
            .sum()
    }

    let norm_a = a.values().map(|&x| x * x).sum::<Score>().sqrt();
    let norm_b = b.values().map(|&x| x * x).sum::<Score>().sqrt();

    let denom = norm_a * norm_b;
    if denom == 0.0 { 0.0 } else { dot(a, b) / denom }
}
