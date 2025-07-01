use cosine::Cosine;
use tf_idf::TfIdf;

pub mod cosine;
pub mod tf_idf;

pub fn similarity(first: &str, second: &str) -> f64 {
    if first.is_empty() || second.is_empty() {
        return 0.0;
    }

    if first == second {
        return 1.0;
    }
    let docs = [first, second];
    let tf_idf = TfIdf::from(docs.as_slice());
    let sim = Cosine::from(tf_idf.get().as_slice());
    sim.get()[0][1]
}
