#![no_std]
use bumpalo::Bump;
use cosine::Cosine;
use tf_idf::TfIdf;

pub mod cosine;
pub mod tf_idf;

pub fn similarity(first: &str, second: &str, arena: &mut [u8]) -> f64 {
    let arena = Bump::from(arena);
    let docs = [first, second];

    let tf_idf = TfIdf::from_corpus(docs.as_slice(), &arena);
    let sim = Cosine::from(tf_idf.get().as_slice(), arena);
    sim.get()[0][1]
}
