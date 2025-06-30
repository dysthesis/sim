use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use libsim::{
    cosine::Cosine,
    similarity,
    tf_idf::{Df, Idf, Term, Tf, TfIdf},
};
use lipsum::lipsum;

fn sim_benchmark(c: &mut Criterion) {
    let doc1 = lipsum(100_000);
    let doc2 = lipsum(100_000);
    let corpus = [doc1.as_str(), doc2.as_str()];
    let corpus_slice = corpus.as_slice();

    let tfidf = TfIdf::from([doc1.as_str(), doc2.as_str()].as_slice());
    let tfidf_slice = tfidf.get().as_slice();

    c.bench_function("TF-IDF with cosine similarity", |b| {
        b.iter(|| black_box(similarity(&doc1, &doc2)))
    });
    c.bench_function("TF-IDF", |b| {
        b.iter(|| black_box(TfIdf::from(corpus_slice)))
    });
    c.bench_function("Cosine similarity", |b| {
        b.iter(|| black_box(Cosine::from(tfidf_slice)));
    });
    c.bench_function("Tokenisation", |b| b.iter(|| black_box(Term::from(&doc1))));
    c.bench_function("Term factor", |b| {
        b.iter(|| black_box(Tf::from(doc1.as_str())))
    });
    c.bench_function("Document factor", |b| {
        b.iter(|| black_box(Df::from([doc1.as_str(), doc2.as_str()].as_slice())))
    });
    c.bench_function("Inverse document factor", |b| {
        b.iter(|| {
            black_box(Idf::from(Df::from(
                [doc1.as_str(), doc2.as_str()].as_slice(),
            )))
        })
    });
}

criterion_group!(benches, sim_benchmark);
criterion_main!(benches);
