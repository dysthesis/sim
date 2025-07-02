use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use libsim::{
    cosine::cosine_similarity_sparse,
    similarity,
    tf_idf::{Df, Idf, Term, Tf, TfIdf},
};
use lipsum::lipsum;

const CORPUS_LEN: usize = 1_000_000;

fn sim_benchmark(c: &mut Criterion) {
    let docs = lipsum(CORPUS_LEN);
    let doc1 = &docs[..CORPUS_LEN];
    let doc2 = &docs[CORPUS_LEN..];
    let corpus = [doc1, doc2];
    let corpus_slice = corpus.as_slice();

    let tfs = [Tf::from(doc1), Tf::from(doc2)];

    let tfidf = TfIdf::from([doc1, doc2].as_slice());

    let vectors = tfidf.get();

    c.bench_function("TF-IDF with cosine similarity", |b| {
        b.iter(|| black_box(similarity(doc1, doc2)))
    });
    c.bench_function("TF-IDF", |b| {
        b.iter(|| black_box(TfIdf::from(corpus_slice)))
    });
    c.bench_function("Cosine similarity", |b| {
        b.iter(|| black_box(cosine_similarity_sparse(&vectors[0], &vectors[1])));
    });
    c.bench_function("Tokenisation", |b| b.iter(|| black_box(Term::from(doc1))));
    c.bench_function("Term factor", |b| b.iter(|| black_box(Tf::from(doc1))));
    c.bench_function("Document factor", |b| {
        b.iter(|| black_box(Df::from(tfs.as_slice())))
    });
    c.bench_function("Inverse document factor", |b| {
        b.iter(|| black_box(Idf::from(Df::from(tfs.as_slice()))))
    });
}

criterion_group!(benches, sim_benchmark);
criterion_main!(benches);
