use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use libsim::similarity;
use lipsum::lipsum;

fn sim_benchmark(c: &mut Criterion) {
    // Load documents once – I/O is *not* part of what we measure
    let doc1 = lipsum(100_000);
    let doc2 = lipsum(100_000);

    c.bench_function("TF-IDF with cosine similarity", |b| {
        b.iter(|| black_box(similarity(&doc1, &doc2)))
    });
}

criterion_group!(benches, sim_benchmark);
criterion_main!(benches);
