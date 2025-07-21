# sim - document similarity checker

Compares the similarity of two text documents using the TF-IDF algorithm.

## Usage

```bash
sim $PATH_TO_DOCUMENT_1 $PATH_TO_DOCUMENT_2
```

## Todo

- [ ] Cache the TF-IDF vector (see [bincode](https://crates.io/crates/bincode))
  - Is `bincode` really the best option? I think a plaintext format like JSON would be more future-proof.
  - Whatever the caching mechanism is, accept the cached vector the exact same way you would a document.
- [ ] Try a different algorithm, such as BM25
- [ ] Implement a separate CLI for tokenising Markdown files.
- [ ] Figure out a more efficient vector representation than a hash map
