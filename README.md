# sim - document similarity checker

Compares the similarity of two text documents using the TF-IDF algorithm.

## Usage

```bash
sim $PATH_TO_DOCUMENT_1 $PATH_TO_DOCUMENT_2
```

## Todo

- [ ] Cache the TF-IDF vector (see [bincode](https://crates.io/crates/bincode))
- [ ] Figure out a more efficient vector representation than a hash map
