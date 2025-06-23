use std::{env, fs, path::PathBuf, process};

use crate::{cosine::Cosine, tf_idf::TfIdf};

mod cosine;
mod tf_idf;
fn main() {
    // Parse CLI arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        // Print an error message to standard error.
        eprintln!("Usage: {} <path1> <path2>", args[0]);
        // Exit the program with a non-zero status code to indicate an error.
        process::exit(1);
    }
    let path1 = PathBuf::from(&args[1]);
    let path2 = PathBuf::from(&args[2]);
    let doc1 = fs::read_to_string(path1).unwrap();
    let doc1 = doc1.as_str();
    let doc2 = fs::read_to_string(path2).unwrap();
    let doc2 = doc2.as_str();
    let docs = vec![doc1, doc2];
    let tf_idf = TfIdf::from(docs.as_slice());
    let sim = Cosine::from(tf_idf.get().as_slice());
    let res = sim.get()[0][1];
    println!("{res}");
}
