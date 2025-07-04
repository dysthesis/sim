use std::{env, fs, path::PathBuf, process};

use libsim::similarity;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    #[cfg(feature = "dhat-ad-hoc")]
    let _profiler = dhat::Profiler::new_ad_hoc();
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

    let res = similarity(doc1, doc2);
    println!("{res}");
}
