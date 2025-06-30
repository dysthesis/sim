use std::collections::HashMap;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

type Score = f64;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct Term(String);
impl Term {
    pub fn from(string: &str) -> Vec<Self> {
        let sanitised: String = string
            .chars()
            .map(|c| if c.is_alphabetic() { c } else { ' ' })
            .collect();
        sanitised
            .split_whitespace()
            .map(|t| Self(t.to_string()))
            .collect()
    }
}
impl From<Term> for String {
    fn from(Term(value): Term) -> Self {
        value
    }
}
pub struct Tf(HashMap<Term, Score>);
impl From<&str> for Tf {
    fn from(value: &str) -> Self {
        let terms = Term::from(value);
        let tf = terms
            .par_iter()
            .fold(
                HashMap::new,
                |mut frequencies: HashMap<Term, Score>, term| {
                    *frequencies.entry(term.clone()).or_default() += 1 as Score;
                    frequencies
                },
            )
            .reduce(HashMap::new, |mut a, b| {
                a.extend(b);
                a
            });
        Self(tf)
    }
}

impl Tf {
    #[inline]
    pub fn get(&self, string: String) -> Option<Score> {
        self.0.get(&Term(string)).map(|val| val.clone())
    }
    #[inline]
    pub fn get_map(&self) -> &HashMap<Term, Score> {
        &self.0
    }
}

pub struct Df {
    map: HashMap<Term, Score>,
    num_docs: usize,
}

impl From<&[&str]> for Df {
    fn from(value: &[&str]) -> Self {
        let num_docs = value.len();
        let map = value
            .par_iter()
            .map(|doc| {
                let mut df: HashMap<Term, Score> = HashMap::new();
                let unique_terms = Term::from(doc);
                for term in unique_terms {
                    *df.entry(term).or_default() += 1 as Score;
                }
                let res = df.iter().map(|(k, v)| (k.clone(), *v)).collect();
                res
            })
            .reduce(
                || HashMap::new(),
                |mut a, b| {
                    a.extend(b);
                    a
                },
            );
        Df { map, num_docs }
    }
}

pub struct Idf(HashMap<Term, Score>);
impl From<Df> for Idf {
    fn from(value: Df) -> Self {
        let res = value
            .map
            .into_iter()
            .map(|(term, df)| {
                let idf = ((value.num_docs as Score + 1.0) / (df as f64 + 1.0)).ln() + 1.0;
                (term, idf)
            })
            .collect();
        Idf(res)
    }
}

impl Idf {
    #[inline]
    pub fn get(&self, string: String) -> Option<Score> {
        self.0.get(&Term(string)).map(|val| val.clone())
    }
}

pub struct TfIdf(Vec<HashMap<Term, Score>>);
impl TfIdf {
    pub fn get(&self) -> &Vec<HashMap<Term, Score>> {
        &self.0
    }
}
impl From<&[&str]> for TfIdf {
    fn from(value: &[&str]) -> Self {
        let idf: Idf = Df::from(value).into();
        let res = value
            .par_iter()
            .map(|doc| {
                let tf = Tf::from(doc.to_owned());
                tf.get_map()
                    .into_iter()
                    .filter_map(|(term, tf)| {
                        idf.get(term.clone().into()).map(|idf| (term, tf * idf))
                    })
                    .map(|(k, v)| (k.clone(), v))
                    .collect::<HashMap<_, _>>()
            })
            .collect();
        Self(res)
    }
}
