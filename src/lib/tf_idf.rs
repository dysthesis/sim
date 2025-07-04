use std::{collections::HashMap, hash::Hash};

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

type Score = f64;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct Term<'a>(&'a str);
impl<'a> Term<'a> {
    pub fn from(string: &'a str) -> impl Iterator<Item = Term<'a>> {
        string
            .split(|c: char| !c.is_alphabetic())
            .filter(|s| !s.is_empty())
            .map(Self)
    }
}

impl<'a> Term<'a> {
    pub fn borrow(self) -> &'a str {
        let Term(str) = self;
        str
    }
}

pub struct Tf<'a>(HashMap<Term<'a>, Score>);
impl<'a> From<&'a str> for Tf<'a> {
    fn from(value: &'a str) -> Self {
        let terms = Term::from(value);
        let (lower, _upper) = terms.size_hint();
        let tf = terms.fold(
            HashMap::with_capacity(lower),
            |mut frequencies: HashMap<Term, Score>, term| {
                *frequencies.entry(term).or_default() += 1 as Score;
                frequencies
            },
        );
        Self(tf)
    }
}

impl<'a> Tf<'a> {
    #[inline]
    pub fn get(&self, string: &'a str) -> Option<Score> {
        self.0.get(&Term(string)).cloned()
    }
    #[inline]
    pub fn borrow_map(&self) -> &HashMap<Term<'a>, Score> {
        &self.0
    }
    #[inline]
    pub fn get_map(self) -> HashMap<Term<'a>, Score> {
        self.0
    }
}

pub struct Df<'a> {
    map: HashMap<Term<'a>, Score>,
    num_docs: usize,
}

impl<'a> From<&'a [&'a str]> for Df<'a> {
    fn from(value: &[&'a str]) -> Self {
        let num_docs = value.len();
        let map = value
            .par_iter()
            .map(|doc| {
                let mut df: HashMap<Term, Score> = HashMap::new();
                let unique_terms = Term::from(doc);
                for term in unique_terms {
                    *df.entry(term).or_default() += 1 as Score;
                }
                df.iter().map(|(k, v)| (k.clone(), *v)).collect()
            })
            .reduce(HashMap::new, |mut a, b| {
                a.extend(b);
                a
            });
        Df { map, num_docs }
    }
}

impl<'a> From<&[Tf<'a>]> for Df<'a> {
    fn from(value: &[Tf<'a>]) -> Self {
        let num_docs = value.len();
        let map = value
            .iter()
            .fold(HashMap::new(), |mut acc: HashMap<Term, Score>, curr| {
                let curr_map = curr.borrow_map();
                curr_map
                    .keys()
                    .for_each(|k| *acc.entry(k.clone()).or_default() += 1 as Score);
                acc
            });
        Self { num_docs, map }
    }
}

pub struct Idf<'a>(HashMap<Term<'a>, Score>);
impl<'a> From<Df<'a>> for Idf<'a> {
    fn from(value: Df<'a>) -> Self {
        let res = value
            .map
            .into_iter()
            .map(|(term, df)| {
                let idf = ((value.num_docs as Score + 1.0) / (df + 1.0)).ln() + 1.0;
                (term, idf)
            })
            .collect();
        Idf(res)
    }
}

impl<'a> Idf<'a> {
    pub fn get(&self, term: &Term<'a>) -> Option<Score> {
        self.0.get(term).copied()
    }
}
pub struct TfIdf<'a>(Vec<HashMap<Term<'a>, Score>>);
impl<'a> TfIdf<'a> {
    pub fn get(&self) -> &Vec<HashMap<Term<'a>, Score>> {
        &self.0
    }
}
impl<'a> From<&[&'a str]> for TfIdf<'a> {
    fn from(corpus: &[&'a str]) -> Self {
        let tf: Vec<Tf<'a>> = corpus.par_iter().map(|doc| Tf::from(*doc)).collect();
        let idf: Idf = Df::from(tf.as_slice()).into();
        let res = tf
            .par_iter()
            .map(|val| {
                val.borrow_map()
                    .into_par_iter()
                    .filter_map(|(term, tf_w)| idf.get(term).map(|idf_w| (term, tf_w * idf_w)))
                    .map(|(k, v)| (k.clone(), v))
                    .collect::<HashMap<_, _>>()
            })
            .collect();
        Self(res)
    }
}
