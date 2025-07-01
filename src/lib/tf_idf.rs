use bumpalo::{
    Bump,
    collections::{CollectIn, Vec},
};
use core::hash::Hash;
use hashbrown::{DefaultHashBuilder, HashMap};
use micromath::F32;

type Score = F32;
// This is the full type we're working with
pub type BumpHashMap<'a, 'bump> = HashMap<Term<'a>, Score, DefaultHashBuilder, &'bump Bump>;

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

pub struct Tf<'a, 'bump>(BumpHashMap<'a, 'bump>);
impl<'a, 'bump> Tf<'a, 'bump> {
    fn new(value: &'a str, alloc: &'bump Bump) -> Self {
        let terms = Term::from(value);
        let (lower, _upper) = terms.size_hint();
        let tf: BumpHashMap<'a, 'bump> = terms.fold(
            BumpHashMap::with_capacity_in(lower, alloc),
            |mut frequencies: BumpHashMap<'a, 'bump>, term| {
                *frequencies.entry(term).or_default() += <f32 as Into<F32>>::into(1f32);
                frequencies
            },
        );
        Self(tf)
    }
}

impl<'a, 'bump> Tf<'a, 'bump> {
    #[inline]
    pub fn get(&self, string: &'a str) -> Option<Score> {
        self.0.get(&Term(string)).cloned()
    }
    #[inline]
    pub fn borrow_map(&self) -> &BumpHashMap<'a, 'bump> {
        &self.0
    }
    #[inline]
    pub fn get_map(self) -> BumpHashMap<'a, 'bump> {
        self.0
    }
}

pub struct Df<'a, 'bump> {
    map: BumpHashMap<'a, 'bump>,
    num_docs: usize,
}

impl<'a, 'bump> Df<'a, 'bump> {
    fn from(corpus: &[&'a str], alloc: &'bump Bump) -> Self {
        let num_docs = corpus.len();
        let mut map = BumpHashMap::new_in(alloc);
        for doc in corpus.iter() {
            let mut unique_terms_in_doc = Vec::new_in(alloc);
            for term in Term::from(*doc) {
                if !unique_terms_in_doc.contains(&term) {
                    unique_terms_in_doc.push(term);
                }
            }
            for term in unique_terms_in_doc {
                *map.entry(term).or_insert(0.0) += 1.0;
            }
        }
        Self { map, num_docs }
    }
}

pub struct Idf<'a, 'bump>(BumpHashMap<'a, 'bump>);
impl<'a, 'bump> Idf<'a, 'bump> {
    fn from(value: Df<'a, 'bump>, alloc: &'bump Bump) -> Self {
        let mut idf_map = BumpHashMap::with_capacity_in(value.map.len(), alloc);
        let num_docs = (value.num_docs as f32).into();
        for (term, df) in value.map {
            let idf_score = ((num_docs + 1.0) / (df + 1.0)).ln() + 1.0;
            idf_map.insert(term, idf_score);
        }
        Self(idf_map)
    }
}

impl<'a, 'bump> Idf<'a, 'bump> {
    pub fn get(&self, term: &Term<'a>) -> Option<Score> {
        self.0.get(term).copied()
    }
}

pub struct TfIdf<'a, 'bump>(Vec<'bump, BumpHashMap<'a, 'bump>>);
impl<'a, 'bump> TfIdf<'a, 'bump> {
    pub fn get(&self) -> &Vec<'bump, BumpHashMap<'a, 'bump>> {
        &self.0
    }
}
impl<'a, 'bump> TfIdf<'a, 'bump> {
    pub fn from_corpus(corpus: &[&'a str], alloc: &'bump Bump) -> Self {
        let df = Df::from(corpus, alloc);
        let idf = Idf::from(df, alloc);

        // Step 2: Use .collect_in(), which now works because .map() produces the correct type.
        let res = corpus
            .iter()
            .map(|doc| {
                // Call our corrected Tf::new constructor, passing the allocator.
                let tf_map = Tf::new(*doc, alloc).get_map();
                let mut tf_idf_map = BumpHashMap::with_capacity_in(tf_map.len(), alloc);

                for (term, tf_w) in tf_map {
                    if let Some(idf_w) = idf.get(&term) {
                        tf_idf_map.insert(term, tf_w * idf_w);
                    }
                }
                // The item produced by .map() is now a BumpHashMap, which collect_in expects.
                tf_idf_map
            })
            .collect_in(alloc);

        Self(res)
    }
}
