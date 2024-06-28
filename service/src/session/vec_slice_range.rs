use std::{ops::Range, usize};

#[derive(Clone, Debug)]
/// 多区间组成的可间断区间
pub(crate) struct VecSliceRange(Vec<Range<usize>>);

impl VecSliceRange {
    /// 检查range 是否按顺序递增，如果不按顺序递增则panic
    pub fn try_from(v: Vec<Range<usize>>) -> Self {
        if v.is_empty() {
            return Self(v);
        }
        let mut iter = v.iter();

        let mut current = iter.next().unwrap();
        assert!(current.start <= current.end);

        for next in iter {
            assert!(next.start <= next.end);
            assert!(current.end <= next.start);
            current = next;
        }
        Self(v)
    }

    /// 返回区间起始位置
    pub fn start(&self) -> usize {
        self.0.first().unwrap().start
    }
    /// 返回区间结束位置
    pub fn end(&self) -> usize {
        self.0.last().unwrap().end
    }
    /// 返回所有区间长度和
    pub fn len(&self) -> usize {
        self.0.iter().map(|range| range.len()).sum::<usize>()
    }
    /// 返回区间是否为空
    #[allow(unused)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// 延长最后一个区间的长度到end，如果end小于最后一个区间的长度则会panic
    pub fn extend_to(&mut self, end: usize) {
        if end < self.end() {
            panic!("extend_to function can't reduce the {self:?} end to {end:?},so it do nothing",);
        } else {
            self.0.last_mut().unwrap().end = end;
        }
    }

    /// 将索引整体减小n
    pub fn sub_overall(&mut self, n: usize) {
        assert!(self.start() >= n);
        self.0
            .iter_mut()
            .for_each(|range| *range = (range.start - n)..(range.end - n))
    }

    /// 尝试将区间的下限减小
    pub fn try_shrink_end(&mut self, new_end: usize) -> bool {
        let mut index = 0;
        if self.0.iter().enumerate().any(|(i, range)| {
            if range.start <= new_end && range.end >= new_end {
                index = i;
                true
            } else {
                false
            }
        }) {
            self.0.truncate(index + 1);
            let last_range = self.0.last_mut().unwrap();
            if last_range.start == new_end {
                self.0.pop();
            } else {
                last_range.end = new_end;
            }
            true
        } else {
            false
        }
    }

    /// 获取区间vec的不可变引用
    pub fn get_ranges(&self) -> &[Range<usize>] {
        &self.0
    }
}

impl From<Range<usize>> for VecSliceRange {
    fn from(value: Range<usize>) -> Self {
        Self(vec![value])
    }
}

#[test]
fn test_vec_slice_range() {
    print!("asdfasdf {:?} asdfasdf", 10..0);
    let mut a = VecSliceRange::try_from(vec![0..10, 20..30]);
    //test len
    assert!(a.len() == 20);
    //test try_shrink_end
    a.try_shrink_end(25);
    assert!(a.0 == VecSliceRange::try_from(vec![0..10, 20..25]).0);

    let mut b = VecSliceRange::try_from(vec![10..20, 30..40]);
    // test sub_overall
    b.sub_overall(10);
    assert!(b.0 == VecSliceRange::try_from(vec![0..10, 20..30]).0);

    //test extend_to
    b.extend_to(40);
    assert!(b.0 == VecSliceRange::try_from(vec![0..10, 20..40]).0);
}
