use super::vec_slice_range::VecSliceRange;
use causal_lm::{CausalLM, QueryContext};
use common::{upos, utok};
use std::cmp::min;
use tensor::Tensor;

pub(super) struct Cache<Storage> {
    /// 可映射的 token 序列。
    tokens: Vec<utok>,
    /// token 序列在整个对话中的位置。
    pos: usize,
    /// 缓存在 token 序列中的范围。
    cached: VecSliceRange,
    /// 计算缓存。
    cache: Tensor<Storage>,
}

impl<Storage> Cache<Storage> {
    /// 生成一个空白的缓存结构，准备填充 `tokens`。
    #[inline]
    pub fn new(t: &impl CausalLM<Storage = Storage>, tokens: Vec<utok>) -> Self {
        Self {
            tokens,
            pos: 0,
            cached: (0..0).into(),
            cache: t.new_cache(),
        }
    }
    /// 复制缓存结构。
    #[inline]
    pub fn duplicate(&self, t: &impl CausalLM<Storage = Storage>) -> Self {
        assert_eq!(self.cached.start(), 0);
        Self {
            tokens: self.tokens.clone(),
            pos: self.pos,
            cached: self.cached.clone(),
            cache: t.duplicate_cache(&self.cache, self.cached.end() as _),
        }
    }
    /// 回滚缓存到 `pos`，并返回剩余的有效缓存长度。
    pub fn revert(&mut self, pos: usize) -> Option<usize> {
        // 只能在闲时回滚，因此 cache 和 tokens 起始位置对齐
        assert_eq!(self.cached.start(), 0);
        // 回滚之后，tokens.len()、cached.end、pos 不能大于新的 pos
        // 1. pos 不大于 pos；
        let len = pos.checked_sub(self.pos)?;
        // 2. cached.end 不大于 pos；
        self.cached.try_shrink_end(len).then_some(())?;
        // 3. tokens.len() 不大于 pos；
        self.tokens.truncate(len);
        // 返回当前的缓存长度
        Some(self.cached.len())
    }
    /// 扩展待填充 token。
    #[inline]
    pub fn extend(&mut self, tokens: &[utok]) {
        self.tokens.extend_from_slice(tokens);
    }
    /// 所有 token 中还没有加入缓存的部分就是这次的查询。
    #[inline]
    pub fn query(&self) -> &[utok] {
        &self.tokens[self.cached.end()..]
    }
    /// 生成对应的查询上下文。
    #[inline]
    pub fn as_ctx(&mut self) -> QueryContext<Storage> {
        let Cache {
            pos: _pos,
            cache,
            tokens,
            cached,
        } = self;
        QueryContext {
            cache: Some(cache),
            range: cached.len() as upos..(tokens.len() - cached.end() + cached.len()) as upos,
        }
    }

    /// 将新采样的值加入缓存。
    #[inline]
    pub fn push(&mut self, token: utok) {
        self.cached.extend_to(self.tokens.len());
        self.tokens.push(token);
    }
    /// 已采样的最后一个词在对话中的位置。
    #[inline]
    pub fn end(&self) -> usize {
        self.pos + self.tokens.len()
    }
    /// 提取尾部词序列。
    #[inline]
    pub fn slice_tail(&self, pos: usize) -> &[utok] {
        let known = pos.checked_sub(self.pos).unwrap();
        &self.tokens[known..]
    }

    /// 重置缓存窗口,并将起始点设置为尾部一部分之前
    #[allow(unused)]
    pub fn reset_within_one_range(&mut self, min: usize, max: usize) {
        if self.tokens.len() - self.cached.end() + self.cached.len() >= max {
            self.cached = (self.tokens.len() - min..self.tokens.len() - min).into()
        }
    }
    /// 重置缓存窗口，保留起始的一部分，并将起始点设置为尾部一部分之前
    pub fn reset_within_start_and_end_range(
        &mut self,
        start_size: usize,
        end_size: usize,
        max: usize,
    ) {
        if self.tokens.len() - self.cached.end() + self.cached.len() >= max {
            let ranges = self.cached.get_ranges();
            let mut first_range = ranges.first().unwrap().clone();
            first_range.end = min(first_range.end, start_size);

            self.cached = VecSliceRange::try_from(vec![
                first_range,
                (self.tokens.len() - end_size..self.tokens.len() - end_size),
            ])
        }
    }
    /// 重置并清空缓存窗口。
    pub fn reset_with(&mut self, tokens: Vec<utok>, pos: usize) {
        self.tokens = tokens;
        self.pos = pos;
        self.cached = (0..0).into();
    }
    /// 清理缓存中在缓存窗口之前的部分。
    pub fn cleanup_before_start(&mut self) {
        let to_remove = self.cached.start();
        if to_remove > 0 {
            self.tokens.copy_within(to_remove.., 0);
            self.pos += to_remove;
            self.tokens.truncate(self.tokens.len() - to_remove);
            self.cached.sub_overall(to_remove);
        }
    }
    /// 获取cached中最后一个区间的长度，如果cached为空则会panic
    pub fn get_last_cached_range_len(&self) -> usize {
        self.cached.get_ranges().last().unwrap().len()
    }
}
