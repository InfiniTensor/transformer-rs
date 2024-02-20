﻿mod broadcast;
mod slice;
mod split;
mod squeeze;
mod transpose;

use crate::{udim, Affine, Shape};
use smallvec::SmallVec;

pub trait Operator {
    fn build(&self, input: &[udim]) -> SmallVec<[(Shape, Affine); 1]>;
}

pub use broadcast::Broadcast;
pub use slice::Slice;
pub use split::Split;
pub use squeeze::Squeeze;
pub use transpose::Transpose;
