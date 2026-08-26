use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;

pub const BOOK_LEVELS: usize = 50;
pub const MODEL_LEVELS: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Level {
    pub price: f64,
    pub qty: f64,
}

impl Level {
    pub fn new(price: f64, qty: f64) -> Self {
        Self { price, qty }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LocalBook {
    bids: Vec<Level>,
    asks: Vec<Level>,
    last_u: Option<u64>,
    valid: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BookApply {
    Applied,
    Replaced,
    Gap,
    Ignored,
}

impl LocalBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_valid(&self) -> bool {
        self.valid && self.top_complete()
    }

    pub fn top_complete(&self) -> bool {
        self.bids.len() >= MODEL_LEVELS && self.asks.len() >= MODEL_LEVELS
    }

    pub fn last_u(&self) -> Option<u64> {
        self.last_u
    }

    pub fn bids(&self) -> &[Level] {
        &self.bids
    }

    pub fn asks(&self) -> &[Level] {
        &self.asks
    }

    pub fn discard(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.last_u = None;
        self.valid = false;
    }

    pub fn top10(&self) -> Option<Top10> {
        if !self.top_complete() {
            return None;
        }
        let mut bids = [Level::new(0.0, 0.0); MODEL_LEVELS];
        let mut asks = [Level::new(0.0, 0.0); MODEL_LEVELS];
        bids.copy_from_slice(&self.bids[..MODEL_LEVELS]);
        asks.copy_from_slice(&self.asks[..MODEL_LEVELS]);
        Some(Top10 { bids, asks })
    }

    pub fn mid(&self) -> Option<f64> {
        let t = self.top10()?;
        Some((t.bids[0].price + t.asks[0].price) * 0.5)
    }

    pub fn apply_snapshot(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], u: u64) -> BookApply {
        self.bids = normalize_side(bids, true);
        self.asks = normalize_side(asks, false);
        self.last_u = Some(u);
        self.valid = true;
        BookApply::Replaced
    }

    pub fn apply_incremental(
        &mut self,
        bids: &[(f64, f64)],
        asks: &[(f64, f64)],
        start_u: u64,
        end_u: u64,
    ) -> BookApply {
        if !self.valid {
            return BookApply::Ignored;
        }
        let Some(local) = self.last_u else {
            self.discard();
            return BookApply::Gap;
        };
        if start_u != local + 1 {
            self.discard();
            return BookApply::Gap;
        }
        apply_updates(&mut self.bids, bids, true);
        apply_updates(&mut self.asks, asks, false);
        self.last_u = Some(end_u);
        BookApply::Applied
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Top10 {
    pub bids: [Level; MODEL_LEVELS],
    pub asks: [Level; MODEL_LEVELS],
}

impl Top10 {
    pub fn from_levels(bids: [Level; MODEL_LEVELS], asks: [Level; MODEL_LEVELS]) -> Self {
        Self { bids, asks }
    }

    pub fn mid(&self) -> f64 {
        (self.bids[0].price + self.asks[0].price) * 0.5
    }
}

fn normalize_side(updates: &[(f64, f64)], bids: bool) -> Vec<Level> {
    let mut levels: Vec<Level> = updates
        .iter()
        .filter(|(_, q)| *q > 0.0)
        .map(|(p, q)| Level::new(*p, *q))
        .collect();
    if bids {
        levels.sort_by(|a, b| b.price.total_cmp(&a.price));
    } else {
        levels.sort_by(|a, b| a.price.total_cmp(&b.price));
    }
    levels.truncate(BOOK_LEVELS);
    levels
}

fn apply_updates(side: &mut Vec<Level>, updates: &[(f64, f64)], bids: bool) {
    for (price, qty) in updates {
        if let Some(pos) = side.iter().position(|l| prices_eq(l.price, *price)) {
            if *qty <= 0.0 {
                side.remove(pos);
            } else {
                side[pos].qty = *qty;
            }
        } else if *qty > 0.0 {
            side.push(Level::new(*price, *qty));
        }
    }
    if bids {
        side.sort_by(|a, b| b.price.total_cmp(&a.price));
    } else {
        side.sort_by(|a, b| a.price.total_cmp(&b.price));
    }
    side.truncate(BOOK_LEVELS);
}

fn prices_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

pub fn parse_tick_size(raw: &str) -> Option<f64> {
    let d: Decimal = raw.parse().ok()?;
    let v = d.to_f64()?;
    if v > 0.0 { Some(v) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lvl(n: i32, px0: f64, step: f64, bid: bool) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let px = if bid {
                    px0 - step * i as f64
                } else {
                    px0 + step * i as f64
                };
                (px, 1.0 + i as f64)
            })
            .collect()
    }

    #[test]
    fn snapshot_builds_50_and_top10() {
        let mut book = LocalBook::new();
        let bids = lvl(50, 100.0, 0.1, true);
        let asks = lvl(50, 100.1, 0.1, false);
        assert_eq!(book.apply_snapshot(&bids, &asks, 10), BookApply::Replaced);
        assert_eq!(book.bids().len(), 50);
        assert_eq!(book.asks().len(), 50);
        let top = book.top10().expect("top10");
        assert!((top.bids[0].price - 100.0).abs() < 1e-9);
        assert!((top.asks[0].price - 100.1).abs() < 1e-9);
        assert_eq!(top.bids.len(), 10);
        assert!(book.is_valid());
    }

    #[test]
    fn incremental_applies_u_u() {
        let mut book = LocalBook::new();
        book.apply_snapshot(&lvl(10, 100.0, 0.1, true), &lvl(10, 100.1, 0.1, false), 5);
        let r = book.apply_incremental(&[(100.0, 9.0)], &[(100.1, 0.0), (100.15, 3.0)], 6, 8);
        assert_eq!(r, BookApply::Applied);
        assert_eq!(book.last_u(), Some(8));
        let top = book.top10().unwrap();
        assert!((top.bids[0].qty - 9.0).abs() < 1e-9);
        assert!((top.asks[0].price - 100.15).abs() < 1e-9);
    }

    #[test]
    fn duplicate_snapshot_replaces() {
        let mut book = LocalBook::new();
        book.apply_snapshot(&[(100.0, 1.0); 10], &[(101.0, 1.0); 10], 1);
        book.apply_snapshot(
            &lvl(10, 200.0, 0.1, true),
            &lvl(10, 200.1, 0.1, false),
            99,
        );
        let top = book.top10().unwrap();
        assert!((top.bids[0].price - 200.0).abs() < 1e-9);
        assert_eq!(book.last_u(), Some(99));
    }

    #[test]
    fn sequence_gap_discards() {
        let mut book = LocalBook::new();
        book.apply_snapshot(&lvl(10, 100.0, 0.1, true), &lvl(10, 100.1, 0.1, false), 10);
        let r = book.apply_incremental(&[(100.0, 2.0)], &[], 12, 12);
        assert_eq!(r, BookApply::Gap);
        assert!(!book.is_valid());
        assert!(book.top10().is_none());
        assert!(book.last_u().is_none());
    }
}
