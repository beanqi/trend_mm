use std::collections::HashSet;

use crate::market::book::{MODEL_LEVELS, Top10};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aggressor {
    Buy,
    Sell,
}

impl Aggressor {
    pub fn from_size_sign(size: f64) -> Option<Self> {
        if size > 0.0 {
            Some(Aggressor::Buy)
        } else if size < 0.0 {
            Some(Aggressor::Sell)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublicTrade {
    pub id: u64,
    pub price: f64,
    pub size: f64,
    pub is_internal: bool,
}

#[derive(Clone, Debug)]
pub struct TradeDeduper {
    seen: HashSet<u64>,
}

impl Default for TradeDeduper {
    fn default() -> Self {
        Self {
            seen: HashSet::new(),
        }
    }
}

impl TradeDeduper {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn accept(&mut self, trade: &PublicTrade) -> Option<(Aggressor, f64)> {
        if trade.is_internal {
            return None;
        }
        if !self.seen.insert(trade.id) {
            return None;
        }
        let agg = Aggressor::from_size_sign(trade.size)?;
        Some((agg, trade.size.abs()))
    }

    pub fn clear(&mut self) {
        self.seen.clear();
    }
}

#[derive(Clone, Debug)]
pub struct PendingTrade {
    pub price: f64,
    pub qty: f64,
    pub aggressor: Aggressor,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MatchedFlow {
    pub add_bid: [f64; MODEL_LEVELS],
    pub add_ask: [f64; MODEL_LEVELS],
    pub cancel_bid: [f64; MODEL_LEVELS],
    pub cancel_ask: [f64; MODEL_LEVELS],
    pub exec_bid: [f64; MODEL_LEVELS],
    pub exec_ask: [f64; MODEL_LEVELS],
    pub n_add_bid: [u32; MODEL_LEVELS],
    pub n_add_ask: [u32; MODEL_LEVELS],
    pub n_cancel_bid: [u32; MODEL_LEVELS],
    pub n_cancel_ask: [u32; MODEL_LEVELS],
    pub n_exec_bid: [u32; MODEL_LEVELS],
    pub n_exec_ask: [u32; MODEL_LEVELS],
}

impl MatchedFlow {
    /// Same-price/side matching: take trade volume from book decreases first;
    /// leftover decrease is treated as cancel.
    pub fn from_delta_and_trades(
        before: &Top10,
        after: &Top10,
        trades: &[PendingTrade],
    ) -> (Self, Vec<PendingTrade>) {
        match_trades(before, after, trades)
    }
}

fn prices_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

fn qty_at(levels: &[crate::market::book::Level; MODEL_LEVELS], price: f64) -> f64 {
    levels
        .iter()
        .find(|l| prices_eq(l.price, price))
        .map(|l| l.qty)
        .unwrap_or(0.0)
}

fn index_of(levels: &[crate::market::book::Level; MODEL_LEVELS], price: f64) -> Option<usize> {
    levels.iter().position(|l| prices_eq(l.price, price))
}

pub fn match_trades(
    before: &Top10,
    after: &Top10,
    trades: &[PendingTrade],
) -> (MatchedFlow, Vec<PendingTrade>) {
    let mut flow = MatchedFlow::default();

    for i in 0..MODEL_LEVELS {
        let old_b = qty_at(&before.bids, after.bids[i].price);
        if after.bids[i].qty > old_b {
            flow.add_bid[i] = after.bids[i].qty - old_b;
            flow.n_add_bid[i] = 1;
        }
        let old_a = qty_at(&before.asks, after.asks[i].price);
        if after.asks[i].qty > old_a {
            flow.add_ask[i] = after.asks[i].qty - old_a;
            flow.n_add_ask[i] = 1;
        }
    }

    let mut dec_bid = [0.0; MODEL_LEVELS];
    let mut dec_ask = [0.0; MODEL_LEVELS];
    for i in 0..MODEL_LEVELS {
        let new_b = qty_at(&after.bids, before.bids[i].price);
        if before.bids[i].qty > new_b {
            dec_bid[i] = before.bids[i].qty - new_b;
        }
        let new_a = qty_at(&after.asks, before.asks[i].price);
        if before.asks[i].qty > new_a {
            dec_ask[i] = before.asks[i].qty - new_a;
        }
    }

    let mut leftover = Vec::new();
    for t in trades {
        let remaining = match t.aggressor {
            Aggressor::Sell => take_from_decrease(
                t,
                &before.bids,
                &mut dec_bid,
                &mut flow.exec_bid,
                &mut flow.n_exec_bid,
            ),
            Aggressor::Buy => take_from_decrease(
                t,
                &before.asks,
                &mut dec_ask,
                &mut flow.exec_ask,
                &mut flow.n_exec_ask,
            ),
        };
        if remaining > 1e-12 {
            leftover.push(PendingTrade {
                price: t.price,
                qty: remaining,
                aggressor: t.aggressor,
            });
        }
    }

    flow.cancel_bid = dec_bid;
    flow.cancel_ask = dec_ask;
    for i in 0..MODEL_LEVELS {
        if flow.cancel_bid[i] > 0.0 {
            flow.n_cancel_bid[i] = 1;
        }
        if flow.cancel_ask[i] > 0.0 {
            flow.n_cancel_ask[i] = 1;
        }
    }
    (flow, leftover)
}

fn take_from_decrease(
    trade: &PendingTrade,
    levels: &[crate::market::book::Level; MODEL_LEVELS],
    dec: &mut [f64; MODEL_LEVELS],
    exec: &mut [f64; MODEL_LEVELS],
    n_exec: &mut [u32; MODEL_LEVELS],
) -> f64 {
    let mut remaining = trade.qty;
    if let Some(i) = index_of(levels, trade.price) {
        let take = remaining.min(dec[i]);
        if take > 0.0 {
            dec[i] -= take;
            exec[i] += take;
            n_exec[i] += 1;
            remaining -= take;
        }
        return remaining;
    }
    if let Some((i, _)) = levels
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (a.price - trade.price).abs().total_cmp(&(b.price - trade.price).abs()))
    {
        let take = remaining.min(dec[i]);
        if take > 0.0 {
            dec[i] -= take;
            exec[i] += take;
            n_exec[i] += 1;
            remaining -= take;
        }
    }
    remaining
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::book::Level;

    fn flat_top(bid_px: f64, ask_px: f64, bid_qty: f64, ask_qty: f64) -> Top10 {
        let mut bids = [Level::new(0.0, 0.0); MODEL_LEVELS];
        let mut asks = [Level::new(0.0, 0.0); MODEL_LEVELS];
        for i in 0..MODEL_LEVELS {
            bids[i] = Level::new(bid_px - i as f64 * 0.1, bid_qty);
            asks[i] = Level::new(ask_px + i as f64 * 0.1, ask_qty);
        }
        Top10 { bids, asks }
    }

    #[test]
    fn trade_dedup_sign_and_internal() {
        let mut d = TradeDeduper::new();
        let buy = PublicTrade {
            id: 1,
            price: 100.0,
            size: 3.0,
            is_internal: false,
        };
        let sell = PublicTrade {
            id: 2,
            price: 100.0,
            size: -2.0,
            is_internal: false,
        };
        let internal = PublicTrade {
            id: 3,
            price: 100.0,
            size: 4.0,
            is_internal: true,
        };
        assert_eq!(d.accept(&buy), Some((Aggressor::Buy, 3.0)));
        assert_eq!(d.accept(&buy), None);
        assert_eq!(d.accept(&sell), Some((Aggressor::Sell, 2.0)));
        assert_eq!(d.accept(&internal), None);
        assert_eq!(Aggressor::from_size_sign(1.0), Some(Aggressor::Buy));
        assert_eq!(Aggressor::from_size_sign(-1.0), Some(Aggressor::Sell));
    }

    #[test]
    fn match_prefers_trade_from_decrease_leftover_is_cancel() {
        let before = flat_top(100.0, 100.1, 10.0, 10.0);
        let mut after = before;
        after.bids[0].qty = 4.0;
        let trades = [PendingTrade {
            price: 100.0,
            qty: 3.0,
            aggressor: Aggressor::Sell,
        }];
        let (flow, left) = MatchedFlow::from_delta_and_trades(&before, &after, &trades);
        assert!(left.is_empty());
        assert!((flow.exec_bid[0] - 3.0).abs() < 1e-9);
        assert!((flow.cancel_bid[0] - 3.0).abs() < 1e-9);
        assert_eq!(flow.exec_ask[0], 0.0);
        assert_eq!(flow.add_bid[0], 0.0);
    }

    #[test]
    fn unmatched_trade_qty_stays_pending_not_forced_exec() {
        let before = flat_top(100.0, 100.1, 10.0, 10.0);
        let mut after = before;
        after.bids[0].qty = 8.0;
        let trades = [PendingTrade {
            price: 100.0,
            qty: 5.0,
            aggressor: Aggressor::Sell,
        }];
        let (flow, left) = MatchedFlow::from_delta_and_trades(&before, &after, &trades);
        assert_eq!(left.len(), 1);
        assert!((left[0].qty - 3.0).abs() < 1e-9);
        assert!((flow.exec_bid[0] - 2.0).abs() < 1e-9);
        assert_eq!(flow.cancel_bid[0], 0.0);
    }

    #[test]
    fn add_is_increase() {
        let before = flat_top(100.0, 100.1, 10.0, 10.0);
        let mut after = before;
        after.asks[0].qty = 15.0;
        let (flow, _) = MatchedFlow::from_delta_and_trades(&before, &after, &[]);
        assert!((flow.add_ask[0] - 5.0).abs() < 1e-9);
        assert_eq!(flow.cancel_ask[0], 0.0);
    }
}
