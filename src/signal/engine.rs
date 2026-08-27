use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

use crate::config::Config;
use crate::market::book::{BookApply, LocalBook, Top10};
use crate::market::flow::{MatchedFlow, PendingTrade, TradeDeduper};
use crate::market::gate::MarketEvent;
use crate::signal::ewma::EwmaNorm;
use crate::signal::factors::{
    DepthAt, WindowAgg, compute_raw, level_weights,
};
use crate::types::{
    FEATURE_COUNT, FittedModel, DataQuality,
};

const SHORT_WINDOW: Duration = Duration::from_millis(10);
const LONG_WINDOW: Duration = Duration::from_millis(50);
const LONG_GAP: Duration = Duration::from_secs(2);

#[derive(Clone, Debug)]
struct TimedFlow {
    at: Instant,
    flow: MatchedFlow,
    depth_after: DepthAt,
}

#[derive(Clone, Debug)]
struct PendingBookFlow {
    before: Top10,
    flow: MatchedFlow,
    depth_after: DepthAt,
}

pub struct SignalEngine {
    cfg: Config,
    book: LocalBook,
    trades: TradeDeduper,
    pending: Vec<PendingTrade>,
    pending_book: Option<PendingBookFlow>,
    events: VecDeque<TimedFlow>,
    ewma: EwmaNorm,
    model: FittedModel,
    tick_size: Option<f64>,
    last_top: Option<Top10>,
    last_event_at: Option<Instant>,
    last_book_at: Option<Instant>,
    warm_started: Instant,
    interrupted: bool,
    last_raw: [f64; FEATURE_COUNT],
    last_x: [f64; FEATURE_COUNT],
    last_pred: [f64; 3],
    last_mid: Option<f64>,
    weights: [f64; 10],
}

#[derive(Clone, Debug)]
pub struct EngineSnapshot {
    pub quality: DataQuality,
    pub mid: Option<f64>,
    pub tick_size: Option<f64>,
    pub factors: [f64; FEATURE_COUNT],
    pub raw: [f64; FEATURE_COUNT],
    pub preds: [f64; 3],
    pub model_version: String,
    pub model: FittedModel,
    pub event_at: Instant,
}

impl SignalEngine {
    pub fn new(cfg: Config, model: FittedModel) -> Self {
        Self {
            cfg,
            book: LocalBook::new(),
            trades: TradeDeduper::new(),
            pending: Vec::new(),
            pending_book: None,
            events: VecDeque::new(),
            ewma: EwmaNorm::new(),
            model,
            tick_size: None,
            last_top: None,
            last_event_at: None,
            last_book_at: None,
            warm_started: Instant::now(),
            interrupted: true,
            last_raw: [0.0; FEATURE_COUNT],
            last_x: [0.0; FEATURE_COUNT],
            last_pred: [0.0; 3],
            last_mid: None,
            weights: level_weights(),
        }
    }

    pub fn set_model(&mut self, model: FittedModel) {
        self.model = model;
        if let (Some(top), Some(tick)) = (self.last_top, self.tick_size) {
            self.recompute(top, tick, Instant::now(), Duration::ZERO);
        }
    }

    pub fn model(&self) -> &FittedModel {
        &self.model
    }

    pub fn tick_size(&self) -> Option<f64> {
        self.tick_size
    }

    pub fn enter_warming(&mut self) {
        self.interrupted = true;
        self.warm_started = Instant::now();
        self.ewma.reset();
        self.events.clear();
        self.pending.clear();
        self.pending_book = None;
        self.trades.clear();
        self.last_top = None;
        self.last_mid = None;
        self.last_event_at = None;
        self.last_book_at = None;
    }

    pub fn apply(&mut self, ev: MarketEvent, now: Instant) -> bool {
        match ev {
            MarketEvent::TickSize(t) => {
                self.tick_size = Some(t);
                false
            }
            MarketEvent::TickSizeFailed(_) => {
                self.tick_size = None;
                self.enter_warming();
                false
            }
            MarketEvent::Disconnected(_) => {
                self.book.discard();
                self.enter_warming();
                false
            }
            MarketEvent::DepthSnapshot {
                bids, asks, u, ..
            } => {
                self.book.apply_snapshot(&bids, &asks, u);
                self.on_book_replaced(now);
                false
            }
            MarketEvent::DepthIncremental {
                bids,
                asks,
                start_u,
                end_u,
                ..
            } => {
                match self.book.apply_incremental(&bids, &asks, start_u, end_u) {
                    BookApply::Gap => {
                        self.enter_warming();
                        true
                    }
                    BookApply::Applied => {
                        self.on_book_updated(now);
                        false
                    }
                    _ => false,
                }
            }
            MarketEvent::Trade(t) => {
                if let Some((agg, qty)) = self.trades.accept(&t) {
                    self.on_trade(PendingTrade {
                        price: t.price,
                        qty,
                        aggressor: agg,
                    }, now);
                }
                false
            }
        }
    }

    fn on_book_replaced(&mut self, now: Instant) {
        self.events.clear();
        self.pending.clear();
        self.pending_book = None;
        self.ewma.reset();
        self.interrupted = true;
        self.warm_started = now;
        if let Some(top) = self.book.top10() {
            self.last_top = Some(top);
            self.last_mid = Some(top.mid());
            self.last_event_at = Some(now);
            self.last_book_at = Some(now);
            if let Some(tick) = self.tick_size {
                self.recompute(top, tick, now, Duration::ZERO);
            }
        } else {
            self.last_top = None;
            self.last_mid = None;
        }
    }

    fn on_trade(&mut self, trade: PendingTrade, now: Instant) {
        let reference_top = self
            .pending_book
            .as_ref()
            .map(|pending| pending.before)
            .or(self.last_top);
        let matched = self.pending_book.as_mut().is_some_and(|pending| {
            pending
                .flow
                .exclude_trade_from_cancels(&pending.before, &trade)
        });
        if !matched {
            self.pending.push(trade.clone());
        }

        let Some(top) = reference_top else {
            return;
        };
        let flow = MatchedFlow::from_trade(&top, &trade);
        let depth = self
            .last_top
            .map(|current| DepthAt::from_top(&current))
            .unwrap_or_else(|| DepthAt::from_top(&top));
        self.push_flow(now, flow, depth);
        if let Some(tick) = self.tick_size {
            let current = self.last_top.unwrap_or(top);
            let dt = self
                .last_event_at
                .map(|at| now.saturating_duration_since(at))
                .unwrap_or(Duration::from_millis(1));
            self.recompute(current, tick, now, dt);
        }
        self.last_event_at = Some(now);
    }

    fn on_book_updated(&mut self, now: Instant) {
        let Some(after) = self.book.top10() else {
            self.last_top = None;
            return;
        };
        let Some(before) = self.last_top else {
            self.last_top = Some(after);
            self.last_mid = Some(after.mid());
            return;
        };
        if self
            .last_book_at
            .is_some_and(|prev| now.duration_since(prev) > LONG_GAP)
        {
            self.enter_warming();
            self.last_top = Some(after);
            self.last_mid = Some(after.mid());
            self.last_event_at = Some(now);
            self.last_book_at = Some(now);
            return;
        }
        if let Some(pending) = self.pending_book.take().filter(|p| p.flow.has_cancels()) {
            self.push_flow(now, pending.flow, pending.depth_after);
        }

        let depth_after = DepthAt::from_top(&after);
        let mut book_flow = MatchedFlow::from_book_delta(&before, &after);
        let additions = book_flow.take_additions();
        if additions.has_additions() {
            self.push_flow(now, additions, depth_after);
        }
        let mut pending_book = PendingBookFlow {
            before,
            flow: book_flow,
            depth_after,
        };
        for trade in std::mem::take(&mut self.pending) {
            pending_book
                .flow
                .exclude_trade_from_cancels(&pending_book.before, &trade);
        }
        self.pending_book = Some(pending_book);
        self.last_top = Some(after);
        self.last_mid = Some(after.mid());
        if let Some(tick) = self.tick_size {
            let dt = self
                .last_event_at
                .map(|t| now.saturating_duration_since(t))
                .unwrap_or(Duration::from_millis(20));
            self.recompute(after, tick, now, dt);
        }
        self.last_event_at = Some(now);
        self.last_book_at = Some(now);
    }

    fn push_flow(&mut self, now: Instant, flow: MatchedFlow, depth: DepthAt) {
        self.events.push_back(TimedFlow {
            at: now,
            flow,
            depth_after: depth,
        });
        let keep = LONG_WINDOW + Duration::from_millis(20);
        while let Some(front) = self.events.front() {
            if now.duration_since(front.at) > keep {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    fn window(&self, now: Instant, len: Duration) -> WindowAgg {
        let mut agg = WindowAgg::default();
        for ev in &self.events {
            if now.duration_since(ev.at) <= len {
                agg.add_flow(&ev.flow);
            }
        }
        agg
    }

    fn depth_ago(&self, now: Instant, ago: Duration) -> DepthAt {
        let target = now.checked_sub(ago).unwrap_or(now);
        let mut last = self.last_top.map(|t| DepthAt::from_top(&t)).unwrap_or_default();
        for ev in &self.events {
            if ev.at <= target {
                last = ev.depth_after;
            }
        }
        last
    }

    fn recompute(&mut self, top: Top10, tick: f64, now: Instant, dt: Duration) {
        let short = self.window(now, SHORT_WINDOW);
        let long = self.window(now, LONG_WINDOW);
        let depth_s = self.depth_ago(now, SHORT_WINDOW);
        self.last_raw = compute_raw(&top, tick, &short, &long, &depth_s, &self.weights);
        self.last_x = self.ewma.update(&self.last_raw, dt);
        self.last_pred = self.model.predict_all(&self.last_x);
        if self.interrupted && self.ewma.accumulated() >= self.cfg.warm_period && self.tick_size.is_some() && self.book.is_valid() {
            self.interrupted = false;
        }
    }

    pub fn quality(&self) -> DataQuality {
        if self.tick_size.is_none() {
            return DataQuality::NoTickSize;
        }
        if !self.book.is_valid() || self.last_top.is_none() {
            return DataQuality::IncompleteBook;
        }
        if self.interrupted || self.ewma.accumulated() < self.cfg.warm_period {
            return DataQuality::Warming;
        }
        if self.last_book_at.is_some_and(|t| t.elapsed() > LONG_GAP) {
            return DataQuality::Stale;
        }
        DataQuality::Ok
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot {
            quality: self.quality(),
            mid: self.last_mid,
            tick_size: self.tick_size,
            factors: self.last_x,
            raw: self.last_raw,
            preds: self.last_pred,
            model_version: self.model.version.clone(),
            model: self.model.clone(),
            event_at: self.last_event_at.unwrap_or_else(Instant::now),
        }
    }

    pub fn force_warmed_for_test(&mut self) {
        self.interrupted = false;
        self.ewma = {
            let mut e = EwmaNorm::new();
            let raw = self.last_raw;
            e.update(&raw, Duration::from_millis(1));
            for _ in 0..20 {
                e.update(&raw, Duration::from_secs(10));
            }
            e
        };
    }
}

pub async fn apply_and_snapshot(
    engine: &Arc<RwLock<SignalEngine>>,
    ev: MarketEvent,
    now: Instant,
) -> (EngineSnapshot, bool) {
    let mut g = engine.write().await;
    let gap = g.apply(ev, now);
    (g.snapshot(), gap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::book::Level;
    use crate::market::flow::PublicTrade;
    use crate::signal::model::cold_start_model;
    use crate::types::ModelStatus;

    fn cfg() -> Config {
        let mut c = Config::default();
        c.warm_period = Duration::from_millis(50);
        c
    }

    fn full_book(bid1: f64, ask1: f64, bq: f64, aq: f64) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let bids = (0..50)
            .map(|i| (bid1 - i as f64 * 0.1, bq))
            .collect();
        let asks = (0..50)
            .map(|i| (ask1 + i as f64 * 0.1, aq))
            .collect();
        (bids, asks)
    }

    #[test]
    fn warming_refuses_valid_samples_until_ready() {
        let mut e = SignalEngine::new(cfg(), cold_start_model());
        assert_eq!(e.quality(), DataQuality::NoTickSize);
        assert!(!e.quality().allows_valid_sample());
        let now = Instant::now();
        e.apply(MarketEvent::TickSize(0.1), now);
        let (b, a) = full_book(100.0, 100.1, 5.0, 5.0);
        e.apply(
            MarketEvent::DepthSnapshot {
                bids: b,
                asks: a,
                u: 1,
                exch_time_ms: 0,
            },
            now,
        );
        assert_eq!(e.quality(), DataQuality::Warming);
        e.apply(MarketEvent::Disconnected("gap".into()), now);
        assert!(!e.quality().allows_valid_sample());
        assert_eq!(e.model().status, ModelStatus::Provisional);
        for v in e.snapshot().factors {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn reconnect_and_gap_clear_windows() {
        let mut e = SignalEngine::new(cfg(), cold_start_model());
        let now = Instant::now();
        e.apply(MarketEvent::TickSize(0.1), now);
        let (b, a) = full_book(100.0, 100.1, 5.0, 5.0);
        e.apply(
            MarketEvent::DepthSnapshot {
                bids: b.clone(),
                asks: a.clone(),
                u: 1,
                exch_time_ms: 0,
            },
            now,
        );
        e.apply(
            MarketEvent::Trade(PublicTrade {
                id: 1,
                price: 100.0,
                size: -1.0,
                is_internal: false,
            }),
            now,
        );
        let gap = e.apply(
            MarketEvent::DepthIncremental {
                bids: vec![(100.0, 4.0)],
                asks: vec![],
                start_u: 3,
                end_u: 3,
                exch_time_ms: 0,
            },
            now + Duration::from_millis(5),
        );
        assert!(gap);
        assert!(!e.quality().allows_valid_sample());
        assert!(e.events.is_empty());
        let _ = Level::new(0.0, 0.0);
    }

    #[test]
    fn trade_is_immediate_and_matching_decrease_is_not_double_counted() {
        let mut e = SignalEngine::new(cfg(), cold_start_model());
        let now = Instant::now();
        e.apply(MarketEvent::TickSize(0.1), now);
        let (b, a) = full_book(100.0, 100.1, 10.0, 10.0);
        e.apply(
            MarketEvent::DepthSnapshot {
                bids: b,
                asks: a,
                u: 1,
                exch_time_ms: 0,
            },
            now,
        );
        e.apply(
            MarketEvent::Trade(PublicTrade {
                id: 11,
                price: 100.0,
                size: -3.0,
                is_internal: false,
            }),
            now + Duration::from_millis(1),
        );
        assert_eq!(e.pending.len(), 1);
        assert_eq!(e.events.len(), 1, "public trade must affect factors immediately");
        assert!((e.events[0].flow.exec_bid[0] - 3.0).abs() < 1e-9);
        e.apply(
            MarketEvent::DepthIncremental {
                bids: vec![(100.0, 4.0)],
                asks: vec![],
                start_u: 2,
                end_u: 2,
                exch_time_ms: 0,
            },
            now + Duration::from_millis(2),
        );
        assert!(e.pending.is_empty());
        assert_eq!(e.events.len(), 1);
        assert!((e.pending_book.as_ref().unwrap().flow.cancel_bid[0] - 3.0).abs() < 1e-9);
        e.apply(
            MarketEvent::DepthIncremental {
                bids: vec![],
                asks: vec![],
                start_u: 3,
                end_u: 3,
                exch_time_ms: 0,
            },
            now + Duration::from_millis(22),
        );
        assert!((e.events.back().unwrap().flow.cancel_bid[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn trade_larger_than_net_decrease_keeps_full_execution() {
        let mut e = SignalEngine::new(cfg(), cold_start_model());
        let now = Instant::now();
        e.apply(MarketEvent::TickSize(0.1), now);
        let (b, a) = full_book(100.0, 100.1, 10.0, 10.0);
        e.apply(
            MarketEvent::DepthSnapshot {
                bids: b,
                asks: a,
                u: 1,
                exch_time_ms: 0,
            },
            now,
        );
        e.apply(
            MarketEvent::Trade(PublicTrade {
                id: 21,
                price: 100.0,
                size: -5.0,
                is_internal: false,
            }),
            now + Duration::from_millis(1),
        );
        e.apply(
            MarketEvent::DepthIncremental {
                bids: vec![(100.0, 8.0)],
                asks: vec![],
                start_u: 2,
                end_u: 2,
                exch_time_ms: 0,
            },
            now + Duration::from_millis(2),
        );
        assert!(e.pending.is_empty());
        assert!((e.events[0].flow.exec_bid[0] - 5.0).abs() < 1e-9);
        assert_eq!(e.pending_book.as_ref().unwrap().flow.cancel_bid[0], 0.0);
        e.apply(
            MarketEvent::DepthIncremental {
                bids: vec![(100.0, 3.0)],
                asks: vec![],
                start_u: 3,
                end_u: 3,
                exch_time_ms: 0,
            },
            now + Duration::from_millis(3),
        );
        assert!((e.pending_book.as_ref().unwrap().flow.cancel_bid[0] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn trade_arriving_after_book_update_reclassifies_the_decrease() {
        let mut e = SignalEngine::new(cfg(), cold_start_model());
        let now = Instant::now();
        e.apply(MarketEvent::TickSize(0.1), now);
        let (b, a) = full_book(100.0, 100.1, 10.0, 10.0);
        e.apply(
            MarketEvent::DepthSnapshot {
                bids: b,
                asks: a,
                u: 1,
                exch_time_ms: 0,
            },
            now,
        );
        e.apply(
            MarketEvent::DepthIncremental {
                bids: vec![(100.0, 4.0)],
                asks: vec![],
                start_u: 2,
                end_u: 2,
                exch_time_ms: 0,
            },
            now + Duration::from_millis(1),
        );
        assert!((e.pending_book.as_ref().unwrap().flow.cancel_bid[0] - 6.0).abs() < 1e-9);
        e.apply(
            MarketEvent::Trade(PublicTrade {
                id: 30,
                price: 100.0,
                size: -3.0,
                is_internal: false,
            }),
            now + Duration::from_millis(2),
        );
        assert!((e.events.back().unwrap().flow.exec_bid[0] - 3.0).abs() < 1e-9);
        assert!((e.pending_book.as_ref().unwrap().flow.cancel_bid[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn multi_level_buy_sweep_immediately_sets_trade_factors() {
        let mut e = SignalEngine::new(cfg(), cold_start_model());
        let now = Instant::now();
        e.apply(MarketEvent::TickSize(0.1), now);
        let (b, a) = full_book(100.0, 100.1, 10.0, 10.0);
        e.apply(
            MarketEvent::DepthSnapshot {
                bids: b,
                asks: a,
                u: 1,
                exch_time_ms: 0,
            },
            now,
        );
        for (id, price, size) in [(31, 100.1, 8.0), (32, 100.2, 12.0), (33, 100.3, 20.0)] {
            e.apply(
                MarketEvent::Trade(PublicTrade {
                    id,
                    price,
                    size,
                    is_internal: false,
                }),
                now + Duration::from_millis(id - 30),
            );
        }
        let executed: f64 = e
            .events
            .iter()
            .flat_map(|event| event.flow.exec_ask)
            .sum();
        assert!((executed - 40.0).abs() < 1e-9);
        assert!(e.snapshot().raw[11] > 0.99, "10ms trade imbalance must see the sweep");
        assert!(e.snapshot().raw[12] > 0.99, "50ms trade imbalance must see the sweep");
    }
}
