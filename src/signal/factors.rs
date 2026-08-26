use crate::market::book::{MODEL_LEVELS, Top10};
use crate::market::flow::MatchedFlow;
use crate::types::{EPS, FEATURE_COUNT};

pub const LEVEL_DECAY: f64 = 0.35;
pub const SHORT_MS: f64 = 10.0;
pub const LONG_MS: f64 = 50.0;

pub fn level_weights() -> [f64; MODEL_LEVELS] {
    let mut w = [0.0; MODEL_LEVELS];
    let mut sum = 0.0;
    for i in 0..MODEL_LEVELS {
        w[i] = (-LEVEL_DECAY * i as f64).exp();
        sum += w[i];
    }
    for i in 0..MODEL_LEVELS {
        w[i] /= sum;
    }
    w
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WindowAgg {
    pub add_bid: [f64; MODEL_LEVELS],
    pub add_ask: [f64; MODEL_LEVELS],
    pub cancel_bid: [f64; MODEL_LEVELS],
    pub cancel_ask: [f64; MODEL_LEVELS],
    pub exec_bid: [f64; MODEL_LEVELS],
    pub exec_ask: [f64; MODEL_LEVELS],
    pub n_add_bid: u32,
    pub n_add_ask: u32,
    pub n_cancel_bid: u32,
    pub n_cancel_ask: u32,
    pub n_exec_bid: u32,
    pub n_exec_ask: u32,
}

impl WindowAgg {
    pub fn add_flow(&mut self, f: &MatchedFlow) {
        for i in 0..MODEL_LEVELS {
            self.add_bid[i] += f.add_bid[i];
            self.add_ask[i] += f.add_ask[i];
            self.cancel_bid[i] += f.cancel_bid[i];
            self.cancel_ask[i] += f.cancel_ask[i];
            self.exec_bid[i] += f.exec_bid[i];
            self.exec_ask[i] += f.exec_ask[i];
            self.n_add_bid += f.n_add_bid[i];
            self.n_add_ask += f.n_add_ask[i];
            self.n_cancel_bid += f.n_cancel_bid[i];
            self.n_cancel_ask += f.n_cancel_ask[i];
            self.n_exec_bid += f.n_exec_bid[i];
            self.n_exec_ask += f.n_exec_ask[i];
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DepthAt {
    pub bid: [f64; MODEL_LEVELS],
    pub ask: [f64; MODEL_LEVELS],
}

impl DepthAt {
    pub fn from_top(top: &Top10) -> Self {
        let mut d = Self::default();
        for i in 0..MODEL_LEVELS {
            d.bid[i] = top.bids[i].qty;
            d.ask[i] = top.asks[i].qty;
        }
        d
    }
}

pub fn compute_raw(
    top: &Top10,
    tick: f64,
    short: &WindowAgg,
    long: &WindowAgg,
    depth_t_short: &DepthAt,
    weights: &[f64; MODEL_LEVELS],
) -> [f64; FEATURE_COUNT] {
    let mut f = [0.0; FEATURE_COUNT];
    f[0] = f1(top);
    f[1] = f2(top, weights);
    f[2] = f3(top, tick);
    f[3] = f4(top, tick, weights);
    f[4] = f5(top);
    f[5] = f6(top, tick);
    f[6] = ofi_rate(short, top, weights, SHORT_MS);
    f[7] = ofi_rate(long, top, weights, LONG_MS);
    f[8] = f9(f[6], f[7]);
    f[9] = f10(short, weights);
    f[10] = f11(short, weights);
    f[11] = trade_imbalance(short);
    f[12] = trade_imbalance(long);
    f[13] = f14(short, depth_t_short, weights);
    f[14] = f15(long, weights);
    f[15] = f16(top, long);
    f[16] = f17(short);
    f
}

fn f1(top: &Top10) -> f64 {
    let qb = top.bids[0].qty;
    let qa = top.asks[0].qty;
    (qb - qa) / (qb + qa + EPS)
}

fn f2(top: &Top10, w: &[f64; MODEL_LEVELS]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..MODEL_LEVELS {
        num += w[i] * (top.bids[i].qty - top.asks[i].qty);
        den += w[i] * (top.bids[i].qty + top.asks[i].qty);
    }
    num / (den + EPS)
}

fn f3(top: &Top10, tick: f64) -> f64 {
    let qb = top.bids[0].qty;
    let qa = top.asks[0].qty;
    let a = top.asks[0].price;
    let b = top.bids[0].price;
    let mp = (a * qb + b * qa) / (qb + qa + EPS);
    let m = (a + b) * 0.5;
    (mp - m) / (tick + EPS)
}

fn f4(top: &Top10, tick: f64, w: &[f64; MODEL_LEVELS]) -> f64 {
    let m = top.mid();
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..MODEL_LEVELS {
        let db = ((m - top.bids[i].price) / (tick + EPS)).max(EPS);
        let da = ((top.asks[i].price - m) / (tick + EPS)).max(EPS);
        let tb = top.bids[i].qty / db;
        let ta = top.asks[i].qty / da;
        num += w[i] * (tb - ta);
        den += w[i] * (tb + ta);
    }
    num / (den + EPS)
}

fn f5(top: &Top10) -> f64 {
    let kb = (0..3).map(|i| top.bids[i].qty).sum::<f64>()
        / ((0..MODEL_LEVELS).map(|i| top.bids[i].qty).sum::<f64>() + EPS);
    let ka = (0..3).map(|i| top.asks[i].qty).sum::<f64>()
        / ((0..MODEL_LEVELS).map(|i| top.asks[i].qty).sum::<f64>() + EPS);
    kb - ka
}

fn f6(top: &Top10, tick: f64) -> f64 {
    let m = top.mid();
    let mut sb = 0.0;
    let mut qb = 0.0;
    let mut sa = 0.0;
    let mut qa = 0.0;
    for i in 0..MODEL_LEVELS {
        let db = (m - top.bids[i].price) / (tick + EPS);
        let da = (top.asks[i].price - m) / (tick + EPS);
        sb += top.bids[i].qty * db;
        qb += top.bids[i].qty;
        sa += top.asks[i].qty * da;
        qa += top.asks[i].qty;
    }
    let db = sb / (qb + EPS);
    let da = sa / (qa + EPS);
    (da - db) / (da + db + EPS)
}

fn phi(w: &WindowAgg, i: usize) -> f64 {
    w.add_bid[i] + w.cancel_ask[i] + w.exec_ask[i]
        - w.add_ask[i]
        - w.cancel_bid[i]
        - w.exec_bid[i]
}

fn ofi_rate(w: &WindowAgg, top: &Top10, weights: &[f64; MODEL_LEVELS], tau_ms: f64) -> f64 {
    let mut num = 0.0;
    let mut depth = 0.0;
    for i in 0..MODEL_LEVELS {
        num += weights[i] * phi(w, i);
        depth += weights[i] * (top.bids[i].qty + top.asks[i].qty);
    }
    let tau_s = tau_ms / 1000.0;
    num / (tau_s * depth + EPS)
}

fn f9(f7: f64, f8: f64) -> f64 {
    (f7 - f8) / (f7.abs() + f8.abs() + EPS)
}

fn f10(w: &WindowAgg, weights: &[f64; MODEL_LEVELS]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..MODEL_LEVELS {
        num += weights[i] * (w.add_bid[i] - w.add_ask[i]);
        den += weights[i] * (w.add_bid[i] + w.add_ask[i]);
    }
    num / (den + EPS)
}

fn f11(w: &WindowAgg, weights: &[f64; MODEL_LEVELS]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..MODEL_LEVELS {
        num += weights[i] * (w.cancel_ask[i] - w.cancel_bid[i]);
        den += weights[i] * (w.cancel_ask[i] + w.cancel_bid[i]);
    }
    num / (den + EPS)
}

fn trade_imbalance(w: &WindowAgg) -> f64 {
    let buy: f64 = w.exec_ask.iter().sum();
    let sell: f64 = w.exec_bid.iter().sum();
    (buy - sell) / (buy + sell + EPS)
}

fn f14(w: &WindowAgg, depth: &DepthAt, weights: &[f64; MODEL_LEVELS]) -> f64 {
    let mut depa_n = 0.0;
    let mut depa_d = 0.0;
    let mut depb_n = 0.0;
    let mut depb_d = 0.0;
    for i in 0..MODEL_LEVELS {
        depa_n += weights[i] * (w.cancel_ask[i] + w.exec_ask[i]);
        depa_d += weights[i] * depth.ask[i];
        depb_n += weights[i] * (w.cancel_bid[i] + w.exec_bid[i]);
        depb_d += weights[i] * depth.bid[i];
    }
    let depa = depa_n / (depa_d + EPS);
    let depb = depb_n / (depb_d + EPS);
    (depa - depb) / (depa + depb + EPS)
}

fn f15(w: &WindowAgg, weights: &[f64; MODEL_LEVELS]) -> f64 {
    let mut rb_n = 0.0;
    let mut rb_d = 0.0;
    let mut ra_n = 0.0;
    let mut ra_d = 0.0;
    for i in 0..MODEL_LEVELS {
        rb_n += weights[i] * w.add_bid[i];
        rb_d += weights[i] * (w.cancel_bid[i] + w.exec_bid[i]);
        ra_n += weights[i] * w.add_ask[i];
        ra_d += weights[i] * (w.cancel_ask[i] + w.exec_ask[i]);
    }
    let rb = rb_n / (rb_d + EPS);
    let ra = ra_n / (ra_d + EPS);
    (rb - ra) / (rb + ra + EPS)
}

fn f16(top: &Top10, long: &WindowAgg) -> f64 {
    let tau = LONG_MS / 1000.0;
    let lb = (long.cancel_bid[0] + long.exec_bid[0]) / tau;
    let la = (long.cancel_ask[0] + long.exec_ask[0]) / tau;
    let tb = top.bids[0].qty / (lb + EPS);
    let ta = top.asks[0].qty / (la + EPS);
    (tb - ta) / (tb + ta + EPS)
}

fn f17(w: &WindowAgg) -> f64 {
    let up = w.n_add_bid + w.n_cancel_ask + w.n_exec_ask;
    let down = w.n_add_ask + w.n_cancel_bid + w.n_exec_bid;
    let up = up as f64;
    let down = down as f64;
    (up - down) / (up + down + EPS)
}

pub const FACTOR_NAMES: [&str; FEATURE_COUNT] = [
    "L1 imbalance",
    "10-level imbalance",
    "Microprice offset",
    "Distance-weighted pressure",
    "Near-depth concentration",
    "Mean-distance asymmetry",
    "10ms multi-level OFI",
    "50ms multi-level OFI",
    "OFI acceleration",
    "10ms add imbalance",
    "10ms cancel imbalance",
    "10ms trade imbalance",
    "50ms trade imbalance",
    "10ms depletion imbalance",
    "50ms replenish imbalance",
    "L1 queue exhaust time",
    "10ms event-count imbalance",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::book::Level;

    fn one_sided_bid_heavy() -> Top10 {
        let mut bids = [Level::new(0.0, 0.0); MODEL_LEVELS];
        let mut asks = [Level::new(0.0, 0.0); MODEL_LEVELS];
        for i in 0..MODEL_LEVELS {
            let near = if i < 3 { 20.0 } else { 1.0 };
            let far = if i < 3 { 1.0 } else { 20.0 };
            bids[i] = Level::new(100.0 - i as f64 * 0.1, near);
            asks[i] = Level::new(100.1 + i as f64 * 0.1, far);
        }
        Top10 { bids, asks }
    }

    fn one_sided_ask_heavy() -> Top10 {
        let mut bids = [Level::new(0.0, 0.0); MODEL_LEVELS];
        let mut asks = [Level::new(0.0, 0.0); MODEL_LEVELS];
        for i in 0..MODEL_LEVELS {
            let near = if i < 3 { 20.0 } else { 1.0 };
            let far = if i < 3 { 1.0 } else { 20.0 };
            bids[i] = Level::new(100.0 - i as f64 * 0.1, far);
            asks[i] = Level::new(100.1 + i as f64 * 0.1, near);
        }
        Top10 { bids, asks }
    }

    fn flow_up() -> WindowAgg {
        let mut w = WindowAgg::default();
        w.add_bid[0] = 5.0;
        w.cancel_ask[0] = 4.0;
        w.exec_ask[0] = 3.0;
        w.n_add_bid = 2;
        w.n_cancel_ask = 2;
        w.n_exec_ask = 2;
        w
    }

    fn flow_down() -> WindowAgg {
        let mut w = WindowAgg::default();
        w.add_ask[0] = 5.0;
        w.cancel_bid[0] = 4.0;
        w.exec_bid[0] = 3.0;
        w.n_add_ask = 2;
        w.n_cancel_bid = 2;
        w.n_exec_bid = 2;
        w
    }

    #[test]
    fn documented_signs_when_one_sided() {
        let w = level_weights();
        let up_book = one_sided_bid_heavy();
        let down_book = one_sided_ask_heavy();
        let depth_up = DepthAt::from_top(&up_book);
        let depth_down = DepthAt::from_top(&down_book);
        let up = compute_raw(&up_book, 0.1, &flow_up(), &flow_up(), &depth_up, &w);
        let down = compute_raw(&down_book, 0.1, &flow_down(), &flow_down(), &depth_down, &w);
        for k in 0..FEATURE_COUNT {
            assert!(
                up[k] > 0.0,
                "factor {} should be + on upside pressure, got {}",
                k + 1,
                up[k]
            );
            assert!(
                down[k] < 0.0,
                "factor {} should be - on downside pressure, got {}",
                k + 1,
                down[k]
            );
        }
    }
}
