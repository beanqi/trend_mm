use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::{RwLock, mpsc, watch};

use crate::config::Config;
use crate::live::LiveState;
use crate::market::gate::MarketEvent;
use crate::signal::engine::SignalEngine;
use crate::storage::WriteCmd;
use crate::trainer::ModelManager;
use crate::types::{
    FEATURE_COUNT, FEATURE_VERSION, HORIZONS_MS, DataQuality, Horizon, RunPhase,
};

const MID_KEEP: Duration = Duration::from_secs(5);
const LABEL_STALE: Duration = Duration::from_millis(200);

#[derive(Clone, Copy, Debug)]
pub struct MidPoint {
    pub mono: Instant,
    pub wall_ms: i64,
    pub mid: f64,
}

#[derive(Clone, Debug)]
pub struct PendingSample {
    pub ts_ms: i64,
    pub sampled_at: Instant,
    pub mid: f64,
    pub tick_size: f64,
    pub factors: [f64; FEATURE_COUNT],
    pub model_version: String,
    pub pred: [f64; 3],
    pub quality: DataQuality,
    pub actual: [Option<f64>; 3],
    pub valid: [bool; 3],
    pub done: [bool; 3],
}

#[derive(Clone, Debug)]
pub struct CompletedSample {
    pub ts_ms: i64,
    pub mid: f64,
    pub tick_size: f64,
    pub factors: [f64; FEATURE_COUNT],
    pub model_version: String,
    pub feature_version: String,
    pub pred: [Option<f64>; 3],
    pub actual: [Option<f64>; 3],
    pub quality: String,
}

#[derive(Clone, Copy, Debug)]
pub struct HorizonLabel {
    pub actual: Option<f64>,
    pub valid: bool,
}

/// Last valid mid with time strictly before `target`. No look-ahead.
pub fn label_from_mids(mids: &[MidPoint], sample_mid: f64, tick: f64, target: Instant) -> HorizonLabel {
    let mut last: Option<f64> = None;
    for m in mids {
        if m.mono < target {
            last = Some(m.mid);
        } else {
            break;
        }
    }
    match last {
        Some(mid) => HorizonLabel {
            actual: Some((mid - sample_mid) / tick),
            valid: true,
        },
        None => HorizonLabel {
            actual: None,
            valid: false,
        },
    }
}

pub fn complete_labels(
    pending: &mut PendingSample,
    mids: &[MidPoint],
    now: Instant,
) -> bool {
    for (i, h) in Horizon::all().iter().enumerate() {
        if pending.done[i] {
            continue;
        }
        let due = pending.sampled_at + Duration::from_millis(h.ms());
        if now < due {
            continue;
        }
        if pending.quality.allows_valid_sample() {
            let lab = label_from_mids(mids, pending.mid, pending.tick_size, due);
            if lab.valid {
                pending.actual[i] = lab.actual;
                pending.valid[i] = true;
            } else if now.duration_since(due) > LABEL_STALE {
                pending.actual[i] = None;
                pending.valid[i] = false;
            } else {
                continue;
            }
        } else {
            pending.actual[i] = None;
            pending.valid[i] = false;
        }
        pending.done[i] = true;
    }
    pending.done.iter().all(|d| *d)
}

pub fn to_completed(p: &PendingSample) -> CompletedSample {
    let mut pred = [None; 3];
    let mut actual = [None; 3];
    for i in 0..3 {
        if p.quality.allows_valid_sample() {
            pred[i] = Some(p.pred[i]);
        }
        if p.valid[i] {
            actual[i] = p.actual[i];
        }
    }
    CompletedSample {
        ts_ms: p.ts_ms,
        mid: p.mid,
        tick_size: p.tick_size,
        factors: p.factors,
        model_version: p.model_version.clone(),
        feature_version: FEATURE_VERSION.to_string(),
        pred,
        actual,
        quality: p.quality.as_str().to_string(),
    }
}

pub fn wall_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub async fn run_signal_and_eval(
    cfg: Config,
    engine: Arc<RwLock<SignalEngine>>,
    live: Arc<LiveState>,
    models: ModelManager,
    mut events: mpsc::Receiver<MarketEvent>,
    write_tx: mpsc::Sender<WriteCmd>,
    mut degraded_rx: watch::Receiver<Option<String>>,
) {
    let mut mids: VecDeque<MidPoint> = VecDeque::new();
    let mut pending: VecDeque<PendingSample> = VecDeque::new();
    let mut last_publish = Instant::now() - cfg.live_publish_period;
    let mut sample_tick = tokio::time::interval(cfg.sample_period);
    let mut label_tick = tokio::time::interval(Duration::from_millis(5));

    loop {
        tokio::select! {
            ev = events.recv() => {
                let Some(ev) = ev else { break };
                let now = Instant::now();
                let conn = connection_after_event(&ev, false);
                let mut g = engine.write().await;
                let gap = g.apply(ev, now);
                live.set_connection(if gap { "resubscribing" } else { conn });
                if let Some(mid) = g.snapshot().mid {
                    mids.push_back(MidPoint { mono: now, wall_ms: wall_ms(), mid });
                }
                prune_mids(&mut mids, now);
                let snap = g.snapshot();
                drop(g);
                if last_publish.elapsed() >= cfg.live_publish_period {
                    live.publish_from_engine(&snap, models.training());
                    last_publish = Instant::now();
                }
                let _ = HORIZONS_MS;
            }
            _ = sample_tick.tick() => {
                let now = Instant::now();
                let current_model = models.current();
                let mut g = engine.write().await;
                if g.model().version != current_model.version {
                    g.set_model(current_model);
                }
                let snap = g.snapshot();
                let sqlite_err = degraded_rx.borrow().clone();
                if sqlite_err.is_some() {
                    live.set_phase(RunPhase::Degraded);
                } else if snap.quality.allows_valid_sample() {
                    live.set_phase(RunPhase::Running);
                } else {
                    live.set_phase(RunPhase::Warming);
                }
                let tick = snap.tick_size.unwrap_or(0.0);
                let mid = snap.mid.unwrap_or(0.0);
                pending.push_back(PendingSample {
                    ts_ms: wall_ms(),
                    sampled_at: now,
                    mid,
                    tick_size: tick,
                    factors: snap.factors,
                    model_version: snap.model_version.clone(),
                    pred: snap.preds,
                    quality: snap.quality,
                    actual: [None; 3],
                    valid: [false; 3],
                    done: [false; 3],
                });
                drop(g);
                live.note_sample();
            }
            _ = label_tick.tick() => {
                let now = Instant::now();
                prune_mids(&mut mids, now);
                let mid_slice: Vec<MidPoint> = mids.iter().copied().collect();
                let mut ready = Vec::new();
                let mut i = 0;
                while i < pending.len() {
                    if complete_labels(&mut pending[i], &mid_slice, now) {
                        ready.push(pending.remove(i).unwrap());
                    } else {
                        i += 1;
                    }
                }
                for p in ready {
                    let row = to_completed(&p);
                    if write_tx.send(WriteCmd::Sample(row)).await.is_err() {
                        return;
                    }
                }
            }
            _ = degraded_rx.changed() => {
                if degraded_rx.borrow().is_some() {
                    live.set_phase(RunPhase::Degraded);
                    live.set_alarm("sqlite write failed; training paused");
                } else {
                    live.clear_alarm();
                }
            }
        }
    }
}

/// Connection label after applying a market event. Disconnected / tick-size
/// failure must not be overwritten to "connected".
pub fn connection_after_event(ev: &MarketEvent, gap: bool) -> &'static str {
    if gap {
        "resubscribing"
    } else {
        match ev {
            MarketEvent::Disconnected(_) => "reconnecting",
            MarketEvent::TickSizeFailed(_) => "tick_size_failed",
            _ => "connected",
        }
    }
}

fn prune_mids(mids: &mut VecDeque<MidPoint>, now: Instant) {
    while let Some(front) = mids.front() {
        if now.duration_since(front.mono) > MID_KEEP {
            mids.pop_front();
        } else {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_uses_last_mid_strictly_before_target() {
        let t0 = Instant::now();
        let mids = [
            MidPoint { mono: t0 + Duration::from_millis(0), wall_ms: 0, mid: 100.0 },
            MidPoint { mono: t0 + Duration::from_millis(8), wall_ms: 8, mid: 100.2 },
            MidPoint { mono: t0 + Duration::from_millis(10), wall_ms: 10, mid: 100.5 },
            MidPoint { mono: t0 + Duration::from_millis(12), wall_ms: 12, mid: 101.0 },
        ];
        let target = t0 + Duration::from_millis(10);
        let lab = label_from_mids(&mids, 100.0, 0.1, target);
        assert!(lab.valid);
        // last mid with time < t+10 is 100.2, not 100.5 at exactly t+h and not 101
        assert!((lab.actual.unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn late_schedule_does_not_take_future_mid() {
        let t0 = Instant::now();
        let mids = [
            MidPoint { mono: t0, wall_ms: 0, mid: 50.0 },
            MidPoint { mono: t0 + Duration::from_millis(25), wall_ms: 25, mid: 51.0 },
        ];
        let lab = label_from_mids(&mids, 50.0, 0.1, t0 + Duration::from_millis(25));
        assert!((lab.actual.unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn missing_mid_is_invalid() {
        let t0 = Instant::now();
        let lab = label_from_mids(&[], 100.0, 0.1, t0 + Duration::from_millis(10));
        assert!(!lab.valid);
        assert!(lab.actual.is_none());
    }

    #[test]
    fn one_second_sample_fields_and_50ms_complete_row() {
        let now = Instant::now();
        let mut pending = PendingSample {
            ts_ms: 1_700_000_050_000,
            sampled_at: now,
            mid: 100.0,
            tick_size: 0.1,
            factors: [0.05; FEATURE_COUNT],
            model_version: "provisional-cold-start".into(),
            pred: [0.1, 0.2, 0.3],
            quality: DataQuality::Ok,
            actual: [None; 3],
            valid: [false; 3],
            done: [false; 3],
        };
        let mids = [
            MidPoint { mono: now, wall_ms: pending.ts_ms, mid: 100.0 },
            MidPoint { mono: now + Duration::from_millis(9), wall_ms: 0, mid: 100.1 },
            MidPoint { mono: now + Duration::from_millis(24), wall_ms: 0, mid: 100.2 },
            MidPoint { mono: now + Duration::from_millis(49), wall_ms: 0, mid: 100.4 },
        ];
        assert!(!complete_labels(&mut pending, &mids, now + Duration::from_millis(20)));
        assert!(complete_labels(&mut pending, &mids, now + Duration::from_millis(50)));
        assert!(pending.done.iter().all(|d| *d));
        let row = to_completed(&pending);
        assert_eq!(row.ts_ms, 1_700_000_050_000);
        assert_eq!(row.mid, 100.0);
        assert_eq!(row.tick_size, 0.1);
        assert_eq!(row.factors.len(), 17);
        assert_eq!(row.model_version, "provisional-cold-start");
        assert_eq!(row.feature_version, FEATURE_VERSION);
        assert_eq!(row.pred, [Some(0.1), Some(0.2), Some(0.3)]);
        assert!((row.actual[0].unwrap() - 1.0).abs() < 1e-9);
        assert!((row.actual[1].unwrap() - 2.0).abs() < 1e-9);
        assert!((row.actual[2].unwrap() - 4.0).abs() < 1e-9);
        assert_eq!(row.quality, "ok");
    }

    #[test]
    fn stale_or_missing_horizon_marked_invalid() {
        let now = Instant::now();
        let mut pending = PendingSample {
            ts_ms: 1,
            sampled_at: now,
            mid: 100.0,
            tick_size: 0.1,
            factors: [0.0; FEATURE_COUNT],
            model_version: "v".into(),
            pred: [0.0; 3],
            quality: DataQuality::Ok,
            actual: [None; 3],
            valid: [false; 3],
            done: [false; 3],
        };
        assert!(complete_labels(
            &mut pending,
            &[],
            now + Duration::from_millis(50) + LABEL_STALE + Duration::from_millis(1),
        ));
        let row = to_completed(&pending);
        assert_eq!(row.actual, [None, None, None]);
        assert!(!pending.valid.iter().any(|v| *v));
    }

    #[test]
    fn disconnect_and_tick_size_failure_are_not_connected() {
        assert_eq!(
            connection_after_event(&MarketEvent::Disconnected("ws closed".into()), false),
            "reconnecting"
        );
        assert_eq!(
            connection_after_event(&MarketEvent::TickSizeFailed("http 500".into()), false),
            "tick_size_failed"
        );
        assert_eq!(
            connection_after_event(&MarketEvent::TickSize(0.1), false),
            "connected"
        );
        assert_eq!(
            connection_after_event(&MarketEvent::Disconnected("gap".into()), true),
            "resubscribing"
        );
    }
}
