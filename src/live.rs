use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use serde::Serialize;
use tokio::sync::watch;

use crate::signal::engine::EngineSnapshot;
use crate::signal::factors::FACTOR_NAMES;
use crate::signal::model::now_ms;
use crate::types::{
    DEPTH_GRANULARITY_MS, DEPTH_GRANULARITY_NOTE, FEATURE_COUNT, FittedModel, Horizon, RunPhase,
};

#[derive(Clone, Debug, Serialize)]
pub struct StatusBody {
    pub phase: RunPhase,
    pub training: bool,
    pub connection: String,
    pub data_age_ms: Option<u64>,
    pub model_version: String,
    pub model_status: String,
    pub next_train_ms: i64,
    pub tick_size: Option<f64>,
    pub mid: Option<f64>,
    pub preds: [f64; 3],
    pub alarm: Option<String>,
    pub depth_granularity_ms: u32,
    pub depth_granularity_note: String,
    pub symbol: String,
    pub feature_version: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct StreamFrame {
    pub ts_ms: i64,
    pub mid: Option<f64>,
    pub tick_size: Option<f64>,
    pub preds: [f64; 3],
    pub factors: [f64; FEATURE_COUNT],
    pub factor_names: [&'static str; FEATURE_COUNT],
    pub weights: [[f64; FEATURE_COUNT]; 3],
    pub contributions: [[f64; FEATURE_COUNT]; 3],
    pub quality: String,
    pub model_version: String,
    pub phase: RunPhase,
}

pub struct LiveState {
    phase: Mutex<RunPhase>,
    connection: Mutex<String>,
    alarm: Mutex<Option<String>>,
    last_event: Mutex<Option<Instant>>,
    last_sample_ms: AtomicI64,
    next_train_ms: AtomicI64,
    training: AtomicBool,
    publish_seq: AtomicU64,
    snapshot: RwLock<EngineSnapshot>,
    status_tx: watch::Sender<StatusBody>,
    stream_tx: watch::Sender<StreamFrame>,
    symbol: String,
}

impl LiveState {
    pub fn new(model: FittedModel) -> Self {
        let snap = EngineSnapshot {
            quality: crate::types::DataQuality::NoTickSize,
            mid: None,
            tick_size: None,
            factors: [0.0; FEATURE_COUNT],
            raw: [0.0; FEATURE_COUNT],
            preds: [0.0; 3],
            model_version: model.version.clone(),
            model: model.clone(),
            event_at: Instant::now(),
        };
        let status = status_from(
            &snap,
            RunPhase::Warming,
            false,
            "disconnected",
            None,
            now_ms() + 24 * 3600 * 1000,
            None,
            "BTC-USDT",
        );
        let frame = frame_from(&snap, RunPhase::Warming);
        let (status_tx, _) = watch::channel(status);
        let (stream_tx, _) = watch::channel(frame);
        Self {
            phase: Mutex::new(RunPhase::Warming),
            connection: Mutex::new("disconnected".into()),
            alarm: Mutex::new(None),
            last_event: Mutex::new(None),
            last_sample_ms: AtomicI64::new(0),
            next_train_ms: AtomicI64::new(now_ms() + 24 * 3600 * 1000),
            training: AtomicBool::new(false),
            publish_seq: AtomicU64::new(0),
            snapshot: RwLock::new(snap),
            status_tx,
            stream_tx,
            symbol: "BTC-USDT".into(),
        }
    }

    pub fn subscribe_status(&self) -> watch::Receiver<StatusBody> {
        self.status_tx.subscribe()
    }

    pub fn subscribe_stream(&self) -> watch::Receiver<StreamFrame> {
        self.stream_tx.subscribe()
    }

    pub fn set_phase(&self, p: RunPhase) {
        *self.phase.lock().expect("phase") = p;
        self.broadcast_status();
    }

    pub fn phase(&self) -> RunPhase {
        *self.phase.lock().expect("phase")
    }

    pub fn set_connection(&self, s: &str) {
        *self.connection.lock().expect("conn") = s.to_string();
    }

    pub fn set_alarm(&self, msg: &str) {
        *self.alarm.lock().expect("alarm") = Some(msg.to_string());
        self.broadcast_status();
    }

    pub fn clear_alarm(&self) {
        *self.alarm.lock().expect("alarm") = None;
    }

    pub fn set_training(&self, on: bool) {
        self.training.store(on, Ordering::SeqCst);
        self.broadcast_status();
    }

    pub fn training(&self) -> bool {
        self.training.load(Ordering::SeqCst)
    }

    pub fn set_next_train_ms(&self, ms: i64) {
        self.next_train_ms.store(ms, Ordering::SeqCst);
        self.broadcast_status();
    }

    pub fn next_train_ms(&self) -> i64 {
        self.next_train_ms.load(Ordering::SeqCst)
    }

    pub fn note_sample(&self) {
        self.last_sample_ms.store(now_ms(), Ordering::SeqCst);
    }

    pub fn publish_from_engine(&self, snap: &EngineSnapshot, training: bool) {
        {
            let mut g = self.snapshot.write().expect("snap");
            *g = snap.clone();
        }
        self.training.store(training, Ordering::SeqCst);
        *self.last_event.lock().expect("evt") = Some(Instant::now());
        self.publish_seq.fetch_add(1, Ordering::Relaxed);
        let phase = self.phase();
        let _ = self.stream_tx.send(frame_from(snap, phase));
        self.broadcast_status();
    }

    pub fn status(&self) -> StatusBody {
        let snap = self.snapshot.read().expect("snap").clone();
        status_from(
            &snap,
            self.phase(),
            self.training.load(Ordering::SeqCst),
            &self.connection.lock().expect("conn"),
            self.data_age_ms(),
            self.next_train_ms.load(Ordering::SeqCst),
            self.alarm.lock().expect("alarm").clone(),
            &self.symbol,
        )
    }

    pub fn data_age_ms(&self) -> Option<u64> {
        self.last_event
            .lock()
            .expect("evt")
            .map(|t| t.elapsed().as_millis() as u64)
    }

    fn broadcast_status(&self) {
        let _ = self.status_tx.send(self.status());
    }
}

fn status_from(
    snap: &EngineSnapshot,
    phase: RunPhase,
    training: bool,
    connection: &str,
    data_age_ms: Option<u64>,
    next_train_ms: i64,
    alarm: Option<String>,
    symbol: &str,
) -> StatusBody {
    StatusBody {
        phase,
        training,
        connection: connection.to_string(),
        data_age_ms,
        model_version: snap.model_version.clone(),
        model_status: snap.model.status.as_str().to_string(),
        next_train_ms,
        tick_size: snap.tick_size,
        mid: snap.mid,
        preds: snap.preds,
        alarm,
        depth_granularity_ms: DEPTH_GRANULARITY_MS,
        depth_granularity_note: DEPTH_GRANULARITY_NOTE.to_string(),
        symbol: symbol.to_string(),
        feature_version: crate::types::FEATURE_VERSION.to_string(),
    }
}

fn frame_from(snap: &EngineSnapshot, phase: RunPhase) -> StreamFrame {
    let mut weights = [[0.0; FEATURE_COUNT]; 3];
    let mut contributions = [[0.0; FEATURE_COUNT]; 3];
    for (i, h) in Horizon::all().iter().enumerate() {
        let (_, w) = snap.model.params(*h).scale_and_weights();
        weights[i] = w;
        for k in 0..FEATURE_COUNT {
            contributions[i][k] = snap.model.params(*h).beta[k] * snap.factors[k];
        }
    }
    StreamFrame {
        ts_ms: now_ms(),
        mid: snap.mid,
        tick_size: snap.tick_size,
        preds: snap.preds,
        factors: snap.factors,
        factor_names: FACTOR_NAMES,
        weights,
        contributions,
        quality: snap.quality.as_str().to_string(),
        model_version: snap.model_version.clone(),
        phase,
    }
}

pub fn shared(model: FittedModel) -> Arc<LiveState> {
    Arc::new(LiveState::new(model))
}
