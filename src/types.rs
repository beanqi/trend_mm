use serde::{Deserialize, Serialize};

pub const FEATURE_COUNT: usize = 17;
pub const FEATURE_VERSION: &str = "factors_1_17_trade_flow_v2";
pub const DEPTH_GRANULARITY_MS: u32 = 20;
pub const DEPTH_GRANULARITY_NOTE: &str =
    "Gate futures.obu 50-level stream pushes ~20ms batches; 10ms order-flow factors are not true 10ms event-time.";

pub const HORIZONS_MS: [u64; 3] = [10, 25, 50];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunPhase {
    Warming,
    Running,
    Degraded,
}

impl RunPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Warming => "warming",
            Self::Running => "running",
            Self::Degraded => "degraded",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    Provisional,
    Active,
    Rejected,
}

impl ModelStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Provisional => "provisional",
            Self::Active => "active",
            Self::Rejected => "rejected",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "provisional" => Some(Self::Provisional),
            "active" => Some(Self::Active),
            "rejected" => Some(Self::Rejected),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Horizon {
    H10,
    H25,
    H50,
}

impl Horizon {
    pub fn all() -> [Horizon; 3] {
        [Horizon::H10, Horizon::H25, Horizon::H50]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::H10 => "10ms",
            Self::H25 => "25ms",
            Self::H50 => "50ms",
        }
    }

    pub fn ms(self) -> u64 {
        match self {
            Self::H10 => 10,
            Self::H25 => 25,
            Self::H50 => 50,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::H10 => 0,
            Self::H25 => 1,
            Self::H50 => 2,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "10ms" | "10" | "h10" => Some(Self::H10),
            "25ms" | "25" | "h25" => Some(Self::H25),
            "50ms" | "50" | "h50" => Some(Self::H50),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataQuality {
    Ok,
    Warming,
    IncompleteBook,
    Stale,
    NoTickSize,
}

impl DataQuality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Warming => "warming",
            Self::IncompleteBook => "incomplete_book",
            Self::Stale => "stale",
            Self::NoTickSize => "no_tick_size",
        }
    }

    pub fn allows_valid_sample(self) -> bool {
        self == Self::Ok
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HorizonParams {
    pub intercept: f64,
    pub beta: [f64; FEATURE_COUNT],
}

impl HorizonParams {
    pub fn scale_and_weights(&self) -> (f64, [f64; FEATURE_COUNT]) {
        let scale: f64 = self.beta.iter().sum();
        let mut weights = [0.0; FEATURE_COUNT];
        if scale > 0.0 {
            for i in 0..FEATURE_COUNT {
                weights[i] = self.beta[i] / scale;
            }
        }
        (scale, weights)
    }

    pub fn predict(&self, x: &[f64; FEATURE_COUNT]) -> f64 {
        let mut s = self.intercept;
        for i in 0..FEATURE_COUNT {
            s += self.beta[i] * x[i];
        }
        s
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FittedModel {
    pub version: String,
    pub status: ModelStatus,
    pub feature_version: String,
    pub h10: HorizonParams,
    pub h25: HorizonParams,
    pub h50: HorizonParams,
    pub train_start_ms: Option<i64>,
    pub train_end_ms: Option<i64>,
    pub valid_start_ms: Option<i64>,
    pub valid_end_ms: Option<i64>,
    pub train_params_json: String,
    pub metrics_json: String,
    pub created_at_ms: i64,
    pub activated_at_ms: Option<i64>,
    pub depth_granularity_note: String,
}

impl FittedModel {
    pub fn params(&self, h: Horizon) -> &HorizonParams {
        match h {
            Horizon::H10 => &self.h10,
            Horizon::H25 => &self.h25,
            Horizon::H50 => &self.h50,
        }
    }

    pub fn predict_all(&self, x: &[f64; FEATURE_COUNT]) -> [f64; 3] {
        [
            self.h10.predict(x),
            self.h25.predict(x),
            self.h50.predict(x),
        ]
    }
}

pub const EPS: f64 = 1e-12;
