use crate::types::{
    DEPTH_GRANULARITY_NOTE, FEATURE_COUNT, FEATURE_VERSION, FittedModel, HorizonParams, ModelStatus,
};

/// Original 10/25/50ms weights for factors 1–17 (masses 75/65/55%).
pub const RAW_W10: [f64; FEATURE_COUNT] = [
    0.04, 0.04, 0.06, 0.04, 0.03, 0.04, 0.08, 0.05, 0.05, 0.03, 0.05, 0.07, 0.04, 0.05, 0.03, 0.03,
    0.02,
];
pub const RAW_W25: [f64; FEATURE_COUNT] = [
    0.03, 0.04, 0.04, 0.03, 0.03, 0.03, 0.04, 0.07, 0.04, 0.03, 0.05, 0.04, 0.05, 0.04, 0.03, 0.03,
    0.03,
];
pub const RAW_W50: [f64; FEATURE_COUNT] = [
    0.02, 0.03, 0.03, 0.02, 0.02, 0.03, 0.02, 0.07, 0.03, 0.03, 0.04, 0.02, 0.07, 0.04, 0.04, 0.02,
    0.02,
];

pub const RENORM_10: f64 = 0.75;
pub const RENORM_25: f64 = 0.65;
pub const RENORM_50: f64 = 0.55;

pub const COLD_START_VERSION: &str = "provisional-cold-start";

pub fn renormalize(raw: &[f64; FEATURE_COUNT], mass: f64) -> [f64; FEATURE_COUNT] {
    let mut w = [0.0; FEATURE_COUNT];
    for i in 0..FEATURE_COUNT {
        w[i] = raw[i] / mass;
    }
    w
}

pub fn cold_start_weights() -> [[f64; FEATURE_COUNT]; 3] {
    [
        renormalize(&RAW_W10, RENORM_10),
        renormalize(&RAW_W25, RENORM_25),
        renormalize(&RAW_W50, RENORM_50),
    ]
}

pub fn params_from_weights(weights: &[f64; FEATURE_COUNT]) -> HorizonParams {
    HorizonParams {
        intercept: 0.0,
        beta: *weights,
    }
}

pub fn cold_start_model() -> FittedModel {
    let w = cold_start_weights();
    FittedModel {
        version: COLD_START_VERSION.to_string(),
        status: ModelStatus::Provisional,
        feature_version: FEATURE_VERSION.to_string(),
        h10: params_from_weights(&w[0]),
        h25: params_from_weights(&w[1]),
        h50: params_from_weights(&w[2]),
        train_start_ms: None,
        train_end_ms: None,
        valid_start_ms: None,
        valid_end_ms: None,
        train_params_json: serde_json::json!({
            "kind": "cold_start",
            "intercept": 0.0,
            "scale": 1.0,
            "renorm": [RENORM_10, RENORM_25, RENORM_50],
        })
        .to_string(),
        metrics_json: "{}".into(),
        created_at_ms: now_ms(),
        activated_at_ms: Some(now_ms()),
        depth_granularity_note: DEPTH_GRANULARITY_NOTE.to_string(),
    }
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_renormalized_sums_and_scale() {
        let w = cold_start_weights();
        let masses = [0.75, 0.65, 0.55];
        let raws = [RAW_W10, RAW_W25, RAW_W50];
        for h in 0..3 {
            let sum: f64 = w[h].iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "horizon {h} sum={sum}");
            for i in 0..FEATURE_COUNT {
                assert!((w[h][i] - raws[h][i] / masses[h]).abs() < 1e-15);
            }
        }
        let m = cold_start_model();
        assert_eq!(m.status, ModelStatus::Provisional);
        assert_eq!(m.h10.intercept, 0.0);
        let (scale, _) = m.h10.scale_and_weights();
        assert!((scale - 1.0).abs() < 1e-12);
        assert_eq!(m.h25.intercept, 0.0);
        assert_eq!(m.h50.intercept, 0.0);
    }
}
