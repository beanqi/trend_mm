use std::time::Duration;

use crate::types::{EPS, FEATURE_COUNT};

const TAU: Duration = Duration::from_secs(120);

#[derive(Clone, Debug)]
pub struct EwmaNorm {
    mean: [f64; FEATURE_COUNT],
    var: [f64; FEATURE_COUNT],
    initialized: bool,
    accumulated: Duration,
}

impl Default for EwmaNorm {
    fn default() -> Self {
        Self {
            mean: [0.0; FEATURE_COUNT],
            var: [0.0; FEATURE_COUNT],
            initialized: false,
            accumulated: Duration::ZERO,
        }
    }
}

impl EwmaNorm {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn accumulated(&self) -> Duration {
        self.accumulated
    }

    pub fn update(&mut self, raw: &[f64; FEATURE_COUNT], dt: Duration) -> [f64; FEATURE_COUNT] {
        if !self.initialized {
            self.mean = *raw;
            self.var = [0.0; FEATURE_COUNT];
            self.initialized = true;
            self.accumulated = Duration::ZERO;
            return [0.0; FEATURE_COUNT];
        }
        let dt = if dt.is_zero() {
            Duration::from_millis(1)
        } else {
            dt
        };
        let alpha = 1.0 - (-dt.as_secs_f64() / TAU.as_secs_f64()).exp();
        for i in 0..FEATURE_COUNT {
            let delta = raw[i] - self.mean[i];
            self.mean[i] += alpha * delta;
            let delta2 = raw[i] - self.mean[i];
            self.var[i] = (1.0 - alpha) * (self.var[i] + alpha * delta * delta2);
        }
        self.accumulated = self.accumulated.saturating_add(dt);
        self.normalize(raw)
    }

    pub fn normalize(&self, raw: &[f64; FEATURE_COUNT]) -> [f64; FEATURE_COUNT] {
        let mut x = [0.0; FEATURE_COUNT];
        if !self.initialized {
            return x;
        }
        for i in 0..FEATURE_COUNT {
            let sigma = self.var[i].max(0.0).sqrt();
            let z = (raw[i] - self.mean[i]) / (sigma + EPS);
            x[i] = (z.clamp(-3.0, 3.0)) / 3.0;
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipped_to_unit_interval() {
        let mut e = EwmaNorm::new();
        let mut raw = [0.0; FEATURE_COUNT];
        e.update(&raw, Duration::from_millis(10));
        raw[0] = 1000.0;
        for _ in 0..50 {
            let x = e.update(&raw, Duration::from_millis(50));
            for v in x {
                assert!((-1.0..=1.0).contains(&v), "{v}");
            }
        }
    }
}
