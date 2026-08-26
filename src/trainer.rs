use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;

use crate::config::Config;
use crate::live::LiveState;
use crate::signal::model::{cold_start_model, now_ms};
use crate::storage::{HorizonMetrics, MetricAcc, SampleRow, Storage, WriteCmd};
use crate::types::{
    DEPTH_GRANULARITY_NOTE, FEATURE_COUNT, FEATURE_VERSION, FittedModel, Horizon, HorizonParams,
    ModelStatus, RunPhase,
};

#[derive(Clone)]
pub struct ModelManager {
    current: Arc<Mutex<FittedModel>>,
    training: Arc<Mutex<bool>>,
}

impl ModelManager {
    pub fn new(model: FittedModel) -> Self {
        Self {
            current: Arc::new(Mutex::new(model)),
            training: Arc::new(Mutex::new(false)),
        }
    }

    pub fn current(&self) -> FittedModel {
        self.current.lock().expect("model").clone()
    }

    pub fn swap_active(&self, mut next: FittedModel) -> FittedModel {
        next.status = ModelStatus::Active;
        next.activated_at_ms = Some(now_ms());
        let mut g = self.current.lock().expect("model");
        *g = next.clone();
        next
    }

    pub fn training(&self) -> bool {
        *self.training.lock().expect("train")
    }

    pub fn set_training(&self, on: bool) {
        *self.training.lock().expect("train") = on;
    }
}

#[derive(Clone, Debug)]
pub struct TrainDecision {
    pub promote: bool,
    pub reason: String,
    pub candidate: FittedModel,
}

pub fn time_split(samples: &[SampleRow]) -> (&[SampleRow], &[SampleRow]) {
    let n = samples.len();
    let cut = (n as f64 * 0.8).floor() as usize;
    let cut = cut.clamp(1, n.saturating_sub(1));
    if n < 2 {
        return (samples, &[]);
    }
    (&samples[..cut], &samples[cut..])
}

pub fn should_train(
    now_ms: i64,
    labeled_span: Option<(i64, i64)>,
    last_train_ms: Option<i64>,
    first_after_ms: i64,
    period_ms: i64,
) -> bool {
    let Some((first, last)) = labeled_span else {
        return false;
    };
    if last.saturating_sub(first) < first_after_ms {
        return false;
    }
    if now_ms < last {
        return false;
    }
    match last_train_ms {
        None => true,
        Some(t) => now_ms - t >= period_ms,
    }
}

pub fn next_train_time(
    now: i64,
    labeled_span: Option<(i64, i64)>,
    last_train_ms: Option<i64>,
    first_after_ms: i64,
    period_ms: i64,
) -> i64 {
    match (labeled_span, last_train_ms) {
        (None, _) => now + first_after_ms,
        (Some((first, last)), None) => {
            let ready_at = first + first_after_ms;
            if last >= first + first_after_ms {
                now
            } else {
                // still accumulating; estimate remaining wall time from current last
                now + (ready_at - last).max(0)
            }
        }
        (_, Some(t)) => t + period_ms,
    }
}

pub fn fit_horizon(
    train: &[SampleRow],
    h: Horizon,
    prior: &HorizonParams,
    lambda: f64,
) -> Option<HorizonParams> {
    let idx = h.index();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for s in train {
        if let Some(y) = s.actual[idx] {
            xs.push(s.factors);
            ys.push(y);
        }
    }
    if xs.len() < 3 {
        return None;
    }
    Some(nonneg_ridge(&xs, &ys, &prior.beta, prior.intercept, lambda))
}

pub fn nonneg_ridge(
    xs: &[[f64; FEATURE_COUNT]],
    ys: &[f64],
    prior: &[f64; FEATURE_COUNT],
    _prior_intercept: f64,
    lambda: f64,
) -> HorizonParams {
    let n = xs.len() as f64;
    let mut xbar = [0.0; FEATURE_COUNT];
    let mut ybar = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        ybar += *y;
        for k in 0..FEATURE_COUNT {
            xbar[k] += x[k];
        }
    }
    ybar /= n;
    for k in 0..FEATURE_COUNT {
        xbar[k] /= n;
    }

    let mut xtx = [[0.0; FEATURE_COUNT]; FEATURE_COUNT];
    let mut xty = [0.0; FEATURE_COUNT];
    for (x, y) in xs.iter().zip(ys.iter()) {
        let mut xc = [0.0; FEATURE_COUNT];
        for k in 0..FEATURE_COUNT {
            xc[k] = x[k] - xbar[k];
        }
        let yc = *y - ybar;
        for k in 0..FEATURE_COUNT {
            xty[k] += xc[k] * yc;
            for j in 0..FEATURE_COUNT {
                xtx[k][j] += xc[k] * xc[j];
            }
        }
    }

    let mut active = [true; FEATURE_COUNT];
    let mut beta = *prior;
    for _ in 0..FEATURE_COUNT + 2 {
        let solved = solve_active(&xtx, &xty, prior, lambda, &active);
        beta = solved;
        let mut clipped = false;
        for k in 0..FEATURE_COUNT {
            if active[k] && beta[k] < 0.0 {
                active[k] = false;
                beta[k] = 0.0;
                clipped = true;
            }
        }
        if !clipped {
            break;
        }
    }
    for k in 0..FEATURE_COUNT {
        beta[k] = beta[k].max(0.0);
    }
    let mut intercept = ybar;
    for k in 0..FEATURE_COUNT {
        intercept -= beta[k] * xbar[k];
    }
    HorizonParams { intercept, beta }
}

fn solve_active(
    xtx: &[[f64; FEATURE_COUNT]; FEATURE_COUNT],
    xty: &[f64; FEATURE_COUNT],
    prior: &[f64; FEATURE_COUNT],
    lambda: f64,
    active: &[bool; FEATURE_COUNT],
) -> [f64; FEATURE_COUNT] {
    let idx: Vec<usize> = (0..FEATURE_COUNT).filter(|&i| active[i]).collect();
    let m = idx.len();
    if m == 0 {
        return [0.0; FEATURE_COUNT];
    }
    let mut a = vec![vec![0.0; m]; m];
    let mut b = vec![0.0; m];
    for (i, &ii) in idx.iter().enumerate() {
        b[i] = xty[ii] + lambda * prior[ii];
        for (j, &jj) in idx.iter().enumerate() {
            a[i][j] = xtx[ii][jj];
            if i == j {
                a[i][j] += lambda;
            }
        }
    }
    let sol = gauss(&mut a, &mut b);
    let mut beta = [0.0; FEATURE_COUNT];
    for (i, &ii) in idx.iter().enumerate() {
        beta[ii] = sol[i];
    }
    beta
}

fn gauss(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        let mut piv = i;
        for r in i + 1..n {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let diag = a[i][i];
        if diag.abs() < 1e-15 {
            continue;
        }
        for c in i..n {
            a[i][c] /= diag;
        }
        b[i] /= diag;
        for r in 0..n {
            if r == i {
                continue;
            }
            let f = a[r][i];
            for c in i..n {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    b.to_vec()
}

pub fn eval_model(model: &FittedModel, valid: &[SampleRow], dir_thresh: f64) -> [HorizonMetrics; 3] {
    let mut acc = [MetricAcc::default(), MetricAcc::default(), MetricAcc::default()];
    for s in valid {
        let pred = model.predict_all(&s.factors);
        for i in 0..3 {
            acc[i].add(Some(pred[i]), s.actual[i], dir_thresh);
        }
    }
    [
        acc[0].to_metrics(Horizon::H10),
        acc[1].to_metrics(Horizon::H25),
        acc[2].to_metrics(Horizon::H50),
    ]
}

pub fn should_promote(
    current: &[HorizonMetrics; 3],
    candidate: &[HorizonMetrics; 3],
    rmse_improve: f64,
    max_ic_drop: f64,
    min_samples: usize,
    min_coverage: f64,
) -> (bool, String) {
    for m in candidate {
        if (m.n_valid as usize) < min_samples {
            return (false, format!("{} too few valid samples {}", m.horizon, m.n_valid));
        }
        if m.coverage < min_coverage {
            return (false, format!("{} coverage {} below {}", m.horizon, m.coverage, min_coverage));
        }
        if m.rmse.is_none() || m.ic.is_none() {
            return (false, format!("{} missing rmse/ic", m.horizon));
        }
    }
    let mut cur_rmse = 0.0;
    let mut cand_rmse = 0.0;
    for i in 0..3 {
        cur_rmse += current[i].rmse.unwrap_or(f64::INFINITY);
        cand_rmse += candidate[i].rmse.unwrap_or(f64::INFINITY);
        let cur_ic = current[i].ic.unwrap_or(0.0);
        let cand_ic = candidate[i].ic.unwrap_or(0.0);
        if cur_ic - cand_ic > max_ic_drop {
            return (
                false,
                format!(
                    "{} IC drop {} > {}",
                    candidate[i].horizon,
                    cur_ic - cand_ic,
                    max_ic_drop
                ),
            );
        }
    }
    cur_rmse /= 3.0;
    cand_rmse /= 3.0;
    if !cand_rmse.is_finite() || !cur_rmse.is_finite() {
        return (false, "rmse not finite".into());
    }
    let improve = (cur_rmse - cand_rmse) / cur_rmse.max(1e-12);
    if improve + 1e-15 < rmse_improve {
        return (
            false,
            format!("mean RMSE improve {improve} < {rmse_improve}"),
        );
    }
    (true, format!("mean RMSE improve {improve}"))
}

pub fn train_candidate(
    samples: &[SampleRow],
    current: &FittedModel,
    lambda: f64,
    cfg: &Config,
) -> Option<TrainDecision> {
    if samples.len() < cfg.min_train_samples {
        return None;
    }
    let (train, valid) = time_split(samples);
    if valid.is_empty() || train.is_empty() {
        return None;
    }
    let prior_src = if current.status == ModelStatus::Provisional {
        &cold_start_model()
    } else {
        current
    };
    let h10 = fit_horizon(train, Horizon::H10, prior_src.params(Horizon::H10), lambda)?;
    let h25 = fit_horizon(train, Horizon::H25, prior_src.params(Horizon::H25), lambda)?;
    let h50 = fit_horizon(train, Horizon::H50, prior_src.params(Horizon::H50), lambda)?;
    for p in [&h10, &h25, &h50] {
        if p.beta.iter().any(|b| !b.is_finite() || *b < -1e-12) {
            return None;
        }
        if !p.intercept.is_finite() {
            return None;
        }
    }
    let mut candidate = FittedModel {
        version: format!("trained-{}", now_ms()),
        status: ModelStatus::Rejected,
        feature_version: FEATURE_VERSION.to_string(),
        h10,
        h25,
        h50,
        train_start_ms: train.first().map(|s| s.ts_ms),
        train_end_ms: train.last().map(|s| s.ts_ms),
        valid_start_ms: valid.first().map(|s| s.ts_ms),
        valid_end_ms: valid.last().map(|s| s.ts_ms),
        train_params_json: serde_json::json!({
            "lambda": lambda,
            "n_train": train.len(),
            "n_valid": valid.len(),
            "prior_version": current.version,
        })
        .to_string(),
        metrics_json: "{}".into(),
        created_at_ms: now_ms(),
        activated_at_ms: None,
        depth_granularity_note: DEPTH_GRANULARITY_NOTE.to_string(),
    };
    let cur_m = eval_model(current, valid, cfg.directional_pred_threshold);
    let cand_m = eval_model(&candidate, valid, cfg.directional_pred_threshold);
    let (ok, reason) = should_promote(
        &cur_m,
        &cand_m,
        cfg.promote_rmse_improve,
        cfg.promote_max_ic_drop,
        (cfg.min_train_samples / 5).max(1),
        cfg.min_coverage,
    );
    candidate.metrics_json = serde_json::json!({
        "current": cur_m,
        "candidate": cand_m,
        "reason": reason,
        "promote": ok,
    })
    .to_string();
    if ok {
        candidate.status = ModelStatus::Active;
    }
    Some(TrainDecision {
        promote: ok,
        reason,
        candidate,
    })
}

pub async fn run_trainer(
    cfg: Config,
    storage: Storage,
    models: ModelManager,
    live: Arc<LiveState>,
    write_tx: mpsc::Sender<WriteCmd>,
) {
    let first_ms = cfg.first_train_after.as_millis() as i64;
    let period_ms = cfg.train_period.as_millis() as i64;
    let window_ms = cfg.train_window.as_millis() as i64;
    let mut last_train: Option<i64> = None;
    let mut tick = tokio::time::interval(Duration::from_secs(30));
    loop {
        tick.tick().await;
        if live.phase() == RunPhase::Degraded {
            continue;
        }
        let now = now_ms();
        let span = storage.valid_labeled_span_ms().ok().flatten();
        live.set_next_train_ms(next_train_time(now, span, last_train, first_ms, period_ms));
        if !should_train(now, span, last_train, first_ms, period_ms) {
            continue;
        }
        models.set_training(true);
        live.set_training(true);
        let since = now - window_ms;
        let storage_c = storage.clone();
        let current = models.current();
        let cfg_c = cfg.clone();
        let decision = tokio::task::spawn_blocking(move || {
            let rows = storage_c.load_train_samples(since, FEATURE_VERSION)?;
            Ok::<_, anyhow::Error>(train_candidate(&rows, &current, cfg_c.ridge_lambda, &cfg_c))
        })
        .await;
        models.set_training(false);
        live.set_training(false);
        match decision {
            Ok(Ok(Some(d))) => {
                last_train = Some(now_ms());
                live.set_next_train_ms(now_ms() + period_ms);
                let mut model = d.candidate;
                if d.promote {
                    model.status = ModelStatus::Active;
                    model.activated_at_ms = Some(now_ms());
                    if let Err(e) = storage.insert_model(&model) {
                        tracing::error!(error = %e, "failed to persist promoted model; keeping current");
                    } else {
                        models.swap_active(model);
                        tracing::info!("promoted model persisted then switched");
                    }
                } else {
                    model.status = ModelStatus::Rejected;
                    tracing::info!(version = %model.version, reason = %d.reason, "rejected candidate");
                    let _ = write_tx.send(WriteCmd::Model(model)).await;
                }
            }
            Ok(Ok(None)) => {
                tracing::info!("training skipped: not enough labeled rows");
            }
            Ok(Err(e)) => {
                last_train = Some(now_ms());
                live.set_next_train_ms(now_ms() + period_ms);
                tracing::error!(error = %e, "training failed; keeping current model");
            }
            Err(e) => {
                last_train = Some(now_ms());
                live.set_next_train_ms(now_ms() + period_ms);
                tracing::error!(error = %e, "training task join failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;

    fn row(ts: i64, x0: f64, y: f64) -> SampleRow {
        let mut factors = [0.0; FEATURE_COUNT];
        factors[0] = x0;
        SampleRow {
            ts_ms: ts,
            mid: 100.0,
            tick_size: 0.1,
            factors,
            model_version: "v".into(),
            feature_version: FEATURE_VERSION.into(),
            pred: [Some(0.0); 3],
            actual: [Some(y), Some(y), Some(y)],
            quality: "ok".into(),
        }
    }

    #[test]
    fn split_is_time_order_80_20() {
        let rows: Vec<_> = (0..10).map(|i| row(i * 1000, 0.0, 0.0)).collect();
        let (tr, va) = time_split(&rows);
        assert_eq!(tr.len(), 8);
        assert_eq!(va.len(), 2);
        assert!(tr.last().unwrap().ts_ms < va.first().unwrap().ts_ms);
        assert!(tr.windows(2).all(|w| w[0].ts_ms <= w[1].ts_ms));
    }

    #[test]
    fn beta_nonnegative_and_prior_cold_start() {
        let mut rows = Vec::new();
        for i in 0..40 {
            let x = (i as f64) / 40.0;
            rows.push(row(i as i64 * 1000, x, 2.0 * x));
        }
        let current = cold_start_model();
        let mut cfg = Config::default();
        cfg.min_train_samples = 5;
        let d = train_candidate(&rows, &current, 0.1, &cfg).unwrap();
        for p in [&d.candidate.h10, &d.candidate.h25, &d.candidate.h50] {
            assert!(p.beta.iter().all(|b| *b >= -1e-12));
            assert!(p.beta[0] > 0.0);
        }
        assert_eq!(d.candidate.h10.beta.len(), FEATURE_COUNT);

        let mut active = d.candidate.clone();
        active.status = ModelStatus::Active;
        let prior_beta = active.h10.beta;
        let d2 = train_candidate(&rows, &active, 10.0, &cfg).unwrap();
        // large lambda keeps the second fit near the active prior, not a fresh cold-start
        let dist_active: f64 = d2.candidate.h10.beta
            .iter()
            .zip(prior_beta.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let cold = cold_start_model().h10.beta;
        let dist_cold: f64 = d2.candidate.h10.beta
            .iter()
            .zip(cold.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(dist_active < dist_cold);
    }

    #[test]
    fn promote_and_reject_rules() {
        fn m(rmse: f64, ic: f64) -> HorizonMetrics {
            HorizonMetrics {
                horizon: "10ms".into(),
                n_total: 100,
                n_valid: 100,
                coverage: 1.0,
                ic: Some(ic),
                mae: Some(rmse),
                rmse: Some(rmse),
                dir_hits: 50,
                dir_n: 100,
                dir_acc: Some(0.5),
                buckets: vec![],
            }
        }
        let cur = [m(1.0, 0.2), m(1.0, 0.2), m(1.0, 0.2)];
        let better = [m(0.98, 0.2), m(0.98, 0.2), m(0.98, 0.2)];
        let (ok, _) = should_promote(&cur, &better, 0.01, 0.01, 10, 0.5);
        assert!(ok);
        let worse_rmse = [m(0.995, 0.2), m(0.995, 0.2), m(0.995, 0.2)];
        let (ok, _) = should_promote(&cur, &worse_rmse, 0.01, 0.01, 10, 0.5);
        assert!(!ok);
        let ic_drop = [m(0.9, 0.18), m(0.9, 0.2), m(0.9, 0.2)];
        let (ok, _) = should_promote(&cur, &ic_drop, 0.01, 0.01, 10, 0.5);
        assert!(!ok);
    }

    #[test]
    fn atomic_switch_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("t.db");
        let st = Storage::open(&db).unwrap();
        let cold = cold_start_model();
        st.insert_model(&cold).unwrap();
        let mgr = ModelManager::new(cold.clone());
        let mut next = cold.clone();
        next.version = "trained-1".into();
        next.h10.intercept = 0.3;
        let swapped = mgr.swap_active(next);
        assert_eq!(mgr.current().version, "trained-1");
        assert_eq!(mgr.current().status, ModelStatus::Active);
        assert_eq!(swapped.status, ModelStatus::Active);
        st.insert_model(&mgr.current()).unwrap();
        let loaded = st.load_last_active_model().unwrap().unwrap();
        assert_eq!(loaded.version, "trained-1");
        assert_eq!(loaded.status, ModelStatus::Active);
        assert!((loaded.h10.intercept - 0.3).abs() < 1e-12);
    }

    #[test]
    fn scheduler_uses_24h_then_6h() {
        let first = 1_000_000;
        let day = 24 * 3600 * 1000;
        let six = 6 * 3600 * 1000;
        // wall clock advanced 24h but only 1s of labeled data — must not train
        assert!(!should_train(first + day, Some((first, first + 1000)), None, day, six));
        // 24h of labeled span — train
        assert!(should_train(first + day, Some((first, first + day)), None, day, six));
        assert!(!should_train(
            first + day + 1,
            Some((first, first + day)),
            Some(first + day),
            day,
            six
        ));
        assert!(should_train(
            first + day + six,
            Some((first, first + day + six)),
            Some(first + day),
            day,
            six
        ));
    }
}
