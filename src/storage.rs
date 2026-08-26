use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

use crate::eval::CompletedSample;
use crate::types::{
    FEATURE_COUNT, FEATURE_VERSION, FittedModel, Horizon, HorizonParams, ModelStatus,
};

#[derive(Clone)]
pub struct Storage {
    conn: Arc<Mutex<Connection>>,
    dir_thresh: f64,
}

pub enum WriteCmd {
    Sample(CompletedSample),
    Model(FittedModel),
    Cleanup { retain_days: i64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleRow {
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

#[derive(Clone, Debug, Serialize, Default)]
pub struct HorizonMetrics {
    pub horizon: String,
    pub n_total: i64,
    pub n_valid: i64,
    pub coverage: f64,
    pub ic: Option<f64>,
    pub mae: Option<f64>,
    pub rmse: Option<f64>,
    pub dir_hits: i64,
    pub dir_n: i64,
    pub dir_acc: Option<f64>,
    pub buckets: Vec<BucketMean>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BucketMean {
    pub lo: f64,
    pub hi: f64,
    pub mean_actual: f64,
    pub n: i64,
}

#[derive(Clone, Debug, Default)]
pub struct MetricAcc {
    pub n_total: i64,
    pub n_valid: i64,
    pub sum_p: f64,
    pub sum_a: f64,
    pub sum_pp: f64,
    pub sum_aa: f64,
    pub sum_pa: f64,
    pub sum_abs: f64,
    pub sum_sq: f64,
    pub dir_hits: i64,
    pub dir_n: i64,
}

impl MetricAcc {
    pub fn add(&mut self, pred: Option<f64>, actual: Option<f64>, dir_thresh: f64) {
        self.n_total += 1;
        let (Some(p), Some(a)) = (pred, actual) else {
            return;
        };
        self.n_valid += 1;
        self.sum_p += p;
        self.sum_a += a;
        self.sum_pp += p * p;
        self.sum_aa += a * a;
        self.sum_pa += p * a;
        self.sum_abs += (p - a).abs();
        self.sum_sq += (p - a) * (p - a);
        if a != 0.0 && p.abs() >= dir_thresh {
            self.dir_n += 1;
            if p.signum() == a.signum() {
                self.dir_hits += 1;
            }
        }
    }

    pub fn coverage(&self) -> f64 {
        if self.n_total == 0 {
            0.0
        } else {
            self.n_valid as f64 / self.n_total as f64
        }
    }

    pub fn pearson_ic(&self) -> Option<f64> {
        pearson(self.n_valid, self.sum_p, self.sum_a, self.sum_pp, self.sum_aa, self.sum_pa)
    }

    pub fn mae(&self) -> Option<f64> {
        if self.n_valid == 0 {
            None
        } else {
            Some(self.sum_abs / self.n_valid as f64)
        }
    }

    pub fn rmse(&self) -> Option<f64> {
        if self.n_valid == 0 {
            None
        } else {
            Some((self.sum_sq / self.n_valid as f64).sqrt())
        }
    }

    pub fn dir_acc(&self) -> Option<f64> {
        if self.dir_n == 0 {
            None
        } else {
            Some(self.dir_hits as f64 / self.dir_n as f64)
        }
    }

    pub fn to_metrics(&self, horizon: Horizon) -> HorizonMetrics {
        HorizonMetrics {
            horizon: horizon.as_str().to_string(),
            n_total: self.n_total,
            n_valid: self.n_valid,
            coverage: self.coverage(),
            ic: self.pearson_ic(),
            mae: self.mae(),
            rmse: self.rmse(),
            dir_hits: self.dir_hits,
            dir_n: self.dir_n,
            dir_acc: self.dir_acc(),
            buckets: Vec::new(),
        }
    }
}

pub fn pearson(n: i64, sum_p: f64, sum_a: f64, sum_pp: f64, sum_aa: f64, sum_pa: f64) -> Option<f64> {
    if n < 2 {
        return None;
    }
    let n = n as f64;
    let num = n * sum_pa - sum_p * sum_a;
    let den = ((n * sum_pp - sum_p * sum_p).max(0.0) * (n * sum_aa - sum_a * sum_a).max(0.0)).sqrt();
    if den == 0.0 {
        None
    } else {
        Some(num / den)
    }
}

pub fn directional_hit(pred: f64, actual: f64, thresh: f64) -> Option<bool> {
    if actual == 0.0 || pred.abs() < thresh {
        None
    } else {
        Some(pred.signum() == actual.signum())
    }
}

impl Storage {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        let conn = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        init_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            dir_thresh: 0.05,
        })
    }

    pub fn with_dir_thresh(mut self, dir_thresh: f64) -> Self {
        self.dir_thresh = dir_thresh;
        self
    }

    pub fn writer_loop(&self, mut rx: mpsc::Receiver<WriteCmd>, degraded: watch::Sender<Option<String>>) {
        while let Some(cmd) = rx.blocking_recv() {
            let res = match cmd {
                WriteCmd::Sample(s) => self.persist_sample(&s),
                WriteCmd::Model(m) => self.insert_model(&m),
                WriteCmd::Cleanup { retain_days } => self.cleanup(retain_days),
            };
            match res {
                Ok(()) => {
                    let _ = degraded.send(None);
                }
                Err(e) => {
                    tracing::error!(error = %e, "sqlite write failed");
                    let _ = degraded.send(Some(e.to_string()));
                }
            }
        }
    }

    pub fn persist_sample(&self, s: &CompletedSample) -> anyhow::Result<()> {
        let mut conn = self.conn.lock().expect("sqlite lock");
        let tx = conn.transaction()?;
        let factors = serde_json::to_string(&s.factors)?;
        tx.execute(
            "INSERT OR REPLACE INTO evaluation_samples
             (ts_ms, mid, tick_size, factors_json, model_version, feature_version,
              pred_10, pred_25, pred_50, actual_10, actual_25, actual_50, quality)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            params![
                s.ts_ms,
                s.mid,
                s.tick_size,
                factors,
                s.model_version,
                s.feature_version,
                s.pred[0],
                s.pred[1],
                s.pred[2],
                s.actual[0],
                s.actual[1],
                s.actual[2],
                s.quality,
            ],
        )?;
        upsert_minute(&tx, s, self.dir_thresh)?;
        tx.commit()?;
        Ok(())
    }

    pub fn insert_model(&self, m: &FittedModel) -> anyhow::Result<()> {
        let conn = self.conn.lock().expect("sqlite lock");
        let (s10, w10) = m.h10.scale_and_weights();
        let (s25, w25) = m.h25.scale_and_weights();
        let (s50, w50) = m.h50.scale_and_weights();
        let coef = serde_json::json!({
            "h10": {"intercept": m.h10.intercept, "beta": m.h10.beta, "scale": s10, "weights": w10},
            "h25": {"intercept": m.h25.intercept, "beta": m.h25.beta, "scale": s25, "weights": w25},
            "h50": {"intercept": m.h50.intercept, "beta": m.h50.beta, "scale": s50, "weights": w50},
        });
        conn.execute(
            "INSERT OR REPLACE INTO model_versions
             (version, status, feature_version, train_start_ms, train_end_ms,
              valid_start_ms, valid_end_ms, coef_json, train_params_json, metrics_json,
              created_at_ms, activated_at_ms, depth_granularity_note)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            params![
                m.version,
                m.status.as_str(),
                m.feature_version,
                m.train_start_ms,
                m.train_end_ms,
                m.valid_start_ms,
                m.valid_end_ms,
                coef.to_string(),
                m.train_params_json,
                m.metrics_json,
                m.created_at_ms,
                m.activated_at_ms,
                m.depth_granularity_note,
            ],
        )?;
        Ok(())
    }

    pub fn load_last_active_model(&self) -> anyhow::Result<Option<FittedModel>> {
        let conn = self.conn.lock().expect("sqlite lock");
        let mut stmt = conn.prepare(
            "SELECT version, status, feature_version, train_start_ms, train_end_ms,
                    valid_start_ms, valid_end_ms, coef_json, train_params_json, metrics_json,
                    created_at_ms, activated_at_ms, depth_granularity_note
             FROM model_versions
             WHERE status IN ('active','provisional')
             ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'provisional' THEN 1 ELSE 2 END,
                      COALESCE(activated_at_ms, created_at_ms) DESC
             LIMIT 1",
        )?;
        let row = stmt
            .query_row([], |r| {
                Ok(ModelRow {
                    version: r.get(0)?,
                    status: r.get(1)?,
                    feature_version: r.get(2)?,
                    train_start_ms: r.get(3)?,
                    train_end_ms: r.get(4)?,
                    valid_start_ms: r.get(5)?,
                    valid_end_ms: r.get(6)?,
                    coef_json: r.get(7)?,
                    train_params_json: r.get(8)?,
                    metrics_json: r.get(9)?,
                    created_at_ms: r.get(10)?,
                    activated_at_ms: r.get(11)?,
                    depth_granularity_note: r.get(12)?,
                })
            })
            .optional()?;
        Ok(row.map(|r| r.into_model()))
    }

    pub fn list_models(&self) -> anyhow::Result<Vec<FittedModel>> {
        let conn = self.conn.lock().expect("sqlite lock");
        let mut stmt = conn.prepare(
            "SELECT version, status, feature_version, train_start_ms, train_end_ms,
                    valid_start_ms, valid_end_ms, coef_json, train_params_json, metrics_json,
                    created_at_ms, activated_at_ms, depth_granularity_note
             FROM model_versions
             ORDER BY created_at_ms DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ModelRow {
                version: r.get(0)?,
                status: r.get(1)?,
                feature_version: r.get(2)?,
                train_start_ms: r.get(3)?,
                train_end_ms: r.get(4)?,
                valid_start_ms: r.get(5)?,
                valid_end_ms: r.get(6)?,
                coef_json: r.get(7)?,
                train_params_json: r.get(8)?,
                metrics_json: r.get(9)?,
                created_at_ms: r.get(10)?,
                activated_at_ms: r.get(11)?,
                depth_granularity_note: r.get(12)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).map(|r| r.into_model()).collect())
    }

    pub fn load_train_samples(
        &self,
        since_ms: i64,
        feature_version: &str,
    ) -> anyhow::Result<Vec<SampleRow>> {
        let conn = self.conn.lock().expect("sqlite lock");
        let mut stmt = conn.prepare(
            "SELECT ts_ms, mid, tick_size, factors_json, model_version, feature_version,
                    pred_10, pred_25, pred_50, actual_10, actual_25, actual_50, quality
             FROM evaluation_samples
             WHERE ts_ms >= ?1
               AND feature_version = ?2
               AND actual_10 IS NOT NULL
               AND actual_25 IS NOT NULL
               AND actual_50 IS NOT NULL
               AND quality = 'ok'
             ORDER BY ts_ms ASC",
        )?;
        let rows = stmt.query_map(params![since_ms, feature_version], map_sample)?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn earliest_valid_ts(&self) -> anyhow::Result<Option<i64>> {
        let conn = self.conn.lock().expect("sqlite lock");
        let ts: Option<i64> = conn.query_row(
            "SELECT MIN(ts_ms) FROM evaluation_samples
             WHERE quality='ok' AND actual_50 IS NOT NULL AND feature_version=?1",
            params![FEATURE_VERSION],
            |r| r.get(0),
        )?;
        Ok(ts)
    }

    /// Span of valid labeled data: last valid ts − first valid ts.
    pub fn valid_labeled_span_ms(&self) -> anyhow::Result<Option<(i64, i64)>> {
        let conn = self.conn.lock().expect("sqlite lock");
        let row: Option<(Option<i64>, Option<i64>)> = conn
            .query_row(
                "SELECT MIN(ts_ms), MAX(ts_ms) FROM evaluation_samples
                 WHERE quality='ok'
                   AND actual_10 IS NOT NULL
                   AND actual_25 IS NOT NULL
                   AND actual_50 IS NOT NULL
                   AND feature_version=?1",
                params![FEATURE_VERSION],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        Ok(match row {
            Some((Some(lo), Some(hi))) => Some((lo, hi)),
            _ => None,
        })
    }

    pub fn query_samples(&self, from_ms: i64, to_ms: i64, max_points: usize) -> anyhow::Result<Vec<SampleRow>> {
        let conn = self.conn.lock().expect("sqlite lock");
        let mut stmt = conn.prepare(
            "SELECT ts_ms, mid, tick_size, factors_json, model_version, feature_version,
                    pred_10, pred_25, pred_50, actual_10, actual_25, actual_50, quality
             FROM evaluation_samples
             WHERE ts_ms >= ?1 AND ts_ms <= ?2
             ORDER BY ts_ms ASC",
        )?;
        let rows: Vec<SampleRow> = stmt
            .query_map(params![from_ms, to_ms], map_sample)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(downsample(rows, max_points))
    }

    pub fn query_metrics(
        &self,
        from_ms: i64,
        to_ms: i64,
        model_version: Option<&str>,
        dir_thresh: f64,
    ) -> anyhow::Result<[HorizonMetrics; 3]> {
        let conn = self.conn.lock().expect("sqlite lock");
        let sql = if model_version.is_some() {
            "SELECT pred_10, pred_25, pred_50, actual_10, actual_25, actual_50
             FROM evaluation_samples
             WHERE ts_ms >= ?1 AND ts_ms <= ?2 AND model_version = ?3"
        } else {
            "SELECT pred_10, pred_25, pred_50, actual_10, actual_25, actual_50
             FROM evaluation_samples
             WHERE ts_ms >= ?1 AND ts_ms <= ?2"
        };
        let mut stmt = conn.prepare(sql)?;
        let mut acc = [MetricAcc::default(), MetricAcc::default(), MetricAcc::default()];
        let mut buckets: [Vec<(f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        {
            let mut rows = if let Some(v) = model_version {
                stmt.query(params![from_ms, to_ms, v])?
            } else {
                stmt.query(params![from_ms, to_ms])?
            };
            while let Some(r) = rows.next()? {
                let pred = [r.get::<_, Option<f64>>(0)?, r.get(1)?, r.get(2)?];
                let actual = [r.get::<_, Option<f64>>(3)?, r.get(4)?, r.get(5)?];
                for i in 0..3 {
                    acc[i].add(pred[i], actual[i], dir_thresh);
                    if let (Some(p), Some(a)) = (pred[i], actual[i]) {
                        buckets[i].push((p, a));
                    }
                }
            }
        }
        let mut out = [
            acc[0].to_metrics(Horizon::H10),
            acc[1].to_metrics(Horizon::H25),
            acc[2].to_metrics(Horizon::H50),
        ];
        for i in 0..3 {
            out[i].buckets = bucket_means(&buckets[i]);
        }
        Ok(out)
    }

    pub fn query_minute_metrics(
        &self,
        from_ms: i64,
        to_ms: i64,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let conn = self.conn.lock().expect("sqlite lock");
        let mut stmt = conn.prepare(
            "SELECT minute_ms, model_version, horizon, n_total, n_valid,
                    sum_p, sum_a, sum_pp, sum_aa, sum_pa, sum_abs, sum_sq, dir_hits, dir_n
             FROM metric_minutes
             WHERE minute_ms >= ?1 AND minute_ms <= ?2
             ORDER BY minute_ms ASC",
        )?;
        let rows = stmt.query_map(params![from_ms, to_ms], |r| {
            Ok(serde_json::json!({
                "minute_ms": r.get::<_, i64>(0)?,
                "model_version": r.get::<_, String>(1)?,
                "horizon": r.get::<_, String>(2)?,
                "n_total": r.get::<_, i64>(3)?,
                "n_valid": r.get::<_, i64>(4)?,
                "sum_p": r.get::<_, f64>(5)?,
                "sum_a": r.get::<_, f64>(6)?,
                "sum_pp": r.get::<_, f64>(7)?,
                "sum_aa": r.get::<_, f64>(8)?,
                "sum_pa": r.get::<_, f64>(9)?,
                "sum_abs": r.get::<_, f64>(10)?,
                "sum_sq": r.get::<_, f64>(11)?,
                "dir_hits": r.get::<_, i64>(12)?,
                "dir_n": r.get::<_, i64>(13)?,
            }))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn cleanup(&self, retain_days: i64) -> anyhow::Result<()> {
        let cutoff = crate::signal::model::now_ms() - retain_days * 24 * 3600 * 1000;
        let mut conn = self.conn.lock().expect("sqlite lock");
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM evaluation_samples
             WHERE ts_ms < ?1
               AND EXISTS (
                 SELECT 1 FROM metric_minutes m
                 WHERE m.minute_ms = (evaluation_samples.ts_ms / 60000) * 60000
               )",
            params![cutoff],
        )?;
        tx.commit()?;
        Ok(())
    }
}

struct ModelRow {
    version: String,
    status: String,
    feature_version: String,
    train_start_ms: Option<i64>,
    train_end_ms: Option<i64>,
    valid_start_ms: Option<i64>,
    valid_end_ms: Option<i64>,
    coef_json: String,
    train_params_json: String,
    metrics_json: String,
    created_at_ms: i64,
    activated_at_ms: Option<i64>,
    depth_granularity_note: String,
}

impl ModelRow {
    fn into_model(self) -> FittedModel {
        let v: serde_json::Value = serde_json::from_str(&self.coef_json).unwrap_or(serde_json::json!({}));
        FittedModel {
            version: self.version,
            status: ModelStatus::parse(&self.status).unwrap_or(ModelStatus::Provisional),
            feature_version: self.feature_version,
            h10: parse_params(&v["h10"]),
            h25: parse_params(&v["h25"]),
            h50: parse_params(&v["h50"]),
            train_start_ms: self.train_start_ms,
            train_end_ms: self.train_end_ms,
            valid_start_ms: self.valid_start_ms,
            valid_end_ms: self.valid_end_ms,
            train_params_json: self.train_params_json,
            metrics_json: self.metrics_json,
            created_at_ms: self.created_at_ms,
            activated_at_ms: self.activated_at_ms,
            depth_granularity_note: self.depth_granularity_note,
        }
    }
}

fn parse_params(v: &serde_json::Value) -> HorizonParams {
    let intercept = v.get("intercept").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let mut beta = [0.0; FEATURE_COUNT];
    if let Some(arr) = v.get("beta").and_then(|x| x.as_array()) {
        for (i, x) in arr.iter().take(FEATURE_COUNT).enumerate() {
            beta[i] = x.as_f64().unwrap_or(0.0);
        }
    }
    HorizonParams { intercept, beta }
}

fn map_sample(r: &rusqlite::Row<'_>) -> rusqlite::Result<SampleRow> {
    let factors_json: String = r.get(3)?;
    let factors = parse_factors(&factors_json);
    Ok(SampleRow {
        ts_ms: r.get(0)?,
        mid: r.get(1)?,
        tick_size: r.get(2)?,
        factors,
        model_version: r.get(4)?,
        feature_version: r.get(5)?,
        pred: [r.get(6)?, r.get(7)?, r.get(8)?],
        actual: [r.get(9)?, r.get(10)?, r.get(11)?],
        quality: r.get(12)?,
    })
}

fn parse_factors(s: &str) -> [f64; FEATURE_COUNT] {
    let v: Vec<f64> = serde_json::from_str(s).unwrap_or_default();
    let mut out = [0.0; FEATURE_COUNT];
    for (i, x) in v.iter().take(FEATURE_COUNT).enumerate() {
        out[i] = *x;
    }
    out
}

fn upsert_minute(
    tx: &rusqlite::Transaction<'_>,
    s: &CompletedSample,
    dir_thresh: f64,
) -> anyhow::Result<()> {
    let minute = (s.ts_ms / 60_000) * 60_000;
    for (i, h) in Horizon::all().iter().enumerate() {
        let pred = s.pred[i];
        let actual = s.actual[i];
        let mut acc = MetricAcc::default();
        acc.add(pred, actual, dir_thresh);
        tx.execute(
            "INSERT INTO metric_minutes
             (minute_ms, model_version, horizon, n_total, n_valid,
              sum_p, sum_a, sum_pp, sum_aa, sum_pa, sum_abs, sum_sq, dir_hits, dir_n)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)
             ON CONFLICT(minute_ms, model_version, horizon) DO UPDATE SET
               n_total = n_total + excluded.n_total,
               n_valid = n_valid + excluded.n_valid,
               sum_p = sum_p + excluded.sum_p,
               sum_a = sum_a + excluded.sum_a,
               sum_pp = sum_pp + excluded.sum_pp,
               sum_aa = sum_aa + excluded.sum_aa,
               sum_pa = sum_pa + excluded.sum_pa,
               sum_abs = sum_abs + excluded.sum_abs,
               sum_sq = sum_sq + excluded.sum_sq,
               dir_hits = dir_hits + excluded.dir_hits,
               dir_n = dir_n + excluded.dir_n",
            params![
                minute,
                s.model_version,
                h.as_str(),
                acc.n_total,
                acc.n_valid,
                acc.sum_p,
                acc.sum_a,
                acc.sum_pp,
                acc.sum_aa,
                acc.sum_pa,
                acc.sum_abs,
                acc.sum_sq,
                acc.dir_hits,
                acc.dir_n,
            ],
        )?;
    }
    Ok(())
}

fn downsample(rows: Vec<SampleRow>, max_points: usize) -> Vec<SampleRow> {
    if rows.len() <= max_points || max_points == 0 {
        return rows;
    }
    let step = rows.len() as f64 / max_points as f64;
    let mut out = Vec::with_capacity(max_points);
    for i in 0..max_points {
        let idx = ((i as f64) * step).floor() as usize;
        out.push(rows[idx.min(rows.len() - 1)].clone());
    }
    out
}

pub fn bucket_means(pairs: &[(f64, f64)]) -> Vec<BucketMean> {
    if pairs.is_empty() {
        return Vec::new();
    }
    const N: usize = 5;
    let edges = [-f64::INFINITY, -0.5, -0.1, 0.1, 0.5, f64::INFINITY];
    let mut sums = [0.0; N];
    let mut ns = [0i64; N];
    for (p, a) in pairs {
        let b = edges.windows(2).position(|w| *p >= w[0] && *p < w[1]).unwrap_or(N - 1);
        sums[b] += *a;
        ns[b] += 1;
    }
    (0..N)
        .filter(|&i| ns[i] > 0)
        .map(|i| BucketMean {
            lo: if edges[i].is_infinite() { -1e9 } else { edges[i] },
            hi: if edges[i + 1].is_infinite() { 1e9 } else { edges[i + 1] },
            mean_actual: sums[i] / ns[i] as f64,
            n: ns[i],
        })
        .collect()
}

fn init_schema(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS evaluation_samples (
            ts_ms INTEGER PRIMARY KEY,
            mid REAL NOT NULL,
            tick_size REAL NOT NULL,
            factors_json TEXT NOT NULL,
            model_version TEXT NOT NULL,
            feature_version TEXT NOT NULL,
            pred_10 REAL,
            pred_25 REAL,
            pred_50 REAL,
            actual_10 REAL,
            actual_25 REAL,
            actual_50 REAL,
            quality TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_versions (
            version TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            feature_version TEXT NOT NULL,
            train_start_ms INTEGER,
            train_end_ms INTEGER,
            valid_start_ms INTEGER,
            valid_end_ms INTEGER,
            coef_json TEXT NOT NULL,
            train_params_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            activated_at_ms INTEGER,
            depth_granularity_note TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metric_minutes (
            minute_ms INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            horizon TEXT NOT NULL,
            n_total INTEGER NOT NULL,
            n_valid INTEGER NOT NULL,
            sum_p REAL NOT NULL,
            sum_a REAL NOT NULL,
            sum_pp REAL NOT NULL,
            sum_aa REAL NOT NULL,
            sum_pa REAL NOT NULL,
            sum_abs REAL NOT NULL,
            sum_sq REAL NOT NULL,
            dir_hits INTEGER NOT NULL,
            dir_n INTEGER NOT NULL,
            PRIMARY KEY (minute_ms, model_version, horizon)
        );
        CREATE INDEX IF NOT EXISTS idx_samples_feat ON evaluation_samples(feature_version, ts_ms);
        "#,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::CompletedSample;

    fn sample(ts: i64, feat: &str, pred: [f64; 3], actual: [Option<f64>; 3]) -> CompletedSample {
        CompletedSample {
            ts_ms: ts,
            mid: 100.0,
            tick_size: 0.1,
            factors: [0.1; FEATURE_COUNT],
            model_version: "provisional-cold-start".into(),
            feature_version: feat.into(),
            pred: [Some(pred[0]), Some(pred[1]), Some(pred[2])],
            actual,
            quality: "ok".into(),
        }
    }

    #[test]
    fn persist_sample_fields_and_50ms_complete() {
        let dir = tempfile::tempdir().unwrap();
        let st = Storage::open(&dir.path().join("e.db")).unwrap();
        let mode: String = {
            let c = st.conn.lock().unwrap();
            c.pragma_query_value(None, "journal_mode", |r| r.get(0)).unwrap()
        };
        assert_eq!(mode.to_lowercase(), "wal");
        let s = sample(1_700_000_000_000, FEATURE_VERSION, [0.2, 0.3, 0.4], [Some(0.1), Some(0.2), Some(0.5)]);
        st.persist_sample(&s).unwrap();
        let rows = st.query_samples(0, 9_000_000_000_000, 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ts_ms, s.ts_ms);
        assert_eq!(rows[0].mid, 100.0);
        assert_eq!(rows[0].tick_size, 0.1);
        assert_eq!(rows[0].factors.len(), 17);
        assert_eq!(rows[0].model_version, "provisional-cold-start");
        assert_eq!(rows[0].feature_version, FEATURE_VERSION);
        assert_eq!(rows[0].pred, [Some(0.2), Some(0.3), Some(0.4)]);
        assert_eq!(rows[0].actual, [Some(0.1), Some(0.2), Some(0.5)]);
        assert_eq!(rows[0].quality, "ok");
    }

    #[test]
    fn valid_labeled_span_uses_first_and_last() {
        let dir = tempfile::tempdir().unwrap();
        let st = Storage::open(&dir.path().join("e.db")).unwrap();
        assert!(st.valid_labeled_span_ms().unwrap().is_none());
        st.persist_sample(&sample(1_000, FEATURE_VERSION, [1.0; 3], [Some(1.0); 3])).unwrap();
        st.persist_sample(&sample(5_000, FEATURE_VERSION, [1.0; 3], [Some(1.0); 3])).unwrap();
        let (lo, hi) = st.valid_labeled_span_ms().unwrap().unwrap();
        assert_eq!(lo, 1_000);
        assert_eq!(hi, 5_000);
    }

    #[test]
    fn feature_version_mismatch_excluded_from_train() {
        let dir = tempfile::tempdir().unwrap();
        let st = Storage::open(&dir.path().join("e.db")).unwrap();
        st.persist_sample(&sample(10, FEATURE_VERSION, [1.0; 3], [Some(1.0); 3])).unwrap();
        st.persist_sample(&sample(20, "old_factors", [1.0; 3], [Some(1.0); 3])).unwrap();
        let rows = st.load_train_samples(0, FEATURE_VERSION).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ts_ms, 10);
        assert_eq!(rows[0].feature_version, FEATURE_VERSION);
    }

    #[test]
    fn minute_aggregates_then_30_day_delete() {
        let dir = tempfile::tempdir().unwrap();
        let st = Storage::open(&dir.path().join("e.db")).unwrap();
        let old = sample(1_000, FEATURE_VERSION, [0.2; 3], [Some(0.1); 3]);
        st.persist_sample(&old).unwrap();
        let minutes = st.query_minute_metrics(0, 60_000).unwrap();
        assert!(!minutes.is_empty());
        st.cleanup(30).unwrap();
        let remaining = st.query_samples(0, 10_000, 10).unwrap();
        assert!(remaining.is_empty(), "expired sample should be deleted after minute aggregate exists");
    }

    #[test]
    fn metrics_coverage_ic_mae_rmse_dir_and_buckets() {
        let dir = tempfile::tempdir().unwrap();
        let st = Storage::open(&dir.path().join("e.db")).unwrap();
        // three valid + one missing actual
        st.persist_sample(&sample(1_000, FEATURE_VERSION, [1.0, 1.0, 1.0], [Some(1.0), Some(1.0), Some(1.0)])).unwrap();
        st.persist_sample(&sample(2_000, FEATURE_VERSION, [2.0, 2.0, 2.0], [Some(2.0), Some(2.0), Some(2.0)])).unwrap();
        st.persist_sample(&sample(3_000, FEATURE_VERSION, [3.0, 3.0, 3.0], [Some(3.0), Some(3.0), Some(3.0)])).unwrap();
        let mut missing = sample(4_000, FEATURE_VERSION, [4.0, 4.0, 4.0], [None, None, None]);
        missing.quality = "stale".into();
        st.persist_sample(&missing).unwrap();

        let m = st.query_metrics(0, 10_000, None, 0.05).unwrap();
        let h = &m[0];
        assert_eq!(h.n_total, 4);
        assert_eq!(h.n_valid, 3);
        assert!((h.coverage - 0.75).abs() < 1e-12);
        let ic = h.ic.unwrap();
        assert!((ic - 1.0).abs() < 1e-9, "perfect correlation, got {ic}");
        assert!((h.mae.unwrap() - 0.0).abs() < 1e-12);
        assert!((h.rmse.unwrap() - 0.0).abs() < 1e-12);
        assert_eq!(h.dir_n, 3);
        assert_eq!(h.dir_hits, 3);
        assert!((h.dir_acc.unwrap() - 1.0).abs() < 1e-12);
        assert!(!h.buckets.is_empty());
        let mean: f64 = h.buckets.iter().map(|b| b.mean_actual * b.n as f64).sum::<f64>()
            / h.buckets.iter().map(|b| b.n as f64).sum::<f64>();
        assert!((mean - 2.0).abs() < 1e-12);
    }

    #[test]
    fn directional_only_when_actual_nonzero_and_pred_above_threshold() {
        assert_eq!(directional_hit(0.2, 1.0, 0.05), Some(true));
        assert_eq!(directional_hit(-0.2, 1.0, 0.05), Some(false));
        assert_eq!(directional_hit(0.2, 0.0, 0.05), None);
        assert_eq!(directional_hit(0.01, 1.0, 0.05), None);
    }
}
