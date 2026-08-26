pub mod config;
pub mod eval;
pub mod live;
pub mod market;
pub mod signal;
pub mod storage;
pub mod trainer;
pub mod types;
pub mod web;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use tokio::sync::{RwLock, mpsc, watch};

use crate::config::Config;
use crate::live::LiveState;
use crate::market::gate::MarketEvent;
use crate::signal::engine::SignalEngine;
use crate::storage::Storage;
use crate::trainer::ModelManager;
use crate::types::FEATURE_VERSION;

pub async fn run(cfg: Config) -> anyhow::Result<()> {
    tracing::info!(bind = %cfg.http_bind, db = %cfg.db_path.display(), "starting trend_mm");

    let storage = Storage::open(&cfg.db_path)
        .context("open sqlite")?
        .with_dir_thresh(cfg.directional_pred_threshold);
    let loaded = storage.load_last_active_model()?;
    let model_manager = match loaded {
        Some(model) => {
            tracing::info!(version = %model.version, "loaded last active model");
            ModelManager::new(model)
        }
        None => {
            let model = crate::signal::model::cold_start_model();
            storage.insert_model(&model)?;
            ModelManager::new(model)
        }
    };

    let live = Arc::new(LiveState::new(model_manager.current()));
    let (write_tx, write_rx) = mpsc::channel(1024);
    let (degraded_tx, degraded_rx) = watch::channel(None);
    let (event_tx, event_rx) = mpsc::channel::<MarketEvent>(4096);

    let storage_writer = storage.clone();
    tokio::task::spawn_blocking(move || {
        storage_writer.writer_loop(write_rx, degraded_tx);
    });

    let engine = Arc::new(RwLock::new(SignalEngine::new(
        cfg.clone(),
        model_manager.current(),
    )));

    {
        let engine = engine.clone();
        let live = live.clone();
        let model_manager = model_manager.clone();
        let write_tx = write_tx.clone();
        let cfg_eval = cfg.clone();
        tokio::spawn(async move {
            crate::eval::run_signal_and_eval(
                cfg_eval,
                engine,
                live,
                model_manager,
                event_rx,
                write_tx,
                degraded_rx,
            )
            .await;
        });
    }

    {
        let cfg_m = cfg.clone();
        tokio::spawn(async move {
            crate::market::gate::run_market(cfg_m, event_tx).await;
        });
    }

    {
        let cfg_t = cfg.clone();
        let storage_t = storage.clone();
        let live_t = live.clone();
        let models = model_manager.clone();
        let write_tx = write_tx.clone();
        tokio::spawn(async move {
            crate::trainer::run_trainer(cfg_t, storage_t, models, live_t, write_tx).await;
        });
    }

    {
        let cfg_c = cfg.clone();
        let write_tx = write_tx.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(24 * 3600));
            loop {
                tick.tick().await;
                let _ = write_tx
                    .send(storage::WriteCmd::Cleanup {
                        retain_days: cfg_c.retain_days,
                    })
                    .await;
            }
        });
    }

    crate::web::serve(cfg, live, storage, model_manager).await
}

pub fn feature_version() -> &'static str {
    FEATURE_VERSION
}
