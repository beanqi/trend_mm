use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{Query, State};
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use serde::Deserialize;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::config::Config;
use crate::live::LiveState;
use crate::signal::model::now_ms;
use crate::storage::Storage;
use crate::trainer::ModelManager;
use crate::types::Horizon;

const INDEX_HTML: &str = include_str!("../static/index.html");
const APP_CSS: &str = include_str!("../static/app.css");
const APP_JS: &str = include_str!("../static/app.js");

#[derive(Clone)]
pub struct AppState {
    pub live: Arc<LiveState>,
    pub storage: Storage,
    pub models: ModelManager,
    pub cfg: Config,
}

pub async fn serve(
    cfg: Config,
    live: Arc<LiveState>,
    storage: Storage,
    models: ModelManager,
) -> anyhow::Result<()> {
    let bind = cfg.http_bind;
    let app = router(AppState {
        live,
        storage,
        models,
        cfg,
    });
    let listener = TcpListener::bind(bind).await?;
    tracing::info!(%bind, "http listening");
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/app.css", get(css))
        .route("/app.js", get(js))
        .route("/api/status", get(api_status))
        .route("/api/stream", get(api_stream))
        .route("/api/samples", get(api_samples))
        .route("/api/metrics", get(api_metrics))
        .route("/api/models", get(api_models))
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], APP_CSS)
}

async fn js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript; charset=utf-8")],
        APP_JS,
    )
}

async fn api_status(State(st): State<AppState>) -> impl IntoResponse {
    axum::Json(st.live.status())
}

async fn api_stream(State(st): State<AppState>) -> impl IntoResponse {
    let mut rx = st.live.subscribe_stream();
    let (tx, out) = mpsc::channel(16);
    tokio::spawn(async move {
        loop {
            let frame = rx.borrow().clone();
            let data = serde_json::to_string(&frame).unwrap_or_else(|_| "{}".into());
            if tx.send(Ok::<Event, std::convert::Infallible>(Event::default().data(data)))
                .await
                .is_err()
            {
                break;
            }
            if rx.changed().await.is_err() {
                break;
            }
        }
    });
    Sse::new(ReceiverStream::new(out)).keep_alive(KeepAlive::new().interval(Duration::from_secs(5)))
}

#[derive(Debug, Deserialize)]
struct SamplesQ {
    from_ms: Option<i64>,
    to_ms: Option<i64>,
    horizon: Option<String>,
    limit: Option<usize>,
}

async fn api_samples(State(st): State<AppState>, Query(q): Query<SamplesQ>) -> impl IntoResponse {
    let to = q.to_ms.unwrap_or_else(now_ms);
    let from = q.from_ms.unwrap_or(to - 5 * 60 * 1000);
    let limit = q.limit.unwrap_or(600).min(2000);
    let horizon = q
        .horizon
        .as_deref()
        .and_then(Horizon::parse)
        .unwrap_or(Horizon::H25);
    let rows = st.storage.query_samples(from, to, limit).unwrap_or_default();
    let points: Vec<serde_json::Value> = rows
        .into_iter()
        .map(|r| {
            let i = horizon.index();
            serde_json::json!({
                "ts_ms": r.ts_ms,
                "mid": r.mid,
                "tick_size": r.tick_size,
                "pred": r.pred[i],
                "actual": r.actual[i],
                "quality": r.quality,
                "model_version": r.model_version,
                "horizon": horizon.as_str(),
            })
        })
        .collect();
    axum::Json(serde_json::json!({
        "horizon": horizon.as_str(),
        "from_ms": from,
        "to_ms": to,
        "points": points,
    }))
}

#[derive(Debug, Deserialize)]
struct MetricsQ {
    window: Option<String>,
    model_version: Option<String>,
}

async fn api_metrics(State(st): State<AppState>, Query(q): Query<MetricsQ>) -> impl IntoResponse {
    let now = now_ms();
    let windows = match q.window.as_deref() {
        Some("1h") => vec![("1h", 3600_000)],
        Some("24h") => vec![("24h", 24 * 3600_000)],
        Some("7d") => vec![("7d", 7 * 24 * 3600_000)],
        _ => vec![
            ("1h", 3600_000),
            ("24h", 24 * 3600_000),
            ("7d", 7 * 24 * 3600_000),
        ],
    };
    let mut out = serde_json::Map::new();
    for (name, dur) in windows {
        let metrics = st
            .storage
            .query_metrics(
                now - dur,
                now,
                q.model_version.as_deref(),
                st.cfg.directional_pred_threshold,
            )
            .unwrap_or_else(|_| Default::default());
        out.insert(name.to_string(), serde_json::to_value(metrics).unwrap_or_default());
    }
    axum::Json(serde_json::Value::Object(out))
}

async fn api_models(State(st): State<AppState>) -> impl IntoResponse {
    let models = st.storage.list_models().unwrap_or_default();
    axum::Json(serde_json::json!({ "models": models }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::model::cold_start_model;
    use crate::types::RunPhase;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn app() -> Router {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::open(&dir.path().join("w.db")).unwrap();
        let model = cold_start_model();
        storage.insert_model(&model).unwrap();
        let models = ModelManager::new(model.clone());
        let live = Arc::new(LiveState::new(model));
        // leak tempdir for process lifetime of this test router
        std::mem::forget(dir);
        router(AppState {
            live,
            storage,
            models,
            cfg: Config::default(),
        })
    }

    async fn body_json(app: Router, uri: &str) -> (StatusCode, serde_json::Value) {
        let res = app
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(serde_json::json!({})))
    }

    #[tokio::test]
    async fn status_has_required_fields() {
        let (code, v) = body_json(app(), "/api/status").await;
        assert_eq!(code, StatusCode::OK);
        let phase = v["phase"].as_str().unwrap();
        assert!(matches!(phase, "warming" | "running" | "degraded"));
        assert!(v["model_version"].as_str().unwrap().len() > 0);
        assert!(v.get("next_train_ms").is_some());
        let note = v["depth_granularity_note"].as_str().unwrap();
        assert!(note.contains("20ms"));
        assert_eq!(v["depth_granularity_ms"], 20);
        assert_eq!(v["model_status"], "provisional");
        let _ = RunPhase::Warming;
    }

    #[tokio::test]
    async fn static_page_lists_dashboard_sections() {
        let res = app()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let html = String::from_utf8(res.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap();
        for needle in [
            "id=\"status-panel\"",
            "id=\"curves-panel\"",
            "id=\"metrics-panel\"",
            "1 hour",
            "24 hours",
            "7 days",
            "id=\"factors-panel\"",
            "id=\"models-panel\"",
            "Last 5 minutes",
        ] {
            assert!(html.contains(needle), "missing {needle}");
        }
    }

    #[tokio::test]
    async fn samples_metrics_models_shapes() {
        let (c, samples) = body_json(app(), "/api/samples?horizon=25ms").await;
        assert_eq!(c, StatusCode::OK);
        assert_eq!(samples["horizon"], "25ms");
        assert!(samples["points"].is_array());
        let (c, metrics) = body_json(app(), "/api/metrics").await;
        assert_eq!(c, StatusCode::OK);
        assert!(metrics.get("1h").is_some());
        assert!(metrics.get("24h").is_some());
        assert!(metrics.get("7d").is_some());
        let (c, models) = body_json(app(), "/api/models").await;
        assert_eq!(c, StatusCode::OK);
        assert!(models["models"].as_array().unwrap().len() >= 1);
        assert_eq!(models["models"][0]["status"], "provisional");
    }
}
