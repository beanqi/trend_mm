use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Config {
    pub symbol_display: String,
    pub symbol_gate: String,
    pub db_path: PathBuf,
    pub http_bind: SocketAddr,
    pub sample_period: Duration,
    pub retain_days: i64,
    pub first_train_after: Duration,
    pub train_period: Duration,
    pub train_window: Duration,
    pub ridge_lambda: f64,
    pub promote_rmse_improve: f64,
    pub promote_max_ic_drop: f64,
    pub directional_pred_threshold: f64,
    pub min_train_samples: usize,
    pub min_coverage: f64,
    pub warm_period: Duration,
    pub live_publish_period: Duration,
    pub rest_base: String,
    pub ws_url: String,
    pub disable_market: bool,
}

impl Config {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = env::var("TREND_MM_BIND") {
            cfg.http_bind = v.parse().expect("TREND_MM_BIND must be host:port");
        }
        if let Ok(v) = env::var("TREND_MM_DB") {
            cfg.db_path = PathBuf::from(v);
        }
        if let Ok(v) = env::var("TREND_MM_SYMBOL") {
            cfg.symbol_display = v.clone();
            cfg.symbol_gate = display_to_gate(&v);
        }
        if let Ok(v) = env::var("TREND_MM_DISABLE_MARKET") {
            cfg.disable_market = v == "1" || v.eq_ignore_ascii_case("true");
        }
        if let Ok(v) = env::var("TREND_MM_RIDGE_LAMBDA") {
            cfg.ridge_lambda = v.parse().expect("TREND_MM_RIDGE_LAMBDA");
        }
        if let Ok(v) = env::var("TREND_MM_WARM_SECS") {
            cfg.warm_period = Duration::from_secs_f64(v.parse().expect("TREND_MM_WARM_SECS"));
        }
        if let Ok(v) = env::var("TREND_MM_FIRST_TRAIN_SECS") {
            cfg.first_train_after =
                Duration::from_secs_f64(v.parse().expect("TREND_MM_FIRST_TRAIN_SECS"));
        }
        if let Ok(v) = env::var("TREND_MM_TRAIN_PERIOD_SECS") {
            cfg.train_period =
                Duration::from_secs_f64(v.parse().expect("TREND_MM_TRAIN_PERIOD_SECS"));
        }
        if let Ok(v) = env::var("TREND_MM_REST_BASE") {
            cfg.rest_base = v;
        }
        if let Ok(v) = env::var("TREND_MM_WS_URL") {
            cfg.ws_url = v;
        }
        cfg
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            symbol_display: "BTC-USDT".to_string(),
            symbol_gate: "BTC_USDT".to_string(),
            db_path: PathBuf::from("trend_mm.db"),
            http_bind: "127.0.0.1:8080".parse().expect("bind"),
            sample_period: Duration::from_secs(1),
            retain_days: 30,
            first_train_after: Duration::from_secs(24 * 3600),
            train_period: Duration::from_secs(6 * 3600),
            train_window: Duration::from_secs(7 * 24 * 3600),
            ridge_lambda: 1.0,
            promote_rmse_improve: 0.01,
            promote_max_ic_drop: 0.01,
            directional_pred_threshold: 0.05,
            min_train_samples: 100,
            min_coverage: 0.5,
            warm_period: Duration::from_secs(120),
            live_publish_period: Duration::from_millis(100),
            rest_base: "https://api.gateio.ws/api/v4".to_string(),
            ws_url: "wss://fx-ws.gateio.ws/v4/ws/usdt".to_string(),
            disable_market: false,
        }
    }
}

pub fn display_to_gate(symbol: &str) -> String {
    symbol.replace('-', "_")
}
