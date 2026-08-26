use std::time::Duration;

use anyhow::Context;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio::time::{Instant, timeout};
use tokio_tungstenite::tungstenite::Message;

use crate::config::Config;
use crate::market::book::{BookApply, LocalBook, parse_tick_size};
use crate::market::flow::PublicTrade;

const PING_EVERY: Duration = Duration::from_secs(15);
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug)]
pub enum MarketEvent {
    TickSize(f64),
    TickSizeFailed(String),
    DepthSnapshot {
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
        u: u64,
        exch_time_ms: i64,
    },
    DepthIncremental {
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
        start_u: u64,
        end_u: u64,
        exch_time_ms: i64,
    },
    Trade(PublicTrade),
    Disconnected(String),
}

pub async fn run_market(cfg: Config, tx: mpsc::Sender<MarketEvent>) {
    if cfg.disable_market {
        tracing::info!("market disabled");
        return;
    }
    let mut backoff = Duration::from_secs(1);
    loop {
        match run_session(&cfg, &tx).await {
            Ok(()) => backoff = Duration::from_secs(1),
            Err(e) => {
                tracing::warn!(error = %e, "market session ended");
                let _ = tx.send(MarketEvent::Disconnected(e.to_string())).await;
            }
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}

async fn run_session(cfg: &Config, tx: &mpsc::Sender<MarketEvent>) -> anyhow::Result<()> {
    match fetch_tick_size(&cfg.rest_base, &cfg.symbol_gate).await {
        Ok(tick) => {
            tx.send(MarketEvent::TickSize(tick)).await.ok();
        }
        Err(e) => {
            tx.send(MarketEvent::TickSizeFailed(e.to_string()))
                .await
                .ok();
            return Err(e);
        }
    }

    let (ws, _) = tokio_tungstenite::connect_async(&cfg.ws_url)
        .await
        .context("ws connect")?;
    let (mut sink, mut stream) = ws.split();

    let now = chrono_secs();
    let obu = format!("ob.{}.50", cfg.symbol_gate);
    let sub_obu = json!({
        "time": now,
        "channel": "futures.obu",
        "event": "subscribe",
        "payload": [obu],
    });
    let sub_trades = json!({
        "time": now,
        "channel": "futures.trades",
        "event": "subscribe",
        "payload": [cfg.symbol_gate],
    });
    sink.send(Message::Text(sub_obu.to_string().into()))
        .await
        .context("sub obu")?;
    sink.send(Message::Text(sub_trades.to_string().into()))
        .await
        .context("sub trades")?;

    let mut last_msg = Instant::now();
    let mut ping = tokio::time::interval(PING_EVERY);
    let mut need_resub = false;
    let mut local = LocalBook::new();

    loop {
        if need_resub {
            local.discard();
            let unsub = json!({
                "time": chrono_secs(),
                "channel": "futures.obu",
                "event": "unsubscribe",
                "payload": [format!("ob.{}.50", cfg.symbol_gate)],
            });
            let sub = json!({
                "time": chrono_secs(),
                "channel": "futures.obu",
                "event": "subscribe",
                "payload": [format!("ob.{}.50", cfg.symbol_gate)],
            });
            sink.send(Message::Text(unsub.to_string().into())).await.ok();
            sink.send(Message::Text(sub.to_string().into()))
                .await
                .context("resubscribe obu")?;
            let _ = tx
                .send(MarketEvent::Disconnected("order book sequence gap".into()))
                .await;
            need_resub = false;
        }

        tokio::select! {
            _ = ping.tick() => {
                if last_msg.elapsed() > IDLE_TIMEOUT {
                    anyhow::bail!("websocket idle timeout");
                }
                let ping_msg = json!({"time": chrono_secs(), "channel": "futures.ping"});
                sink.send(Message::Text(ping_msg.to_string().into())).await.ok();
            }
            msg = timeout(IDLE_TIMEOUT, stream.next()) => {
                let msg = match msg {
                    Ok(Some(m)) => m.context("ws frame")?,
                    Ok(None) => anyhow::bail!("ws closed"),
                    Err(_) => anyhow::bail!("websocket read timeout"),
                };
                last_msg = Instant::now();
                match msg {
                    Message::Text(t) => {
                        for ev in parse_ws_text(&t) {
                            match &ev {
                                MarketEvent::DepthSnapshot { bids, asks, u, .. } => {
                                    local.apply_snapshot(bids, asks, *u);
                                }
                                MarketEvent::DepthIncremental {
                                    bids,
                                    asks,
                                    start_u,
                                    end_u,
                                    ..
                                } => {
                                    if local.apply_incremental(bids, asks, *start_u, *end_u)
                                        == BookApply::Gap
                                    {
                                        need_resub = true;
                                        continue;
                                    }
                                }
                                _ => {}
                            }
                            if tx.send(ev).await.is_err() {
                                return Ok(());
                            }
                        }
                    }
                    Message::Ping(p) => {
                        sink.send(Message::Pong(p)).await.ok();
                    }
                    Message::Close(_) => anyhow::bail!("ws close frame"),
                    _ => {}
                }
            }
        }
    }
}

fn chrono_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub async fn fetch_tick_size(rest_base: &str, contract: &str) -> anyhow::Result<f64> {
    let url = format!("{rest_base}/futures/usdt/contracts/{contract}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let resp: ContractInfo = client
        .get(&url)
        .send()
        .await
        .context("contract http")?
        .error_for_status()
        .context("contract status")?
        .json()
        .await
        .context("contract json")?;
    parse_tick_size(&resp.order_price_round).context("order_price_round missing or invalid")
}

#[derive(Debug, Deserialize)]
struct ContractInfo {
    order_price_round: String,
}

#[derive(Debug, Deserialize)]
struct WsEnvelope {
    channel: Option<String>,
    event: Option<String>,
    result: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ObuResult {
    t: Option<i64>,
    full: Option<bool>,
    #[serde(default)]
    b: Vec<[String; 2]>,
    #[serde(default)]
    a: Vec<[String; 2]>,
    #[serde(rename = "U")]
    start_u: Option<u64>,
    #[serde(rename = "u")]
    end_u: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct TradeRow {
    id: u64,
    price: String,
    size: Value,
    #[serde(default)]
    is_internal: bool,
}

pub fn parse_ws_text(text: &str) -> Vec<MarketEvent> {
    let Ok(env) = serde_json::from_str::<WsEnvelope>(text) else {
        return Vec::new();
    };
    match env.channel.as_deref() {
        Some("futures.obu") if env.event.as_deref() == Some("update") => {
            env.result.and_then(parse_obu).into_iter().collect()
        }
        Some("futures.trades") if env.event.as_deref() == Some("update") => {
            env.result.map(parse_trades).unwrap_or_default()
        }
        _ => Vec::new(),
    }
}

fn parse_obu(result: Value) -> Option<MarketEvent> {
    let r: ObuResult = serde_json::from_value(result).ok()?;
    let bids = parse_levels(&r.b);
    let asks = parse_levels(&r.a);
    let end_u = r.end_u?;
    let exch = r.t.unwrap_or(0);
    if r.full.unwrap_or(false) {
        Some(MarketEvent::DepthSnapshot {
            bids,
            asks,
            u: end_u,
            exch_time_ms: exch,
        })
    } else {
        let start_u = r.start_u?;
        Some(MarketEvent::DepthIncremental {
            bids,
            asks,
            start_u,
            end_u,
            exch_time_ms: exch,
        })
    }
}

fn parse_levels(rows: &[[String; 2]]) -> Vec<(f64, f64)> {
    rows.iter()
        .filter_map(|[p, q]| Some((p.parse().ok()?, q.parse().ok()?)))
        .collect()
}

fn parse_trades(result: Value) -> Vec<MarketEvent> {
    let rows: Vec<TradeRow> = if result.is_array() {
        serde_json::from_value(result).unwrap_or_default()
    } else {
        serde_json::from_value(result).into_iter().collect()
    };
    rows.into_iter().filter_map(trade_from_row).collect()
}

fn trade_from_row(row: TradeRow) -> Option<MarketEvent> {
    let size = match row.size {
        Value::String(s) => s.parse().ok()?,
        Value::Number(n) => n.as_f64()?,
        _ => return None,
    };
    Some(MarketEvent::Trade(PublicTrade {
        id: row.id,
        price: row.price.parse().ok()?,
        size,
        is_internal: row.is_internal,
    }))
}

/// Caller uses this after applying an incremental locally and seeing a gap.
pub fn gap_event() -> MarketEvent {
    MarketEvent::Disconnected("order book sequence gap".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_snapshot_and_incremental() {
        let snap = r#"{"channel":"futures.obu","event":"update","result":{"t":1,"full":true,"s":"ob.BTC_USDT.50","u":10,"b":[["100.0","1"],["99.9","2"]],"a":[["100.1","3"]]}}"#;
        match &parse_ws_text(snap)[..] {
            [MarketEvent::DepthSnapshot { u, bids, asks, .. }] => {
                assert_eq!(*u, 10);
                assert_eq!(bids[0], (100.0, 1.0));
                assert_eq!(asks[0], (100.1, 3.0));
            }
            other => panic!("{other:?}"),
        }
        let inc = r#"{"channel":"futures.obu","event":"update","result":{"t":2,"s":"ob.BTC_USDT.50","U":11,"u":12,"b":[["100.0","0"]],"a":[]}}"#;
        match &parse_ws_text(inc)[..] {
            [MarketEvent::DepthIncremental {
                start_u, end_u, bids, ..
            }] => {
                assert_eq!(*start_u, 11);
                assert_eq!(*end_u, 12);
                assert_eq!(bids[0], (100.0, 0.0));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parse_trade_internal_and_size_sign() {
        let t = r#"{"channel":"futures.trades","event":"update","result":[{"id":7,"price":"100.1","size":"-2.5","contract":"BTC_USDT","is_internal":true}]}"#;
        match &parse_ws_text(t)[..] {
            [MarketEvent::Trade(tr)] => {
                assert_eq!(tr.id, 7);
                assert_eq!(tr.size, -2.5);
                assert!(tr.is_internal);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parse_trade_array_emits_every_row() {
        let t = r#"{"channel":"futures.trades","event":"update","result":[
            {"id":1,"price":"100.0","size":"1.5","contract":"BTC_USDT"},
            {"id":2,"price":"100.1","size":"-2","contract":"BTC_USDT"},
            {"id":3,"price":"100.0","size":"4","contract":"BTC_USDT","is_internal":true}
        ]}"#;
        let evs = parse_ws_text(t);
        assert_eq!(evs.len(), 3, "must not drop trades from a multi-row update");
        match &evs[..] {
            [MarketEvent::Trade(a), MarketEvent::Trade(b), MarketEvent::Trade(c)] => {
                assert_eq!(a.id, 1);
                assert_eq!(a.size, 1.5);
                assert!(!a.is_internal);
                assert_eq!(b.id, 2);
                assert_eq!(b.size, -2.0);
                assert_eq!(c.id, 3);
                assert!(c.is_internal);
            }
            other => panic!("{other:?}"),
        }
    }
}
