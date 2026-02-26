use serde::{Deserialize, Serialize};
use web_sys::Storage;

const KEY: &str = "phyz_results";

/// A cached data point (minimal, just what the viz needs).
#[derive(Clone, Serialize, Deserialize)]
pub struct CachedPoint {
    pub log_g2: f64,
    pub s_ee: f64,
    pub a_cut: f64,
}

fn storage() -> Option<Storage> {
    crate::dom::window().local_storage().ok().flatten()
}

/// Load all cached points from localStorage.
pub fn load() -> Vec<CachedPoint> {
    let Some(s) = storage() else { return vec![] };
    let Some(json) = s.get_item(KEY).ok().flatten() else { return vec![] };
    serde_json::from_str(&json).unwrap_or_default()
}

/// Save all points to localStorage (overwrites).
pub fn save(points: &[CachedPoint]) {
    let Some(s) = storage() else { return };
    if let Ok(json) = serde_json::to_string(points) {
        s.set_item(KEY, &json).ok();
    }
}

/// Append a single point and persist.
pub fn append(point: CachedPoint) {
    let mut points = load();
    points.push(point);
    save(&points);
}

/// Get the number of cached points.
pub fn count() -> usize {
    load().len()
}

const SPLASH_KEY: &str = "phyz_splash_seen";

pub fn has_seen_splash() -> bool {
    storage()
        .and_then(|s| s.get_item(SPLASH_KEY).ok().flatten())
        .map(|v| v == "1")
        .unwrap_or(false)
}

pub fn mark_splash_seen() {
    if let Some(s) = storage() {
        s.set_item(SPLASH_KEY, "1").ok();
    }
}

const EFFORT_KEY: &str = "phyz_effort";

pub fn load_effort() -> u32 {
    storage()
        .and_then(|s| s.get_item(EFFORT_KEY).ok().flatten())
        .and_then(|v| v.parse().ok())
        .unwrap_or(50)
}

pub fn save_effort(pct: u32) {
    if let Some(s) = storage() {
        s.set_item(EFFORT_KEY, &pct.to_string()).ok();
    }
}

const SESSION_KEY: &str = "phyz_session";

pub fn load_session() -> Option<crate::auth::AuthSession> {
    let s = storage()?;
    let json = s.get_item(SESSION_KEY).ok().flatten()?;
    serde_json::from_str(&json).ok()
}

pub fn save_session(session: &crate::auth::AuthSession) {
    let Some(s) = storage() else { return };
    if let Ok(json) = serde_json::to_string(session) {
        s.set_item(SESSION_KEY, &json).ok();
    }
}

pub fn clear_session() {
    if let Some(s) = storage() {
        s.remove_item(SESSION_KEY).ok();
    }
}
