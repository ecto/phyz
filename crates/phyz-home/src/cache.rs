use std::cell::RefCell;

use serde::{Deserialize, Serialize};
use web_sys::Storage;

const KEY: &str = "phyz_results_v2";

/// A cached data point (minimal, just what the viz needs).
#[derive(Clone, Serialize, Deserialize)]
pub struct CachedPoint {
    pub log_g2: f64,
    pub s_ee: f64,
    pub a_cut: f64,
    #[serde(default)]
    pub partition_index: usize,
}

fn storage() -> Option<Storage> {
    crate::dom::window().local_storage().ok().flatten()
}

// ── In-memory results cache with lazy persistence ──────────────────────

thread_local! {
    static CACHE: RefCell<Option<Vec<CachedPoint>>> = const { RefCell::new(None) };
    static DIRTY: RefCell<bool> = const { RefCell::new(false) };
}

/// Ensure the in-memory cache is initialized from localStorage (once).
fn ensure_loaded() {
    CACHE.with(|c| {
        if c.borrow().is_none() {
            let points = load_from_storage();
            *c.borrow_mut() = Some(points);
        }
    });
}

fn load_from_storage() -> Vec<CachedPoint> {
    let Some(s) = storage() else { return vec![] };
    let Some(json) = s.get_item(KEY).ok().flatten() else { return vec![] };
    serde_json::from_str(&json).unwrap_or_default()
}

fn save_to_storage(points: &[CachedPoint]) {
    let Some(s) = storage() else { return };
    if let Ok(json) = serde_json::to_string(points) {
        s.set_item(KEY, &json).ok();
    }
}

/// Load all cached points (from memory; first call reads localStorage).
pub fn load() -> Vec<CachedPoint> {
    ensure_loaded();
    CACHE.with(|c| c.borrow().as_ref().unwrap().clone())
}

/// Overwrite the cache (used by startup merge).
pub fn save(points: &[CachedPoint]) {
    CACHE.with(|c| {
        *c.borrow_mut() = Some(points.to_vec());
    });
    DIRTY.with(|d| *d.borrow_mut() = true);
}

/// Append a single point (in-memory only — call flush() to persist).
pub fn append(point: CachedPoint) {
    ensure_loaded();
    CACHE.with(|c| {
        c.borrow_mut().as_mut().unwrap().push(point);
    });
    DIRTY.with(|d| *d.borrow_mut() = true);
}

/// Append multiple points (in-memory only — call flush() to persist).
pub fn append_batch(new_points: &[CachedPoint]) {
    if new_points.is_empty() {
        return;
    }
    ensure_loaded();
    CACHE.with(|c| {
        c.borrow_mut().as_mut().unwrap().extend_from_slice(new_points);
    });
    DIRTY.with(|d| *d.borrow_mut() = true);
}

/// Get the number of cached points (no deserialization).
pub fn count() -> usize {
    ensure_loaded();
    CACHE.with(|c| c.borrow().as_ref().unwrap().len())
}

/// Flush dirty cache to localStorage. Call this on a timer, not every frame.
pub fn flush() {
    let dirty = DIRTY.with(|d| {
        let v = *d.borrow();
        if v {
            *d.borrow_mut() = false;
        }
        v
    });
    if dirty {
        CACHE.with(|c| {
            if let Some(ref pts) = *c.borrow() {
                save_to_storage(pts);
            }
        });
    }
}

// ── Non-results keys (small values, no caching needed) ─────────────────

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

const MUTED_KEY: &str = "phyz_muted";

pub fn load_muted() -> bool {
    storage()
        .and_then(|s| s.get_item(MUTED_KEY).ok().flatten())
        .map(|v| v == "1")
        .unwrap_or(false)
}

pub fn save_muted(muted: bool) {
    if let Some(s) = storage() {
        s.set_item(MUTED_KEY, if muted { "1" } else { "0" }).ok();
    }
}
