use std::cell::{Cell, RefCell};
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CloseEvent, ErrorEvent, MessageEvent, WebSocket};

use crate::dom;

/// Connection status for the UI indicator.
#[derive(Clone, Copy, PartialEq)]
pub enum Status {
    Connecting,
    Live,
    Reconnecting,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Connecting => "connecting\u{2026}",
            Status::Live => "live",
            Status::Reconnecting => "reconnecting\u{2026}",
        }
    }
}

/// Supabase Realtime WebSocket client using Phoenix Channels protocol.
/// Subscribes to both `contributors` (leaderboard) and `results` (viz feed) changes.
pub struct RealtimeClient {
    ws: RefCell<Option<WebSocket>>,
    status: Rc<Cell<Status>>,
    ref_counter: Rc<Cell<u32>>,
    heartbeat_id: RefCell<Option<i32>>,
    debounce_id: RefCell<Option<i32>>,
    debounce_results_id: RefCell<Option<i32>>,
    reconnect_delay: Rc<Cell<u32>>,
    on_contributors_change: RefCell<Option<Rc<dyn Fn()>>>,
    on_results_change: RefCell<Option<Rc<dyn Fn()>>>,
    // Store closures to prevent GC
    _closures: RefCell<Vec<Box<dyn std::any::Any>>>,
}

impl RealtimeClient {
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            ws: RefCell::new(None),
            status: Rc::new(Cell::new(Status::Connecting)),
            ref_counter: Rc::new(Cell::new(0)),
            heartbeat_id: RefCell::new(None),
            debounce_id: RefCell::new(None),
            debounce_results_id: RefCell::new(None),
            reconnect_delay: Rc::new(Cell::new(1000)),
            on_contributors_change: RefCell::new(None),
            on_results_change: RefCell::new(None),
            _closures: RefCell::new(Vec::new()),
        })
    }

    pub fn status(&self) -> Status {
        self.status.get()
    }

    /// Connect to Supabase Realtime and subscribe to `contributors` + `results` changes.
    pub fn connect(self: &Rc<Self>, on_contributors: Rc<dyn Fn()>, on_results: Rc<dyn Fn()>) {
        *self.on_contributors_change.borrow_mut() = Some(on_contributors);
        *self.on_results_change.borrow_mut() = Some(on_results);
        self.do_connect();
    }

    fn do_connect(self: &Rc<Self>) {
        // Close existing connection if any
        if let Some(ws) = self.ws.borrow().as_ref() {
            ws.close().ok();
        }
        self.clear_heartbeat();

        self.set_status(Status::Connecting);

        let key = crate::supabase::anon_key();
        let project = crate::supabase::project_ref();
        let url =
            format!("wss://{project}.supabase.co/realtime/v1/websocket?apikey={key}&vsn=1.0.0");

        let ws = match WebSocket::new(&url) {
            Ok(ws) => ws,
            Err(e) => {
                web_sys::console::warn_1(
                    &format!("realtime: WebSocket creation failed: {e:?}").into(),
                );
                self.schedule_reconnect();
                return;
            }
        };

        let this = Rc::clone(self);
        let onopen = Closure::wrap(Box::new(move |_: JsValue| {
            web_sys::console::log_1(&"realtime: websocket open, joining channels".into());
            this.reconnect_delay.set(1000);
            this.join_channels();
            this.start_heartbeat();
        }) as Box<dyn FnMut(JsValue)>);
        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));

        let this2 = Rc::clone(self);
        let onmessage = Closure::wrap(Box::new(move |e: MessageEvent| {
            if let Some(text) = e.data().as_string() {
                this2.handle_message(&text);
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

        let this3 = Rc::clone(self);
        let onclose = Closure::wrap(Box::new(move |_: CloseEvent| {
            web_sys::console::log_1(&"realtime: connection closed".into());
            this3.clear_heartbeat();
            this3.set_status(Status::Reconnecting);
            this3.schedule_reconnect();
        }) as Box<dyn FnMut(CloseEvent)>);
        ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));

        let this4 = Rc::clone(self);
        let onerror = Closure::wrap(Box::new(move |_: ErrorEvent| {
            web_sys::console::warn_1(&"realtime: WebSocket error".into());
            this4.set_status(Status::Reconnecting);
        }) as Box<dyn FnMut(ErrorEvent)>);
        ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));

        *self.ws.borrow_mut() = Some(ws);

        // Store closures to prevent GC
        let mut closures = self._closures.borrow_mut();
        closures.clear();
        closures.push(Box::new(onopen));
        closures.push(Box::new(onmessage));
        closures.push(Box::new(onclose));
        closures.push(Box::new(onerror));
    }

    fn next_ref(&self) -> u32 {
        let r = self.ref_counter.get() + 1;
        self.ref_counter.set(r);
        r
    }

    fn send(&self, msg: &str) {
        if let Some(ws) = self.ws.borrow().as_ref() {
            ws.send_with_str(msg).ok();
        }
    }

    /// Join both channels: `contributors` (leaderboard) and `results` (viz feed).
    fn join_channels(&self) {
        let key = crate::supabase::anon_key();

        // Contributors channel (leaderboard updates)
        let r1 = self.next_ref();
        let msg1 = serde_json::json!({
            "topic": "realtime:public:contributors",
            "event": "phx_join",
            "payload": {
                "config": {
                    "broadcast": { "self": false },
                    "presence": { "key": "" },
                    "postgres_changes": [{
                        "event": "*",
                        "schema": "public",
                        "table": "contributors"
                    }]
                },
                "access_token": key
            },
            "ref": r1.to_string()
        });
        self.send(&msg1.to_string());

        // Results channel (live viz feed)
        let r2 = self.next_ref();
        let msg2 = serde_json::json!({
            "topic": "realtime:public:results",
            "event": "phx_join",
            "payload": {
                "config": {
                    "broadcast": { "self": false },
                    "presence": { "key": "" },
                    "postgres_changes": [{
                        "event": "INSERT",
                        "schema": "public",
                        "table": "results"
                    }]
                },
                "access_token": key
            },
            "ref": r2.to_string()
        });
        self.send(&msg2.to_string());
    }

    /// Start heartbeat every 30s.
    fn start_heartbeat(&self) {
        self.clear_heartbeat();

        let ws_ref = Rc::new(self.ws.borrow().clone());
        let ref_counter = self.ref_counter.clone();

        let cb = Closure::wrap(Box::new(move || {
            if let Some(ws) = ws_ref.as_ref() {
                let r = ref_counter.get() + 1;
                ref_counter.set(r);
                let msg = serde_json::json!({
                    "topic": "phoenix",
                    "event": "heartbeat",
                    "payload": {},
                    "ref": r.to_string()
                });
                ws.send_with_str(&msg.to_string()).ok();
            }
        }) as Box<dyn FnMut()>);

        let id = dom::window()
            .set_interval_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                30_000,
            )
            .unwrap_or(0);
        cb.forget();
        *self.heartbeat_id.borrow_mut() = Some(id);
    }

    fn clear_heartbeat(&self) {
        if let Some(id) = self.heartbeat_id.borrow_mut().take() {
            dom::window().clear_interval_with_handle(id);
        }
    }

    /// Handle incoming Phoenix message — dispatch by topic.
    fn handle_message(&self, text: &str) {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(text) else {
            return;
        };

        let event = msg.get("event").and_then(|e| e.as_str()).unwrap_or("");
        let topic = msg.get("topic").and_then(|t| t.as_str()).unwrap_or("");

        match event {
            "postgres_changes" => {
                if topic.contains("contributors") {
                    self.debounce_contributors();
                } else if topic.contains("results") {
                    self.debounce_results();
                }
            }
            "phx_reply" => {
                let status = msg
                    .get("payload")
                    .and_then(|p| p.get("status"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("");
                if status == "ok" {
                    web_sys::console::log_1(&format!("realtime: channel joined: {topic}").into());
                    self.set_status(Status::Live);
                } else if status == "error" {
                    let reason = msg
                        .get("payload")
                        .and_then(|p| p.get("response"))
                        .map(|r| r.to_string())
                        .unwrap_or_default();
                    web_sys::console::warn_1(
                        &format!("realtime: join error ({topic}): {reason}").into(),
                    );
                }
            }
            "phx_error" => {
                web_sys::console::warn_1(&"realtime: channel error, reconnecting".into());
                self.set_status(Status::Reconnecting);
            }
            "system" => {
                web_sys::console::log_1(
                    &format!(
                        "realtime: system: {}",
                        msg.get("payload").unwrap_or(&serde_json::Value::Null)
                    )
                    .into(),
                );
            }
            _ => {}
        }
    }

    /// Debounce contributors change events by 500ms.
    fn debounce_contributors(&self) {
        if let Some(id) = self.debounce_id.borrow_mut().take() {
            dom::window().clear_timeout_with_handle(id);
        }

        let cb_fn = self.on_contributors_change.borrow().clone();
        let cb = Closure::once(move || {
            if let Some(f) = cb_fn {
                f();
            }
        });

        let id = dom::window()
            .set_timeout_with_callback_and_timeout_and_arguments_0(cb.as_ref().unchecked_ref(), 500)
            .unwrap_or(0);
        cb.forget();
        *self.debounce_id.borrow_mut() = Some(id);
    }

    /// Debounce results change events by 2s (batches multiple inserts into one feed fetch).
    fn debounce_results(&self) {
        if let Some(id) = self.debounce_results_id.borrow_mut().take() {
            dom::window().clear_timeout_with_handle(id);
        }

        let cb_fn = self.on_results_change.borrow().clone();
        let cb = Closure::once(move || {
            if let Some(f) = cb_fn {
                f();
            }
        });

        let id = dom::window()
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                2000,
            )
            .unwrap_or(0);
        cb.forget();
        *self.debounce_results_id.borrow_mut() = Some(id);
    }

    /// Schedule reconnect with exponential backoff (1s → 30s cap).
    fn schedule_reconnect(self: &Rc<Self>) {
        let delay = self.reconnect_delay.get();
        let next = (delay * 2).min(30_000);
        self.reconnect_delay.set(next);

        let this = Rc::clone(self);
        let cb = Closure::once(move || {
            this.do_connect();
        });

        dom::window()
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                delay as i32,
            )
            .ok();
        cb.forget();
    }

    fn set_status(&self, s: Status) {
        self.status.set(s);
        update_status_ui(s);
    }
}

fn update_status_ui(status: Status) {
    let (dot_class, label) = match status {
        Status::Connecting => ("rt-dot connecting", status.label()),
        Status::Live => ("rt-dot live", status.label()),
        Status::Reconnecting => ("rt-dot reconnecting", status.label()),
    };
    dom::set_class("rt-dot", dot_class);
    dom::set_text("rt-status", label);
}
