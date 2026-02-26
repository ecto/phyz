use std::cell::{Cell, RefCell};
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::auth::{self, AuthSession};
use crate::coordinator::Coordinator;
use crate::dom;
use crate::supabase::{self, SupabaseClient};
use crate::viz::Renderer;
use crate::worker::WorkerPool;

/// Main application entry point.
pub async fn run() {
    web_sys::console::log_1(&"phyz@home starting...".into());

    // Initialize canvas2d renderer
    let canvas = dom::canvas();
    let renderer = match Renderer::new(canvas) {
        Ok(r) => r,
        Err(e) => {
            web_sys::console::error_1(&format!("renderer init failed: {e}").into());
            dom::set_text("gn", "Canvas not supported");
            return;
        }
    };
    let renderer = Rc::new(RefCell::new(renderer));

    // Check for magic link callback (returning from email click)
    let session: Rc<RefCell<Option<AuthSession>>> = Rc::new(RefCell::new(
        auth::check_callback().or_else(auth::load_session),
    ));

    // Initialize Supabase client
    let client = Rc::new(SupabaseClient::new());

    // Register contributor
    let fingerprint = supabase::browser_fingerprint();
    let contributor_id = match client.upsert_contributor(&fingerprint).await {
        Ok(id) => id,
        Err(e) => {
            web_sys::console::warn_1(&format!("contributor registration: {e}").into());
            fingerprint.clone() // fallback to fingerprint as ID
        }
    };
    let contributor_id_rc = Rc::new(contributor_id.clone());

    // Load cached results immediately (survives refresh)
    let cached = crate::cache::load();
    if !cached.is_empty() {
        let mut r = renderer.borrow_mut();
        for cp in &cached {
            r.add_point(cp.log_g2, cp.s_ee, cp.a_cut);
        }
        // Mark cached points as settled (not fresh)
        for p in &mut r.points {
            p.age = 3.0;
        }
        web_sys::console::log_1(
            &format!("restored {} cached results", cached.len()).into(),
        );
    }

    // Fetch global results feed with contributor names
    let last_feed_ts: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    match client.fetch_results_feed(200, None).await {
        Ok(results) => {
            let n_before = renderer.borrow().points.len();
            let mut r = renderer.borrow_mut();

            // Build a HashSet of existing (log_g2_bits, partition_index) for O(1) dedup
            use std::collections::HashSet;
            let existing: HashSet<(u32, usize)> = r.points.iter().map(|p| {
                (p.log_g2.to_bits(), p.a_cut as usize)
            }).collect();

            // Track newest timestamp for incremental realtime fetching
            if let Some(first) = results.first() {
                *last_feed_ts.borrow_mut() = Some(first.submitted_at.clone());
            }

            let mut new_cache_points = Vec::new();
            for res in &results {
                let log_g2 = res.coupling_g2.log10();
                let log_g2_bits = (log_g2 as f32).to_bits();
                let has_areas = !res.result_data.boundary_area_per_partition.is_empty();
                for (i, &s_ee) in res.result_data.entropy_per_partition.iter().enumerate() {
                    if !existing.contains(&(log_g2_bits, i)) {
                        let a_cut = if has_areas {
                            res.result_data.boundary_area_per_partition[i]
                        } else {
                            i as f64
                        };
                        // Label first partition with contributor name
                        let label = if i == 0 {
                            Some(res.contributor_name.clone())
                        } else {
                            None
                        };
                        r.add_point_labeled(log_g2, s_ee, a_cut, label);
                        r.points.last_mut().unwrap().age = 3.0;
                        new_cache_points.push(crate::cache::CachedPoint {
                            log_g2, s_ee, a_cut,
                            partition_index: i,
                        });
                    }
                }
            }
            let n_new = r.points.len() - n_before;
            if n_new > 0 {
                drop(r);
                crate::cache::append_batch(&new_cache_points);
                web_sys::console::log_1(
                    &format!("merged {} new results from server ({} with names)", n_new, results.len()).into(),
                );
            }
        }
        Err(e) => {
            web_sys::console::warn_1(&format!("fetch feed: {e}").into());
            // If no cache and no server, seed demo data
            if cached.is_empty() {
                let mut r = renderer.borrow_mut();
                seed_demo_data(&mut r);
            }
        }
    }

    // Get contributor count (shown in leaderboard panel status)
    if let Ok(count) = client.contributor_count().await {
        dom::set_text("n-contributors", &count.to_string());
    }

    // Server-side unit count (persisted across sessions)
    let base_units: Rc<Cell<i64>> = Rc::new(Cell::new(0));

    // Auth UI setup
    {
        let sess = session.borrow();
        if let Some(ref s) = *sess {
            // Authenticated: link auth to contributor
            if let Err(e) = client
                .link_auth(&contributor_id, &s.user_id, &s.access_token)
                .await
            {
                web_sys::console::warn_1(&format!("link_auth: {e}").into());
            }

            // Fetch server stats to check if display_name is set
            let display_name = match client.fetch_my_stats(&contributor_id).await {
                Ok(stats) => {
                    base_units.set(stats.total_units);
                    dom::set_text("my-count", &stats.total_units.to_string());
                    dom::set_text("user-points", &format!("({})", stats.total_units));
                    stats.display_name
                }
                Err(_) => None,
            };

            if display_name.is_none() {
                // Show display name prompt
                dom::set_class("name-modal", "");
            }

            show_auth_ui(display_name.as_deref().unwrap_or(&s.email));
        }
    }

    // Wire sign-in button → open auth modal
    let sign_in_cb = Closure::wrap(Box::new(move || {
        dom::set_class("auth-modal", "");
        dom::set_text("auth-status", "");
    }) as Box<dyn FnMut()>);
    dom::get_el("sign-in-btn")
        .set_onclick(Some(sign_in_cb.as_ref().unchecked_ref()));
    sign_in_cb.forget();

    // Wire auth modal close
    let auth_close_cb = Closure::wrap(Box::new(move || {
        dom::set_class("auth-modal", "hidden");
    }) as Box<dyn FnMut()>);
    dom::get_el("auth-close")
        .set_onclick(Some(auth_close_cb.as_ref().unchecked_ref()));
    auth_close_cb.forget();

    // Shared magic link send logic
    let send_magic_link = {
        let client_ml = client.clone();
        Rc::new(move || {
            let email_el: web_sys::HtmlInputElement = dom::get_el("auth-email")
                .dyn_into()
                .unwrap();
            let email = email_el.value();
            if email.is_empty() || !email.contains('@') {
                dom::set_text("auth-status", "enter a valid email");
                return;
            }
            // Disable inputs
            email_el.set_disabled(true);
            let send_el: web_sys::HtmlButtonElement = dom::get_el("auth-send")
                .dyn_into()
                .unwrap();
            send_el.set_disabled(true);
            dom::set_text("auth-status", "sending...");
            let client = client_ml.clone();
            wasm_bindgen_futures::spawn_local(async move {
                match client.send_magic_link(&email).await {
                    Ok(()) => dom::set_text("auth-status", "check your email!"),
                    Err(e) => {
                        dom::set_text("auth-status", &format!("error: {e}"));
                        // Re-enable on error
                        let el: web_sys::HtmlInputElement = dom::get_el("auth-email")
                            .dyn_into()
                            .unwrap();
                        el.set_disabled(false);
                        let btn: web_sys::HtmlButtonElement = dom::get_el("auth-send")
                            .dyn_into()
                            .unwrap();
                        btn.set_disabled(false);
                    }
                }
            });
        })
    };

    // Wire send button click
    let send_fn = send_magic_link.clone();
    let send_cb = Closure::wrap(Box::new(move || send_fn()) as Box<dyn FnMut()>);
    dom::get_el("auth-send")
        .set_onclick(Some(send_cb.as_ref().unchecked_ref()));
    send_cb.forget();

    // Wire Enter key on email input
    let send_fn2 = send_magic_link.clone();
    let keydown_cb = Closure::wrap(Box::new(move |e: web_sys::KeyboardEvent| {
        if e.key() == "Enter" {
            send_fn2();
        }
    }) as Box<dyn FnMut(web_sys::KeyboardEvent)>);
    dom::get_el("auth-email")
        .add_event_listener_with_callback("keydown", keydown_cb.as_ref().unchecked_ref())
        .ok();
    keydown_cb.forget();

    // Wire sign-out button
    let session_so = session.clone();
    let sign_out_cb = Closure::wrap(Box::new(move || {
        auth::sign_out();
        *session_so.borrow_mut() = None;
        dom::set_class("auth-anon", "");
        dom::set_class("auth-user", "hidden");
    }) as Box<dyn FnMut()>);
    dom::get_el("sign-out-btn")
        .set_onclick(Some(sign_out_cb.as_ref().unchecked_ref()));
    sign_out_cb.forget();

    // Shared display name save logic
    let save_display_name = {
        let client_name = client.clone();
        let cid_name = contributor_id_rc.clone();
        let session_name = session.clone();
        Rc::new(move || {
            let name_el: web_sys::HtmlInputElement = dom::get_el("name-input")
                .dyn_into()
                .unwrap();
            let name = name_el.value().trim().to_string();
            if name.is_empty() {
                dom::set_text("name-status", "enter a name");
                return;
            }
            name_el.set_disabled(true);
            let save_el: web_sys::HtmlButtonElement = dom::get_el("name-save")
                .dyn_into()
                .unwrap();
            save_el.set_disabled(true);
            dom::set_text("name-status", "saving...");
            let client = client_name.clone();
            let cid = cid_name.clone();
            let sess = session_name.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let jwt = sess.borrow().as_ref().map(|s| s.access_token.clone());
                if let Some(jwt) = jwt {
                    match client.set_display_name(&cid, &name, &jwt).await {
                        Ok(()) => {
                            dom::set_text("user-name", &name);
                            dom::set_class("name-modal", "hidden");
                            dom::set_text("name-status", "");
                        }
                        Err(e) => {
                            dom::set_text("name-status", &format!("error: {e}"));
                            let el: web_sys::HtmlInputElement = dom::get_el("name-input")
                                .dyn_into()
                                .unwrap();
                            el.set_disabled(false);
                            let btn: web_sys::HtmlButtonElement = dom::get_el("name-save")
                                .dyn_into()
                                .unwrap();
                            btn.set_disabled(false);
                        }
                    }
                }
            });
        })
    };

    // Wire save button click
    let save_fn = save_display_name.clone();
    let name_save_cb = Closure::wrap(Box::new(move || save_fn()) as Box<dyn FnMut()>);
    dom::get_el("name-save")
        .set_onclick(Some(name_save_cb.as_ref().unchecked_ref()));
    name_save_cb.forget();

    // Wire Enter key on name input
    let save_fn2 = save_display_name.clone();
    let name_key_cb = Closure::wrap(Box::new(move |e: web_sys::KeyboardEvent| {
        if e.key() == "Enter" {
            save_fn2();
        }
    }) as Box<dyn FnMut(web_sys::KeyboardEvent)>);
    dom::get_el("name-input")
        .add_event_listener_with_callback("keydown", name_key_cb.as_ref().unchecked_ref())
        .ok();
    name_key_cb.forget();

    // Initialize leaderboard state
    let lb_state = Rc::new(RefCell::new(LeaderboardState::new(
        contributor_id_rc.as_ref().clone(),
    )));

    // Initial leaderboard fetch (REST, before WebSocket connects)
    {
        let client_lb = client.clone();
        let session_lb = session.clone();
        let lb = lb_state.clone();
        wasm_bindgen_futures::spawn_local(async move {
            match client_lb.fetch_leaderboard().await {
                Ok(entries) => {
                    let my_id = session_lb.borrow().as_ref().map(|s| s.user_id.clone());
                    let mut state = lb.borrow_mut();
                    state.update(&entries, my_id.as_deref());
                    render_leaderboard_panel(&state);
                }
                Err(e) => {
                    dom::set_inner_html(
                        "leaderboard",
                        &format!("<div class=\"lb-empty\">{e}</div>"),
                    );
                }
            }
        });
    }

    // Connect Supabase Realtime for live leaderboard + results feed
    let rt_client = crate::realtime::RealtimeClient::new();
    {
        // Leaderboard callback (contributors changes)
        let client_rt = client.clone();
        let session_rt = session.clone();
        let lb_rt = lb_state.clone();
        let on_contributors: Rc<dyn Fn()> = Rc::new(move || {
            let client = client_rt.clone();
            let sess = session_rt.clone();
            let lb = lb_rt.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Ok(entries) = client.fetch_leaderboard().await {
                    let my_id = sess.borrow().as_ref().map(|s| s.user_id.clone());
                    let mut state = lb.borrow_mut();
                    state.update(&entries, my_id.as_deref());
                    render_leaderboard_panel(&state);
                }
            });
        });

        // Results feed callback (new results from other volunteers)
        let client_feed = client.clone();
        let renderer_feed = renderer.clone();
        let my_cid = contributor_id_rc.clone();
        let feed_ts = last_feed_ts.clone();
        let on_results: Rc<dyn Fn()> = Rc::new(move || {
            let client = client_feed.clone();
            let renderer = renderer_feed.clone();
            let my_cid = my_cid.clone();
            let feed_ts = feed_ts.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let after = feed_ts.borrow().clone();
                match client.fetch_results_feed(50, after.as_deref()).await {
                    Ok(results) => {
                        if results.is_empty() {
                            return;
                        }
                        // Update timestamp to newest result
                        if let Some(first) = results.first() {
                            *feed_ts.borrow_mut() = Some(first.submitted_at.clone());
                        }
                        let mut r = renderer.borrow_mut();
                        let mut n_added = 0usize;
                        for res in &results {
                            // Skip own results (already shown by coordinator)
                            if res.contributor_id == *my_cid {
                                continue;
                            }
                            let log_g2 = res.coupling_g2.log10();
                            let has_areas =
                                !res.result_data.boundary_area_per_partition.is_empty();
                            for (i, &s_ee) in
                                res.result_data.entropy_per_partition.iter().enumerate()
                            {
                                let a_cut = if has_areas {
                                    res.result_data.boundary_area_per_partition[i]
                                } else {
                                    i as f64
                                };
                                // Label first partition with contributor name
                                let label = if i == 0 {
                                    Some(res.contributor_name.clone())
                                } else {
                                    None
                                };
                                r.add_point_labeled(log_g2, s_ee, a_cut, label);
                                n_added += 1;
                            }
                        }
                        if n_added > 0 {
                            web_sys::console::log_1(
                                &format!("realtime: added {} points from other volunteers", n_added)
                                    .into(),
                            );
                        }
                    }
                    Err(e) => {
                        web_sys::console::warn_1(
                            &format!("realtime feed: {e}").into(),
                        );
                    }
                }
            });
        });

        rt_client.connect(on_contributors, on_results);
    }

    // Wire leaderboard panel collapse toggle (header click)
    let lb_collapsed = Rc::new(Cell::new(false));
    {
        let collapsed = lb_collapsed.clone();
        let toggle_cb = Closure::wrap(Box::new(move || {
            let new_val = !collapsed.get();
            collapsed.set(new_val);
            if new_val {
                dom::set_class("lb-panel", "collapsed");
            } else {
                dom::set_class("lb-panel", "");
            }
        }) as Box<dyn FnMut()>);
        dom::get_el("lb-panel-header")
            .set_onclick(Some(toggle_cb.as_ref().unchecked_ref()));
        toggle_cb.forget();
    }

    // Initialize worker pool (Rc, not Rc<RefCell> — pool uses interior mutability)
    let pool_size = WorkerPool::recommended_size();
    let pool = match WorkerPool::new(pool_size) {
        Ok(p) => {
            web_sys::console::log_1(
                &format!("worker pool: {pool_size} workers").into(),
            );
            Rc::new(p)
        }
        Err(e) => {
            web_sys::console::error_1(&format!("worker pool: {e}").into());
            // Continue without workers — viz still works
            dom::set_text("toggle", "\u{25B6}");
            start_render_loop(renderer.clone(), None, client.clone(), Rc::new(Cell::new(false)), Rc::new(Cell::new(0i64)), lb_state.clone());
            return;
        }
    };

    // Create coordinator
    let coordinator = Rc::new(Coordinator::new(
        client.clone(),
        pool.clone(),
        contributor_id,
        renderer.clone(),
    ));

    // Gate auto-start behind splash flag
    if crate::cache::has_seen_splash() {
        coordinator.start();
        dom::set_text("toggle", "\u{23F8}");
        dom::set_class("toggle", "running");
        dom::set_text("status-text", "running");
        dom::set_class("activity-dot", "indicator active");
        // splash stays hidden (default in HTML)
    } else {
        dom::set_text("toggle", "\u{25B6}");
        dom::set_class("toggle", "");
        dom::set_text("status-text", "idle");
        dom::set_class("activity-dot", "indicator idle");
        // Show splash only on first visit
        dom::set_class("splash-backdrop", "");
    }

    // Wire up toggle button
    let coord = coordinator.clone();
    let toggle_cb = Closure::wrap(Box::new(move || {
        if coord.is_complete() {
            return;
        }
        if coord.is_running() {
            coord.stop();
            dom::set_text("toggle", "\u{25B6}");
            dom::set_class("toggle", "");
            dom::set_class("activity-dot", "indicator idle");
            dom::set_text("status-text", "idle");
        } else {
            coord.start();
            dom::set_text("toggle", "\u{23F8}");
            dom::set_class("toggle", "running");
            dom::set_class("activity-dot", "indicator active");
            dom::set_text("status-text", "running");
        }
    }) as Box<dyn FnMut()>);

    dom::get_el("toggle")
        .set_onclick(Some(toggle_cb.as_ref().unchecked_ref()));
    toggle_cb.forget();

    // Wire up splash CTA button
    let coord2 = coordinator.clone();
    let cta_cb = Closure::wrap(Box::new(move || {
        crate::cache::mark_splash_seen();
        if !coord2.is_running() {
            coord2.start();
            dom::set_text("toggle", "\u{23F8}");
            dom::set_class("toggle", "running");
            dom::set_class("activity-dot", "indicator active");
            dom::set_text("status-text", "running");
        }
        dom::set_class("splash-backdrop", "hidden");
    }) as Box<dyn FnMut()>);
    dom::get_el("splash-cta")
        .set_onclick(Some(cta_cb.as_ref().unchecked_ref()));
    cta_cb.forget();

    // Wire up mute button
    let muted = Rc::new(Cell::new(crate::cache::load_muted()));
    if muted.get() {
        dom::set_class("mute-btn", "muted");
    }
    {
        let muted = muted.clone();
        let mute_cb = Closure::wrap(Box::new(move || {
            let new_val = !muted.get();
            muted.set(new_val);
            crate::cache::save_muted(new_val);
            if new_val {
                dom::set_class("mute-btn", "muted");
            } else {
                dom::set_class("mute-btn", "");
            }
        }) as Box<dyn FnMut()>);
        dom::get_el("mute-btn")
            .set_onclick(Some(mute_cb.as_ref().unchecked_ref()));
        mute_cb.forget();
    }

    // Wire up convergence bar click — opens splash modal
    let info_cb = Closure::wrap(Box::new(move || {
        dom::set_class("splash-backdrop", "");
    }) as Box<dyn FnMut()>);
    dom::get_el("convergence-wrap")
        .set_onclick(Some(info_cb.as_ref().unchecked_ref()));
    info_cb.forget();

    // Wire up ? button — opens splash modal
    let info_btn_cb = Closure::wrap(Box::new(move || {
        dom::set_class("splash-backdrop", "");
    }) as Box<dyn FnMut()>);
    dom::get_el("info-btn")
        .set_onclick(Some(info_btn_cb.as_ref().unchecked_ref()));
    info_btn_cb.forget();

    // Wire up × close button — hides splash modal
    let close_cb = Closure::wrap(Box::new(move || {
        dom::set_class("splash-backdrop", "hidden");
    }) as Box<dyn FnMut()>);
    dom::get_el("splash-close")
        .set_onclick(Some(close_cb.as_ref().unchecked_ref()));
    close_cb.forget();

    // Click backdrop (outside card) to dismiss splash
    let backdrop_cb = Closure::wrap(Box::new(move |e: web_sys::MouseEvent| {
        let target = e.target().unwrap();
        let el: &web_sys::Element = target.unchecked_ref();
        if el.id() == "splash-backdrop" {
            dom::set_class("splash-backdrop", "hidden");
        }
    }) as Box<dyn FnMut(web_sys::MouseEvent)>);
    dom::get_el("splash-backdrop")
        .add_event_listener_with_callback("click", backdrop_cb.as_ref().unchecked_ref())
        .ok();
    backdrop_cb.forget();

    // Escape key dismisses splash modal
    let esc_cb = Closure::wrap(Box::new(move |e: web_sys::KeyboardEvent| {
        if e.key() == "Escape" {
            dom::set_class("splash-backdrop", "hidden");
        }
    }) as Box<dyn FnMut(web_sys::KeyboardEvent)>);
    dom::window()
        .add_event_listener_with_callback("keydown", esc_cb.as_ref().unchecked_ref())
        .ok();
    esc_cb.forget();

    // Wire up effort slider
    let saved_effort = crate::cache::load_effort();
    coordinator.set_effort(saved_effort);
    {
        let el = dom::get_el("effort");
        el.set_attribute("value", &saved_effort.to_string()).ok();
    }
    dom::set_text("effort-val", &format!("{saved_effort}%"));

    let coord_effort = coordinator.clone();
    let effort_cb = Closure::wrap(Box::new(move |e: web_sys::Event| {
        let target = e.target().unwrap();
        let input: &web_sys::HtmlInputElement = target.unchecked_ref();
        let val: u32 = input.value().parse().unwrap_or(50);
        coord_effort.set_effort(val);
        crate::cache::save_effort(val);
        dom::set_text("effort-val", &format!("{val}%"));
    }) as Box<dyn FnMut(web_sys::Event)>);
    dom::get_el("effort")
        .add_event_listener_with_callback("input", effort_cb.as_ref().unchecked_ref())
        .ok();
    effort_cb.forget();

    // Wire up mouse drag for orbit camera
    wire_mouse_controls(&renderer);

    // Flush cache on page close
    let unload_cb = Closure::wrap(Box::new(move |_: web_sys::Event| {
        crate::cache::flush();
    }) as Box<dyn FnMut(web_sys::Event)>);
    dom::window()
        .add_event_listener_with_callback("beforeunload", unload_cb.as_ref().unchecked_ref())
        .ok();
    unload_cb.forget();

    // Start render loop with coordinator tick
    start_render_loop(renderer, Some(coordinator), client, muted, base_units, lb_state);
}

fn wire_mouse_controls(renderer: &Rc<RefCell<Renderer>>) {
    let r = renderer.clone();
    let canvas_el = dom::canvas();
    let dragging = Rc::new(RefCell::new(false));
    let last_pos = Rc::new(RefCell::new((0.0f32, 0.0f32)));

    // Mouse controls
    let d1 = dragging.clone();
    let lp1 = last_pos.clone();
    let r1 = r.clone();
    let mousedown = Closure::wrap(Box::new(move |e: web_sys::MouseEvent| {
        *d1.borrow_mut() = true;
        *lp1.borrow_mut() = (e.client_x() as f32, e.client_y() as f32);
        let mut r = r1.borrow_mut();
        r.camera.user_dragging = true;
        r.clear_hover();
    }) as Box<dyn FnMut(web_sys::MouseEvent)>);
    canvas_el
        .add_event_listener_with_callback("mousedown", mousedown.as_ref().unchecked_ref())
        .ok();
    mousedown.forget();

    let d2 = dragging.clone();
    let r2 = r.clone();
    let mouseup = Closure::wrap(Box::new(move |_: web_sys::MouseEvent| {
        *d2.borrow_mut() = false;
        r2.borrow_mut().camera.user_dragging = false;
    }) as Box<dyn FnMut(web_sys::MouseEvent)>);
    canvas_el
        .add_event_listener_with_callback("mouseup", mouseup.as_ref().unchecked_ref())
        .ok();
    mouseup.forget();

    let d3 = dragging.clone();
    let lp3 = last_pos.clone();
    let r3 = r.clone();
    let mousemove = Closure::wrap(Box::new(move |e: web_sys::MouseEvent| {
        let mx = e.client_x() as f32;
        let my = e.client_y() as f32;
        if *d3.borrow() {
            let (lx, ly) = *lp3.borrow();
            let dx = mx - lx;
            let dy = my - ly;
            *lp3.borrow_mut() = (mx, my);
            r3.borrow_mut().camera.rotate(dx, dy);
        } else {
            // Update hover position for tooltip
            r3.borrow_mut().set_hover(mx, my);
        }
    }) as Box<dyn FnMut(web_sys::MouseEvent)>);
    canvas_el
        .add_event_listener_with_callback("mousemove", mousemove.as_ref().unchecked_ref())
        .ok();
    mousemove.forget();

    // Clear hover on mouse leave
    let r_leave = r.clone();
    let mouseleave = Closure::wrap(Box::new(move |_: web_sys::MouseEvent| {
        r_leave.borrow_mut().clear_hover();
    }) as Box<dyn FnMut(web_sys::MouseEvent)>);
    canvas_el
        .add_event_listener_with_callback("mouseleave", mouseleave.as_ref().unchecked_ref())
        .ok();
    mouseleave.forget();

    let r4 = r.clone();
    let wheel = Closure::wrap(Box::new(move |e: web_sys::WheelEvent| {
        r4.borrow_mut().camera.zoom(e.delta_y() as f32);
    }) as Box<dyn FnMut(web_sys::WheelEvent)>);
    canvas_el
        .add_event_listener_with_callback("wheel", wheel.as_ref().unchecked_ref())
        .ok();
    wheel.forget();

    // Touch controls for mobile
    let touch_pos = Rc::new(RefCell::new((0.0f32, 0.0f32)));
    let touch_dist = Rc::new(RefCell::new(0.0f32));

    let tp1 = touch_pos.clone();
    let rt1 = r.clone();
    let touchstart = Closure::wrap(Box::new(move |e: web_sys::TouchEvent| {
        e.prevent_default();
        rt1.borrow_mut().camera.user_dragging = true;
        if let Some(t) = e.touches().get(0) {
            *tp1.borrow_mut() = (t.client_x() as f32, t.client_y() as f32);
        }
    }) as Box<dyn FnMut(web_sys::TouchEvent)>);
    canvas_el
        .add_event_listener_with_callback("touchstart", touchstart.as_ref().unchecked_ref())
        .ok();
    touchstart.forget();

    let tp2 = touch_pos.clone();
    let td2 = touch_dist.clone();
    let r5 = r.clone();
    let touchmove = Closure::wrap(Box::new(move |e: web_sys::TouchEvent| {
        e.prevent_default();
        let touches = e.touches();
        if touches.length() == 2 {
            // Pinch to zoom
            if let (Some(t0), Some(t1)) = (touches.get(0), touches.get(1)) {
                let dx = (t1.client_x() - t0.client_x()) as f32;
                let dy = (t1.client_y() - t0.client_y()) as f32;
                let dist = (dx * dx + dy * dy).sqrt();
                let prev = *td2.borrow();
                if prev > 0.0 {
                    let delta = (prev - dist) * 2.0;
                    r5.borrow_mut().camera.zoom(delta);
                }
                *td2.borrow_mut() = dist;
            }
        } else if let Some(t) = touches.get(0) {
            // Single finger drag to orbit
            let (lx, ly) = *tp2.borrow();
            let dx = t.client_x() as f32 - lx;
            let dy = t.client_y() as f32 - ly;
            *tp2.borrow_mut() = (t.client_x() as f32, t.client_y() as f32);
            r5.borrow_mut().camera.rotate(dx, dy);
        }
    }) as Box<dyn FnMut(web_sys::TouchEvent)>);
    canvas_el
        .add_event_listener_with_callback("touchmove", touchmove.as_ref().unchecked_ref())
        .ok();
    touchmove.forget();

    let td3 = touch_dist;
    let rt3 = r.clone();
    let touchend = Closure::wrap(Box::new(move |_: web_sys::TouchEvent| {
        *td3.borrow_mut() = 0.0;
        rt3.borrow_mut().camera.user_dragging = false;
    }) as Box<dyn FnMut(web_sys::TouchEvent)>);
    canvas_el
        .add_event_listener_with_callback("touchend", touchend.as_ref().unchecked_ref())
        .ok();
    touchend.forget();
}

fn start_render_loop(
    renderer: Rc<RefCell<Renderer>>,
    coordinator: Option<Rc<Coordinator>>,
    client: Rc<SupabaseClient>,
    muted: Rc<Cell<bool>>,
    base_units: Rc<Cell<i64>>,
    lb_state: Rc<RefCell<LeaderboardState>>,
) {
    let r = renderer;
    let coord = coordinator;
    let frame_count = Rc::new(RefCell::new(0u32));
    let last_count = Rc::new(RefCell::new(0u32));
    let last_rate_time = Rc::new(RefCell::new(0.0f64));
    let completion_handled = Rc::new(RefCell::new(false));
    let last_rounds = Rc::new(Cell::new(0u32));
    let anim = Rc::new(RefCell::new(AnimState::new()));
    // Experiment progress — polled from server
    let exp_submitted: Rc<Cell<usize>> = Rc::new(Cell::new(0));
    let exp_consensus: Rc<Cell<usize>> = Rc::new(Cell::new(0));
    let exp_total: Rc<Cell<usize>> = Rc::new(Cell::new(0));
    let exp_fetching: Rc<Cell<bool>> = Rc::new(Cell::new(false));

    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();

    let fc = frame_count.clone();
    let lc = last_count.clone();
    let lrt = last_rate_time.clone();
    let ch = completion_handled.clone();
    let an = anim.clone();
    let es = exp_submitted.clone();
    let ec = exp_consensus.clone();
    let et = exp_total.clone();
    let ef = exp_fetching.clone();
    let cl = client.clone();
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        // Render
        r.borrow_mut().render();

        // Tick coordinator (processes results, dispatches work)
        if let Some(ref c) = coord {
            c.tick();
        }

        // Animate numbers every frame
        {
            let mut a = an.borrow_mut();
            a.tick();
            dom::set_text("progress-num", &format!("{}", a.progress.as_int()));
            dom::set_text("my-rate", &format!("{:.1}", a.rate.current));
            dom::set_text("my-count", &format!("{}", a.completed.as_int()));
            dom::set_text("user-points", &format!("({})", a.completed.as_int()));
            if a.gn.target != 0.0 {
                let gn_str = format!("{:.1}", a.gn.current);
                dom::set_text("gn", &gn_str);
                dom::set_text("gn-splash", &gn_str);
            }
            if a.r2.target != 0.0 {
                dom::set_text("r2", &format!("{:.3}", a.r2.current));
            }
            // Experiment progress (target updated from async poll)
            a.exp_submitted.set(es.get() as f64);
            a.exp_consensus.set(ec.get() as f64);
            a.exp_total.set(et.get() as f64);
            dom::set_text("exp-submitted", &format!("{}", a.exp_submitted.as_int()));
            dom::set_text("exp-total", &format!("{}", a.exp_total.as_int()));
            let consensus = a.exp_consensus.as_int();
            dom::set_text("exp-consensus", &format!("{}", consensus));
            if consensus > 0 {
                dom::set_class("exp-consensus-wrap", "");
            }
        }

        // Tick leaderboard animations and re-render if in-flight
        {
            let mut lb = lb_state.borrow_mut();
            if lb.tick() {
                render_leaderboard_panel(&lb);
            }
        }

        // Update stats periodically (~1s)
        let count = {
            let mut c = fc.borrow_mut();
            *c += 1;
            *c
        };

        // Update worker lanes every ~10 frames (~167ms)
        if count % 10 == 0 {
            if let Some(ref c) = coord {
                let slots = c.worker_slots();
                let mut html = String::new();
                let mut n_active = 0usize;
                let mut n_idle = 0usize;
                for (i, slot) in slots.iter().enumerate() {
                    if slot.active {
                        n_active += 1;
                        let pct = if slot.total > 0 {
                            (slot.done as f64 / slot.total as f64 * 100.0).min(100.0)
                        } else {
                            0.0
                        };
                        let frac = format!("{}/{}", slot.done, slot.total);
                        let result_html = slot.last_result.as_deref().unwrap_or("");
                        html.push_str(&format!(
                            "<div class=\"wk\" title=\"Worker {i} — computing eigenvalues for this parameter set\">\
                             <span class=\"wk-idx\">w{i}</span>\
                             <span class=\"wk-params\" title=\"Level, coupling g², perturbation direction\">{}</span>\
                             <span class=\"wk-bar\" title=\"Batch progress: {}/{}\">\
                             <span class=\"wk-fill\" style=\"width:{pct:.0}%\"></span></span>\
                             <span class=\"wk-frac\">{frac}</span>\
                             <span class=\"wk-result\" title=\"Last result: ground-state energy, partition count, wall time\">{result_html}</span>\
                             </div>",
                            slot.label, slot.done, slot.total,
                        ));
                    } else {
                        n_idle += 1;
                    }
                }
                if n_idle > 0 && n_active > 0 {
                    html.push_str(&format!(
                        "<div class=\"wk-idle-summary\">+ {n_idle} idle</div>"
                    ));
                }
                dom::set_inner_html("workers", &html);
                let total = n_active + n_idle;
                dom::set_text("worker-count", &format!("{n_active}/{total}w"));
            }
        }

        // Flush cache to localStorage every ~5s (not every frame)
        if count % 300 == 0 {
            crate::cache::flush();
        }

        // Poll experiment progress every ~30s (or immediately on first frame)
        if (count == 1 || count % 1800 == 0) && !ef.get() {
            ef.set(true);
            let client_ep = cl.clone();
            let es2 = es.clone();
            let ec2 = ec.clone();
            let et2 = et.clone();
            let ef2 = ef.clone();
            wasm_bindgen_futures::spawn_local(async move {
                if let Ok(p) = client_ep.experiment_progress().await {
                    es2.set(p.submitted);
                    ec2.set(p.consensus);
                    et2.set(p.total);
                }
                ef2.set(false);
            });
        }

        if count % 60 == 0 {
            let mut a = an.borrow_mut();

            if let Some(ref c) = coord {
                // Points count — only updates on round submission
                a.points.set(c.visible_points() as f64);
                // User's personal points: server total + session completions
                let my_units = (base_units.get() as u32 + c.completed_count()) as usize;
                a.completed.set(my_units as f64);

                // Round progress
                let (round_done, round_total) = c.round_progress();
                a.progress.set(round_done as f64);
                dom::set_text("progress-total", &round_total.to_string());
                dom::set_text("round-status", c.state_label());
                let pct = if round_total > 0 {
                    (round_done as f64 / round_total as f64 * 100.0).min(100.0)
                } else {
                    0.0
                };
                dom::set_style("progress-fill", &format!("width:{pct:.1}%"));

                // Celebrate round completion
                let rounds_now = c.rounds_completed();
                if rounds_now > last_rounds.get() {
                    last_rounds.set(rounds_now);
                    spawn_confetti();
                    if !muted.get() {
                        play_success_sound();
                    }
                }

                // Compute rate (session-only)
                let session_count = c.completed_count();
                let now = js_sys::Date::now();
                let prev_time = *lrt.borrow();
                let prev_count = *lc.borrow();
                if prev_time > 0.0 {
                    let dt = (now - prev_time) / 1000.0;
                    if dt > 0.0 {
                        let rate = (session_count - prev_count) as f64 / dt;
                        a.rate.set(rate);
                    }
                }
                *lrt.borrow_mut() = now;
                *lc.borrow_mut() = session_count;

                // Compute G_N from all data + update convergence bar
                if r.borrow().points.len() >= 10 {
                    let (slope, r2) = linear_regression(&r.borrow().points);
                    let pct = (r2 * 100.0).min(100.0);
                    dom::set_style("convergence-fill", &format!("width:{pct:.1}%"));
                    a.r2.set(r2);
                    if slope > 0.0 {
                        let g_n = 1.0 / (4.0 * slope);
                        a.gn.set(g_n);
                    }
                }

                // Snap on first update to avoid animating from 0 on page load
                if !a.initialized {
                    a.snap_all();
                    a.initialized = true;
                }

                // Detect experiment completion
                if c.is_complete() && !*ch.borrow() {
                    *ch.borrow_mut() = true;
                    dom::set_text("status-text", "complete");
                    dom::set_text("toggle", "\u{2713}");
                    dom::set_class("toggle", "complete");
                    dom::set_class("activity-dot", "indicator done");
                    a.rate.set(0.0);
                    a.rate.snap();
                }
            } else {
                // No coordinator (demo/no-worker mode) — use renderer directly
                a.points.set(r.borrow().points.len() as f64);
            }
        }

        // Request next frame
        dom::request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));

    dom::request_animation_frame(g.borrow().as_ref().unwrap());
}

fn spawn_confetti() {
    let doc = dom::document();
    let Some(container) = doc.get_element_by_id("confetti-container") else {
        return;
    };
    let colors = ["#44cc66", "#ccaa33", "#cc4444", "#4488cc", "#cc44cc", "#44cccc"];

    let mut seed: u64 = js_sys::Date::now() as u64;
    let mut rng = |max: f64| -> f64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64 / u64::MAX as f64) * max
    };

    for _ in 0..60 {
        let el = doc.create_element("div").unwrap();
        el.set_class_name("confetti-particle");
        let x = rng(100.0);
        let size = 4.0 + rng(4.0);
        let delay = rng(0.5);
        let duration = 1.5 + rng(1.0);
        let color = colors[rng(colors.len() as f64) as usize % colors.len()];
        el.set_attribute(
            "style",
            &format!(
                "left:{x:.0}%;width:{size:.0}px;height:{size:.0}px;\
                 background:{color};animation-duration:{duration:.2}s;\
                 animation-delay:{delay:.2}s"
            ),
        )
        .ok();
        container.append_child(&el).ok();
    }

    let window = dom::window();
    let clear_cb = Closure::once(move || {
        container.set_inner_html("");
    });
    window
        .set_timeout_with_callback_and_timeout_and_arguments_0(
            clear_cb.as_ref().unchecked_ref(),
            3000,
        )
        .ok();
    clear_cb.forget();
}

fn play_success_sound() {
    let ctx = match web_sys::AudioContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };

    let gain = ctx.create_gain().unwrap();
    gain.gain().set_value(0.15);
    gain.connect_with_audio_node(&ctx.destination()).ok();

    let osc1 = ctx.create_oscillator().unwrap();
    osc1.set_type(web_sys::OscillatorType::Sine);
    osc1.frequency().set_value(523.0);
    osc1.connect_with_audio_node(&gain).ok();
    let now = ctx.current_time();
    osc1.start_with_when(now).ok();
    osc1.stop_with_when(now + 0.1).ok();

    let osc2 = ctx.create_oscillator().unwrap();
    osc2.set_type(web_sys::OscillatorType::Sine);
    osc2.frequency().set_value(659.0);
    osc2.connect_with_audio_node(&gain).ok();
    osc2.start_with_when(now + 0.1).ok();
    osc2.stop_with_when(now + 0.2).ok();

    let close_cb = Closure::once(move || {
        ctx.close().ok();
    });
    dom::window()
        .set_timeout_with_callback_and_timeout_and_arguments_0(
            close_cb.as_ref().unchecked_ref(),
            300,
        )
        .ok();
    close_cb.forget();
}

/// Seed demo data points that mimic the S_EE ~ (1/4G_N) * A_cut relationship.
fn seed_demo_data(renderer: &mut crate::viz::Renderer) {
    let g2_values: Vec<f64> = (0..40)
        .map(|i| {
            let t = i as f64 / 39.0;
            10.0_f64.powf(-2.0 + t * 4.0)
        })
        .collect();

    let mut seed: u64 = 42;
    let mut rng = || -> f64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64 / u64::MAX as f64) - 0.5
    };

    for &g2 in &g2_values {
        let log_g2 = g2.log10();
        let g_n = 200.0 + 50.0 * (-(log_g2).powi(2) / 2.0).exp();
        let slope = 1.0 / (4.0 * g_n);

        for a_cut_base in &[0.5, 1.0, 1.5, 2.0, 2.5] {
            let a_cut = a_cut_base + rng() * 0.1;
            let s_ee = slope * a_cut + rng() * 0.0005;
            renderer.add_point(log_g2, s_ee.max(0.0), a_cut);
        }
    }

    let (slope, r2) = linear_regression(&renderer.points);
    let pct = (r2 * 100.0).min(100.0);
    dom::set_style("convergence-fill", &format!("width:{pct:.1}%"));
    dom::set_text("r2", &format!("{r2:.3} (demo)"));
    if slope > 0.0 {
        let g_n = 1.0 / (4.0 * slope);
        let gn_str = format!("{g_n:.0}");
        dom::set_text("gn", &gn_str);
        dom::set_text("gn-splash", &gn_str);
    }
}

fn show_auth_ui(name: &str) {
    dom::set_class("auth-anon", "hidden");
    dom::set_class("auth-user", "");
    dom::set_text("user-name", name);
    dom::set_class("auth-modal", "hidden");
}

/// A single row in the leaderboard with animation state.
struct LeaderboardRow {
    player_id: String,
    name: String,
    units: Anim,
    prev_rank: Option<u32>,
    rank_change_age: f64,
}

/// Manages the full leaderboard state with animated transitions.
struct LeaderboardState {
    rows: Vec<LeaderboardRow>,
    my_id: String,
    my_rank: Option<u32>,
    initialized: bool,
}

impl LeaderboardState {
    fn new(my_id: String) -> Self {
        Self {
            rows: Vec::new(),
            my_id,
            my_rank: None,
            initialized: false,
        }
    }

    /// Update with new entries from REST. Computes rank changes and preserves animation state.
    fn update(&mut self, entries: &[crate::supabase::LeaderboardEntry], auth_id: Option<&str>) {
        use std::collections::HashMap;

        // Build lookup of existing rows by player_id
        let old_ranks: HashMap<String, u32> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.player_id.clone(), i as u32 + 1))
            .collect();

        let mut new_rows = Vec::with_capacity(entries.len());
        self.my_rank = None;

        for (i, entry) in entries.iter().enumerate() {
            let rank = i as u32 + 1;
            let is_me = auth_id.map_or(false, |id| id == entry.player_id)
                || entry.player_id == self.my_id;

            if is_me {
                self.my_rank = Some(rank);
            }

            // Find existing row to preserve animation state
            let prev_rank = old_ranks.get(&entry.player_id).copied();
            let (units_anim, rank_change_age) =
                if let Some(old_row) = self.rows.iter().find(|r| r.player_id == entry.player_id) {
                    let mut a = Anim {
                        current: old_row.units.current,
                        target: old_row.units.target,
                    };
                    a.set(entry.units as f64);
                    // New rank change if rank actually changed
                    let age = if prev_rank != Some(rank) && self.initialized {
                        0.0
                    } else {
                        old_row.rank_change_age
                    };
                    (a, age)
                } else {
                    let mut a = Anim::new();
                    a.set(entry.units as f64);
                    if !self.initialized {
                        a.snap();
                    }
                    (a, 300.0) // no change indicator for new entries
                };

            new_rows.push(LeaderboardRow {
                player_id: entry.player_id.clone(),
                name: entry.name.clone().unwrap_or_else(|| "anonymous".to_string()),
                units: units_anim,
                prev_rank,
                rank_change_age,
            });
        }

        self.rows = new_rows;
        self.initialized = true;
    }

    /// Tick all animations. Returns true if any animation is still in-flight.
    fn tick(&mut self) -> bool {
        let mut any_active = false;
        for row in &mut self.rows {
            row.units.tick(0.5);
            if (row.units.current - row.units.target).abs() > 0.5 {
                any_active = true;
            }
            if row.rank_change_age < 300.0 {
                row.rank_change_age += 1.0;
                any_active = true;
            }
        }
        any_active
    }
}

fn render_leaderboard_panel(state: &LeaderboardState) {
    if state.rows.is_empty() {
        dom::set_inner_html("leaderboard", "<div class=\"lb-empty\">no entries yet</div>");
        dom::set_inner_html("lb-me", "");
        return;
    }

    let mut html = String::new();
    let mut found_me = false;

    for (i, row) in state.rows.iter().enumerate() {
        let rank = i + 1;
        let is_me = row.player_id == state.my_id;
        if is_me {
            found_me = true;
        }
        let me_class = if is_me { " me" } else { "" };

        // Rank change indicator (fades over ~5s = ~300 frames)
        let change_html = if let Some(prev) = row.prev_rank {
            let current_rank = rank as u32;
            if prev > current_rank && row.rank_change_age < 300.0 {
                let opacity = 1.0 - (row.rank_change_age / 300.0);
                format!(
                    "<span class=\"lb-change up\" style=\"opacity:{opacity:.2}\">\u{25B2}</span>"
                )
            } else if prev < current_rank && row.rank_change_age < 300.0 {
                let opacity = 1.0 - (row.rank_change_age / 300.0);
                format!(
                    "<span class=\"lb-change down\" style=\"opacity:{opacity:.2}\">\u{25BC}</span>"
                )
            } else {
                "<span class=\"lb-change\"></span>".to_string()
            }
        } else {
            "<span class=\"lb-change\"></span>".to_string()
        };

        let you_badge = if is_me {
            "<span class=\"lb-you\">you</span>"
        } else {
            ""
        };

        html.push_str(&format!(
            "<div class=\"lb-row{me_class}\">\
             <span class=\"lb-rank\">{rank}</span>\
             {change_html}\
             <span class=\"lb-name\">{}{you_badge}</span>\
             <span class=\"lb-units\">{}</span>\
             </div>",
            row.name,
            row.units.as_int(),
        ));
    }
    dom::set_inner_html("leaderboard", &html);

    // Personal rank footer when user is outside visible top N
    if let Some(my_rank) = state.my_rank {
        if !found_me {
            dom::set_inner_html(
                "lb-me",
                &format!("you \u{2014} #{my_rank}"),
            );
        } else {
            dom::set_inner_html("lb-me", "");
        }
    } else {
        dom::set_inner_html("lb-me", "");
    }
}

/// Smoothly animated numeric display value.
struct Anim {
    current: f64,
    target: f64,
}

impl Anim {
    fn new() -> Self {
        Anim { current: 0.0, target: 0.0 }
    }

    fn set(&mut self, target: f64) {
        self.target = target;
    }

    /// Snap current to target (skip animation).
    fn snap(&mut self) {
        self.current = self.target;
    }

    /// Lerp toward target. Call once per frame. `snap` controls integer rounding threshold.
    fn tick(&mut self, snap_dist: f64) {
        self.current += (self.target - self.current) * 0.12;
        if (self.current - self.target).abs() < snap_dist {
            self.current = self.target;
        }
    }

    fn as_int(&self) -> usize {
        self.current as usize
    }
}

/// All animated counters for the UI.
struct AnimState {
    points: Anim,
    completed: Anim,
    progress: Anim,
    rate: Anim,
    gn: Anim,
    r2: Anim,
    exp_submitted: Anim,
    exp_consensus: Anim,
    exp_total: Anim,
    initialized: bool,
}

impl AnimState {
    fn new() -> Self {
        AnimState {
            points: Anim::new(),
            completed: Anim::new(),
            progress: Anim::new(),
            rate: Anim::new(),
            gn: Anim::new(),
            r2: Anim::new(),
            exp_submitted: Anim::new(),
            exp_consensus: Anim::new(),
            exp_total: Anim::new(),
            initialized: false,
        }
    }

    fn tick(&mut self) {
        self.points.tick(0.5);
        self.completed.tick(0.5);
        self.progress.tick(0.5);
        self.rate.tick(0.01);
        self.gn.tick(0.01);
        self.r2.tick(0.0001);
        self.exp_submitted.tick(0.5);
        self.exp_consensus.tick(0.5);
        self.exp_total.tick(0.5);
    }

    fn snap_all(&mut self) {
        self.points.snap();
        self.completed.snap();
        self.progress.snap();
        self.rate.snap();
        self.gn.snap();
        self.r2.snap();
        self.exp_submitted.snap();
        self.exp_consensus.snap();
        self.exp_total.snap();
    }
}

/// Linear regression: S_EE = slope * A_cut + intercept. Returns (slope, R²).
fn linear_regression(points: &[crate::viz::DataPoint]) -> (f64, f64) {
    if points.len() < 2 {
        return (0.0, 0.0);
    }
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|p| p.a_cut as f64).sum();
    let sum_y: f64 = points.iter().map(|p| p.s_ee as f64).sum();
    let sum_xy: f64 = points.iter().map(|p| p.a_cut as f64 * p.s_ee as f64).sum();
    let sum_xx: f64 = points.iter().map(|p| (p.a_cut as f64).powi(2)).sum();
    let sum_yy: f64 = points.iter().map(|p| (p.s_ee as f64).powi(2)).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-20 {
        return (0.0, 0.0);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;

    // R² = (correlation coefficient)²
    let num_r = n * sum_xy - sum_x * sum_y;
    let den_r = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();
    let r2 = if den_r.abs() < 1e-20 {
        0.0
    } else {
        (num_r / den_r).powi(2)
    };

    (slope, r2)
}
