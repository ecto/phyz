use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::supabase::{PendingResult, ResultPayload, SupabaseClient, WorkUnit};
use crate::viz::Renderer;
use crate::worker::WorkerPool;

const BATCH_SIZE: usize = 25;
const ROUND_SIZE: usize = 1024;

/// Round lifecycle state machine.
enum RoundState {
    Idle,
    Fetching,
    Computing {
        round_total: usize,
        results: Vec<PendingResult>,
    },
    Submitting {
        round_total: usize,
    },
    Draining {
        results: Vec<PendingResult>,
    },
    Complete,
}

/// State of a single worker slot for the UI.
pub struct WorkerSlot {
    pub active: bool,
    pub label: String,
    pub done: usize,
    pub total: usize,
    pub last_result: Option<String>,
}

/// Compute slope and R² from a linear fit of y = slope * x + intercept.
/// Returns (slope, r_squared). If fewer than 2 points or zero variance, returns (0, 0).
fn linear_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len().min(y.len());
    if n < 2 {
        return (0.0, 0.0);
    }
    let nf = n as f64;
    let x_mean: f64 = x[..n].iter().sum::<f64>() / nf;
    let y_mean: f64 = y[..n].iter().sum::<f64>() / nf;
    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }
    let slope = if ss_xx > 1e-30 { ss_xy / ss_xx } else { 0.0 };
    let r2 = if ss_xx > 1e-30 && ss_yy > 1e-30 {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    } else {
        0.0
    };
    (slope, r2)
}

/// Client-side coordinator: fetches work, dispatches to workers, submits results.
pub struct Coordinator {
    client: Rc<SupabaseClient>,
    pool: Rc<WorkerPool>,
    contributor_id: String,
    state: Rc<RefCell<RoundState>>,
    round_queue: RefCell<VecDeque<WorkUnit>>,
    dispatched: RefCell<HashMap<String, WorkUnit>>,
    completed_count: Rc<RefCell<u32>>,
    running: Rc<RefCell<bool>>,
    renderer: Rc<RefCell<Renderer>>,
    /// Per-worker label for the lane UI (worker_idx → label).
    worker_labels: RefCell<Vec<String>>,
    /// Per-worker last result summary (worker_idx → short description).
    worker_results: RefCell<Vec<Option<String>>>,
    /// Maps work unit ID → worker index for attributing results.
    unit_worker: RefCell<HashMap<String, usize>>,
    /// Effort level 1-100 — controls how many workers are active.
    effort: Rc<RefCell<u32>>,
    /// Number of rounds successfully submitted (for celebration triggers).
    rounds_completed: Rc<RefCell<u32>>,
    /// Worker batch sizes: worker_idx → items dispatched in current batch.
    batch_sizes: RefCell<HashMap<usize, usize>>,
    /// Work units from completed worker batches (for round progress display).
    round_computed: RefCell<usize>,
    /// Points count shown in UI — only updates on round submission.
    visible_points: Rc<RefCell<usize>>,
}

impl Coordinator {
    pub fn new(
        client: Rc<SupabaseClient>,
        pool: Rc<WorkerPool>,
        contributor_id: String,
        renderer: Rc<RefCell<Renderer>>,
    ) -> Self {
        let n = pool.pool_size();
        let initial_points = renderer.borrow().points.len();
        Coordinator {
            client,
            pool,
            contributor_id,
            state: Rc::new(RefCell::new(RoundState::Idle)),
            round_queue: RefCell::new(VecDeque::new()),
            dispatched: RefCell::new(HashMap::new()),
            completed_count: Rc::new(RefCell::new(0)),
            running: Rc::new(RefCell::new(false)),
            renderer,
            worker_labels: RefCell::new(vec![String::new(); n]),
            worker_results: RefCell::new(vec![None; n]),
            unit_worker: RefCell::new(HashMap::new()),
            effort: Rc::new(RefCell::new(50)),
            rounds_completed: Rc::new(RefCell::new(0)),
            batch_sizes: RefCell::new(HashMap::new()),
            round_computed: RefCell::new(0),
            visible_points: Rc::new(RefCell::new(initial_points)),
        }
    }

    pub fn is_running(&self) -> bool {
        *self.running.borrow()
    }

    pub fn is_complete(&self) -> bool {
        matches!(*self.state.borrow(), RoundState::Complete)
    }

    pub fn completed_count(&self) -> u32 {
        *self.completed_count.borrow()
    }

    /// Points count for display — only updates on round submission.
    pub fn visible_points(&self) -> usize {
        *self.visible_points.borrow()
    }

    /// Number of rounds successfully submitted.
    pub fn rounds_completed(&self) -> u32 {
        *self.rounds_completed.borrow()
    }

    pub fn start(&self) {
        *self.running.borrow_mut() = true;
        web_sys::console::log_1(&"coordinator: started".into());
    }

    pub fn stop(&self) {
        *self.running.borrow_mut() = false;
        web_sys::console::log_1(&"coordinator: stopped".into());
    }

    /// Enter draining state: stop fetching/dispatching, let in-flight workers finish.
    pub fn drain(&self) {
        let mut state = self.state.borrow_mut();
        match &*state {
            RoundState::Draining { .. } | RoundState::Complete => return,
            _ => {}
        }
        // Take accumulated results from Computing state if present
        let results = match std::mem::replace(&mut *state, RoundState::Idle) {
            RoundState::Computing { results, .. } => results,
            _ => Vec::new(),
        };
        *state = RoundState::Draining { results };
        crate::dom::set_text("status-text", "finishing up");
        web_sys::console::log_1(&"coordinator: draining".into());
    }

    pub fn set_effort(&self, pct: u32) {
        *self.effort.borrow_mut() = pct.clamp(1, 100);
    }

    pub fn effort(&self) -> u32 {
        *self.effort.borrow()
    }

    /// Max active workers based on effort percentage.
    pub fn max_workers(&self) -> usize {
        let effort = *self.effort.borrow() as f64;
        let total = self.pool.pool_size() as f64;
        (total * effort / 100.0).ceil().max(1.0) as usize
    }

    /// (computed, round_total) for round progress display.
    /// Only counts work units from fully-completed worker batches.
    pub fn round_progress(&self) -> (usize, usize) {
        match &*self.state.borrow() {
            RoundState::Computing { round_total, .. } => {
                (*self.round_computed.borrow(), *round_total)
            }
            RoundState::Submitting { round_total } => (*round_total, *round_total),
            _ => (0, 0),
        }
    }

    /// Human-readable state label for the UI.
    pub fn state_label(&self) -> &'static str {
        match &*self.state.borrow() {
            RoundState::Idle => "idle",
            RoundState::Fetching => "fetching",
            RoundState::Computing { .. } => "computing",
            RoundState::Submitting { .. } => "submitting",
            RoundState::Draining { .. } => "finishing",
            RoundState::Complete => "complete",
        }
    }

    /// Build worker slot info for the UI.
    pub fn worker_slots(&self) -> Vec<WorkerSlot> {
        let progress = self.pool.worker_progress();
        let labels = self.worker_labels.borrow();
        let results = self.worker_results.borrow();
        progress
            .iter()
            .enumerate()
            .map(|(i, &(done, total))| WorkerSlot {
                active: total > 0,
                label: labels.get(i).cloned().unwrap_or_default(),
                done,
                total,
                last_result: results.get(i).cloned().flatten(),
            })
            .collect()
    }

    /// Called periodically (e.g. every 200ms) from the main app loop.
    pub fn tick(&self) {
        if !*self.running.borrow() {
            return;
        }

        // Always drain worker results first
        let worker_results = self.pool.drain_results();
        for resp in worker_results {
            self.handle_result(resp);
        }

        // Credit completed worker batches to round_computed
        {
            let progress = self.pool.worker_progress();
            let mut bs = self.batch_sizes.borrow_mut();
            let mut rc = self.round_computed.borrow_mut();
            // A worker with progress (0,0) is idle — if it had a batch, it's done
            bs.retain(|&idx, &mut size| {
                let (done, total) = progress.get(idx).copied().unwrap_or((0, 0));
                if total == 0 {
                    // Worker is idle — batch finished
                    *rc += size;
                    false
                } else if done >= total {
                    // About to become idle (drain_results will mark it next tick)
                    // Don't credit yet — wait until actually idle
                    true
                } else {
                    true
                }
            });
        }

        // State machine transitions
        let current_state = {
            // Brief borrow to determine what to do
            std::mem::discriminant(&*self.state.borrow())
        };

        match current_state {
            d if d == std::mem::discriminant(&RoundState::Idle) => {
                self.start_fetch();
            }
            d if d == std::mem::discriminant(&RoundState::Fetching) => {
                // no-op: async fetch callback will transition us
            }
            d if d
                == std::mem::discriminant(&RoundState::Computing {
                    round_total: 0,
                    results: Vec::new(),
                }) =>
            {
                self.tick_computing();
            }
            d if d == std::mem::discriminant(&RoundState::Submitting { round_total: 0 }) => {
                // no-op: async submit callback will transition us
            }
            d if d
                == std::mem::discriminant(&RoundState::Draining {
                    results: Vec::new(),
                }) =>
            {
                self.tick_draining();
            }
            _ => {
                // Complete — no-op
            }
        }
    }

    fn start_fetch(&self) {
        *self.state.borrow_mut() = RoundState::Fetching;
        *self.round_computed.borrow_mut() = 0;
        self.batch_sizes.borrow_mut().clear();

        let client = self.client.clone();
        let state = self.state.clone();

        wasm_bindgen_futures::spawn_local(async move {
            match client.fetch_pending_work(ROUND_SIZE).await {
                Ok(mut units) => {
                    // Shuffle so different volunteers naturally spread across
                    // the parameter space instead of all computing the same
                    // units in insertion order.
                    for i in (1..units.len()).rev() {
                        let j = (js_sys::Math::random() * (i + 1) as f64) as usize;
                        units.swap(i, j);
                    }
                    let count = units.len();
                    if count == 0 {
                        *state.borrow_mut() = RoundState::Complete;
                        web_sys::console::log_1(&"no more work — complete".into());
                    } else {
                        // Check if we were stopped/drained while fetching
                        let is_fetching =
                            matches!(*state.borrow(), RoundState::Fetching);
                        if is_fetching {
                            web_sys::console::log_1(
                                &format!("fetched {count} work units").into(),
                            );
                            // We'll set up Computing state — queue goes into round_queue
                            // but we can't access round_queue from here (not Send).
                            // Instead, stash units in a temporary state variant.
                            *state.borrow_mut() = RoundState::Computing {
                                round_total: count,
                                results: Vec::new(),
                            };
                        }
                        // If state was changed to Draining while we were fetching,
                        // the units are lost — server will re-serve them as pending.
                        // Push units into round_queue via a separate mechanism.
                        // Since we're in an async block that captured state but not
                        // round_queue, we need to store them somewhere accessible.
                        // Solution: we'll use a shared stash.
                        if matches!(*state.borrow(), RoundState::Computing { .. }) {
                            // Store units in a JS-side stash via a RefCell we captured
                            FETCH_STASH.with(|s| {
                                *s.borrow_mut() = units;
                            });
                        }
                    }
                }
                Err(e) => {
                    web_sys::console::warn_1(&format!("fetch work: {e}").into());
                    // Go back to idle to retry next tick
                    *state.borrow_mut() = RoundState::Idle;
                }
            }
        });
    }

    fn tick_computing(&self) {
        // Pull any stashed fetch results into round_queue
        FETCH_STASH.with(|s| {
            let mut stash = s.borrow_mut();
            if !stash.is_empty() {
                let mut rq = self.round_queue.borrow_mut();
                for u in stash.drain(..) {
                    rq.push_back(u);
                }
            }
        });

        // Dispatch batches to idle workers
        let max_active = self.max_workers();
        while self.pool.available() > 0
            && (self.pool.pool_size() - self.pool.available()) < max_active
        {
            let mut batch_units = Vec::new();
            {
                let mut rq = self.round_queue.borrow_mut();
                for _ in 0..BATCH_SIZE {
                    match rq.pop_front() {
                        Some(u) => batch_units.push(u),
                        None => break,
                    }
                }
            }
            if batch_units.is_empty() {
                break;
            }

            let items: Vec<(String, crate::supabase::WorkParams)> = batch_units
                .iter()
                .map(|u| (u.id.clone(), u.params.clone()))
                .collect();

            // Build label from first item
            let first = &batch_units[0];
            let pert_str = match &first.params.perturbation {
                crate::supabase::Perturbation::Base => "base".to_string(),
                crate::supabase::Perturbation::Edge { index, .. } => format!("e{index}"),
            };
            let label = format!(
                "L{} g²={:.1e} {}",
                first.params.level, first.params.coupling_g2, pert_str
            );

            // Track dispatched units
            for u in &batch_units {
                self.dispatched
                    .borrow_mut()
                    .insert(u.id.clone(), u.clone());
            }

            if let Some(worker_idx) = self.pool.dispatch_batch(&items) {
                self.worker_labels.borrow_mut()[worker_idx] = label;
                self.batch_sizes
                    .borrow_mut()
                    .insert(worker_idx, batch_units.len());
                let mut uw = self.unit_worker.borrow_mut();
                for u in &batch_units {
                    uw.insert(u.id.clone(), worker_idx);
                }
            }
        }

        // Check if all results for this round are in
        let (round_total, n_results) = {
            let state = self.state.borrow();
            match &*state {
                RoundState::Computing {
                    round_total,
                    results,
                } => (*round_total, results.len()),
                _ => return,
            }
        };

        if n_results >= round_total {
            // All done — take results and submit
            let results = match std::mem::replace(
                &mut *self.state.borrow_mut(),
                RoundState::Submitting { round_total },
            ) {
                RoundState::Computing { results, .. } => results,
                _ => unreachable!(),
            };
            self.submit_batch(results, round_total);
        }
    }

    fn tick_draining(&self) {
        // When all workers are idle and dispatched is empty, submit and complete
        if self.pool.available() == self.pool.pool_size()
            && self.dispatched.borrow().is_empty()
        {
            let results = match std::mem::replace(
                &mut *self.state.borrow_mut(),
                RoundState::Complete,
            ) {
                RoundState::Draining { results } => results,
                _ => Vec::new(),
            };
            if !results.is_empty() {
                let client = self.client.clone();
                let count = self.completed_count.clone();
                let vis_pts = self.visible_points.clone();
                let n = results.len();
                let pts_now = self.renderer.borrow().points.len();
                wasm_bindgen_futures::spawn_local(async move {
                    match client.submit_results_batch(&results).await {
                        Ok(()) => {
                            *count.borrow_mut() += n as u32;
                            *vis_pts.borrow_mut() = pts_now;
                            web_sys::console::log_1(
                                &format!("drain: submitted {n} results").into(),
                            );
                        }
                        Err(e) => {
                            web_sys::console::warn_1(
                                &format!("drain submit: {e}").into(),
                            );
                        }
                    }
                });
            }
            *self.running.borrow_mut() = false;
            web_sys::console::log_1(&"coordinator: complete".into());
        }
    }

    fn submit_batch(&self, results: Vec<PendingResult>, round_total: usize) {
        let client = self.client.clone();
        let state = self.state.clone();
        let count = self.completed_count.clone();
        let rounds = self.rounds_completed.clone();
        let vis_pts = self.visible_points.clone();
        let n = results.len();
        // Snapshot renderer point count now (all results already added)
        let pts_now = self.renderer.borrow().points.len();

        wasm_bindgen_futures::spawn_local(async move {
            match client.submit_results_batch(&results).await {
                Ok(()) => {
                    *count.borrow_mut() += n as u32;
                    *rounds.borrow_mut() += 1;
                    *vis_pts.borrow_mut() = pts_now;
                    web_sys::console::log_1(
                        &format!("submitted {n}/{round_total} results").into(),
                    );
                    *state.borrow_mut() = RoundState::Idle;
                }
                Err(e) => {
                    web_sys::console::warn_1(&format!("batch submit: {e}").into());
                    // Go back to idle — server will re-serve those units
                    *state.borrow_mut() = RoundState::Idle;
                }
            }
        });
    }

    fn handle_result(&self, resp: crate::worker::WorkerResponse) {
        if let Some(error) = &resp.error {
            web_sys::console::warn_1(&format!("worker error: {error}").into());
            return;
        }

        // Look up which worker produced this result
        let worker_idx = self.unit_worker.borrow_mut().remove(&resp.id);

        let Some(result_val) = resp.result else {
            return;
        };
        let Ok(payload) = serde_json::from_value::<ResultPayload>(result_val) else {
            web_sys::console::warn_1(&"failed to parse result payload".into());
            return;
        };

        // Look up original work params for viz
        let unit = self.dispatched.borrow_mut().remove(&resp.id);
        if let Some(ref unit) = unit {
            let log_g2 = unit.params.coupling_g2.log10();
            let n_partitions = payload.entropy_per_partition.len();

            // Expand: one viz point + cache entry per partition entropy
            let has_areas = !payload.boundary_area_per_partition.is_empty();
            let mut cache_batch = Vec::with_capacity(n_partitions);
            for (i, &s_ee) in payload.entropy_per_partition.iter().enumerate() {
                let a_cut = if has_areas {
                    payload.boundary_area_per_partition[i]
                } else {
                    i as f64
                };
                self.renderer
                    .borrow_mut()
                    .add_point(log_g2, s_ee, a_cut);

                cache_batch.push(crate::cache::CachedPoint {
                    log_g2,
                    s_ee,
                    a_cut,
                    partition_index: i,
                });
            }
            crate::cache::append_batch(&cache_batch);

            // Store per-worker last result summary with slope + R²
            if let Some(idx) = worker_idx {
                let short = if has_areas && n_partitions >= 2 {
                    let (slope, r2) = linear_fit(
                        &payload.boundary_area_per_partition,
                        &payload.entropy_per_partition,
                    );
                    format!(
                        "E\u{2080}={:.3} slope={:.3} R\u{00B2}={:.2} {:.0}ms",
                        payload.ground_state_energy, slope, r2, payload.walltime_ms,
                    )
                } else {
                    format!(
                        "E\u{2080}={:.3} S\u{00D7}{} {:.0}ms",
                        payload.ground_state_energy, n_partitions, payload.walltime_ms,
                    )
                };
                self.worker_results.borrow_mut()[idx] = Some(short);
            }
        }

        // Push result into the state's results vec (for batch submission)
        let pending = PendingResult {
            work_unit_id: resp.id,
            contributor_id: self.contributor_id.clone(),
            result: payload,
        };

        let mut state = self.state.borrow_mut();
        match &mut *state {
            RoundState::Computing { results, .. } => {
                results.push(pending);
            }
            RoundState::Draining { results } => {
                results.push(pending);
            }
            _ => {
                // Result arrived in unexpected state — log and drop
                web_sys::console::warn_1(
                    &format!("result in unexpected state: {}", pending.work_unit_id).into(),
                );
            }
        }
    }
}

thread_local! {
    /// Stash for work units fetched asynchronously, consumed by tick_computing.
    static FETCH_STASH: RefCell<Vec<WorkUnit>> = RefCell::new(Vec::new());
}
