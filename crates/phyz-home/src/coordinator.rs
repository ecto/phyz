use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::supabase::{ResultPayload, SupabaseClient, WorkUnit};
use crate::viz::Renderer;
use crate::worker::WorkerPool;

const BATCH_SIZE: usize = 25;

/// State of a single worker slot for the UI.
pub struct WorkerSlot {
    pub active: bool,
    pub label: String,
    pub done: usize,
    pub total: usize,
    pub last_result: Option<String>,
}

/// Client-side coordinator: fetches work, dispatches to workers, submits results.
pub struct Coordinator {
    client: Rc<SupabaseClient>,
    pool: Rc<WorkerPool>,
    contributor_id: String,
    queue: Rc<RefCell<VecDeque<WorkUnit>>>,
    dispatched: RefCell<HashMap<String, WorkUnit>>,
    completed_count: Rc<RefCell<u32>>,
    running: Rc<RefCell<bool>>,
    fetching: Rc<RefCell<bool>>,
    no_work: Rc<RefCell<bool>>,
    complete: Rc<RefCell<bool>>,
    renderer: Rc<RefCell<Renderer>>,
    /// When true, stop fetching/dispatching but keep draining in-flight work.
    draining: Rc<RefCell<bool>>,
    /// Per-worker label for the lane UI (worker_idx → label).
    worker_labels: RefCell<Vec<String>>,
    /// Per-worker last result summary (worker_idx → short description).
    worker_results: RefCell<Vec<Option<String>>>,
    /// Maps work unit ID → worker index for attributing results.
    unit_worker: RefCell<HashMap<String, usize>>,
    /// Effort level 1-100 — controls how many workers are active.
    effort: Rc<RefCell<u32>>,
}

impl Coordinator {
    pub fn new(
        client: Rc<SupabaseClient>,
        pool: Rc<WorkerPool>,
        contributor_id: String,
        renderer: Rc<RefCell<Renderer>>,
    ) -> Self {
        let n = pool.pool_size();
        Coordinator {
            client,
            pool,
            contributor_id,
            queue: Rc::new(RefCell::new(VecDeque::new())),
            dispatched: RefCell::new(HashMap::new()),
            completed_count: Rc::new(RefCell::new(0)),
            running: Rc::new(RefCell::new(false)),
            fetching: Rc::new(RefCell::new(false)),
            no_work: Rc::new(RefCell::new(false)),
            complete: Rc::new(RefCell::new(false)),
            renderer,
            draining: Rc::new(RefCell::new(false)),
            worker_labels: RefCell::new(vec![String::new(); n]),
            worker_results: RefCell::new(vec![None; n]),
            unit_worker: RefCell::new(HashMap::new()),
            effort: Rc::new(RefCell::new(50)),
        }
    }

    pub fn is_running(&self) -> bool {
        *self.running.borrow()
    }

    pub fn is_complete(&self) -> bool {
        *self.complete.borrow()
    }

    pub fn completed_count(&self) -> u32 {
        *self.completed_count.borrow()
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
        if *self.draining.borrow() {
            return;
        }
        *self.draining.borrow_mut() = true;
        crate::dom::set_text("status-text", "finishing up");
        web_sys::console::log_1(&"coordinator: draining".into());
    }

    pub fn is_draining(&self) -> bool {
        *self.draining.borrow()
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

        // 1. Always process completed results from workers (even when draining)
        let results = self.pool.drain_results();
        for resp in results {
            self.handle_result(resp);
        }

        let draining = *self.draining.borrow();

        if draining {
            // Draining: skip dispatch + fetch. When all workers idle → complete.
            if self.pool.available() == self.pool.pool_size()
                && self.dispatched.borrow().is_empty()
            {
                if !*self.complete.borrow() {
                    *self.complete.borrow_mut() = true;
                    *self.running.borrow_mut() = false;
                    web_sys::console::log_1(&"experiment complete!".into());
                }
            }
            return;
        }

        // 2. Dispatch batches to idle workers (limited by effort)
        let max_active = self.max_workers();
        while self.pool.available() > 0
            && (self.pool.pool_size() - self.pool.available()) < max_active
        {
            let mut batch_units = Vec::new();
            {
                let mut q = self.queue.borrow_mut();
                for _ in 0..BATCH_SIZE {
                    match q.pop_front() {
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
                let mut uw = self.unit_worker.borrow_mut();
                for u in &batch_units {
                    uw.insert(u.id.clone(), worker_idx);
                }
            }
        }

        // 3. Enter drain when server says no more work and local queue is empty
        if self.queue.borrow().is_empty()
            && *self.no_work.borrow()
            && !*self.fetching.borrow()
        {
            self.drain();
            return;
        }

        // 4. Fetch more work if queue is low and not already fetching
        if self.queue.borrow().len() < BATCH_SIZE * 2 && !*self.fetching.borrow() {
            *self.fetching.borrow_mut() = true;

            let client = self.client.clone();
            let queue = self.queue.clone();
            let fetching = self.fetching.clone();
            let no_work = self.no_work.clone();

            wasm_bindgen_futures::spawn_local(async move {
                match client.fetch_pending_work(BATCH_SIZE * 4).await {
                    Ok(units) => {
                        let count = units.len();
                        *no_work.borrow_mut() = count == 0;
                        let mut q = queue.borrow_mut();
                        for u in units {
                            q.push_back(u);
                        }
                        if count > 0 {
                            web_sys::console::log_1(
                                &format!("fetched {count} work units").into(),
                            );
                        }
                    }
                    Err(e) => {
                        web_sys::console::warn_1(&format!("fetch work: {e}").into());
                    }
                }
                *fetching.borrow_mut() = false;
            });
        }
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

            // Store per-worker last result summary
            if let Some(idx) = worker_idx {
                let short = format!(
                    "E\u{2080}={:.3} S\u{00D7}{} {:.0}ms",
                    payload.ground_state_energy, n_partitions, payload.walltime_ms,
                );
                self.worker_results.borrow_mut()[idx] = Some(short);
            }
        }

        // Submit result to Supabase — only count as completed on success
        let client = self.client.clone();
        let uid = resp.id;
        let cid = self.contributor_id.clone();
        let count = self.completed_count.clone();
        wasm_bindgen_futures::spawn_local(async move {
            match client.submit_result(&uid, &cid, &payload).await {
                Ok(()) => {
                    *count.borrow_mut() += 1;
                }
                Err(e) => {
                    web_sys::console::warn_1(&format!("submit: {e}").into());
                }
            }
        });
    }
}
