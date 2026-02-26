use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{MessageEvent, Worker};

use crate::supabase::WorkParams;

/// A single item in a batch request.
#[derive(serde::Serialize)]
pub struct BatchItem {
    pub id: String,
    pub params: WorkParams,
}

/// Batch message sent to a compute worker.
#[derive(serde::Serialize)]
pub struct WorkerBatchRequest {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub items: Vec<BatchItem>,
}

/// Progress of the current batch (sent with each result).
#[derive(serde::Deserialize, Debug, Clone)]
pub struct BatchProgress {
    pub done: usize,
    pub total: usize,
}

/// Message received from a compute worker.
#[derive(serde::Deserialize, Debug, Clone)]
pub struct WorkerResponse {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub id: String,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub progress: Option<BatchProgress>,
}

/// Manages a pool of Web Workers for compute.
/// Uses interior mutability so it can be behind a plain Rc<WorkerPool>.
pub struct WorkerPool {
    workers: Vec<Worker>,
    idle: RefCell<Vec<usize>>,
    /// Shared with the onmessage closures — results accumulate here.
    results: Rc<RefCell<Vec<WorkerResponse>>>,
    /// Per-worker batch progress (done, total). Shared with onmessage closures.
    progress: Rc<RefCell<Vec<(usize, usize)>>>,
}

impl WorkerPool {
    /// Spawn `n` workers with persistent message handlers.
    pub fn new(n: usize) -> Result<Self, String> {
        let idle = RefCell::new((0..n).collect());
        let results: Rc<RefCell<Vec<WorkerResponse>>> = Rc::new(RefCell::new(Vec::new()));
        let progress: Rc<RefCell<Vec<(usize, usize)>>> = Rc::new(RefCell::new(vec![(0, 0); n]));

        let mut workers = Vec::with_capacity(n);
        for i in 0..n {
            let w = Worker::new("/workers/compute.js").map_err(|e| format!("{e:?}"))?;

            let results_ref = results.clone();
            let progress_ref = progress.clone();
            let handler = Closure::wrap(Box::new(move |e: MessageEvent| {
                let data = e.data().as_string().unwrap_or_default();
                match serde_json::from_str::<WorkerResponse>(&data) {
                    Ok(resp) => {
                        // Update per-worker progress
                        if let Some(ref prog) = resp.progress {
                            progress_ref.borrow_mut()[i] = (prog.done, prog.total);
                        } else {
                            // Single-item dispatch (no batch progress) — mark complete
                            progress_ref.borrow_mut()[i] = (1, 1);
                        }
                        results_ref.borrow_mut().push(resp);
                    }
                    Err(err) => {
                        web_sys::console::warn_1(&format!("worker {i} parse error: {err}").into());
                    }
                }
            }) as Box<dyn FnMut(MessageEvent)>);

            w.set_onmessage(Some(handler.as_ref().unchecked_ref()));
            handler.forget();

            workers.push(w);
        }

        Ok(WorkerPool {
            workers,
            idle,
            results,
            progress,
        })
    }

    /// Number of idle workers.
    pub fn available(&self) -> usize {
        self.idle.borrow().len()
    }

    /// Dispatch a batch of work items to an idle worker. Returns the worker index.
    pub fn dispatch_batch(&self, items: &[(String, WorkParams)]) -> Option<usize> {
        let idx = self.idle.borrow_mut().pop()?;

        let msg = WorkerBatchRequest {
            msg_type: "solve_batch".to_string(),
            items: items
                .iter()
                .map(|(id, params)| BatchItem {
                    id: id.clone(),
                    params: params.clone(),
                })
                .collect(),
        };
        let json = serde_json::to_string(&msg).unwrap();

        self.workers[idx]
            .post_message(&JsValue::from_str(&json))
            .ok();

        Some(idx)
    }

    /// Drain all pending results. Workers are marked idle when their batch completes.
    pub fn drain_results(&self) -> Vec<WorkerResponse> {
        let results = std::mem::take(&mut *self.results.borrow_mut());
        let mut idle = self.idle.borrow_mut();
        let mut progress = self.progress.borrow_mut();

        // Check all workers — mark idle when batch is done
        for i in 0..self.workers.len() {
            let (done, total) = progress[i];
            if total > 0 && done >= total && !idle.contains(&i) {
                idle.push(i);
                progress[i] = (0, 0);
            }
        }

        results
    }

    /// Total number of workers in the pool.
    pub fn pool_size(&self) -> usize {
        self.workers.len()
    }

    /// Per-worker batch progress: (done, total). total=0 means idle.
    pub fn worker_progress(&self) -> Vec<(usize, usize)> {
        self.progress.borrow().clone()
    }

    /// Recommended pool size: hardwareConcurrency - 1, min 1.
    pub fn recommended_size() -> usize {
        let nav = crate::dom::window().navigator();
        let cores = nav.hardware_concurrency() as usize;
        cores.saturating_sub(1).max(1)
    }
}
