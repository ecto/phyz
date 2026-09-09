//! What a step costs the HOST: a count of the launches, transfers and syncs
//! [`super::BatchSim`] issues.
//!
//! The device collector this crate feeds is *launch-bound* — 16x the worlds
//! costs 1.6x the time, and the host's `issue` time dominates. A count is
//! therefore a more useful number than a duration: it is exact, it is the
//! same on every machine, and it says directly how much a graph capture can
//! remove. The counters live here rather than in a backend because
//! [`super::BatchSim`] issues the same sequence to every backend, so a census
//! taken on the host mirror (`cuda-host`, no GPU needed) is the count the
//! CUDA path issues too.
//!
//! Counting is a thread-local increment per issued call, so it costs nothing
//! next to a launch and is always on. Take a reading with [`reset`] then
//! [`snapshot`]; the counters are per-thread, so two simulators on two
//! threads do not contaminate each other.

use std::cell::Cell;

/// A tally of host-issued work. Every field counts CALLS, not bytes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LaunchCensus {
    /// Kernel launches issued individually. Under graph capture these are
    /// *recorded*, not executed — the capture pays them once and every later
    /// period pays `graph_launches` instead.
    pub kernel_launches: u64,
    /// Replays of a captured span (`cudaGraphLaunch`). One per step under
    /// `step`, one per control period under `step_many`.
    pub graph_launches: u64,
    /// Captures performed. A capture is amortised over every replay that
    /// follows it, and a stale [`super::GraphKey`] forces a new one.
    pub graph_captures: u64,
    /// Host-to-device transfers.
    pub uploads: u64,
    /// Device-to-host transfers.
    pub downloads: u64,
    /// Blocking waits on the device. Each one drains the pipeline, so a sync
    /// in the inner loop costs far more than its count suggests.
    pub syncs: u64,
}

thread_local! {
    static CENSUS: Cell<LaunchCensus> = const { Cell::new(LaunchCensus {
        kernel_launches: 0,
        graph_launches: 0,
        graph_captures: 0,
        uploads: 0,
        downloads: 0,
        syncs: 0,
    }) };
}

fn bump(f: impl FnOnce(&mut LaunchCensus)) {
    CENSUS.with(|c| {
        let mut v = c.get();
        f(&mut v);
        c.set(v);
    });
}

pub(crate) fn kernel() {
    bump(|c| c.kernel_launches += 1);
}
pub(crate) fn graph() {
    bump(|c| c.graph_launches += 1);
}
pub(crate) fn capture() {
    bump(|c| c.graph_captures += 1);
}
pub(crate) fn upload() {
    bump(|c| c.uploads += 1);
}
pub(crate) fn download() {
    bump(|c| c.downloads += 1);
}
pub(crate) fn sync() {
    bump(|c| c.syncs += 1);
}

/// The counts issued by this thread so far.
pub fn snapshot() -> LaunchCensus {
    CENSUS.with(|c| c.get())
}

/// Zero this thread's counters. Call before the span being measured.
pub fn reset() {
    CENSUS.with(|c| c.set(LaunchCensus::default()));
}
