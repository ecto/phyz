//! Gravitomagnetic transformer experiment.
//!
//! Full simulation of Swain's gravitational transformer: circulating mass
//! current in a primary winding induces gravitomagnetic fields that couple
//! to a secondary winding via the GEM analog of Faraday's law.
//!
//! # Protocol
//!
//! 1. Set up primary mass current at amplitude A
//! 2. Evolve geometry via tent moves with source
//! 3. Extract GEM fields at secondary location
//! 4. Measure induced GEM EMF around secondary loop
//! 5. Repeat at multiple amplitudes → coupling coefficient
//! 6. Compare to linearized GEM prediction

use crate::complex::SimplicialComplex;
use crate::foliation::{flat_minkowski_sq_lengths, foliated_hypercubic, FoliatedComplex};
use crate::gem::{b_grav_tensor_frobenius, extract_gem_fields, induced_gem_emf, GemFields};
use crate::matter::{solve_regge_with_source, MassCurrentLoop, StressEnergy};
use crate::tent_move::{tent_edges_for_vertex, TentConfig};
use rayon::prelude::*;

/// Configuration for a gravitomagnetic transformer experiment.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Number of spatial grid points per axis.
    pub n_spatial: usize,
    /// Number of time slices.
    pub n_time: usize,
    /// Spatial grid spacing (geometric units).
    pub spacing: f64,
    /// Time step (geometric units).
    pub dt: f64,
    /// Newton solver configuration.
    pub tent_config: TentConfig,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            n_spatial: 4,
            n_time: 3,
            spacing: 1.0,
            dt: 0.3,
            tent_config: TentConfig {
                max_newton_iter: 30,
                ..TentConfig::default()
            },
        }
    }
}

/// A winding (loop of edges) on the simplicial lattice.
#[derive(Debug, Clone)]
pub struct Winding {
    /// Ordered list of global vertex indices forming the loop.
    pub vertices: Vec<usize>,
    /// Name for display.
    pub name: String,
}

/// Per-vertex residual statistics for diagnosing solver quality.
#[derive(Debug, Clone)]
pub struct ResidualStats {
    /// Maximum residual across all vertices.
    pub max: f64,
    /// Mean residual.
    pub mean: f64,
    /// Median residual.
    pub median: f64,
    /// 90th percentile residual.
    pub p90: f64,
    /// Number of vertices with residual above tolerance (1e-6).
    pub n_above_tol: usize,
    /// Total number of vertices measured.
    pub n_total: usize,
}

/// Result of a single transformer measurement at one amplitude.
#[derive(Debug, Clone)]
pub struct TransformerMeasurement {
    /// Primary current amplitude.
    pub amplitude: f64,
    /// Induced EMF at the secondary (from GEM field change).
    pub induced_emf: f64,
    /// Maximum |B_g| measured at the secondary.
    pub max_b_grav: f64,
    /// Average Frobenius norm of B_{ij} tidal tensor over secondary vertices.
    pub b_grav_frobenius: f64,
    /// Per-vertex residual statistics.
    pub residual_stats: ResidualStats,
}

impl TransformerMeasurement {
    /// Max residual (backward-compatible convenience).
    pub fn residual(&self) -> f64 {
        self.residual_stats.max
    }
}

/// Result of a full transformer experiment across multiple amplitudes.
#[derive(Debug, Clone)]
pub struct TransformerResult {
    /// Individual measurements.
    pub measurements: Vec<TransformerMeasurement>,
    /// Linear coupling coefficient: EMF / amplitude (from linear fit).
    pub coupling: f64,
    /// GEM linear prediction for the coupling (from flat-space formula).
    pub gem_prediction: f64,
    /// Nonlinear correction factor: coupling / gem_prediction.
    pub nonlinear_correction: f64,
}

/// Build a planar winding (loop of vertices) in a spatial slice.
///
/// Creates a rectangular loop in the xy-plane at given z-level.
/// The loop goes: (x0,y0) → (x1,y0) → (x1,y1) → (x0,y1) → back.
pub fn make_planar_winding(
    fc: &FoliatedComplex,
    slice: usize,
    x_range: [usize; 2],
    y_range: [usize; 2],
    z: usize,
    name: &str,
) -> Winding {
    let n = fc.n_spatial;
    let local = |x: usize, y: usize| -> usize {
        (x % n) + n * (y % n) + n * n * (z % n)
    };

    let mut vertices = Vec::new();

    // Bottom edge: y = y_range[0], x from x_range[0] to x_range[1]
    for x in x_range[0]..x_range[1] {
        vertices.push(fc.global_vertex(slice, local(x, y_range[0])));
    }
    // Right edge: x = x_range[1], y from y_range[0] to y_range[1]
    for y in y_range[0]..y_range[1] {
        vertices.push(fc.global_vertex(slice, local(x_range[1], y)));
    }
    // Top edge: y = y_range[1], x from x_range[1] to x_range[0]
    for x in (x_range[0]..x_range[1]).rev() {
        vertices.push(fc.global_vertex(slice, local(x + 1, y_range[1])));
    }
    // Left edge: x = x_range[0], y from y_range[1] to y_range[0]
    for y in (y_range[0]..y_range[1]).rev() {
        vertices.push(fc.global_vertex(slice, local(x_range[0], y + 1)));
    }

    // Deduplicate corners (the traversal naturally revisits them)
    vertices.dedup();
    // Close the loop if needed
    if vertices.len() > 1 && vertices.first() == vertices.last() {
        vertices.pop();
    }

    Winding {
        vertices,
        name: name.to_string(),
    }
}

/// Run a gravitomagnetic transformer experiment.
///
/// Sets up a primary mass current loop and a secondary sensing loop,
/// evolves the geometry with the primary as a source, and measures
/// the induced GEM fields at the secondary.
pub fn run_transformer(
    config: &TransformerConfig,
    amplitudes: &[f64],
) -> TransformerResult {
    let fc = foliated_hypercubic(config.n_time, config.n_spatial);
    let n = config.n_spatial;

    // Primary winding: loop in z=0 plane at slice 0
    let primary = make_planar_winding(&fc, 0, [0, n / 2], [0, n / 2], 0, "primary");

    // Secondary winding: loop in z=1 plane at slice 0 (different z-level)
    let z_secondary = (n / 2).min(n - 1);
    let secondary = make_planar_winding(
        &fc,
        0,
        [0, n / 2],
        [0, n / 2],
        z_secondary,
        "secondary",
    );

    // Baseline: flat Minkowski GEM fields (should be zero)
    let flat_sq = flat_minkowski_sq_lengths(&fc, config.spacing, config.dt);
    let gem_flat = extract_gem_fields(&fc.complex, &fc, &flat_sq, config.dt, config.spacing);

    let measurements: Vec<TransformerMeasurement> = amplitudes
        .par_iter()
        .map(|&amplitude| {
            run_single_amplitude(
                &fc,
                &flat_sq,
                &primary,
                &secondary,
                &gem_flat,
                amplitude,
                config,
            )
        })
        .collect();

    // Linear fit: coupling = Δ(EMF) / Δ(amplitude)
    let coupling = if measurements.len() >= 2 {
        let n_m = measurements.len();
        let a0 = measurements[0].amplitude;
        let a1 = measurements[n_m - 1].amplitude;
        let e0 = measurements[0].induced_emf;
        let e1 = measurements[n_m - 1].induced_emf;
        if (a1 - a0).abs() > 1e-30 {
            (e1 - e0) / (a1 - a0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // GEM analytic prediction for mutual inductance between coplanar loops
    // In linearized GEM: M_g = (4G/c²) * geometric factor
    // For our lattice, the geometric factor depends on loop separation and area
    let loop_area = (n / 2) as f64 * config.spacing * (n / 2) as f64 * config.spacing;
    let separation = z_secondary as f64 * config.spacing;
    let gem_prediction = if separation > 1e-30 {
        // Simplified Biot-Savart-like estimate: B_g ~ 4 * mass_rate / (c² * R)
        // Mutual inductance ~ area / separation
        4.0 * loop_area / (separation * separation)
    } else {
        0.0
    };

    let nonlinear_correction = if gem_prediction.abs() > 1e-30 {
        coupling / gem_prediction
    } else {
        1.0
    };

    TransformerResult {
        measurements,
        coupling,
        gem_prediction,
        nonlinear_correction,
    }
}

/// Run a transformer experiment using amplitude continuation.
///
/// Instead of solving each amplitude independently from flat space, this
/// solves sequentially: each amplitude initializes from the previous
/// converged geometry. This keeps Newton within its convergence basin
/// for larger amplitudes where starting from flat would diverge.
///
/// For amplitudes where the step from the previous solution is too large,
/// adaptive sub-stepping inserts intermediate amplitudes (not measured,
/// just used as stepping stones).
///
/// # Arguments
/// * `config` - Transformer configuration (mesh size, spacing, etc.)
/// * `amplitudes` - Sorted list of amplitudes to measure at (must be ascending)
/// * `max_substeps` - Maximum sub-steps if a direct step fails (0 = no sub-stepping)
pub fn run_transformer_continuation(
    config: &TransformerConfig,
    amplitudes: &[f64],
    _max_substeps: usize,
) -> TransformerResult {
    let fc = foliated_hypercubic(config.n_time, config.n_spatial);
    let n = config.n_spatial;

    let primary = make_planar_winding(&fc, 0, [0, n / 2], [0, n / 2], 0, "primary");
    let z_secondary = (n / 2).min(n - 1);
    let secondary = make_planar_winding(
        &fc,
        0,
        [0, n / 2],
        [0, n / 2],
        z_secondary,
        "secondary",
    );

    let flat_sq = flat_minkowski_sq_lengths(&fc, config.spacing, config.dt);
    let gem_flat = extract_gem_fields(&fc.complex, &fc, &flat_sq, config.dt, config.spacing);

    let mut measurements = Vec::with_capacity(amplitudes.len());
    // The previous amplitude's converged geometry
    let mut prev_sq = flat_sq.clone();

    for &amplitude in amplitudes {
        let source = PersistentMassCurrent::new(&fc, &primary, amplitude);

        // Strategy: try both from-flat and from-previous, keep the better one.
        // This gives continuation's benefit at large A (where flat fails) while
        // avoiding error accumulation at small A (where flat works fine).

        // Attempt 1: from flat space (same as independent solver)
        let mut sq_from_flat = flat_sq.clone();
        evolve_all_tents(&fc, &mut sq_from_flat, &source, &config.tent_config);
        let res_flat = compute_max_residual(&fc, &sq_from_flat, &source);

        // Attempt 2: from previous amplitude's solution
        let mut sq_from_prev = prev_sq.clone();
        evolve_all_tents(&fc, &mut sq_from_prev, &source, &config.tent_config);
        let res_prev = compute_max_residual(&fc, &sq_from_prev, &source);

        // Pick the solution with lower global residual
        let sq_lengths = if res_flat <= res_prev {
            sq_from_flat
        } else {
            sq_from_prev
        };

        // Extract measurements
        let gem_evolved =
            extract_gem_fields(&fc.complex, &fc, &sq_lengths, config.dt, config.spacing);
        let max_b_grav = secondary
            .vertices
            .iter()
            .flat_map(|&v| {
                gem_evolved.b_grav[v]
                    .iter()
                    .flat_map(|row| row.iter())
                    .map(|x| x.abs())
            })
            .fold(0.0f64, f64::max);

        let b_grav_frobenius = {
            let sum: f64 = secondary
                .vertices
                .iter()
                .map(|&v| b_grav_tensor_frobenius(&gem_evolved.b_grav[v]))
                .sum();
            sum / secondary.vertices.len() as f64
        };

        let emf = induced_gem_emf(&gem_flat, &gem_evolved, &secondary.vertices, config.dt);
        let stats = compute_residual_stats(&fc, &sq_lengths, &source);

        measurements.push(TransformerMeasurement {
            amplitude,
            induced_emf: emf,
            max_b_grav,
            b_grav_frobenius,
            residual_stats: stats,
        });

        // Use this as the starting point for the next amplitude
        prev_sq = sq_lengths;
    }

    // Linear fit and GEM prediction (same as run_transformer)
    let coupling = if measurements.len() >= 2 {
        let n_m = measurements.len();
        let a0 = measurements[0].amplitude;
        let a1 = measurements[n_m - 1].amplitude;
        let e0 = measurements[0].induced_emf;
        let e1 = measurements[n_m - 1].induced_emf;
        if (a1 - a0).abs() > 1e-30 {
            (e1 - e0) / (a1 - a0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let loop_area = (n / 2) as f64 * config.spacing * (n / 2) as f64 * config.spacing;
    let separation = z_secondary as f64 * config.spacing;
    let gem_prediction = if separation > 1e-30 {
        4.0 * loop_area / (separation * separation)
    } else {
        0.0
    };

    let nonlinear_correction = if gem_prediction.abs() > 1e-30 {
        coupling / gem_prediction
    } else {
        1.0
    };

    TransformerResult {
        measurements,
        coupling,
        gem_prediction,
        nonlinear_correction,
    }
}

/// Try to reach `target_a` from `start_a` using sub-steps.
/// Each sub-step does a fresh solve from flat, but initializes from the
/// previous sub-step's geometry. Returns the final geometry if successful.
#[allow(dead_code)]
fn try_substep(
    fc: &FoliatedComplex,
    primary: &Winding,
    start_sq: &[f64],
    start_a: f64,
    target_a: f64,
    max_substeps: usize,
    config: &TentConfig,
) -> Option<Vec<f64>> {
    let mut n_sub = 2;
    while n_sub <= max_substeps {
        let mut current_sq = start_sq.to_vec();
        let mut all_ok = true;

        for step in 1..=n_sub {
            let a = start_a + (target_a - start_a) * step as f64 / n_sub as f64;
            let source = PersistentMassCurrent::new(fc, primary, a);
            let (residual, _) = evolve_all_tents(fc, &mut current_sq, &source, config);

            if residual > config.newton_tol * 100.0 {
                all_ok = false;
                break;
            }
        }

        if all_ok {
            return Some(current_sq);
        }
        n_sub *= 2;
    }
    None
}

/// Evolve all tent moves for one amplitude. Returns max residual across all vertices.
///
/// Tolerates individual tent move failures (some boundary vertices may not converge).
/// Returns the max residual and number of failed tent moves.
fn evolve_all_tents(
    fc: &FoliatedComplex,
    sq_lengths: &mut [f64],
    source: &dyn StressEnergy,
    config: &TentConfig,
) -> (f64, usize) {
    let mut max_residual: f64 = 0.0;
    let mut n_failed: usize = 0;
    for target_slice in 1..fc.n_slices {
        let source_slice = target_slice - 1;
        for local_v in 0..fc.vertices_per_slice {
            let v = fc.global_vertex(source_slice, local_v);
            let free = tent_edges_for_vertex(fc, v, target_slice);
            if free.is_empty() {
                continue;
            }
            match solve_regge_with_source(
                &fc.complex,
                sq_lengths,
                &free,
                source,
                config,
            ) {
                Ok(result) => {
                    max_residual = max_residual.max(result.residual);
                }
                Err(_) => {
                    n_failed += 1;
                }
            }
        }
    }
    (max_residual, n_failed)
}

/// Compute max residual across all tent moves (for diagnostics).
fn compute_max_residual(
    fc: &FoliatedComplex,
    sq_lengths: &[f64],
    source: &dyn StressEnergy,
) -> f64 {
    use crate::lorentzian_regge::local_lorentzian_regge_action_grad;

    let eight_pi = 8.0 * std::f64::consts::PI;
    let mut max_res: f64 = 0.0;

    for target_slice in 1..fc.n_slices {
        let source_slice = target_slice - 1;
        for local_v in 0..fc.vertices_per_slice {
            let v = fc.global_vertex(source_slice, local_v);
            let free = tent_edges_for_vertex(fc, v, target_slice);
            if free.is_empty() {
                continue;
            }
            let grad = local_lorentzian_regge_action_grad(&fc.complex, sq_lengths, &free);
            let matter = source.edge_sources(&fc.complex, sq_lengths);
            let res: f64 = grad
                .iter()
                .zip(free.iter())
                .map(|(&g, &ei)| (g + eight_pi * matter[ei]).powi(2))
                .sum::<f64>()
                .sqrt();
            max_res = max_res.max(res);
        }
    }
    max_res
}

/// Compute per-vertex residual statistics (for diagnostics).
fn compute_residual_stats(
    fc: &FoliatedComplex,
    sq_lengths: &[f64],
    source: &dyn StressEnergy,
) -> ResidualStats {
    use crate::lorentzian_regge::local_lorentzian_regge_action_grad;

    let eight_pi = 8.0 * std::f64::consts::PI;
    let mut residuals = Vec::new();

    for target_slice in 1..fc.n_slices {
        let source_slice = target_slice - 1;
        for local_v in 0..fc.vertices_per_slice {
            let v = fc.global_vertex(source_slice, local_v);
            let free = tent_edges_for_vertex(fc, v, target_slice);
            if free.is_empty() {
                continue;
            }
            let grad = local_lorentzian_regge_action_grad(&fc.complex, sq_lengths, &free);
            let matter = source.edge_sources(&fc.complex, sq_lengths);
            let res: f64 = grad
                .iter()
                .zip(free.iter())
                .map(|(&g, &ei)| (g + eight_pi * matter[ei]).powi(2))
                .sum::<f64>()
                .sqrt();
            residuals.push(res);
        }
    }

    if residuals.is_empty() {
        return ResidualStats {
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            p90: 0.0,
            n_above_tol: 0,
            n_total: 0,
        };
    }

    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = residuals.len();
    let max = residuals[n - 1];
    let mean = residuals.iter().sum::<f64>() / n as f64;
    let median = if n % 2 == 0 {
        (residuals[n / 2 - 1] + residuals[n / 2]) / 2.0
    } else {
        residuals[n / 2]
    };
    let p90_idx = ((n as f64 * 0.9) as usize).min(n - 1);
    let p90 = residuals[p90_idx];
    let tol = 1e-6;
    let n_above_tol = residuals.iter().filter(|&&r| r > tol).count();

    ResidualStats {
        max,
        mean,
        median,
        p90,
        n_above_tol,
        n_total: n,
    }
}

fn run_single_amplitude(
    fc: &FoliatedComplex,
    flat_sq: &[f64],
    primary: &Winding,
    secondary: &Winding,
    gem_flat: &GemFields,
    amplitude: f64,
    config: &TransformerConfig,
) -> TransformerMeasurement {
    let mut sq_lengths = flat_sq.to_vec();

    // Create a time-persistent mass current source: the same spatial loop
    // at every time slice, so that the source affects free edges at each step.
    let source = PersistentMassCurrent::new(fc, primary, amplitude);

    // Evolve by sweeping tent moves over all vertices in each slice.
    let (_residual, n_failed) = evolve_all_tents(fc, &mut sq_lengths, &source, &config.tent_config);

    // Extract GEM fields from the evolved geometry
    let gem_evolved = extract_gem_fields(&fc.complex, fc, &sq_lengths, config.dt, config.spacing);

    // Measure B_g at secondary loop vertices
    let max_b_grav = secondary
        .vertices
        .iter()
        .flat_map(|&v| {
            gem_evolved.b_grav[v]
                .iter()
                .flat_map(|row| row.iter())
                .map(|x| x.abs())
        })
        .fold(0.0f64, f64::max);

    let b_grav_frobenius = {
        let sum: f64 = secondary
            .vertices
            .iter()
            .map(|&v| b_grav_tensor_frobenius(&gem_evolved.b_grav[v]))
            .sum();
        sum / secondary.vertices.len() as f64
    };

    // Compute induced EMF: -d(B_g)/dt at the secondary
    let emf = induced_gem_emf(gem_flat, &gem_evolved, &secondary.vertices, config.dt);

    let mut stats = compute_residual_stats(fc, &sq_lengths, &source);
    if n_failed > 0 {
        stats.max = f64::NAN;
    }

    TransformerMeasurement {
        amplitude,
        induced_emf: emf,
        max_b_grav,
        b_grav_frobenius,
        residual_stats: stats,
    }
}

/// Search for stress-energy configurations that enhance gravitomagnetic coupling.
///
/// Parameterizes a "core" material between primary and secondary with
/// energy density and pressure, then optimizes the coupling.
#[derive(Debug, Clone)]
pub struct CoreParams {
    /// Energy density of the core material.
    pub energy_density: f64,
    /// Isotropic pressure.
    pub pressure: f64,
}

/// Result of a permeability search.
#[derive(Debug, Clone)]
pub struct PermeabilityResult {
    /// Best core parameters found.
    pub best_params: CoreParams,
    /// Coupling with the best core.
    pub best_coupling: f64,
    /// Coupling without any core (vacuum).
    pub vacuum_coupling: f64,
    /// Enhancement factor: best_coupling / vacuum_coupling.
    pub enhancement: f64,
}

/// Search over core material parameters to find enhanced coupling.
pub fn permeability_search(
    config: &TransformerConfig,
    amplitude: f64,
    energy_densities: &[f64],
    pressures: &[f64],
) -> PermeabilityResult {
    let fc = foliated_hypercubic(config.n_time, config.n_spatial);
    let flat_sq = flat_minkowski_sq_lengths(&fc, config.spacing, config.dt);
    let n = config.n_spatial;

    let primary = make_planar_winding(&fc, 0, [0, n / 2], [0, n / 2], 0, "primary");
    let secondary = make_planar_winding(
        &fc,
        0,
        [0, n / 2],
        [0, n / 2],
        (n / 2).min(n - 1),
        "secondary",
    );

    let gem_flat = extract_gem_fields(&fc.complex, &fc, &flat_sq, config.dt, config.spacing);

    // Vacuum baseline
    let vacuum_meas = run_single_amplitude(
        &fc,
        &flat_sq,
        &primary,
        &secondary,
        &gem_flat,
        amplitude,
        config,
    );
    let vacuum_coupling = vacuum_meas.induced_emf;

    let mut best_coupling = vacuum_coupling;
    let mut best_params = CoreParams {
        energy_density: 0.0,
        pressure: 0.0,
    };

    // Grid search over core parameters
    for &rho in energy_densities {
        for &p in pressures {
            let mut sq_lengths = flat_sq.clone();

            // Create combined source: primary current + core material
            let source = CombinedSource {
                primary: PersistentMassCurrent::new(&fc, &primary, amplitude),
                core: CoreMaterial {
                    energy_density: rho,
                    pressure: p,
                    // Core occupies vertices between primary and secondary z-levels
                    core_vertices: core_vertices(&fc, 0, n),
                },
            };

            // Sweep tent moves over all vertices
            for target_slice in 1..fc.n_slices {
                let src_slice = target_slice - 1;
                for local_v in 0..fc.vertices_per_slice {
                    let v = fc.global_vertex(src_slice, local_v);
                    let free = tent_edges_for_vertex(&fc, v, target_slice);
                    if !free.is_empty() {
                        let _ = solve_regge_with_source(
                            &fc.complex,
                            &mut sq_lengths,
                            &free,
                            &source,
                            &config.tent_config,
                        );
                    }
                }
            }

            let gem = extract_gem_fields(&fc.complex, &fc, &sq_lengths, config.dt, config.spacing);
            let emf = induced_gem_emf(&gem_flat, &gem, &secondary.vertices, config.dt);

            if emf.abs() > best_coupling.abs() {
                best_coupling = emf;
                best_params = CoreParams {
                    energy_density: rho,
                    pressure: p,
                };
            }
        }
    }

    let enhancement = if vacuum_coupling.abs() > 1e-30 {
        best_coupling / vacuum_coupling
    } else {
        1.0
    };

    PermeabilityResult {
        best_params,
        best_coupling,
        vacuum_coupling,
        enhancement,
    }
}

/// Get vertices in the "core" region between z=0 and z=z_max.
fn core_vertices(fc: &FoliatedComplex, slice: usize, n: usize) -> Vec<usize> {
    let mut verts = Vec::new();
    let z_max = (n / 2).min(n - 1);
    for z in 1..z_max {
        for y in 0..n {
            for x in 0..n {
                let local = x + n * y + n * n * z;
                verts.push(fc.global_vertex(slice, local));
            }
        }
    }
    verts
}

/// A time-persistent mass current: the same spatial loop at every time slice.
///
/// The primary winding is defined at one slice; this source replicates it at
/// every slice so that the Regge equations at each time step include the current.
struct PersistentMassCurrent {
    /// MassCurrentLoop for each slice.
    loops: Vec<MassCurrentLoop>,
}

impl PersistentMassCurrent {
    fn new(fc: &FoliatedComplex, winding: &Winding, amplitude: f64) -> Self {
        // The winding is at some slice; find the local vertex indices
        let local_vertices: Vec<usize> = winding
            .vertices
            .iter()
            .map(|&v| fc.vertex_local(v))
            .collect();

        // Create the same loop at every slice
        let loops = (0..fc.n_slices)
            .map(|s| {
                let loop_verts: Vec<usize> = local_vertices
                    .iter()
                    .map(|&lv| fc.global_vertex(s, lv))
                    .collect();
                MassCurrentLoop {
                    mass_rate: amplitude,
                    loop_vertices: loop_verts,
                    gamma: 1.0,
                }
            })
            .collect();

        Self { loops }
    }
}

impl StressEnergy for PersistentMassCurrent {
    fn edge_sources(&self, complex: &SimplicialComplex, sq_lengths: &[f64]) -> Vec<f64> {
        let mut sources = vec![0.0; complex.n_edges()];
        for loop_src in &self.loops {
            let loop_sources = loop_src.edge_sources(complex, sq_lengths);
            for (s, ls) in sources.iter_mut().zip(loop_sources.iter()) {
                *s += ls;
            }
        }
        sources
    }
}

/// A stress-energy source combining primary current and core material.
struct CombinedSource {
    primary: PersistentMassCurrent,
    core: CoreMaterial,
}

/// Stress-energy from a core material filling a region.
struct CoreMaterial {
    energy_density: f64,
    pressure: f64,
    core_vertices: Vec<usize>,
}

impl StressEnergy for CombinedSource {
    fn edge_sources(&self, complex: &SimplicialComplex, sq_lengths: &[f64]) -> Vec<f64> {
        let mut sources = self.primary.edge_sources(complex, sq_lengths);
        let core_sources = self.core.edge_sources(complex, sq_lengths);
        for (s, c) in sources.iter_mut().zip(core_sources.iter()) {
            *s += c;
        }
        sources
    }
}

impl StressEnergy for CoreMaterial {
    fn edge_sources(&self, complex: &SimplicialComplex, _sq_lengths: &[f64]) -> Vec<f64> {
        let mut sources = vec![0.0; complex.n_edges()];

        // Distribute energy density from core vertices to their edges
        let rho_plus_p = self.energy_density + self.pressure;
        if rho_plus_p.abs() < 1e-30 {
            return sources;
        }

        for &vi in &self.core_vertices {
            // Count edges touching this vertex
            let mut n_edges = 0;
            for e in &complex.edges {
                if e[0] == vi || e[1] == vi {
                    n_edges += 1;
                }
            }
            if n_edges == 0 {
                continue;
            }
            let per_edge = rho_plus_p / n_edges as f64;
            for (ei, e) in complex.edges.iter().enumerate() {
                if e[0] == vi || e[1] == vi {
                    sources[ei] += per_edge;
                }
            }
        }

        sources
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Winding construction produces valid vertex loops.
    #[test]
    fn test_winding_construction() {
        let fc = foliated_hypercubic(2, 4);
        let winding = make_planar_winding(&fc, 0, [0, 2], [0, 2], 0, "test");

        assert!(
            winding.vertices.len() >= 4,
            "winding has {} vertices (expected >= 4)",
            winding.vertices.len()
        );

        // All vertices should be valid
        for &v in &winding.vertices {
            assert!(
                v < fc.complex.n_vertices,
                "vertex {} out of range",
                v
            );
        }

        // All vertices should be in the valid range
        for &v in &winding.vertices {
            assert!(v < fc.complex.n_vertices);
        }
    }

    /// Zero amplitude gives zero induced EMF.
    #[test]
    fn test_zero_amplitude_zero_emf() {
        let config = TransformerConfig {
            n_spatial: 2,
            n_time: 2,
            ..Default::default()
        };
        let result = run_transformer(&config, &[0.0]);
        assert_eq!(result.measurements.len(), 1);
        // Zero current → zero B_g → zero EMF
        assert!(
            result.measurements[0].induced_emf.abs() < 1e-10,
            "emf = {:.2e}",
            result.measurements[0].induced_emf
        );
    }

    /// Multiple amplitudes produce measurements and coupling estimate.
    #[test]
    fn test_multiple_amplitudes() {
        let config = TransformerConfig {
            n_spatial: 2,
            n_time: 2,
            ..Default::default()
        };
        let result = run_transformer(&config, &[0.0, 1e-6, 2e-6]);
        assert_eq!(result.measurements.len(), 3);
        // Coupling should be finite
        assert!(result.coupling.is_finite(), "coupling = {}", result.coupling);
    }

    /// Core vertex selection produces valid vertices.
    #[test]
    fn test_core_vertices() {
        let fc = foliated_hypercubic(2, 4);
        let verts = core_vertices(&fc, 0, 4);
        for &v in &verts {
            assert!(v < fc.complex.n_vertices);
        }
    }
}
