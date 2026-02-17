/* tslint:disable */
/* eslint-disable */

/**
 * Fusion viz: two kernels side-by-side, fuse them, animate the merge.
 */
export class WasmCompileFusionSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    a_count(): number;
    a_names(): string[];
    b_count(): number;
    b_names(): string[];
    fused_count(): number;
    fused_names(): string[];
    static new(): WasmCompileFusionSim;
    progress(): number;
    step_n(_n: number): void;
    time(): number;
}

/**
 * Kernel IR: build a simple physics kernel and return its DAG structure.
 */
export class WasmCompileIrSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static new(): WasmCompileIrSim;
    node_depths(): Uint32Array;
    node_names(): string[];
    node_parents(): Int32Array;
    node_types(): string[];
    num_nodes(): number;
    time(): number;
}

/**
 * WGSL output: display generated WGSL source for different physics ops.
 */
export class WasmCompileWgslSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    all_labels(): string[];
    current_label(): string;
    current_source(): string;
    static new(): WasmCompileWgslSim;
    /**
     * Cycle to next kernel
     */
    next(): void;
    num_kernels(): number;
    source_lines(): number;
    time(): number;
}

export class WasmCradleSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Bob positions as flat [x0,y0,z0, ...].
     */
    bob_positions(): Float64Array;
    bob_radius(): number;
    static new(): WasmCradleSim;
    num_bobs(): number;
    /**
     * Pivot positions as flat [x0,y0,z0, ...].
     */
    pivot_positions(): Float64Array;
    step_n(steps: number): void;
    time(): number;
}

/**
 * Gradient descent optimization: find initial angle to hit target x position.
 */
export class WasmDiffGradientSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    current_loss(): number;
    current_x(): number;
    current_y(): number;
    iteration(): number;
    loss_history(): Float64Array;
    static new(): WasmDiffGradientSim;
    /**
     * Run one gradient descent step: simulate forward, compute loss gradient, update theta0.
     */
    step_n(n: number): void;
    target_x(): number;
    theta0(): number;
    time(): number;
}

/**
 * Jacobian heatmap: pendulum simulation with live Jacobian matrix display.
 */
export class WasmDiffJacobianSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    angle(): number;
    bob_x(): number;
    bob_y(): number;
    jacobian(): Float64Array;
    static new(): WasmDiffJacobianSim;
    step_n(n: number): void;
    time(): number;
}

/**
 * Sensitivity: two pendulums with perturbed ICs, analytical gradient prediction vs actual divergence.
 */
export class WasmDiffSensitivitySim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    actual_divergence(): Float64Array;
    static new(): WasmDiffSensitivitySim;
    pos_a(): Float64Array;
    pos_b(): Float64Array;
    predicted_divergence(): Float64Array;
    step_n(n: number): void;
    time(): number;
}

export class WasmEmSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static dipole(): WasmEmSim;
    static double_slit(): WasmEmSim;
    field(): Float64Array;
    grid_size(): number;
    mask_data(): Float64Array;
    step_n(steps: number): void;
    time(): number;
    static waveguide(): WasmEmSim;
}

/**
 * 100 double pendulums with tiny perturbations showing chaotic divergence.
 */
export class WasmEnsembleSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Second bob endpoint positions as flat [x0,y0,z0, ...].
     */
    endpoint_positions(): Float64Array;
    static new(): WasmEnsembleSim;
    num_instances(): number;
    step_n(steps: number): void;
    time(): number;
}

export class WasmGravitySim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Two bodies in mutual orbit.
     */
    static binary_orbit(): WasmGravitySim;
    masses(): Float64Array;
    num_bodies(): number;
    positions(): Float64Array;
    /**
     * Mercury-like orbit with post-Newtonian precession.
     */
    static precession(): WasmGravitySim;
    /**
     * 5 planets orbiting a central mass.
     */
    static solar_system(): WasmGravitySim;
    step_n(steps: number): void;
    time(): number;
    trail_for(idx: number): Float64Array;
    trail_len(idx: number): number;
}

export class WasmGuardianSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Fixed vs adaptive timestep comparison.
     */
    static adaptive_dt(): WasmGuardianSim;
    /**
     * Correction demo with energy injection and guardian fix.
     */
    static correction_demo(): WasmGuardianSim;
    e0(): number;
    /**
     * Pendulum with energy monitoring gauge.
     */
    static energy_monitor(): WasmGuardianSim;
    ke(): number;
    ke_history(): Float64Array;
    length(): number;
    pe(): number;
    pe_history(): Float64Array;
    q2_val(): number;
    q_val(): number;
    step_n(steps: number): void;
    time(): number;
    total_energy(): number;
    total_history(): Float64Array;
    total_history2(): Float64Array;
    v_val(): number;
}

export class WasmLbmSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Lid-driven cavity flow.
     */
    static cavity_flow(): WasmLbmSim;
    /**
     * Poiseuille channel flow.
     */
    static channel_flow(): WasmLbmSim;
    grid_nx(): number;
    grid_ny(): number;
    step_n(steps: number): void;
    time(): number;
    /**
     * Velocity magnitude field as flat array [nx*ny].
     */
    velocity_field(): Float64Array;
    /**
     * Flow around obstacle (von Karman vortex street).
     */
    static vortex_street(): WasmLbmSim;
    /**
     * Vorticity field (curl of velocity).
     */
    vorticity_field(): Float64Array;
}

export class WasmLorentzSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * E x B drift: crossed electric and magnetic fields.
     */
    static crossed_fields(): WasmLorentzSim;
    /**
     * Magnetic mirror: converging B field lines.
     */
    static magnetic_mirror(): WasmLorentzSim;
    position(): Float64Array;
    speed(): number;
    /**
     * Helical spiral in uniform B field along z.
     */
    static spiral(): WasmLorentzSim;
    step_n(steps: number): void;
    time(): number;
    trail(): Float64Array;
    trail_len(): number;
}

export class WasmMdSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static argon_gas(): WasmMdSim;
    bond_endpoints(): Float64Array;
    box_size(): number;
    static crystal(): WasmMdSim;
    num_bonds(): number;
    num_particles(): number;
    static polymer(): WasmMdSim;
    positions(): Float64Array;
    step_n(steps: number): void;
    temperature(): number;
    time(): number;
    velocities(): Float64Array;
}

/**
 * Ant model: multi-body spider with 8 legs.
 */
export class WasmMjcfAntSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    body_endpoint_positions(): Float64Array;
    joint_positions(): Float64Array;
    nbodies(): number;
    static new(): WasmMjcfAntSim;
    step_n(n: number): void;
    time(): number;
}

/**
 * Cartpole: cart + inverted pendulum (classic control benchmark).
 */
export class WasmMjcfCartpoleSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    cart_x(): number;
    static new(): WasmMjcfCartpoleSim;
    pole_angle(): number;
    pole_tip_x(): number;
    pole_tip_y(): number;
    step_n(n: number): void;
    time(): number;
}

/**
 * MJCF editor: parse a simple body description and return skeleton.
 */
export class WasmMjcfEditorSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    n_bodies(): number;
    /**
     * Parse a simple MJCF-like string. Format: one body per line, "name length mass".
     */
    static parse(xml: string): WasmMjcfEditorSim;
    positions(): Float64Array;
}

export class WasmMpmSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static elastic_blob(): WasmMpmSim;
    static fluid_dam(): WasmMpmSim;
    static fluid_slosh(): WasmMpmSim;
    static granular_column(): WasmMpmSim;
    num_particles(): number;
    num_springs(): number;
    particle_radius(): number;
    positions(): Float64Array;
    spring_endpoints(): Float64Array;
    step_n(steps: number): void;
    time(): number;
    velocities(): Float64Array;
}

export class WasmParticleSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Three spheres with different bounce coefficients.
     */
    static bouncing_spheres(): WasmParticleSim;
    num_particles(): number;
    /**
     * Flat [x0,y0,z0, x1,y1,z1, ...] positions.
     */
    positions(): Float64Array;
    radii(): Float64Array;
    /**
     * 20 spheres cascading into a pile.
     */
    static sphere_cascade(): WasmParticleSim;
    step_n(steps: number): void;
    time(): number;
}

/**
 * 100 pendulums with different PD controllers for swing-up.
 */
export class WasmPolicyGridSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    angles(): Float64Array;
    static new(): WasmPolicyGridSim;
    num_envs(): number;
    /**
     * Reward: 1 when inverted (q=π), 0 when hanging (q=0).
     */
    rewards(): Float64Array;
    step_n(steps: number): void;
    time(): number;
}

export class WasmProbSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Ensemble of bouncing balls with parameter uncertainty.
     */
    static monte_carlo(): WasmProbSim;
    num_particles(): number;
    positions(): Float64Array;
    spread(): number;
    step_n(steps: number): void;
    /**
     * SVGD particles converging to a target distribution (2D Gaussian mixture).
     */
    static svgd_particles(): WasmProbSim;
    time(): number;
    trail_flat(): Float64Array;
    /**
     * Ensemble trajectories diverging from near-identical ICs.
     * No gravity — pure spreading to show uncertainty cone.
     */
    static uncertainty_cone(): WasmProbSim;
}

export class WasmQftSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    average_plaquette(): number;
    beta(): number;
    lattice_size(): number;
    static phase_scan(): WasmQftSim;
    plaq_history(): Float64Array;
    plaquettes(): Float64Array;
    step_n(metro_sweeps: number): void;
    time(): number;
    static u1_plaquette(): WasmQftSim;
    wilson_loop(r: number, t: number): number;
    static wilson_loops(): WasmQftSim;
}

/**
 * Adam vs GD: compare two optimizers on the same problem.
 */
export class WasmReal2SimAdamVsGdSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    adam_length(): number;
    adam_loss(): Float64Array;
    adam_mass(): number;
    gd_length(): number;
    gd_loss(): Float64Array;
    gd_mass(): number;
    iteration(): number;
    static new(): WasmReal2SimAdamVsGdSim;
    step_n(n: number): void;
    time(): number;
}

/**
 * Parameter fitting: gradient descent to match a reference trajectory.
 */
export class WasmReal2SimFitSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    current_loss(): number;
    est_length(): number;
    est_mass(): number;
    /**
     * Current estimated trajectory (for overlay rendering)
     */
    est_trajectory(): Float64Array;
    iteration(): number;
    loss_history(): Float64Array;
    static new(): WasmReal2SimFitSim;
    ref_trajectory(): Float64Array;
    step_n(n: number): void;
    time(): number;
    true_length(): number;
    true_mass(): number;
}

/**
 * Loss landscape: 2D contour of loss over (mass, length) parameter grid.
 */
export class WasmReal2SimLandscapeSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    est_length(): number;
    est_mass(): number;
    grid(): Float64Array;
    grid_size(): number;
    static new(): WasmReal2SimLandscapeSim;
    path_l(): Float64Array;
    path_m(): Float64Array;
    step_n(n: number): void;
    time(): number;
}

/**
 * Action landscape: Einstein-Maxwell action vs uniform scale factor, with gradient descent.
 */
export class WasmReggeActionSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    actions(): Float64Array;
    current_action(): number;
    current_scale(): number;
    gd_path(): Float64Array;
    static new(): WasmReggeActionSim;
    scales(): Float64Array;
    step_n(n: number): void;
    time(): number;
}

/**
 * Curvature slice: Reissner-Nordström edge lengths as color heatmap.
 */
export class WasmReggeCurvatureSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Compute edge lengths for Reissner-Nordström mesh.
     * Returns flat array of edge length values (one per edge).
     */
    compute(mass: number, charge: number): Float64Array;
    /**
     * Grid size for heatmap rendering.
     */
    grid_size(): number;
    /**
     * Returns number of edges for the current mesh size.
     */
    n_edges(): number;
    static new(): WasmReggeCurvatureSim;
    time(): number;
}

/**
 * Symmetry bars: Noether current norms for different spacetimes.
 */
export class WasmReggeSymmetrySim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Compute Noether current norms for flat, RN, and Kerr spacetimes.
     * Returns [flat_gauge, flat_translation, rn_gauge, rn_translation, kerr_gauge, kerr_translation].
     */
    compute(): Float64Array;
    static new(): WasmReggeSymmetrySim;
    time(): number;
}

export class WasmSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    body_endpoint_positions(): Float64Array;
    /**
     * N-link chain with viscous damping.
     */
    static chain(n: number): WasmSim;
    /**
     * Double pendulum with viscous damping.
     */
    static double_pendulum(): WasmSim;
    joint_angles(): Float64Array;
    joint_positions(): Float64Array;
    nbodies(): number;
    /**
     * Single pendulum with viscous damping.
     */
    static pendulum(): WasmSim;
    /**
     * Step the simulation forward n times using RK4.
     */
    step_n(n: number): void;
    time(): number;
    total_energy(): number;
}

/**
 * 1024 pendulums with varying lengths creating wave interference.
 */
export class WasmWaveFieldSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    angles(): Float64Array;
    static new(): WasmWaveFieldSim;
    num_pendulums(): number;
    step_n(steps: number): void;
    time(): number;
    velocities(): Float64Array;
}

export class WasmWorldSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    body_endpoint_positions(): Float64Array;
    link_lengths(): Float64Array;
    nbodies(): number;
    static phase_portrait(): WasmWorldSim;
    q(): number;
    static random_chain(): WasmWorldSim;
    step_n(steps: number): void;
    static tendon_actuated(): WasmWorldSim;
    time(): number;
    v(): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmcompilefusionsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmcompileirsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmcompilewgslsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmcradlesim_free: (a: number, b: number) => void;
    readonly __wbg_wasmdiffgradientsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmdiffjacobiansim_free: (a: number, b: number) => void;
    readonly __wbg_wasmdiffsensitivitysim_free: (a: number, b: number) => void;
    readonly __wbg_wasmemsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmensemblesim_free: (a: number, b: number) => void;
    readonly __wbg_wasmgravitysim_free: (a: number, b: number) => void;
    readonly __wbg_wasmguardiansim_free: (a: number, b: number) => void;
    readonly __wbg_wasmlbmsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmlorentzsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmmdsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmmjcfantsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmmjcfcartpolesim_free: (a: number, b: number) => void;
    readonly __wbg_wasmmjcfeditorsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmmpmsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmparticlesim_free: (a: number, b: number) => void;
    readonly __wbg_wasmpolicygridsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmprobsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmqftsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmreal2simadamvsgdsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmreal2simfitsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmreal2simlandscapesim_free: (a: number, b: number) => void;
    readonly __wbg_wasmreggeactionsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmreggecurvaturesim_free: (a: number, b: number) => void;
    readonly __wbg_wasmreggesymmetrysim_free: (a: number, b: number) => void;
    readonly __wbg_wasmsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmwavefieldsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmworldsim_free: (a: number, b: number) => void;
    readonly wasmcompilefusionsim_a_count: (a: number) => number;
    readonly wasmcompilefusionsim_a_names: (a: number) => [number, number];
    readonly wasmcompilefusionsim_b_count: (a: number) => number;
    readonly wasmcompilefusionsim_b_names: (a: number) => [number, number];
    readonly wasmcompilefusionsim_fused_count: (a: number) => number;
    readonly wasmcompilefusionsim_fused_names: (a: number) => [number, number];
    readonly wasmcompilefusionsim_new: () => number;
    readonly wasmcompilefusionsim_progress: (a: number) => number;
    readonly wasmcompilefusionsim_step_n: (a: number, b: number) => void;
    readonly wasmcompilefusionsim_time: (a: number) => number;
    readonly wasmcompileirsim_new: () => number;
    readonly wasmcompileirsim_node_depths: (a: number) => [number, number];
    readonly wasmcompileirsim_node_names: (a: number) => [number, number];
    readonly wasmcompileirsim_node_parents: (a: number) => [number, number];
    readonly wasmcompileirsim_node_types: (a: number) => [number, number];
    readonly wasmcompileirsim_num_nodes: (a: number) => number;
    readonly wasmcompileirsim_time: (a: number) => number;
    readonly wasmcompilewgslsim_all_labels: (a: number) => [number, number];
    readonly wasmcompilewgslsim_current_label: (a: number) => [number, number];
    readonly wasmcompilewgslsim_current_source: (a: number) => [number, number];
    readonly wasmcompilewgslsim_new: () => number;
    readonly wasmcompilewgslsim_next: (a: number) => void;
    readonly wasmcompilewgslsim_source_lines: (a: number) => number;
    readonly wasmcompilewgslsim_time: (a: number) => number;
    readonly wasmcradlesim_bob_positions: (a: number) => [number, number];
    readonly wasmcradlesim_bob_radius: (a: number) => number;
    readonly wasmcradlesim_new: () => number;
    readonly wasmcradlesim_num_bobs: (a: number) => number;
    readonly wasmcradlesim_pivot_positions: (a: number) => [number, number];
    readonly wasmcradlesim_step_n: (a: number, b: number) => void;
    readonly wasmcradlesim_time: (a: number) => number;
    readonly wasmdiffgradientsim_current_loss: (a: number) => number;
    readonly wasmdiffgradientsim_current_x: (a: number) => number;
    readonly wasmdiffgradientsim_current_y: (a: number) => number;
    readonly wasmdiffgradientsim_iteration: (a: number) => number;
    readonly wasmdiffgradientsim_loss_history: (a: number) => [number, number];
    readonly wasmdiffgradientsim_new: () => number;
    readonly wasmdiffgradientsim_step_n: (a: number, b: number) => void;
    readonly wasmdiffgradientsim_target_x: (a: number) => number;
    readonly wasmdiffgradientsim_theta0: (a: number) => number;
    readonly wasmdiffgradientsim_time: (a: number) => number;
    readonly wasmdiffjacobiansim_angle: (a: number) => number;
    readonly wasmdiffjacobiansim_bob_x: (a: number) => number;
    readonly wasmdiffjacobiansim_bob_y: (a: number) => number;
    readonly wasmdiffjacobiansim_jacobian: (a: number) => [number, number];
    readonly wasmdiffjacobiansim_new: () => number;
    readonly wasmdiffjacobiansim_step_n: (a: number, b: number) => void;
    readonly wasmdiffsensitivitysim_actual_divergence: (a: number) => [number, number];
    readonly wasmdiffsensitivitysim_new: () => number;
    readonly wasmdiffsensitivitysim_pos_a: (a: number) => [number, number];
    readonly wasmdiffsensitivitysim_pos_b: (a: number) => [number, number];
    readonly wasmdiffsensitivitysim_predicted_divergence: (a: number) => [number, number];
    readonly wasmdiffsensitivitysim_step_n: (a: number, b: number) => void;
    readonly wasmemsim_dipole: () => number;
    readonly wasmemsim_double_slit: () => number;
    readonly wasmemsim_field: (a: number) => [number, number];
    readonly wasmemsim_grid_size: (a: number) => number;
    readonly wasmemsim_mask_data: (a: number) => [number, number];
    readonly wasmemsim_step_n: (a: number, b: number) => void;
    readonly wasmemsim_time: (a: number) => number;
    readonly wasmemsim_waveguide: () => number;
    readonly wasmensemblesim_endpoint_positions: (a: number) => [number, number];
    readonly wasmensemblesim_new: () => number;
    readonly wasmensemblesim_num_instances: (a: number) => number;
    readonly wasmensemblesim_step_n: (a: number, b: number) => void;
    readonly wasmensemblesim_time: (a: number) => number;
    readonly wasmgravitysim_binary_orbit: () => number;
    readonly wasmgravitysim_masses: (a: number) => [number, number];
    readonly wasmgravitysim_num_bodies: (a: number) => number;
    readonly wasmgravitysim_positions: (a: number) => [number, number];
    readonly wasmgravitysim_precession: () => number;
    readonly wasmgravitysim_solar_system: () => number;
    readonly wasmgravitysim_step_n: (a: number, b: number) => void;
    readonly wasmgravitysim_trail_for: (a: number, b: number) => [number, number];
    readonly wasmgravitysim_trail_len: (a: number, b: number) => number;
    readonly wasmguardiansim_adaptive_dt: () => number;
    readonly wasmguardiansim_correction_demo: () => number;
    readonly wasmguardiansim_energy_monitor: () => number;
    readonly wasmguardiansim_ke: (a: number) => number;
    readonly wasmguardiansim_ke_history: (a: number) => [number, number];
    readonly wasmguardiansim_length: (a: number) => number;
    readonly wasmguardiansim_pe: (a: number) => number;
    readonly wasmguardiansim_pe_history: (a: number) => [number, number];
    readonly wasmguardiansim_q2_val: (a: number) => number;
    readonly wasmguardiansim_step_n: (a: number, b: number) => void;
    readonly wasmguardiansim_total_energy: (a: number) => number;
    readonly wasmguardiansim_total_history: (a: number) => [number, number];
    readonly wasmguardiansim_total_history2: (a: number) => [number, number];
    readonly wasmlbmsim_cavity_flow: () => number;
    readonly wasmlbmsim_channel_flow: () => number;
    readonly wasmlbmsim_grid_ny: (a: number) => number;
    readonly wasmlbmsim_step_n: (a: number, b: number) => void;
    readonly wasmlbmsim_velocity_field: (a: number) => [number, number];
    readonly wasmlbmsim_vortex_street: () => number;
    readonly wasmlbmsim_vorticity_field: (a: number) => [number, number];
    readonly wasmlorentzsim_crossed_fields: () => number;
    readonly wasmlorentzsim_magnetic_mirror: () => number;
    readonly wasmlorentzsim_position: (a: number) => [number, number];
    readonly wasmlorentzsim_speed: (a: number) => number;
    readonly wasmlorentzsim_spiral: () => number;
    readonly wasmlorentzsim_step_n: (a: number, b: number) => void;
    readonly wasmlorentzsim_time: (a: number) => number;
    readonly wasmlorentzsim_trail: (a: number) => [number, number];
    readonly wasmlorentzsim_trail_len: (a: number) => number;
    readonly wasmmdsim_argon_gas: () => number;
    readonly wasmmdsim_bond_endpoints: (a: number) => [number, number];
    readonly wasmmdsim_crystal: () => number;
    readonly wasmmdsim_num_bonds: (a: number) => number;
    readonly wasmmdsim_num_particles: (a: number) => number;
    readonly wasmmdsim_polymer: () => number;
    readonly wasmmdsim_positions: (a: number) => [number, number];
    readonly wasmmdsim_step_n: (a: number, b: number) => void;
    readonly wasmmdsim_temperature: (a: number) => number;
    readonly wasmmdsim_time: (a: number) => number;
    readonly wasmmdsim_velocities: (a: number) => [number, number];
    readonly wasmmjcfantsim_body_endpoint_positions: (a: number) => [number, number];
    readonly wasmmjcfantsim_joint_positions: (a: number) => [number, number];
    readonly wasmmjcfantsim_nbodies: (a: number) => number;
    readonly wasmmjcfantsim_new: () => number;
    readonly wasmmjcfantsim_step_n: (a: number, b: number) => void;
    readonly wasmmjcfcartpolesim_new: () => number;
    readonly wasmmjcfcartpolesim_pole_angle: (a: number) => number;
    readonly wasmmjcfcartpolesim_pole_tip_x: (a: number) => number;
    readonly wasmmjcfcartpolesim_pole_tip_y: (a: number) => number;
    readonly wasmmjcfcartpolesim_step_n: (a: number, b: number) => void;
    readonly wasmmjcfeditorsim_parse: (a: number, b: number) => number;
    readonly wasmmjcfeditorsim_positions: (a: number) => [number, number];
    readonly wasmmpmsim_elastic_blob: () => number;
    readonly wasmmpmsim_fluid_dam: () => number;
    readonly wasmmpmsim_fluid_slosh: () => number;
    readonly wasmmpmsim_granular_column: () => number;
    readonly wasmmpmsim_num_particles: (a: number) => number;
    readonly wasmmpmsim_num_springs: (a: number) => number;
    readonly wasmmpmsim_positions: (a: number) => [number, number];
    readonly wasmmpmsim_spring_endpoints: (a: number) => [number, number];
    readonly wasmmpmsim_step_n: (a: number, b: number) => void;
    readonly wasmmpmsim_velocities: (a: number) => [number, number];
    readonly wasmparticlesim_bouncing_spheres: () => number;
    readonly wasmparticlesim_num_particles: (a: number) => number;
    readonly wasmparticlesim_positions: (a: number) => [number, number];
    readonly wasmparticlesim_radii: (a: number) => [number, number];
    readonly wasmparticlesim_sphere_cascade: () => number;
    readonly wasmparticlesim_step_n: (a: number, b: number) => void;
    readonly wasmpolicygridsim_angles: (a: number) => [number, number];
    readonly wasmpolicygridsim_new: () => number;
    readonly wasmpolicygridsim_rewards: (a: number) => [number, number];
    readonly wasmpolicygridsim_step_n: (a: number, b: number) => void;
    readonly wasmprobsim_monte_carlo: () => number;
    readonly wasmprobsim_positions: (a: number) => [number, number];
    readonly wasmprobsim_spread: (a: number) => number;
    readonly wasmprobsim_step_n: (a: number, b: number) => void;
    readonly wasmprobsim_svgd_particles: () => number;
    readonly wasmprobsim_trail_flat: (a: number) => [number, number];
    readonly wasmprobsim_uncertainty_cone: () => number;
    readonly wasmqftsim_average_plaquette: (a: number) => number;
    readonly wasmqftsim_phase_scan: () => number;
    readonly wasmqftsim_plaq_history: (a: number) => [number, number];
    readonly wasmqftsim_plaquettes: (a: number) => [number, number];
    readonly wasmqftsim_step_n: (a: number, b: number) => void;
    readonly wasmqftsim_time: (a: number) => number;
    readonly wasmqftsim_u1_plaquette: () => number;
    readonly wasmqftsim_wilson_loop: (a: number, b: number, c: number) => number;
    readonly wasmqftsim_wilson_loops: () => number;
    readonly wasmreal2simadamvsgdsim_adam_loss: (a: number) => [number, number];
    readonly wasmreal2simadamvsgdsim_adam_mass: (a: number) => number;
    readonly wasmreal2simadamvsgdsim_gd_loss: (a: number) => [number, number];
    readonly wasmreal2simadamvsgdsim_new: () => number;
    readonly wasmreal2simadamvsgdsim_step_n: (a: number, b: number) => void;
    readonly wasmreal2simadamvsgdsim_time: (a: number) => number;
    readonly wasmreal2simfitsim_current_loss: (a: number) => number;
    readonly wasmreal2simfitsim_est_trajectory: (a: number) => [number, number];
    readonly wasmreal2simfitsim_loss_history: (a: number) => [number, number];
    readonly wasmreal2simfitsim_new: () => number;
    readonly wasmreal2simfitsim_ref_trajectory: (a: number) => [number, number];
    readonly wasmreal2simfitsim_step_n: (a: number, b: number) => void;
    readonly wasmreal2simfitsim_time: (a: number) => number;
    readonly wasmreal2simlandscapesim_grid: (a: number) => [number, number];
    readonly wasmreal2simlandscapesim_new: () => number;
    readonly wasmreal2simlandscapesim_path_l: (a: number) => [number, number];
    readonly wasmreal2simlandscapesim_path_m: (a: number) => [number, number];
    readonly wasmreal2simlandscapesim_step_n: (a: number, b: number) => void;
    readonly wasmreal2simlandscapesim_time: (a: number) => number;
    readonly wasmreggeactionsim_actions: (a: number) => [number, number];
    readonly wasmreggeactionsim_current_action: (a: number) => number;
    readonly wasmreggeactionsim_gd_path: (a: number) => [number, number];
    readonly wasmreggeactionsim_new: () => number;
    readonly wasmreggeactionsim_scales: (a: number) => [number, number];
    readonly wasmreggeactionsim_step_n: (a: number, b: number) => void;
    readonly wasmreggeactionsim_time: (a: number) => number;
    readonly wasmreggecurvaturesim_compute: (a: number, b: number, c: number) => [number, number];
    readonly wasmreggecurvaturesim_grid_size: (a: number) => number;
    readonly wasmreggecurvaturesim_n_edges: (a: number) => number;
    readonly wasmreggecurvaturesim_new: () => number;
    readonly wasmreggesymmetrysim_compute: (a: number) => [number, number];
    readonly wasmreggesymmetrysim_new: () => number;
    readonly wasmsim_body_endpoint_positions: (a: number) => [number, number];
    readonly wasmsim_chain: (a: number) => number;
    readonly wasmsim_double_pendulum: () => number;
    readonly wasmsim_joint_angles: (a: number) => [number, number];
    readonly wasmsim_joint_positions: (a: number) => [number, number];
    readonly wasmsim_pendulum: () => number;
    readonly wasmsim_step_n: (a: number, b: number) => void;
    readonly wasmsim_total_energy: (a: number) => number;
    readonly wasmwavefieldsim_angles: (a: number) => [number, number];
    readonly wasmwavefieldsim_new: () => number;
    readonly wasmwavefieldsim_step_n: (a: number, b: number) => void;
    readonly wasmwavefieldsim_velocities: (a: number) => [number, number];
    readonly wasmworldsim_body_endpoint_positions: (a: number) => [number, number];
    readonly wasmworldsim_link_lengths: (a: number) => [number, number];
    readonly wasmworldsim_phase_portrait: () => number;
    readonly wasmworldsim_q: (a: number) => number;
    readonly wasmworldsim_random_chain: () => number;
    readonly wasmworldsim_step_n: (a: number, b: number) => void;
    readonly wasmworldsim_tendon_actuated: () => number;
    readonly wasmworldsim_v: (a: number) => number;
    readonly wasmcompilewgslsim_num_kernels: (a: number) => number;
    readonly wasmsim_nbodies: (a: number) => number;
    readonly wasmworldsim_nbodies: (a: number) => number;
    readonly wasmreggecurvaturesim_time: (a: number) => number;
    readonly wasmreggesymmetrysim_time: (a: number) => number;
    readonly wasmdiffjacobiansim_time: (a: number) => number;
    readonly wasmdiffsensitivitysim_time: (a: number) => number;
    readonly wasmgravitysim_time: (a: number) => number;
    readonly wasmguardiansim_e0: (a: number) => number;
    readonly wasmguardiansim_q_val: (a: number) => number;
    readonly wasmguardiansim_time: (a: number) => number;
    readonly wasmguardiansim_v_val: (a: number) => number;
    readonly wasmlbmsim_grid_nx: (a: number) => number;
    readonly wasmlbmsim_time: (a: number) => number;
    readonly wasmmdsim_box_size: (a: number) => number;
    readonly wasmmjcfantsim_time: (a: number) => number;
    readonly wasmmjcfcartpolesim_cart_x: (a: number) => number;
    readonly wasmmjcfcartpolesim_time: (a: number) => number;
    readonly wasmmjcfeditorsim_n_bodies: (a: number) => number;
    readonly wasmmpmsim_particle_radius: (a: number) => number;
    readonly wasmmpmsim_time: (a: number) => number;
    readonly wasmparticlesim_time: (a: number) => number;
    readonly wasmpolicygridsim_num_envs: (a: number) => number;
    readonly wasmpolicygridsim_time: (a: number) => number;
    readonly wasmprobsim_num_particles: (a: number) => number;
    readonly wasmprobsim_time: (a: number) => number;
    readonly wasmqftsim_beta: (a: number) => number;
    readonly wasmqftsim_lattice_size: (a: number) => number;
    readonly wasmreal2simadamvsgdsim_adam_length: (a: number) => number;
    readonly wasmreal2simadamvsgdsim_gd_length: (a: number) => number;
    readonly wasmreal2simadamvsgdsim_gd_mass: (a: number) => number;
    readonly wasmreal2simadamvsgdsim_iteration: (a: number) => number;
    readonly wasmreal2simfitsim_est_length: (a: number) => number;
    readonly wasmreal2simfitsim_est_mass: (a: number) => number;
    readonly wasmreal2simfitsim_iteration: (a: number) => number;
    readonly wasmreal2simfitsim_true_length: (a: number) => number;
    readonly wasmreal2simfitsim_true_mass: (a: number) => number;
    readonly wasmreal2simlandscapesim_est_length: (a: number) => number;
    readonly wasmreal2simlandscapesim_est_mass: (a: number) => number;
    readonly wasmreal2simlandscapesim_grid_size: (a: number) => number;
    readonly wasmreggeactionsim_current_scale: (a: number) => number;
    readonly wasmsim_time: (a: number) => number;
    readonly wasmwavefieldsim_num_pendulums: (a: number) => number;
    readonly wasmwavefieldsim_time: (a: number) => number;
    readonly wasmworldsim_time: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_drop_slice: (a: number, b: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
