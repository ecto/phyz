/* tslint:disable */
/* eslint-disable */

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

export class WasmMpmSim {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static elastic_blob(): WasmMpmSim;
    static fluid_dam(): WasmMpmSim;
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
    readonly __wbg_wasmcradlesim_free: (a: number, b: number) => void;
    readonly __wbg_wasmemsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmensemblesim_free: (a: number, b: number) => void;
    readonly __wbg_wasmgravitysim_free: (a: number, b: number) => void;
    readonly __wbg_wasmguardiansim_free: (a: number, b: number) => void;
    readonly __wbg_wasmlbmsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmlorentzsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmmdsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmmpmsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmparticlesim_free: (a: number, b: number) => void;
    readonly __wbg_wasmpolicygridsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmprobsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmqftsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmwavefieldsim_free: (a: number, b: number) => void;
    readonly __wbg_wasmworldsim_free: (a: number, b: number) => void;
    readonly wasmcradlesim_bob_positions: (a: number) => [number, number];
    readonly wasmcradlesim_bob_radius: (a: number) => number;
    readonly wasmcradlesim_new: () => number;
    readonly wasmcradlesim_num_bobs: (a: number) => number;
    readonly wasmcradlesim_pivot_positions: (a: number) => [number, number];
    readonly wasmcradlesim_step_n: (a: number, b: number) => void;
    readonly wasmcradlesim_time: (a: number) => number;
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
    readonly wasmgravitysim_time: (a: number) => number;
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
    readonly wasmguardiansim_q_val: (a: number) => number;
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
    readonly wasmmpmsim_elastic_blob: () => number;
    readonly wasmmpmsim_fluid_dam: () => number;
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
    readonly wasmpolicygridsim_time: (a: number) => number;
    readonly wasmprobsim_monte_carlo: () => number;
    readonly wasmprobsim_num_particles: (a: number) => number;
    readonly wasmprobsim_positions: (a: number) => [number, number];
    readonly wasmprobsim_spread: (a: number) => number;
    readonly wasmprobsim_step_n: (a: number, b: number) => void;
    readonly wasmprobsim_svgd_particles: () => number;
    readonly wasmprobsim_trail_flat: (a: number) => [number, number];
    readonly wasmprobsim_uncertainty_cone: () => number;
    readonly wasmqftsim_average_plaquette: (a: number) => number;
    readonly wasmqftsim_lattice_size: (a: number) => number;
    readonly wasmqftsim_phase_scan: () => number;
    readonly wasmqftsim_plaq_history: (a: number) => [number, number];
    readonly wasmqftsim_plaquettes: (a: number) => [number, number];
    readonly wasmqftsim_step_n: (a: number, b: number) => void;
    readonly wasmqftsim_time: (a: number) => number;
    readonly wasmqftsim_u1_plaquette: () => number;
    readonly wasmqftsim_wilson_loop: (a: number, b: number, c: number) => number;
    readonly wasmqftsim_wilson_loops: () => number;
    readonly wasmsim_body_endpoint_positions: (a: number) => [number, number];
    readonly wasmsim_chain: (a: number) => number;
    readonly wasmsim_double_pendulum: () => number;
    readonly wasmsim_joint_angles: (a: number) => [number, number];
    readonly wasmsim_joint_positions: (a: number) => [number, number];
    readonly wasmsim_nbodies: (a: number) => number;
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
    readonly wasmworldsim_nbodies: (a: number) => number;
    readonly wasmguardiansim_e0: (a: number) => number;
    readonly wasmguardiansim_time: (a: number) => number;
    readonly wasmguardiansim_v_val: (a: number) => number;
    readonly wasmlbmsim_grid_nx: (a: number) => number;
    readonly wasmlbmsim_time: (a: number) => number;
    readonly wasmmdsim_box_size: (a: number) => number;
    readonly wasmmpmsim_particle_radius: (a: number) => number;
    readonly wasmmpmsim_time: (a: number) => number;
    readonly wasmparticlesim_time: (a: number) => number;
    readonly wasmpolicygridsim_num_envs: (a: number) => number;
    readonly wasmprobsim_time: (a: number) => number;
    readonly wasmqftsim_beta: (a: number) => number;
    readonly wasmsim_time: (a: number) => number;
    readonly wasmwavefieldsim_num_pendulums: (a: number) => number;
    readonly wasmwavefieldsim_time: (a: number) => number;
    readonly wasmworldsim_time: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
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
