/* @ts-self-types="./tau_wasm.d.ts" */

export class WasmCradleSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmCradleSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmCradleSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmCradleSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmcradlesim_free(ptr, 0);
    }
    /**
     * Bob positions as flat [x0,y0,z0, ...].
     * @returns {Float64Array}
     */
    bob_positions() {
        const ret = wasm.wasmcradlesim_bob_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    bob_radius() {
        const ret = wasm.wasmcradlesim_bob_radius(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {WasmCradleSim}
     */
    static new() {
        const ret = wasm.wasmcradlesim_new();
        return WasmCradleSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_bobs() {
        const ret = wasm.wasmcradlesim_num_bobs(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Pivot positions as flat [x0,y0,z0, ...].
     * @returns {Float64Array}
     */
    pivot_positions() {
        const ret = wasm.wasmcradlesim_pivot_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmcradlesim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcradlesim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmCradleSim.prototype[Symbol.dispose] = WasmCradleSim.prototype.free;

export class WasmEmSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmEmSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmEmSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmEmSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmemsim_free(ptr, 0);
    }
    /**
     * @returns {WasmEmSim}
     */
    static dipole() {
        const ret = wasm.wasmemsim_dipole();
        return WasmEmSim.__wrap(ret);
    }
    /**
     * @returns {WasmEmSim}
     */
    static double_slit() {
        const ret = wasm.wasmemsim_double_slit();
        return WasmEmSim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    field() {
        const ret = wasm.wasmemsim_field(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    grid_size() {
        const ret = wasm.wasmemsim_grid_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Float64Array}
     */
    mask_data() {
        const ret = wasm.wasmemsim_mask_data(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmemsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmemsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {WasmEmSim}
     */
    static waveguide() {
        const ret = wasm.wasmemsim_waveguide();
        return WasmEmSim.__wrap(ret);
    }
}
if (Symbol.dispose) WasmEmSim.prototype[Symbol.dispose] = WasmEmSim.prototype.free;

/**
 * 100 double pendulums with tiny perturbations showing chaotic divergence.
 */
export class WasmEnsembleSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmEnsembleSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmEnsembleSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmEnsembleSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmensemblesim_free(ptr, 0);
    }
    /**
     * Second bob endpoint positions as flat [x0,y0,z0, ...].
     * @returns {Float64Array}
     */
    endpoint_positions() {
        const ret = wasm.wasmensemblesim_endpoint_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmEnsembleSim}
     */
    static new() {
        const ret = wasm.wasmensemblesim_new();
        return WasmEnsembleSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_instances() {
        const ret = wasm.wasmensemblesim_num_instances(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmensemblesim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmensemblesim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmEnsembleSim.prototype[Symbol.dispose] = WasmEnsembleSim.prototype.free;

export class WasmGravitySim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmGravitySim.prototype);
        obj.__wbg_ptr = ptr;
        WasmGravitySimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmGravitySimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmgravitysim_free(ptr, 0);
    }
    /**
     * Two bodies in mutual orbit.
     * @returns {WasmGravitySim}
     */
    static binary_orbit() {
        const ret = wasm.wasmgravitysim_binary_orbit();
        return WasmGravitySim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    masses() {
        const ret = wasm.wasmgravitysim_masses(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    num_bodies() {
        const ret = wasm.wasmgravitysim_num_bodies(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Float64Array}
     */
    positions() {
        const ret = wasm.wasmgravitysim_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Mercury-like orbit with post-Newtonian precession.
     * @returns {WasmGravitySim}
     */
    static precession() {
        const ret = wasm.wasmgravitysim_precession();
        return WasmGravitySim.__wrap(ret);
    }
    /**
     * 5 planets orbiting a central mass.
     * @returns {WasmGravitySim}
     */
    static solar_system() {
        const ret = wasm.wasmgravitysim_solar_system();
        return WasmGravitySim.__wrap(ret);
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmgravitysim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmgravitysim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} idx
     * @returns {Float64Array}
     */
    trail_for(idx) {
        const ret = wasm.wasmgravitysim_trail_for(this.__wbg_ptr, idx);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} idx
     * @returns {number}
     */
    trail_len(idx) {
        const ret = wasm.wasmgravitysim_trail_len(this.__wbg_ptr, idx);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WasmGravitySim.prototype[Symbol.dispose] = WasmGravitySim.prototype.free;

export class WasmGuardianSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmGuardianSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmGuardianSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmGuardianSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmguardiansim_free(ptr, 0);
    }
    /**
     * Fixed vs adaptive timestep comparison.
     * @returns {WasmGuardianSim}
     */
    static adaptive_dt() {
        const ret = wasm.wasmguardiansim_adaptive_dt();
        return WasmGuardianSim.__wrap(ret);
    }
    /**
     * Correction demo with energy injection and guardian fix.
     * @returns {WasmGuardianSim}
     */
    static correction_demo() {
        const ret = wasm.wasmguardiansim_correction_demo();
        return WasmGuardianSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    e0() {
        const ret = wasm.wasmcradlesim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * Pendulum with energy monitoring gauge.
     * @returns {WasmGuardianSim}
     */
    static energy_monitor() {
        const ret = wasm.wasmguardiansim_energy_monitor();
        return WasmGuardianSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    ke() {
        const ret = wasm.wasmguardiansim_ke(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    ke_history() {
        const ret = wasm.wasmguardiansim_ke_history(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    length() {
        const ret = wasm.wasmguardiansim_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    pe() {
        const ret = wasm.wasmguardiansim_pe(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    pe_history() {
        const ret = wasm.wasmguardiansim_pe_history(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    q2_val() {
        const ret = wasm.wasmguardiansim_q2_val(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    q_val() {
        const ret = wasm.wasmguardiansim_q_val(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmguardiansim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmemsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    total_energy() {
        const ret = wasm.wasmguardiansim_total_energy(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    total_history() {
        const ret = wasm.wasmguardiansim_total_history(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    total_history2() {
        const ret = wasm.wasmguardiansim_total_history2(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    v_val() {
        const ret = wasm.wasmgravitysim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmGuardianSim.prototype[Symbol.dispose] = WasmGuardianSim.prototype.free;

export class WasmLbmSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmLbmSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmLbmSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLbmSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlbmsim_free(ptr, 0);
    }
    /**
     * Lid-driven cavity flow.
     * @returns {WasmLbmSim}
     */
    static cavity_flow() {
        const ret = wasm.wasmlbmsim_cavity_flow();
        return WasmLbmSim.__wrap(ret);
    }
    /**
     * Poiseuille channel flow.
     * @returns {WasmLbmSim}
     */
    static channel_flow() {
        const ret = wasm.wasmlbmsim_channel_flow();
        return WasmLbmSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    grid_nx() {
        const ret = wasm.wasmgravitysim_num_bodies(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    grid_ny() {
        const ret = wasm.wasmlbmsim_grid_ny(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmlbmsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmgravitysim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * Velocity magnitude field as flat array [nx*ny].
     * @returns {Float64Array}
     */
    velocity_field() {
        const ret = wasm.wasmlbmsim_velocity_field(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Flow around obstacle (von Karman vortex street).
     * @returns {WasmLbmSim}
     */
    static vortex_street() {
        const ret = wasm.wasmlbmsim_vortex_street();
        return WasmLbmSim.__wrap(ret);
    }
    /**
     * Vorticity field (curl of velocity).
     * @returns {Float64Array}
     */
    vorticity_field() {
        const ret = wasm.wasmlbmsim_vorticity_field(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) WasmLbmSim.prototype[Symbol.dispose] = WasmLbmSim.prototype.free;

export class WasmLorentzSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmLorentzSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmLorentzSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLorentzSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlorentzsim_free(ptr, 0);
    }
    /**
     * E x B drift: crossed electric and magnetic fields.
     * @returns {WasmLorentzSim}
     */
    static crossed_fields() {
        const ret = wasm.wasmlorentzsim_crossed_fields();
        return WasmLorentzSim.__wrap(ret);
    }
    /**
     * Magnetic mirror: converging B field lines.
     * @returns {WasmLorentzSim}
     */
    static magnetic_mirror() {
        const ret = wasm.wasmlorentzsim_magnetic_mirror();
        return WasmLorentzSim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    position() {
        const ret = wasm.wasmlorentzsim_position(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    speed() {
        const ret = wasm.wasmlorentzsim_speed(this.__wbg_ptr);
        return ret;
    }
    /**
     * Helical spiral in uniform B field along z.
     * @returns {WasmLorentzSim}
     */
    static spiral() {
        const ret = wasm.wasmlorentzsim_spiral();
        return WasmLorentzSim.__wrap(ret);
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmlorentzsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmlorentzsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    trail() {
        const ret = wasm.wasmlorentzsim_trail(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    trail_len() {
        const ret = wasm.wasmlorentzsim_trail_len(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WasmLorentzSim.prototype[Symbol.dispose] = WasmLorentzSim.prototype.free;

export class WasmMdSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmMdSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmMdSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMdSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmdsim_free(ptr, 0);
    }
    /**
     * @returns {WasmMdSim}
     */
    static argon_gas() {
        const ret = wasm.wasmmdsim_argon_gas();
        return WasmMdSim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    bond_endpoints() {
        const ret = wasm.wasmmdsim_bond_endpoints(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    box_size() {
        const ret = wasm.wasmemsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {WasmMdSim}
     */
    static crystal() {
        const ret = wasm.wasmmdsim_crystal();
        return WasmMdSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_bonds() {
        const ret = wasm.wasmmdsim_num_bonds(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    num_particles() {
        const ret = wasm.wasmmdsim_num_particles(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {WasmMdSim}
     */
    static polymer() {
        const ret = wasm.wasmmdsim_polymer();
        return WasmMdSim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    positions() {
        const ret = wasm.wasmmdsim_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmmdsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    temperature() {
        const ret = wasm.wasmmdsim_temperature(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmmdsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    velocities() {
        const ret = wasm.wasmmdsim_velocities(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) WasmMdSim.prototype[Symbol.dispose] = WasmMdSim.prototype.free;

export class WasmMpmSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmMpmSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmMpmSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMpmSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmpmsim_free(ptr, 0);
    }
    /**
     * @returns {WasmMpmSim}
     */
    static elastic_blob() {
        const ret = wasm.wasmmpmsim_elastic_blob();
        return WasmMpmSim.__wrap(ret);
    }
    /**
     * @returns {WasmMpmSim}
     */
    static fluid_dam() {
        const ret = wasm.wasmmpmsim_fluid_dam();
        return WasmMpmSim.__wrap(ret);
    }
    /**
     * @returns {WasmMpmSim}
     */
    static granular_column() {
        const ret = wasm.wasmmpmsim_granular_column();
        return WasmMpmSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_particles() {
        const ret = wasm.wasmmpmsim_num_particles(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    num_springs() {
        const ret = wasm.wasmmpmsim_num_springs(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    particle_radius() {
        const ret = wasm.wasmguardiansim_q_val(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    positions() {
        const ret = wasm.wasmmpmsim_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    spring_endpoints() {
        const ret = wasm.wasmmpmsim_spring_endpoints(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmmpmsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcradlesim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    velocities() {
        const ret = wasm.wasmmpmsim_velocities(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) WasmMpmSim.prototype[Symbol.dispose] = WasmMpmSim.prototype.free;

export class WasmParticleSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmParticleSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmParticleSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmParticleSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmparticlesim_free(ptr, 0);
    }
    /**
     * Three spheres with different bounce coefficients.
     * @returns {WasmParticleSim}
     */
    static bouncing_spheres() {
        const ret = wasm.wasmparticlesim_bouncing_spheres();
        return WasmParticleSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_particles() {
        const ret = wasm.wasmparticlesim_num_particles(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Flat [x0,y0,z0, x1,y1,z1, ...] positions.
     * @returns {Float64Array}
     */
    positions() {
        const ret = wasm.wasmparticlesim_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    radii() {
        const ret = wasm.wasmparticlesim_radii(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * 20 spheres cascading into a pile.
     * @returns {WasmParticleSim}
     */
    static sphere_cascade() {
        const ret = wasm.wasmparticlesim_sphere_cascade();
        return WasmParticleSim.__wrap(ret);
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmparticlesim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmemsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmParticleSim.prototype[Symbol.dispose] = WasmParticleSim.prototype.free;

/**
 * 100 pendulums with different PD controllers for swing-up.
 */
export class WasmPolicyGridSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmPolicyGridSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmPolicyGridSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmPolicyGridSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmpolicygridsim_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    angles() {
        const ret = wasm.wasmpolicygridsim_angles(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmPolicyGridSim}
     */
    static new() {
        const ret = wasm.wasmpolicygridsim_new();
        return WasmPolicyGridSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_envs() {
        const ret = wasm.wasmemsim_grid_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Reward: 1 when inverted (q=π), 0 when hanging (q=0).
     * @returns {Float64Array}
     */
    rewards() {
        const ret = wasm.wasmpolicygridsim_rewards(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmpolicygridsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmpolicygridsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmPolicyGridSim.prototype[Symbol.dispose] = WasmPolicyGridSim.prototype.free;

export class WasmProbSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmProbSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmProbSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmProbSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmprobsim_free(ptr, 0);
    }
    /**
     * Ensemble of bouncing balls with parameter uncertainty.
     * @returns {WasmProbSim}
     */
    static monte_carlo() {
        const ret = wasm.wasmprobsim_monte_carlo();
        return WasmProbSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_particles() {
        const ret = wasm.wasmprobsim_num_particles(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Float64Array}
     */
    positions() {
        const ret = wasm.wasmprobsim_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    spread() {
        const ret = wasm.wasmprobsim_spread(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmprobsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * SVGD particles converging to a target distribution (2D Gaussian mixture).
     * @returns {WasmProbSim}
     */
    static svgd_particles() {
        const ret = wasm.wasmprobsim_svgd_particles();
        return WasmProbSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmguardiansim_q_val(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    trail_flat() {
        const ret = wasm.wasmprobsim_trail_flat(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Ensemble trajectories diverging from near-identical ICs.
     * No gravity — pure spreading to show uncertainty cone.
     * @returns {WasmProbSim}
     */
    static uncertainty_cone() {
        const ret = wasm.wasmprobsim_uncertainty_cone();
        return WasmProbSim.__wrap(ret);
    }
}
if (Symbol.dispose) WasmProbSim.prototype[Symbol.dispose] = WasmProbSim.prototype.free;

export class WasmQftSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmQftSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmQftSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmQftSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmqftsim_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    average_plaquette() {
        const ret = wasm.wasmqftsim_average_plaquette(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    beta() {
        const ret = wasm.wasmguardiansim_q_val(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    lattice_size() {
        const ret = wasm.wasmqftsim_lattice_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {WasmQftSim}
     */
    static phase_scan() {
        const ret = wasm.wasmqftsim_phase_scan();
        return WasmQftSim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    plaq_history() {
        const ret = wasm.wasmqftsim_plaq_history(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    plaquettes() {
        const ret = wasm.wasmqftsim_plaquettes(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} metro_sweeps
     */
    step_n(metro_sweeps) {
        wasm.wasmqftsim_step_n(this.__wbg_ptr, metro_sweeps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmqftsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {WasmQftSim}
     */
    static u1_plaquette() {
        const ret = wasm.wasmqftsim_u1_plaquette();
        return WasmQftSim.__wrap(ret);
    }
    /**
     * @param {number} r
     * @param {number} t
     * @returns {number}
     */
    wilson_loop(r, t) {
        const ret = wasm.wasmqftsim_wilson_loop(this.__wbg_ptr, r, t);
        return ret;
    }
    /**
     * @returns {WasmQftSim}
     */
    static wilson_loops() {
        const ret = wasm.wasmqftsim_wilson_loops();
        return WasmQftSim.__wrap(ret);
    }
}
if (Symbol.dispose) WasmQftSim.prototype[Symbol.dispose] = WasmQftSim.prototype.free;

export class WasmSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmsim_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    body_endpoint_positions() {
        const ret = wasm.wasmsim_body_endpoint_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * N-link chain with viscous damping.
     * @param {number} n
     * @returns {WasmSim}
     */
    static chain(n) {
        const ret = wasm.wasmsim_chain(n);
        return WasmSim.__wrap(ret);
    }
    /**
     * Double pendulum with viscous damping.
     * @returns {WasmSim}
     */
    static double_pendulum() {
        const ret = wasm.wasmsim_double_pendulum();
        return WasmSim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    joint_angles() {
        const ret = wasm.wasmsim_joint_angles(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    joint_positions() {
        const ret = wasm.wasmsim_joint_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    nbodies() {
        const ret = wasm.wasmsim_nbodies(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Single pendulum with viscous damping.
     * @returns {WasmSim}
     */
    static pendulum() {
        const ret = wasm.wasmsim_pendulum();
        return WasmSim.__wrap(ret);
    }
    /**
     * Step the simulation forward n times using RK4.
     * @param {number} n
     */
    step_n(n) {
        wasm.wasmsim_step_n(this.__wbg_ptr, n);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcradlesim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    total_energy() {
        const ret = wasm.wasmsim_total_energy(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmSim.prototype[Symbol.dispose] = WasmSim.prototype.free;

/**
 * 1024 pendulums with varying lengths creating wave interference.
 */
export class WasmWaveFieldSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmWaveFieldSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmWaveFieldSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmWaveFieldSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmwavefieldsim_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    angles() {
        const ret = wasm.wasmwavefieldsim_angles(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmWaveFieldSim}
     */
    static new() {
        const ret = wasm.wasmwavefieldsim_new();
        return WasmWaveFieldSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_pendulums() {
        const ret = wasm.wasmqftsim_lattice_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmwavefieldsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmguardiansim_q_val(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    velocities() {
        const ret = wasm.wasmwavefieldsim_velocities(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) WasmWaveFieldSim.prototype[Symbol.dispose] = WasmWaveFieldSim.prototype.free;

export class WasmWorldSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmWorldSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmWorldSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmWorldSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmworldsim_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    body_endpoint_positions() {
        const ret = wasm.wasmworldsim_body_endpoint_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    link_lengths() {
        const ret = wasm.wasmworldsim_link_lengths(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    nbodies() {
        const ret = wasm.wasmsim_nbodies(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {WasmWorldSim}
     */
    static phase_portrait() {
        const ret = wasm.wasmworldsim_phase_portrait();
        return WasmWorldSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    q() {
        const ret = wasm.wasmworldsim_q(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {WasmWorldSim}
     */
    static random_chain() {
        const ret = wasm.wasmworldsim_random_chain();
        return WasmWorldSim.__wrap(ret);
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmworldsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {WasmWorldSim}
     */
    static tendon_actuated() {
        const ret = wasm.wasmworldsim_tendon_actuated();
        return WasmWorldSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcradlesim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    v() {
        const ret = wasm.wasmworldsim_v(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmWorldSim.prototype[Symbol.dispose] = WasmWorldSim.prototype.free;

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_be289d5034ed271b: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./tau_wasm_bg.js": import0,
    };
}

const WasmCradleSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcradlesim_free(ptr >>> 0, 1));
const WasmEmSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmemsim_free(ptr >>> 0, 1));
const WasmEnsembleSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmensemblesim_free(ptr >>> 0, 1));
const WasmGravitySimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmgravitysim_free(ptr >>> 0, 1));
const WasmGuardianSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmguardiansim_free(ptr >>> 0, 1));
const WasmLbmSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlbmsim_free(ptr >>> 0, 1));
const WasmLorentzSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlorentzsim_free(ptr >>> 0, 1));
const WasmMdSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmdsim_free(ptr >>> 0, 1));
const WasmMpmSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmpmsim_free(ptr >>> 0, 1));
const WasmParticleSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmparticlesim_free(ptr >>> 0, 1));
const WasmPolicyGridSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmpolicygridsim_free(ptr >>> 0, 1));
const WasmProbSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmprobsim_free(ptr >>> 0, 1));
const WasmQftSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmqftsim_free(ptr >>> 0, 1));
const WasmSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmsim_free(ptr >>> 0, 1));
const WasmWaveFieldSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmwavefieldsim_free(ptr >>> 0, 1));
const WasmWorldSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmworldsim_free(ptr >>> 0, 1));

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat64ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('tau_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
