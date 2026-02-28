/* @ts-self-types="./phyz_wasm.d.ts" */

export class QuantumSolver {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        QuantumSolverFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_quantumsolver_free(ptr, 0);
    }
    /**
     * @returns {string}
     */
    info() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.quantumsolver_info(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} triangulation
     */
    constructor(triangulation) {
        const ptr0 = passStringToWasm0(triangulation, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.quantumsolver_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0] >>> 0;
        QuantumSolverFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Solve for all partitions at once. Returns JSON with ground_state_energy,
     * entropy_per_partition, and walltime_ms.
     * @param {number} coupling_g2
     * @param {bigint} geometry_seed
     * @param {string} perturbation_type
     * @param {number} perturbation_index
     * @param {number} perturbation_direction
     * @param {number} fd_epsilon
     * @returns {string}
     */
    solve_all_partitions(coupling_g2, geometry_seed, perturbation_type, perturbation_index, perturbation_direction, fd_epsilon) {
        let deferred3_0;
        let deferred3_1;
        try {
            const ptr0 = passStringToWasm0(perturbation_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.quantumsolver_solve_all_partitions(this.__wbg_ptr, coupling_g2, geometry_seed, ptr0, len0, perturbation_index, perturbation_direction, fd_epsilon);
            var ptr2 = ret[0];
            var len2 = ret[1];
            if (ret[3]) {
                ptr2 = 0; len2 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred3_0 = ptr2;
            deferred3_1 = len2;
            return getStringFromWasm0(ptr2, len2);
        } finally {
            wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
        }
    }
    /**
     * GPU-accelerated solve via WebGPU. Falls back to CPU on failure.
     * Returns JSON identical to [`solve_all_partitions`].
     * @param {number} coupling_g2
     * @param {bigint} geometry_seed
     * @param {string} perturbation_type
     * @param {number} perturbation_index
     * @param {number} perturbation_direction
     * @param {number} fd_epsilon
     * @returns {Promise<string>}
     */
    solve_all_partitions_gpu(coupling_g2, geometry_seed, perturbation_type, perturbation_index, perturbation_direction, fd_epsilon) {
        const ptr0 = passStringToWasm0(perturbation_type, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.quantumsolver_solve_all_partitions_gpu(this.__wbg_ptr, coupling_g2, geometry_seed, ptr0, len0, perturbation_index, perturbation_direction, fd_epsilon);
        return ret;
    }
}
if (Symbol.dispose) QuantumSolver.prototype[Symbol.dispose] = QuantumSolver.prototype.free;

/**
 * Fusion viz: two kernels side-by-side, fuse them, animate the merge.
 */
export class WasmCompileFusionSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmCompileFusionSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmCompileFusionSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmCompileFusionSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmcompilefusionsim_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    a_count() {
        const ret = wasm.wasmcompilefusionsim_a_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string[]}
     */
    a_names() {
        const ret = wasm.wasmcompilefusionsim_a_names(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    b_count() {
        const ret = wasm.wasmcompilefusionsim_b_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string[]}
     */
    b_names() {
        const ret = wasm.wasmcompilefusionsim_b_names(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    fused_count() {
        const ret = wasm.wasmcompilefusionsim_fused_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {string[]}
     */
    fused_names() {
        const ret = wasm.wasmcompilefusionsim_fused_names(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {WasmCompileFusionSim}
     */
    static new() {
        const ret = wasm.wasmcompilefusionsim_new();
        return WasmCompileFusionSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    progress() {
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} _n
     */
    step_n(_n) {
        wasm.wasmcompilefusionsim_step_n(this.__wbg_ptr, _n);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcompilefusionsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmCompileFusionSim.prototype[Symbol.dispose] = WasmCompileFusionSim.prototype.free;

/**
 * Kernel IR: build a simple physics kernel and return its DAG structure.
 */
export class WasmCompileIrSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmCompileIrSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmCompileIrSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmCompileIrSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmcompileirsim_free(ptr, 0);
    }
    /**
     * @returns {WasmCompileIrSim}
     */
    static new() {
        const ret = wasm.wasmcompileirsim_new();
        return WasmCompileIrSim.__wrap(ret);
    }
    /**
     * @returns {Uint32Array}
     */
    node_depths() {
        const ret = wasm.wasmcompileirsim_node_depths(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string[]}
     */
    node_names() {
        const ret = wasm.wasmcompileirsim_node_names(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Int32Array}
     */
    node_parents() {
        const ret = wasm.wasmcompileirsim_node_parents(this.__wbg_ptr);
        var v1 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string[]}
     */
    node_types() {
        const ret = wasm.wasmcompileirsim_node_types(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    num_nodes() {
        const ret = wasm.wasmcompileirsim_num_nodes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcompileirsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmCompileIrSim.prototype[Symbol.dispose] = WasmCompileIrSim.prototype.free;

/**
 * WGSL output: display generated WGSL source for different physics ops.
 */
export class WasmCompileWgslSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmCompileWgslSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmCompileWgslSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmCompileWgslSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmcompilewgslsim_free(ptr, 0);
    }
    /**
     * @returns {string[]}
     */
    all_labels() {
        const ret = wasm.wasmcompilewgslsim_all_labels(this.__wbg_ptr);
        var v1 = getArrayJsValueFromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string}
     */
    current_label() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmcompilewgslsim_current_label(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {string}
     */
    current_source() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmcompilewgslsim_current_source(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @returns {WasmCompileWgslSim}
     */
    static new() {
        const ret = wasm.wasmcompilewgslsim_new();
        return WasmCompileWgslSim.__wrap(ret);
    }
    /**
     * Cycle to next kernel
     */
    next() {
        wasm.wasmcompilewgslsim_next(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    num_kernels() {
        const ret = wasm.wasmcompileirsim_num_nodes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    source_lines() {
        const ret = wasm.wasmcompilewgslsim_source_lines(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcompilewgslsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmCompileWgslSim.prototype[Symbol.dispose] = WasmCompileWgslSim.prototype.free;

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
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmCradleSim.prototype[Symbol.dispose] = WasmCradleSim.prototype.free;

/**
 * Gradient descent optimization: find initial angle to hit target x position.
 */
export class WasmDiffGradientSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmDiffGradientSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmDiffGradientSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmDiffGradientSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmdiffgradientsim_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    current_loss() {
        const ret = wasm.wasmdiffgradientsim_current_loss(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    current_x() {
        const ret = wasm.wasmdiffgradientsim_current_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    current_y() {
        const ret = wasm.wasmdiffgradientsim_current_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    iteration() {
        const ret = wasm.wasmdiffgradientsim_iteration(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Float64Array}
     */
    loss_history() {
        const ret = wasm.wasmdiffgradientsim_loss_history(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmDiffGradientSim}
     */
    static new() {
        const ret = wasm.wasmdiffgradientsim_new();
        return WasmDiffGradientSim.__wrap(ret);
    }
    /**
     * Run one gradient descent step: simulate forward, compute loss gradient, update theta0.
     * @param {number} n
     */
    step_n(n) {
        wasm.wasmdiffgradientsim_step_n(this.__wbg_ptr, n);
    }
    /**
     * @returns {number}
     */
    target_x() {
        const ret = wasm.wasmdiffgradientsim_target_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    theta0() {
        const ret = wasm.wasmdiffgradientsim_theta0(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmdiffgradientsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmDiffGradientSim.prototype[Symbol.dispose] = WasmDiffGradientSim.prototype.free;

/**
 * Jacobian heatmap: pendulum simulation with live Jacobian matrix display.
 */
export class WasmDiffJacobianSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmDiffJacobianSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmDiffJacobianSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmDiffJacobianSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmdiffjacobiansim_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    angle() {
        const ret = wasm.wasmdiffjacobiansim_angle(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    bob_x() {
        const ret = wasm.wasmdiffjacobiansim_bob_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    bob_y() {
        const ret = wasm.wasmdiffjacobiansim_bob_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    jacobian() {
        const ret = wasm.wasmdiffjacobiansim_jacobian(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmDiffJacobianSim}
     */
    static new() {
        const ret = wasm.wasmdiffjacobiansim_new();
        return WasmDiffJacobianSim.__wrap(ret);
    }
    /**
     * @param {number} n
     */
    step_n(n) {
        wasm.wasmdiffjacobiansim_step_n(this.__wbg_ptr, n);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmDiffJacobianSim.prototype[Symbol.dispose] = WasmDiffJacobianSim.prototype.free;

/**
 * Sensitivity: two pendulums with perturbed ICs, analytical gradient prediction vs actual divergence.
 */
export class WasmDiffSensitivitySim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmDiffSensitivitySim.prototype);
        obj.__wbg_ptr = ptr;
        WasmDiffSensitivitySimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmDiffSensitivitySimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmdiffsensitivitysim_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    actual_divergence() {
        const ret = wasm.wasmdiffsensitivitysim_actual_divergence(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmDiffSensitivitySim}
     */
    static new() {
        const ret = wasm.wasmdiffsensitivitysim_new();
        return WasmDiffSensitivitySim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    pos_a() {
        const ret = wasm.wasmdiffsensitivitysim_pos_a(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    pos_b() {
        const ret = wasm.wasmdiffsensitivitysim_pos_b(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    predicted_divergence() {
        const ret = wasm.wasmdiffsensitivitysim_predicted_divergence(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} n
     */
    step_n(n) {
        wasm.wasmdiffsensitivitysim_step_n(this.__wbg_ptr, n);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmDiffSensitivitySim.prototype[Symbol.dispose] = WasmDiffSensitivitySim.prototype.free;

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

export class WasmGravitySandboxSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmGravitySandboxSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmGravitySandboxSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmGravitySandboxSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmgravitysandboxsim_free(ptr, 0);
    }
    /**
     * Add a new body at (x, y) with velocity (vx, vy) and given mass.
     * @param {number} x
     * @param {number} y
     * @param {number} vx
     * @param {number} vy
     * @param {number} m
     */
    add_body(x, y, vx, vy, m) {
        wasm.wasmgravitysandboxsim_add_body(this.__wbg_ptr, x, y, vx, vy, m);
    }
    /**
     * @returns {Float64Array}
     */
    masses() {
        const ret = wasm.wasmgravitysandboxsim_masses(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmGravitySandboxSim}
     */
    static new() {
        const ret = wasm.wasmgravitysandboxsim_new();
        return WasmGravitySandboxSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_bodies() {
        const ret = wasm.wasmgravitysandboxsim_num_bodies(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Float64Array}
     */
    positions() {
        const ret = wasm.wasmgravitysandboxsim_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmgravitysandboxsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmemsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} idx
     * @returns {Float64Array}
     */
    trail_for(idx) {
        const ret = wasm.wasmgravitysandboxsim_trail_for(this.__wbg_ptr, idx);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) WasmGravitySandboxSim.prototype[Symbol.dispose] = WasmGravitySandboxSim.prototype.free;

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
     * Two disk galaxies colliding — tidal tails form as they merge.
     * @returns {WasmGravitySim}
     */
    static galaxy_collision() {
        const ret = wasm.wasmgravitysim_galaxy_collision();
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
     * Return interleaved velocity magnitudes for coloring by speed.
     * @returns {Float64Array}
     */
    speeds() {
        const ret = wasm.wasmgravitysim_speeds(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
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
        const ret = wasm.wasmcompilefusionsim_time(this.__wbg_ptr);
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

export class WasmGripperSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmGripperSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmGripperSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmGripperSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmgrippersim_free(ptr, 0);
    }
    /**
     * @returns {WasmGripperSim}
     */
    static new() {
        const ret = wasm.wasmgrippersim_new();
        return WasmGripperSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_particles() {
        const ret = wasm.wasmcompilefusionsim_a_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Float64Array}
     */
    positions() {
        const ret = wasm.wasmgrippersim_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmgrippersim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmGripperSim.prototype[Symbol.dispose] = WasmGripperSim.prototype.free;

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
        const ret = wasm.wasmguardiansim_e0(this.__wbg_ptr);
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
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
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
        const ret = wasm.wasmcompilefusionsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmGuardianSim.prototype[Symbol.dispose] = WasmGuardianSim.prototype.free;

export class WasmHourglassSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmHourglassSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmHourglassSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmHourglassSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmhourglasssim_free(ptr, 0);
    }
    /**
     * @returns {WasmHourglassSim}
     */
    static new() {
        const ret = wasm.wasmhourglasssim_new();
        return WasmHourglassSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_particles() {
        const ret = wasm.wasmhourglasssim_num_particles(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Hourglass outline as [x0,y0, x1,y1, ...] for rendering the glass shape.
     * @returns {Float64Array}
     */
    outline() {
        const ret = wasm.wasmhourglasssim_outline(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    particle_radius() {
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    positions() {
        const ret = wasm.wasmhourglasssim_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmhourglasssim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmdiffgradientsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmHourglassSim.prototype[Symbol.dispose] = WasmHourglassSim.prototype.free;

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
        const ret = wasm.wasmcompilefusionsim_time(this.__wbg_ptr);
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
     * Hot gas that gradually cools to form a crystal via LJ attraction.
     * @returns {WasmMdSim}
     */
    static cooling_gas() {
        const ret = wasm.wasmmdsim_cooling_gas();
        return WasmMdSim.__wrap(ret);
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

/**
 * Ant model: multi-body spider with 8 legs.
 */
export class WasmMjcfAntSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmMjcfAntSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmMjcfAntSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMjcfAntSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmjcfantsim_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    body_endpoint_positions() {
        const ret = wasm.wasmmjcfantsim_body_endpoint_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    joint_positions() {
        const ret = wasm.wasmmjcfantsim_joint_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    nbodies() {
        const ret = wasm.wasmmjcfantsim_nbodies(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {WasmMjcfAntSim}
     */
    static new() {
        const ret = wasm.wasmmjcfantsim_new();
        return WasmMjcfAntSim.__wrap(ret);
    }
    /**
     * @param {number} n
     */
    step_n(n) {
        wasm.wasmmjcfantsim_step_n(this.__wbg_ptr, n);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmMjcfAntSim.prototype[Symbol.dispose] = WasmMjcfAntSim.prototype.free;

/**
 * Cartpole: cart + inverted pendulum (classic control benchmark).
 */
export class WasmMjcfCartpoleSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmMjcfCartpoleSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmMjcfCartpoleSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMjcfCartpoleSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmjcfcartpolesim_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    cart_x() {
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {WasmMjcfCartpoleSim}
     */
    static new() {
        const ret = wasm.wasmmjcfcartpolesim_new();
        return WasmMjcfCartpoleSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    pole_angle() {
        const ret = wasm.wasmmjcfcartpolesim_pole_angle(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    pole_tip_x() {
        const ret = wasm.wasmmjcfcartpolesim_pole_tip_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    pole_tip_y() {
        const ret = wasm.wasmmjcfcartpolesim_pole_tip_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} n
     */
    step_n(n) {
        wasm.wasmmjcfcartpolesim_step_n(this.__wbg_ptr, n);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmguardiansim_length(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmMjcfCartpoleSim.prototype[Symbol.dispose] = WasmMjcfCartpoleSim.prototype.free;

/**
 * MJCF editor: parse a simple body description and return skeleton.
 */
export class WasmMjcfEditorSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmMjcfEditorSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmMjcfEditorSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMjcfEditorSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmjcfeditorsim_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    n_bodies() {
        const ret = wasm.wasmcompilefusionsim_b_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Parse a simple MJCF-like string. Format: one body per line, "name length mass".
     * @param {string} xml
     * @returns {WasmMjcfEditorSim}
     */
    static parse(xml) {
        const ptr0 = passStringToWasm0(xml, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmmjcfeditorsim_parse(ptr0, len0);
        return WasmMjcfEditorSim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    positions() {
        const ret = wasm.wasmmjcfeditorsim_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) WasmMjcfEditorSim.prototype[Symbol.dispose] = WasmMjcfEditorSim.prototype.free;

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
    static fluid_slosh() {
        const ret = wasm.wasmmpmsim_fluid_slosh();
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
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
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
        const ret = wasm.wasmguardiansim_e0(this.__wbg_ptr);
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
        const ret = wasm.wasmmjcfcartpolesim_pole_angle(this.__wbg_ptr);
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
        const ret = wasm.wasmcompilefusionsim_fused_count(this.__wbg_ptr);
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
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
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
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    lattice_size() {
        const ret = wasm.wasmcompilefusionsim_a_count(this.__wbg_ptr);
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

export class WasmRagdollSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmRagdollSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmRagdollSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmRagdollSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmragdollsim_free(ptr, 0);
    }
    /**
     * Constraint endpoints as flat [ax0,ay0, bx0,by0, ax1,ay1, ...]
     * @returns {Float64Array}
     */
    constraint_endpoints() {
        const ret = wasm.wasmragdollsim_constraint_endpoints(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmRagdollSim}
     */
    static new() {
        const ret = wasm.wasmragdollsim_new();
        return WasmRagdollSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_constraints() {
        const ret = wasm.wasmragdollsim_num_constraints(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    num_particles() {
        const ret = wasm.wasmcompilefusionsim_fused_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    num_steps() {
        const ret = wasm.wasmragdollsim_num_steps(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Particle positions as flat [x0,y0, x1,y1, ...]
     * @returns {Float64Array}
     */
    positions() {
        const ret = wasm.wasmragdollsim_positions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Stair geometry as flat [x0,y0, x1,y1, ...] — each pair of points is one step (left-top, right-top).
     * @returns {Float64Array}
     */
    stair_geometry() {
        const ret = wasm.wasmragdollsim_stair_geometry(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmragdollsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcompilefusionsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmRagdollSim.prototype[Symbol.dispose] = WasmRagdollSim.prototype.free;

/**
 * Adam vs GD: compare two optimizers on the same problem.
 */
export class WasmReal2SimAdamVsGdSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmReal2SimAdamVsGdSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmReal2SimAdamVsGdSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmReal2SimAdamVsGdSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmreal2simadamvsgdsim_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    adam_length() {
        const ret = wasm.wasmguardiansim_e0(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    adam_loss() {
        const ret = wasm.wasmreal2simadamvsgdsim_adam_loss(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    adam_mass() {
        const ret = wasm.wasmmdsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    gd_length() {
        const ret = wasm.wasmreal2simadamvsgdsim_gd_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    gd_loss() {
        const ret = wasm.wasmreal2simadamvsgdsim_gd_loss(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    gd_mass() {
        const ret = wasm.wasmguardiansim_q2_val(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    iteration() {
        const ret = wasm.wasmmpmsim_num_particles(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {WasmReal2SimAdamVsGdSim}
     */
    static new() {
        const ret = wasm.wasmreal2simadamvsgdsim_new();
        return WasmReal2SimAdamVsGdSim.__wrap(ret);
    }
    /**
     * @param {number} n
     */
    step_n(n) {
        wasm.wasmreal2simadamvsgdsim_step_n(this.__wbg_ptr, n);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmreal2simadamvsgdsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmReal2SimAdamVsGdSim.prototype[Symbol.dispose] = WasmReal2SimAdamVsGdSim.prototype.free;

/**
 * Parameter fitting: gradient descent to match a reference trajectory.
 */
export class WasmReal2SimFitSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmReal2SimFitSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmReal2SimFitSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmReal2SimFitSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmreal2simfitsim_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    current_loss() {
        const ret = wasm.wasmreal2simfitsim_current_loss(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    est_length() {
        const ret = wasm.wasmcompilefusionsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    est_mass() {
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * Current estimated trajectory (for overlay rendering)
     * @returns {Float64Array}
     */
    est_trajectory() {
        const ret = wasm.wasmreal2simfitsim_est_trajectory(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    iteration() {
        const ret = wasm.wasmparticlesim_num_particles(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Float64Array}
     */
    loss_history() {
        const ret = wasm.wasmreal2simfitsim_loss_history(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmReal2SimFitSim}
     */
    static new() {
        const ret = wasm.wasmreal2simfitsim_new();
        return WasmReal2SimFitSim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    ref_trajectory() {
        const ret = wasm.wasmreal2simfitsim_ref_trajectory(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} n
     */
    step_n(n) {
        wasm.wasmreal2simfitsim_step_n(this.__wbg_ptr, n);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmreal2simfitsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    true_length() {
        const ret = wasm.wasmemsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    true_mass() {
        const ret = wasm.wasmmjcfcartpolesim_pole_angle(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmReal2SimFitSim.prototype[Symbol.dispose] = WasmReal2SimFitSim.prototype.free;

/**
 * Loss landscape: 2D contour of loss over (mass, length) parameter grid.
 */
export class WasmReal2SimLandscapeSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmReal2SimLandscapeSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmReal2SimLandscapeSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmReal2SimLandscapeSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmreal2simlandscapesim_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    est_length() {
        const ret = wasm.wasmcompilefusionsim_time(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    est_mass() {
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    grid() {
        const ret = wasm.wasmreal2simlandscapesim_grid(this.__wbg_ptr);
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
     * @returns {WasmReal2SimLandscapeSim}
     */
    static new() {
        const ret = wasm.wasmreal2simlandscapesim_new();
        return WasmReal2SimLandscapeSim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    path_l() {
        const ret = wasm.wasmreal2simlandscapesim_path_l(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    path_m() {
        const ret = wasm.wasmreal2simlandscapesim_path_m(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} n
     */
    step_n(n) {
        wasm.wasmreal2simlandscapesim_step_n(this.__wbg_ptr, n);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmreal2simlandscapesim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmReal2SimLandscapeSim.prototype[Symbol.dispose] = WasmReal2SimLandscapeSim.prototype.free;

/**
 * Action landscape: Einstein-Maxwell action vs uniform scale factor, with gradient descent.
 */
export class WasmReggeActionSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmReggeActionSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmReggeActionSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmReggeActionSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmreggeactionsim_free(ptr, 0);
    }
    /**
     * @returns {Float64Array}
     */
    actions() {
        const ret = wasm.wasmreggeactionsim_actions(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {number}
     */
    current_action() {
        const ret = wasm.wasmreggeactionsim_current_action(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    current_scale() {
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Float64Array}
     */
    gd_path() {
        const ret = wasm.wasmreggeactionsim_gd_path(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmReggeActionSim}
     */
    static new() {
        const ret = wasm.wasmreggeactionsim_new();
        return WasmReggeActionSim.__wrap(ret);
    }
    /**
     * @returns {Float64Array}
     */
    scales() {
        const ret = wasm.wasmreggeactionsim_scales(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} n
     */
    step_n(n) {
        wasm.wasmreggeactionsim_step_n(this.__wbg_ptr, n);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmreggeactionsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmReggeActionSim.prototype[Symbol.dispose] = WasmReggeActionSim.prototype.free;

/**
 * Curvature slice: Reissner-Nordström edge lengths as color heatmap.
 */
export class WasmReggeCurvatureSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmReggeCurvatureSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmReggeCurvatureSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmReggeCurvatureSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmreggecurvaturesim_free(ptr, 0);
    }
    /**
     * Compute edge lengths for Reissner-Nordström mesh.
     * Returns flat array of edge length values (one per edge).
     * @param {number} mass
     * @param {number} charge
     * @returns {Float64Array}
     */
    compute(mass, charge) {
        const ret = wasm.wasmreggecurvaturesim_compute(this.__wbg_ptr, mass, charge);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Grid size for heatmap rendering.
     * @returns {number}
     */
    grid_size() {
        const ret = wasm.wasmreggecurvaturesim_grid_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Returns number of edges for the current mesh size.
     * @returns {number}
     */
    n_edges() {
        const ret = wasm.wasmreggecurvaturesim_n_edges(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {WasmReggeCurvatureSim}
     */
    static new() {
        const ret = wasm.wasmreggecurvaturesim_new();
        return WasmReggeCurvatureSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcompileirsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmReggeCurvatureSim.prototype[Symbol.dispose] = WasmReggeCurvatureSim.prototype.free;

/**
 * Symmetry bars: Noether current norms for different spacetimes.
 */
export class WasmReggeSymmetrySim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmReggeSymmetrySim.prototype);
        obj.__wbg_ptr = ptr;
        WasmReggeSymmetrySimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmReggeSymmetrySimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmreggesymmetrysim_free(ptr, 0);
    }
    /**
     * Compute Noether current norms for flat, RN, and Kerr spacetimes.
     * Returns [flat_gauge, flat_translation, rn_gauge, rn_translation, kerr_gauge, kerr_translation].
     * @returns {Float64Array}
     */
    compute() {
        const ret = wasm.wasmreggesymmetrysim_compute(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {WasmReggeSymmetrySim}
     */
    static new() {
        const ret = wasm.wasmreggesymmetrysim_new();
        return WasmReggeSymmetrySim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmcompileirsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmReggeSymmetrySim.prototype[Symbol.dispose] = WasmReggeSymmetrySim.prototype.free;

export class WasmRubeGoldbergSim {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(WasmRubeGoldbergSim.prototype);
        obj.__wbg_ptr = ptr;
        WasmRubeGoldbergSimFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmRubeGoldbergSimFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmrubegoldbergsim_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    domino_height() {
        const ret = wasm.wasmreal2simadamvsgdsim_gd_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    domino_width() {
        const ret = wasm.wasmguardiansim_q2_val(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {WasmRubeGoldbergSim}
     */
    static new() {
        const ret = wasm.wasmrubegoldbergsim_new();
        return WasmRubeGoldbergSim.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    num_dominoes() {
        const ret = wasm.wasmrubegoldbergsim_num_dominoes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Returns all renderable state as flat array:
     * [ball1_x, ball1_y, ball2_x, ball2_y,
     *  pend_anchor_x, pend_anchor_y, pend_bob_x, pend_bob_y,
     *  dom0_x, dom0_y, dom0_angle, dom1_x, ...,
     *  ramp_x0, ramp_y0, ramp_x1, ramp_y1,
     *  bucket_x, bucket_y, bucket_w]
     * @returns {Float64Array}
     */
    state() {
        const ret = wasm.wasmrubegoldbergsim_state(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {number} steps
     */
    step_n(steps) {
        wasm.wasmrubegoldbergsim_step_n(this.__wbg_ptr, steps);
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.wasmrubegoldbergsim_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmRubeGoldbergSim.prototype[Symbol.dispose] = WasmRubeGoldbergSim.prototype.free;

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
        const ret = wasm.wasmmjcfantsim_nbodies(this.__wbg_ptr);
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
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
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
        const ret = wasm.wasmcompilefusionsim_a_count(this.__wbg_ptr);
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
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
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
        const ret = wasm.wasmmjcfantsim_nbodies(this.__wbg_ptr);
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
        const ret = wasm.wasmcompilefusionsim_progress(this.__wbg_ptr);
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
        __wbg_Error_8c4e43fe74559d73: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_Window_cf5b693340a7c469: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_354364d1b0bd06e5: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_debug_string_0bc8482c6e3508ae: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_0095a73b8b156f76: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_ac34f5003991759a: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_object_5ae8e5880f2c1fbd: function(arg0) {
            const val = arg0;
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_undefined_9e4d92534c42d778: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_string_get_72fb696202c56729: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_be289d5034ed271b: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_d9b87ff7982e3b21: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_beginComputePass_90d5303e604970cb: function(arg0, arg1) {
            const ret = arg0.beginComputePass(arg1);
            return ret;
        },
        __wbg_beginRenderPass_9739520c601001c3: function(arg0, arg1) {
            const ret = arg0.beginRenderPass(arg1);
            return ret;
        },
        __wbg_buffer_26d0910f3a5bc899: function(arg0) {
            const ret = arg0.buffer;
            return ret;
        },
        __wbg_call_389efe28435a9388: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.call(arg1);
            return ret;
        }, arguments); },
        __wbg_call_4708e0c13bdc8e95: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_clearBuffer_6164fc25d22b25cc: function(arg0, arg1, arg2, arg3) {
            arg0.clearBuffer(arg1, arg2, arg3);
        },
        __wbg_clearBuffer_cfcaaf1fb2baa885: function(arg0, arg1, arg2) {
            arg0.clearBuffer(arg1, arg2);
        },
        __wbg_configure_2414aed971d368cd: function(arg0, arg1) {
            arg0.configure(arg1);
        },
        __wbg_copyBufferToBuffer_1ba67191114656a1: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4, arg5);
        },
        __wbg_copyBufferToTexture_878d31d479e48f28: function(arg0, arg1, arg2, arg3) {
            arg0.copyBufferToTexture(arg1, arg2, arg3);
        },
        __wbg_copyExternalImageToTexture_7878d196c0b60d39: function(arg0, arg1, arg2, arg3) {
            arg0.copyExternalImageToTexture(arg1, arg2, arg3);
        },
        __wbg_copyTextureToBuffer_6a8fe0e90f0a663d: function(arg0, arg1, arg2, arg3) {
            arg0.copyTextureToBuffer(arg1, arg2, arg3);
        },
        __wbg_copyTextureToTexture_0a06a393d6726b4a: function(arg0, arg1, arg2, arg3) {
            arg0.copyTextureToTexture(arg1, arg2, arg3);
        },
        __wbg_createBindGroupLayout_1d93b6d41c87ba9d: function(arg0, arg1) {
            const ret = arg0.createBindGroupLayout(arg1);
            return ret;
        },
        __wbg_createBindGroup_61cd07ec9d423432: function(arg0, arg1) {
            const ret = arg0.createBindGroup(arg1);
            return ret;
        },
        __wbg_createBuffer_963aa00d5fe859e4: function(arg0, arg1) {
            const ret = arg0.createBuffer(arg1);
            return ret;
        },
        __wbg_createCommandEncoder_f0e1613e9a2dc1eb: function(arg0, arg1) {
            const ret = arg0.createCommandEncoder(arg1);
            return ret;
        },
        __wbg_createComputePipeline_b9616b9fe2f4eb2f: function(arg0, arg1) {
            const ret = arg0.createComputePipeline(arg1);
            return ret;
        },
        __wbg_createPipelineLayout_56c6cf983f892d2b: function(arg0, arg1) {
            const ret = arg0.createPipelineLayout(arg1);
            return ret;
        },
        __wbg_createQuerySet_c14be802adf7c207: function(arg0, arg1) {
            const ret = arg0.createQuerySet(arg1);
            return ret;
        },
        __wbg_createRenderBundleEncoder_8e4bdffea72f8c1f: function(arg0, arg1) {
            const ret = arg0.createRenderBundleEncoder(arg1);
            return ret;
        },
        __wbg_createRenderPipeline_079a88a0601fcce1: function(arg0, arg1) {
            const ret = arg0.createRenderPipeline(arg1);
            return ret;
        },
        __wbg_createSampler_ef5578990df3baf7: function(arg0, arg1) {
            const ret = arg0.createSampler(arg1);
            return ret;
        },
        __wbg_createShaderModule_17f451ea25cae47c: function(arg0, arg1) {
            const ret = arg0.createShaderModule(arg1);
            return ret;
        },
        __wbg_createTexture_01cc1cd2fea732d9: function(arg0, arg1) {
            const ret = arg0.createTexture(arg1);
            return ret;
        },
        __wbg_createView_04701884291e1ccc: function(arg0, arg1) {
            const ret = arg0.createView(arg1);
            return ret;
        },
        __wbg_destroy_35f94012e5bb9c17: function(arg0) {
            arg0.destroy();
        },
        __wbg_destroy_767d9dde1008e293: function(arg0) {
            arg0.destroy();
        },
        __wbg_destroy_c6af4226dda95dbd: function(arg0) {
            arg0.destroy();
        },
        __wbg_dispatchWorkgroupsIndirect_8b25efab93a7a433: function(arg0, arg1, arg2) {
            arg0.dispatchWorkgroupsIndirect(arg1, arg2);
        },
        __wbg_dispatchWorkgroups_c102fa81b955935d: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_document_ee35a3d3ae34ef6c: function(arg0) {
            const ret = arg0.document;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_drawIndexedIndirect_34484fc6227c7bc8: function(arg0, arg1, arg2) {
            arg0.drawIndexedIndirect(arg1, arg2);
        },
        __wbg_drawIndexedIndirect_5a7c30bb5f1d5b67: function(arg0, arg1, arg2) {
            arg0.drawIndexedIndirect(arg1, arg2);
        },
        __wbg_drawIndexed_115af1449b52a948: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.drawIndexed(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4, arg5 >>> 0);
        },
        __wbg_drawIndexed_a587cce4c317791f: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.drawIndexed(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4, arg5 >>> 0);
        },
        __wbg_drawIndirect_036d71498a21f1a3: function(arg0, arg1, arg2) {
            arg0.drawIndirect(arg1, arg2);
        },
        __wbg_drawIndirect_a1d7c5e893aa5756: function(arg0, arg1, arg2) {
            arg0.drawIndirect(arg1, arg2);
        },
        __wbg_draw_5351b12033166aca: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.draw(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_draw_e2a7c5d66fb2d244: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.draw(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_end_0ac71677a5c1717a: function(arg0) {
            arg0.end();
        },
        __wbg_end_6f776519f1faa582: function(arg0) {
            arg0.end();
        },
        __wbg_error_e98e6aadd08e0b94: function(arg0) {
            const ret = arg0.error;
            return ret;
        },
        __wbg_executeBundles_8e6c0614da2805d4: function(arg0, arg1) {
            arg0.executeBundles(arg1);
        },
        __wbg_features_1b464383ea8a7691: function(arg0) {
            const ret = arg0.features;
            return ret;
        },
        __wbg_features_e5fbbc2760867852: function(arg0) {
            const ret = arg0.features;
            return ret;
        },
        __wbg_finish_20711371c58df61c: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_finish_34b2c54329c8719f: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_finish_a9ab917e756ea00c: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_finish_e0a6c97c0622f843: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_getBindGroupLayout_4a94df6108ac6667: function(arg0, arg1) {
            const ret = arg0.getBindGroupLayout(arg1 >>> 0);
            return ret;
        },
        __wbg_getBindGroupLayout_80e803d942962f6a: function(arg0, arg1) {
            const ret = arg0.getBindGroupLayout(arg1 >>> 0);
            return ret;
        },
        __wbg_getCompilationInfo_2af3ecdfeda551a3: function(arg0) {
            const ret = arg0.getCompilationInfo();
            return ret;
        },
        __wbg_getContext_2966500392030d63: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getContext_2a5764d48600bc43: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getCurrentTexture_5a79cda2ff36e1ee: function(arg0) {
            const ret = arg0.getCurrentTexture();
            return ret;
        },
        __wbg_getMappedRange_932dd043ae22ee0a: function(arg0, arg1, arg2) {
            const ret = arg0.getMappedRange(arg1, arg2);
            return ret;
        },
        __wbg_getPreferredCanvasFormat_de73c02773a5209e: function(arg0) {
            const ret = arg0.getPreferredCanvasFormat();
            return (__wbindgen_enum_GpuTextureFormat.indexOf(ret) + 1 || 96) - 1;
        },
        __wbg_get_9b94d73e6221f75c: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_get_d8db2ad31d529ff8: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_gpu_87871e8f7ace8fee: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_has_624cbf0451d880e8: function(arg0, arg1, arg2) {
            const ret = arg0.has(getStringFromWasm0(arg1, arg2));
            return ret;
        },
        __wbg_instanceof_GpuAdapter_0731153d2b08720b: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUAdapter;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_GpuCanvasContext_d14121c7bd72fcef: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUCanvasContext;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_GpuDeviceLostInfo_a3677ebb8241d800: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUDeviceLostInfo;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_GpuOutOfMemoryError_391d9a08edbfa04b: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUOutOfMemoryError;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_GpuValidationError_f4d803c383da3c92: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUValidationError;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Object_1c6af87502b733ed: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Object;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Window_ed49b2db8df90359: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_label_2082ab37d2ad170d: function(arg0, arg1) {
            const ret = arg1.label;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_length_32ed9a279acd054c: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_35a7bace40f36eac: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_length_9df32f7add647235: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_limits_2dd632c891786ddf: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_limits_f6411f884b0b2d62: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_lineNum_0246de1e072ffe19: function(arg0) {
            const ret = arg0.lineNum;
            return ret;
        },
        __wbg_lost_6e4d29847ce2a34a: function(arg0) {
            const ret = arg0.lost;
            return ret;
        },
        __wbg_mapAsync_37f5e03edf2e1352: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.mapAsync(arg1 >>> 0, arg2, arg3);
            return ret;
        },
        __wbg_maxBindGroups_768ca5e8623bf450: function(arg0) {
            const ret = arg0.maxBindGroups;
            return ret;
        },
        __wbg_maxBindingsPerBindGroup_057972d600d69719: function(arg0) {
            const ret = arg0.maxBindingsPerBindGroup;
            return ret;
        },
        __wbg_maxBufferSize_e237b44f19a5a62b: function(arg0) {
            const ret = arg0.maxBufferSize;
            return ret;
        },
        __wbg_maxColorAttachmentBytesPerSample_d6c7b4051d22c6d6: function(arg0) {
            const ret = arg0.maxColorAttachmentBytesPerSample;
            return ret;
        },
        __wbg_maxColorAttachments_7a18ba24c05edcfd: function(arg0) {
            const ret = arg0.maxColorAttachments;
            return ret;
        },
        __wbg_maxComputeInvocationsPerWorkgroup_b99c2f3611633992: function(arg0) {
            const ret = arg0.maxComputeInvocationsPerWorkgroup;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeX_adb26da9ed7f77f7: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeX;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeY_cc217559c98be33b: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeY;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeZ_66606a80e2cf2309: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeZ;
            return ret;
        },
        __wbg_maxComputeWorkgroupStorageSize_cb6235497b8c4997: function(arg0) {
            const ret = arg0.maxComputeWorkgroupStorageSize;
            return ret;
        },
        __wbg_maxComputeWorkgroupsPerDimension_6bf550b5f21d57cf: function(arg0) {
            const ret = arg0.maxComputeWorkgroupsPerDimension;
            return ret;
        },
        __wbg_maxDynamicStorageBuffersPerPipelineLayout_c6ac20334e328b47: function(arg0) {
            const ret = arg0.maxDynamicStorageBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxDynamicUniformBuffersPerPipelineLayout_aa8f14a74b440f01: function(arg0) {
            const ret = arg0.maxDynamicUniformBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxSampledTexturesPerShaderStage_db7c4922cc60144a: function(arg0) {
            const ret = arg0.maxSampledTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxSamplersPerShaderStage_538705fe2263e710: function(arg0) {
            const ret = arg0.maxSamplersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageBufferBindingSize_32178c0f5f7f85cb: function(arg0) {
            const ret = arg0.maxStorageBufferBindingSize;
            return ret;
        },
        __wbg_maxStorageBuffersPerShaderStage_9f67e9eae0089f77: function(arg0) {
            const ret = arg0.maxStorageBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageTexturesPerShaderStage_57239664936031cf: function(arg0) {
            const ret = arg0.maxStorageTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxTextureArrayLayers_db5d4e486c78ae04: function(arg0) {
            const ret = arg0.maxTextureArrayLayers;
            return ret;
        },
        __wbg_maxTextureDimension1D_3475085ffacabbdc: function(arg0) {
            const ret = arg0.maxTextureDimension1D;
            return ret;
        },
        __wbg_maxTextureDimension2D_7c8d5ecf09eb8519: function(arg0) {
            const ret = arg0.maxTextureDimension2D;
            return ret;
        },
        __wbg_maxTextureDimension3D_8bd976677a0f91d4: function(arg0) {
            const ret = arg0.maxTextureDimension3D;
            return ret;
        },
        __wbg_maxUniformBufferBindingSize_95b1a54e7e4a0f0f: function(arg0) {
            const ret = arg0.maxUniformBufferBindingSize;
            return ret;
        },
        __wbg_maxUniformBuffersPerShaderStage_5f475d9a453af14d: function(arg0) {
            const ret = arg0.maxUniformBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxVertexAttributes_4c48ca2f5d32f860: function(arg0) {
            const ret = arg0.maxVertexAttributes;
            return ret;
        },
        __wbg_maxVertexBufferArrayStride_2233f6933ecc5a16: function(arg0) {
            const ret = arg0.maxVertexBufferArrayStride;
            return ret;
        },
        __wbg_maxVertexBuffers_c47e508cd7348554: function(arg0) {
            const ret = arg0.maxVertexBuffers;
            return ret;
        },
        __wbg_message_0762358e59db7ed6: function(arg0, arg1) {
            const ret = arg1.message;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_message_7957ab09f64c6822: function(arg0, arg1) {
            const ret = arg1.message;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_message_b163994503433c9e: function(arg0, arg1) {
            const ret = arg1.message;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_messages_da071582f72bc978: function(arg0) {
            const ret = arg0.messages;
            return ret;
        },
        __wbg_minStorageBufferOffsetAlignment_51b4801fac3a58de: function(arg0) {
            const ret = arg0.minStorageBufferOffsetAlignment;
            return ret;
        },
        __wbg_minUniformBufferOffsetAlignment_5d62a77924b2335f: function(arg0) {
            const ret = arg0.minUniformBufferOffsetAlignment;
            return ret;
        },
        __wbg_navigator_43be698ba96fc088: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_4478931f32ebca57: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_361308b2356cecd0: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_3eb36ae241fe6f44: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_b5d9e2fb389fef91: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen__convert__closures_____invoke__hc6248a957b6196c7(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = state0.b = 0;
            }
        },
        __wbg_new_from_slice_a3d2629dc1826784: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_no_args_1c7c842f08d00ebb: function(arg0, arg1) {
            const ret = new Function(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_with_byte_offset_and_length_aa261d9c9da49eb1: function(arg0, arg1, arg2) {
            const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_now_a3af9a2f4bbaa4d1: function() {
            const ret = Date.now();
            return ret;
        },
        __wbg_offset_336f14c993863b76: function(arg0) {
            const ret = arg0.offset;
            return ret;
        },
        __wbg_popErrorScope_af0b22f136a861d6: function(arg0) {
            const ret = arg0.popErrorScope();
            return ret;
        },
        __wbg_prototypesetcall_bdcdcc5842e4d77d: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_pushErrorScope_b52914ff10ba6ce3: function(arg0, arg1) {
            arg0.pushErrorScope(__wbindgen_enum_GpuErrorFilter[arg1]);
        },
        __wbg_push_8ffdcb2063340ba5: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_querySelectorAll_1283aae52043a951: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.querySelectorAll(getStringFromWasm0(arg1, arg2));
            return ret;
        }, arguments); },
        __wbg_queueMicrotask_0aa0a927f78f5d98: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_5bb536982f78a56f: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queue_bea4017efaaf9904: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_reason_43acd39cce242b50: function(arg0) {
            const ret = arg0.reason;
            return (__wbindgen_enum_GpuDeviceLostReason.indexOf(ret) + 1 || 3) - 1;
        },
        __wbg_requestAdapter_e6dcfac497cafa7a: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_03b802707d5a382c: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolveQuerySet_811661fb23f3b699: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.resolveQuerySet(arg1, arg2 >>> 0, arg3 >>> 0, arg4, arg5 >>> 0);
        },
        __wbg_resolve_002c4b7d9d8f6b64: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_setBindGroup_62a3045b0921e429: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        },
        __wbg_setBindGroup_6c0fd18e9a53a945: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setBindGroup_7f3b61f1f482133b: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setBindGroup_bf767a5aa46a33ce: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        },
        __wbg_setBindGroup_c4aaff14063226b4: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setBindGroup(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4), arg5, arg6 >>> 0);
        },
        __wbg_setBindGroup_f82e771dc1b69093: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setBlendConstant_016723821cfb3aa4: function(arg0, arg1) {
            arg0.setBlendConstant(arg1);
        },
        __wbg_setIndexBuffer_286a40afdff411b7: function(arg0, arg1, arg2, arg3) {
            arg0.setIndexBuffer(arg1, __wbindgen_enum_GpuIndexFormat[arg2], arg3);
        },
        __wbg_setIndexBuffer_7efd0b7a40c65fb9: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.setIndexBuffer(arg1, __wbindgen_enum_GpuIndexFormat[arg2], arg3, arg4);
        },
        __wbg_setIndexBuffer_e091a9673bb575e2: function(arg0, arg1, arg2, arg3) {
            arg0.setIndexBuffer(arg1, __wbindgen_enum_GpuIndexFormat[arg2], arg3);
        },
        __wbg_setIndexBuffer_f0759f00036f615f: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.setIndexBuffer(arg1, __wbindgen_enum_GpuIndexFormat[arg2], arg3, arg4);
        },
        __wbg_setPipeline_ba92070b8ee81cf9: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_setPipeline_c344f76bae58c4d6: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_setPipeline_d76451c50a121598: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_setScissorRect_0b6ee0852ef0b6b9: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.setScissorRect(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_setStencilReference_34fd3d59673a5a9d: function(arg0, arg1) {
            arg0.setStencilReference(arg1 >>> 0);
        },
        __wbg_setVertexBuffer_06a90dc78e1ad9c4: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.setVertexBuffer(arg1 >>> 0, arg2, arg3, arg4);
        },
        __wbg_setVertexBuffer_1540e9118b6c451d: function(arg0, arg1, arg2, arg3) {
            arg0.setVertexBuffer(arg1 >>> 0, arg2, arg3);
        },
        __wbg_setVertexBuffer_5166eedc06450701: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.setVertexBuffer(arg1 >>> 0, arg2, arg3, arg4);
        },
        __wbg_setVertexBuffer_8621784e5014065b: function(arg0, arg1, arg2, arg3) {
            arg0.setVertexBuffer(arg1 >>> 0, arg2, arg3);
        },
        __wbg_setViewport_731ad30abb13f744: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.setViewport(arg1, arg2, arg3, arg4, arg5, arg6);
        },
        __wbg_set_25cf9deff6bf0ea8: function(arg0, arg1, arg2) {
            arg0.set(arg1, arg2 >>> 0);
        },
        __wbg_set_6cb8631f80447a67: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_height_b386c0f603610637: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_height_f21f985387070100: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_onuncapturederror_19541466822d790b: function(arg0, arg1) {
            arg0.onuncapturederror = arg1;
        },
        __wbg_set_width_7f07715a20503914: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_set_width_d60bc4f2f20c56a4: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_size_661bddb3f9898121: function(arg0) {
            const ret = arg0.size;
            return ret;
        },
        __wbg_static_accessor_GLOBAL_12837167ad935116: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_e628e89ab3b1c95f: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_a621d3dfbb60d0ce: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_f8727f0cf888e0bd: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_submit_f635072bb3d05faa: function(arg0, arg1) {
            arg0.submit(arg1);
        },
        __wbg_then_0d9fe2c7b1857d32: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_b9e7b3b5f1a9e1b5: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_type_c0d5d83032e9858a: function(arg0) {
            const ret = arg0.type;
            return (__wbindgen_enum_GpuCompilationMessageType.indexOf(ret) + 1 || 4) - 1;
        },
        __wbg_unmap_8c2e8131b2aaa844: function(arg0) {
            arg0.unmap();
        },
        __wbg_usage_13caa02888040e9f: function(arg0) {
            const ret = arg0.usage;
            return ret;
        },
        __wbg_valueOf_3c28600026e653c4: function(arg0) {
            const ret = arg0.valueOf();
            return ret;
        },
        __wbg_warn_f7ae1b2e66ccb930: function(arg0) {
            console.warn(arg0);
        },
        __wbg_writeBuffer_5ca4981365eb5ac0: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.writeBuffer(arg1, arg2, arg3, arg4, arg5);
        },
        __wbg_writeTexture_246118eb2f5a1592: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.writeTexture(arg1, arg2, arg3, arg4);
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { dtor_idx: 149, function: Function { arguments: [NamedExternref("GPUUncapturedErrorEvent")], shim_idx: 150, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm.wasm_bindgen__closure__destroy__h06795fcff948ae6e, wasm_bindgen__convert__closures_____invoke__h4bdbd0b6193ff327);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { dtor_idx: 273, function: Function { arguments: [Externref], shim_idx: 274, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm.wasm_bindgen__closure__destroy__hafda76d1930f64ad, wasm_bindgen__convert__closures_____invoke__hbcd8af0807c9e587);
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
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
        "./phyz_wasm_bg.js": import0,
    };
}

function wasm_bindgen__convert__closures_____invoke__h4bdbd0b6193ff327(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__h4bdbd0b6193ff327(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__hbcd8af0807c9e587(arg0, arg1, arg2) {
    wasm.wasm_bindgen__convert__closures_____invoke__hbcd8af0807c9e587(arg0, arg1, arg2);
}

function wasm_bindgen__convert__closures_____invoke__hc6248a957b6196c7(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen__convert__closures_____invoke__hc6248a957b6196c7(arg0, arg1, arg2, arg3);
}


const __wbindgen_enum_GpuCompilationMessageType = ["error", "warning", "info"];


const __wbindgen_enum_GpuDeviceLostReason = ["unknown", "destroyed"];


const __wbindgen_enum_GpuErrorFilter = ["validation", "out-of-memory", "internal"];


const __wbindgen_enum_GpuIndexFormat = ["uint16", "uint32"];


const __wbindgen_enum_GpuTextureFormat = ["r8unorm", "r8snorm", "r8uint", "r8sint", "r16uint", "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint", "r32float", "rg16uint", "rg16sint", "rg16float", "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb", "rgb9e5ufloat", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint", "rg32float", "rgba16uint", "rgba16sint", "rgba16float", "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm", "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm", "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb", "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb", "etc2-rgba8unorm", "etc2-rgba8unorm-srgb", "eac-r11unorm", "eac-r11snorm", "eac-rg11unorm", "eac-rg11snorm", "astc-4x4-unorm", "astc-4x4-unorm-srgb", "astc-5x4-unorm", "astc-5x4-unorm-srgb", "astc-5x5-unorm", "astc-5x5-unorm-srgb", "astc-6x5-unorm", "astc-6x5-unorm-srgb", "astc-6x6-unorm", "astc-6x6-unorm-srgb", "astc-8x5-unorm", "astc-8x5-unorm-srgb", "astc-8x6-unorm", "astc-8x6-unorm-srgb", "astc-8x8-unorm", "astc-8x8-unorm-srgb", "astc-10x5-unorm", "astc-10x5-unorm-srgb", "astc-10x6-unorm", "astc-10x6-unorm-srgb", "astc-10x8-unorm", "astc-10x8-unorm-srgb", "astc-10x10-unorm", "astc-10x10-unorm-srgb", "astc-12x10-unorm", "astc-12x10-unorm-srgb", "astc-12x12-unorm", "astc-12x12-unorm-srgb"];
const QuantumSolverFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_quantumsolver_free(ptr >>> 0, 1));
const WasmCompileFusionSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcompilefusionsim_free(ptr >>> 0, 1));
const WasmCompileIrSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcompileirsim_free(ptr >>> 0, 1));
const WasmCompileWgslSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcompilewgslsim_free(ptr >>> 0, 1));
const WasmCradleSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcradlesim_free(ptr >>> 0, 1));
const WasmDiffGradientSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmdiffgradientsim_free(ptr >>> 0, 1));
const WasmDiffJacobianSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmdiffjacobiansim_free(ptr >>> 0, 1));
const WasmDiffSensitivitySimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmdiffsensitivitysim_free(ptr >>> 0, 1));
const WasmEmSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmemsim_free(ptr >>> 0, 1));
const WasmEnsembleSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmensemblesim_free(ptr >>> 0, 1));
const WasmGravitySandboxSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmgravitysandboxsim_free(ptr >>> 0, 1));
const WasmGravitySimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmgravitysim_free(ptr >>> 0, 1));
const WasmGripperSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmgrippersim_free(ptr >>> 0, 1));
const WasmGuardianSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmguardiansim_free(ptr >>> 0, 1));
const WasmHourglassSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmhourglasssim_free(ptr >>> 0, 1));
const WasmLbmSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlbmsim_free(ptr >>> 0, 1));
const WasmLorentzSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlorentzsim_free(ptr >>> 0, 1));
const WasmMdSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmdsim_free(ptr >>> 0, 1));
const WasmMjcfAntSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmjcfantsim_free(ptr >>> 0, 1));
const WasmMjcfCartpoleSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmjcfcartpolesim_free(ptr >>> 0, 1));
const WasmMjcfEditorSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmjcfeditorsim_free(ptr >>> 0, 1));
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
const WasmRagdollSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmragdollsim_free(ptr >>> 0, 1));
const WasmReal2SimAdamVsGdSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmreal2simadamvsgdsim_free(ptr >>> 0, 1));
const WasmReal2SimFitSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmreal2simfitsim_free(ptr >>> 0, 1));
const WasmReal2SimLandscapeSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmreal2simlandscapesim_free(ptr >>> 0, 1));
const WasmReggeActionSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmreggeactionsim_free(ptr >>> 0, 1));
const WasmReggeCurvatureSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmreggecurvaturesim_free(ptr >>> 0, 1));
const WasmReggeSymmetrySimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmreggesymmetrysim_free(ptr >>> 0, 1));
const WasmRubeGoldbergSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmrubegoldbergsim_free(ptr >>> 0, 1));
const WasmSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmsim_free(ptr >>> 0, 1));
const WasmWaveFieldSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmwavefieldsim_free(ptr >>> 0, 1));
const WasmWorldSimFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmworldsim_free(ptr >>> 0, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => state.dtor(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayJsValueFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    wasm.__externref_drop_slice(ptr, len);
    return result;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

let cachedInt32ArrayMemory0 = null;
function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, dtor, f) {
    const state = { a: arg0, b: arg1, cnt: 1, dtor };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            state.dtor(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
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

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
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
        module_or_path = new URL('phyz_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
