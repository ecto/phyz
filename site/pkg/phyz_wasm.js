/* @ts-self-types="./phyz_wasm.d.ts" */

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
        const ret = wasm.wasmcradlesim_time(this.__wbg_ptr);
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
        const ret = wasm.wasmcradlesim_time(this.__wbg_ptr);
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
        const ret = wasm.wasmcradlesim_time(this.__wbg_ptr);
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
        const ret = wasm.wasmhourglasssim_time(this.__wbg_ptr);
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
        const ret = wasm.wasmcradlesim_time(this.__wbg_ptr);
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
        const ret = wasm.wasmcradlesim_time(this.__wbg_ptr);
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
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
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
