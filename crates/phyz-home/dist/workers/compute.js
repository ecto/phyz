// Web Worker shim: loads phyz_wasm and bridges postMessage.
// SRI hash ensures WASM integrity — browser refuses tampered modules.

let hasWebGPU = false;
let hasGpuMethod = false;

async function init() {
  const { default: init_wasm, QuantumSolver } = await import('/wasm/phyz_wasm.js');
  await init_wasm({ module_or_path: '/wasm/phyz_wasm_bg.wasm' });

  // Detect WebGPU support — wrapped defensively because Safari iOS has
  // experimental/partial WebGPU that can crash on probe.
  try {
    hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
    if (hasWebGPU) {
      // Verify we can actually get an adapter (Safari may throw here)
      const adapter = await navigator.gpu.requestAdapter();
      hasWebGPU = !!adapter;
    }
  } catch (_) {
    hasWebGPU = false;
  }

  // Probe the WASM module for GPU method — only if WebGPU is confirmed working
  if (hasWebGPU) {
    try {
      const probe = new QuantumSolver('s4_level0');
      hasGpuMethod = typeof probe.solve_all_partitions_gpu === 'function';
      probe.free();
    } catch (_) {
      hasGpuMethod = false;
    }
  }

  if (hasWebGPU && hasGpuMethod) {
    console.log('[compute] WebGPU + GPU solver available');
  } else {
    console.log('[compute] CPU solver');
  }

  return QuantumSolver;
}

let QuantumSolverClass = null;
const solverCache = {};

function get_solver(params) {
  const tri = `s4_level${params.level}`;
  if (!solverCache[tri]) {
    solverCache[tri] = new QuantumSolverClass(tri);
  }
  return solverCache[tri];
}

function solve_one_cpu(solver, params) {
  const { coupling_g2, geometry_seed, perturbation } = params;
  if (perturbation.type === 'base') {
    return JSON.parse(solver.solve_all_partitions(
      coupling_g2, BigInt(geometry_seed), "base", -1, 0.0, 0.0
    ));
  } else {
    return JSON.parse(solver.solve_all_partitions(
      coupling_g2, BigInt(geometry_seed), "edge",
      perturbation.index, perturbation.direction, 1e-4
    ));
  }
}

async function solve_one_gpu(solver, params) {
  const { coupling_g2, geometry_seed, perturbation } = params;
  if (perturbation.type === 'base') {
    return JSON.parse(await solver.solve_all_partitions_gpu(
      coupling_g2, BigInt(geometry_seed), "base", -1, 0.0, 0.0
    ));
  } else {
    return JSON.parse(await solver.solve_all_partitions_gpu(
      coupling_g2, BigInt(geometry_seed), "edge",
      perturbation.index, perturbation.direction, 1e-4
    ));
  }
}

async function solve_one(params) {
  const solver = get_solver(params);
  if (hasWebGPU && hasGpuMethod) {
    try {
      return await solve_one_gpu(solver, params);
    } catch (err) {
      console.warn('[compute] GPU solve failed, falling back to CPU:', err);
    }
  }
  return solve_one_cpu(solver, params);
}

self.onmessage = async function(e) {
  try {
    const msg = JSON.parse(e.data);

    if (!QuantumSolverClass) {
      QuantumSolverClass = await init();
    }

    if (msg.type === 'solve') {
      const result = await solve_one(msg.params);
      self.postMessage(JSON.stringify({
        type: 'result',
        id: msg.id,
        result,
      }));
    } else if (msg.type === 'solve_batch') {
      const total = msg.items.length;
      for (let i = 0; i < total; i++) {
        const item = msg.items[i];
        try {
          const result = await solve_one(item.params);
          self.postMessage(JSON.stringify({
            type: 'result',
            id: item.id,
            result,
            progress: { done: i + 1, total },
          }));
        } catch (err) {
          self.postMessage(JSON.stringify({
            type: 'result',
            id: item.id,
            error: err.toString(),
            progress: { done: i + 1, total },
          }));
        }
      }
    }
  } catch (err) {
    self.postMessage(JSON.stringify({
      type: 'error',
      id: (typeof msg !== 'undefined' && msg.id) || '',
      error: err.toString(),
    }));
  }
};
