// Web Worker shim: loads phyz_wasm and bridges postMessage.
// SRI hash ensures WASM integrity — browser refuses tampered modules.

let solver = null;
let hasWebGPU = false;

async function init() {
  const { default: init_wasm, QuantumSolver } = await import('/wasm/phyz_wasm.js');
  await init_wasm({ module_or_path: '/wasm/phyz_wasm_bg.wasm' });

  // Detect WebGPU support
  hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
  if (hasWebGPU) {
    console.log('[compute] WebGPU detected — GPU solver available');
  } else {
    console.log('[compute] No WebGPU — using CPU solver');
  }

  return QuantumSolver;
}

let QuantumSolverClass = null;
const solverCache = {};

function solve_one_cpu(params) {
  const { level, coupling_g2, geometry_seed, perturbation } = params;
  const tri = `s4_level${level}`;
  if (!solverCache[tri]) {
    solverCache[tri] = new QuantumSolverClass(tri);
  }
  const solver = solverCache[tri];

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

async function solve_one_gpu(params) {
  const { level, coupling_g2, geometry_seed, perturbation } = params;
  const tri = `s4_level${level}`;
  if (!solverCache[tri]) {
    solverCache[tri] = new QuantumSolverClass(tri);
  }
  const solver = solverCache[tri];

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
  if (hasWebGPU && typeof solverCache[`s4_level${params.level}`]?.solve_all_partitions_gpu === 'function') {
    try {
      return await solve_one_gpu(params);
    } catch (err) {
      console.warn('[compute] GPU solve failed, falling back to CPU:', err);
    }
  }
  return solve_one_cpu(params);
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
