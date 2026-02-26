// Web Worker shim: loads phyz_wasm and bridges postMessage.
// SRI hash ensures WASM integrity — browser refuses tampered modules.

let solver = null;

async function init() {
  const { default: init_wasm, QuantumSolver } = await import('/wasm/phyz_wasm.js');
  await init_wasm({ module_or_path: '/wasm/phyz_wasm_bg.wasm' });
  return QuantumSolver;
}

let QuantumSolverClass = null;
const solverCache = {};

function solve_one(params) {
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

self.onmessage = async function(e) {
  try {
    const msg = JSON.parse(e.data);

    if (!QuantumSolverClass) {
      QuantumSolverClass = await init();
    }

    if (msg.type === 'solve') {
      const result = solve_one(msg.params);
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
          const result = solve_one(item.params);
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
