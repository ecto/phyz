"""Gymnasium environment for tau physics engine."""

import numpy as np
try:
    import gymnasium as gym
except ImportError:
    gym = None
import tau


class TauEnv:
    """Gymnasium-compatible environment for tau physics engine.

    Args:
        model_path: Path to MJCF model file
        dt: Timestep for simulation (overrides model dt if provided)
        solver: Solver type ('euler' or 'rk4')
    """

    def __init__(self, model_path: str, dt: float = None, solver: str = 'euler'):
        # Load model
        loader = tau.MjcfLoader(model_path)
        self.model = loader.build_model()
        self.state = self.model.default_state()

        # Create simulator
        if solver == 'rk4':
            self.sim = tau.Simulator.rk4()
        else:
            self.sim = tau.Simulator()

        # Define observation and action spaces
        if gym is not None:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.model.nq + self.model.nv,),
                dtype=np.float64,
            )
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.model.nv,),
                dtype=np.float64,
            )

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.state = self.model.default_state()
        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        """Take one step in the environment.

        Args:
            action: Control input (must match action_space shape)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Set control
        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] != self.model.nv:
            raise ValueError(f"Action dimension {action.shape[0]} != nv {self.model.nv}")
        self.state.ctrl = action

        # Step simulation
        self.sim.step(self.model, self.state)

        # Compute reward (can be overridden in subclasses)
        reward = self._compute_reward()

        # Check termination
        terminated = self._is_terminated()
        truncated = False

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Get observation from current state."""
        return np.concatenate([
            np.array(self.state.q),
            np.array(self.state.v),
        ])

    def _compute_reward(self):
        """Compute reward (to be overridden in subclasses)."""
        return 0.0

    def _is_terminated(self):
        """Check if episode is terminated (to be overridden in subclasses)."""
        return False

    def render(self, mode='human'):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass


class AntEnv(TauEnv):
    """Ant locomotion environment.

    Reward is forward progress minus control cost.
    """

    def __init__(self, model_path: str = "ant.xml"):
        super().__init__(model_path)
        self.prev_x = 0.0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        # Assume first DOF is x position
        self.prev_x = np.array(self.state.q)[0] if self.model.nq > 0 else 0.0
        return obs, info

    def _compute_reward(self):
        """Reward forward progress minus control cost."""
        q = np.array(self.state.q)
        v = np.array(self.state.v)
        ctrl = np.array(self.state.ctrl)

        # Forward progress (assume first DOF is x)
        x = q[0] if len(q) > 0 else 0.0
        forward_reward = (x - self.prev_x) / self.model.dt
        self.prev_x = x

        # Control cost
        ctrl_cost = 0.5 * np.sum(ctrl ** 2)

        # Survival bonus
        survive_reward = 1.0

        return forward_reward + survive_reward - 0.01 * ctrl_cost

    def _is_terminated(self):
        """Episode ends if ant falls over."""
        q = np.array(self.state.q)
        # Assume second DOF is z height
        if len(q) > 1:
            z = q[1]
            return z < 0.2 or z > 1.0
        return False
