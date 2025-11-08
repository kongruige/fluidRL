import numpy as np

class FluidEnv:
    """
    A simple 2D continuous-state environment.
    
    State: [x_position, y_position]
    Actions: 5 discrete actions (push up/down/left/right, or do nothing)
    Reward: -1 per step, +100 for reaching the goal.
    """
    def __init__(self):
        # --- World Parameters ---
        # The agent's state is its [x, y] position.
        # The world is a box from (-1, -1) to (1, 1)
        self.world_bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        
        # --- Goal ---
        self.goal_pos = np.array([0.8, 0.8])
        self.goal_radius = 0.1
        
        # --- Physics Parameters ---
        self.dt = 0.05  # Time step for physics simulation
        self.max_steps = 200 # Max steps per episode
        
        # --- Action Space (for DQN, Lec 7) ---
        # We need discrete actions. We'll map integers (0-4) to force vectors.
        self.action_space_size = 5
        action_force_magnitude = 1.0
        self.action_map = {
            0: np.array([0, 0]),           # Do nothing
            1: np.array([0, action_force_magnitude]),  # Push Up
            2: np.array([0, -action_force_magnitude]), # Push Down
            3: np.array([-action_force_magnitude, 0]), # Push Left
            4: np.array([action_force_magnitude, 0]),  # Push Right
        }
        
        # --- Internal State ---
        self.state = None
        self.current_step = 0

    def _get_flow_force(self, state):
        """
        Calculates the background "vortex" force at a given state.
        This is the "fluid" part of the environment.
        Force = (-y, x)
        """
        x, y = state
        # We scale the force to make it more/less challenging
        flow_force = np.array([-y, x]) * 2.0 
        return flow_force

    def reset(self):
        """
        Resets the environment to a new episode.
        Returns:
            state (np.array): The initial state [x, y]
        """
        # Start at a fixed position, e.g., (0, 0)
        self.state = np.array([0.0, 0.0])
        self.current_step = 0
        return self.state

    def step(self, action_index):
        """
        Takes one step in the environment.
        
        Args:
            action_index (int): The discrete action (0-4) to take.
            
        Returns:
            (next_state, reward, done)
        """
        if self.state is None:
            raise Exception("Must call reset() before step()")
            
        # 1. Get forces
        action_force = self.action_map[action_index]
        flow_force = self._get_flow_force(self.state)
        
        # 2. Update physics (Simple Euler integration)
        # new_velocity = action_force + flow_force (we assume mass=1)
        # new_state = old_state + new_velocity * dt
        new_state = self.state + (action_force + flow_force) * self.dt
        
        # 3. Clip state to stay within world bounds
        new_state = np.clip(
            new_state,
            self.world_bounds[:, 0], # min_x, min_y
            self.world_bounds[:, 1]  # max_x, max_y
        )
        
        self.state = new_state
        self.current_step += 1
        
        # 4. Calculate reward and done flag
        dist_to_goal = np.linalg.norm(self.state - self.goal_pos)
        
        if dist_to_goal <= self.goal_radius:
            # Reached goal
            reward = 100.0
            done = True
        elif self.current_step >= self.max_steps:
            # Ran out of time
            reward = -10.0 # Extra penalty for failure
            done = True
        else:
            # Standard step cost
            reward = -1.0 
            done = False
            
        return self.state, reward, done

# --- Main execution block to test the environment ---
if __name__ == "__main__":
    env = FluidEnv()
    state = env.reset()
    print(f"Initial State: {state}")
    
    # Test a few random steps
    for _ in range(5):
        # Pick a random action (0, 1, 2, 3, or 4)
        action = np.random.randint(0, env.action_space_size) 
        next_state, reward, done = env.step(action)
        
        print(f"Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
        
        if done:
            print("Episode finished. Resetting...")
            state = env.reset()