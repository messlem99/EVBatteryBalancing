import numpy as np 
import gymnasium as gym
from gymnasium import spaces
from scipy import interpolate
import pandas as pd

# Define the Battery Management MDP Environment for cell balancing.
class BatteryBalancingEnv(gym.Env):
    def __init__(self, driving_cycle_filepath="data.csv", noise_variance=0, include_noise=True,
                 history_length=5, smoothing_factor=0.1):
        super().__init__()
        # Battery and pack configuration parameters.
        self.cell_count = 10
        self.cell_nominal_capacity = 2.0  # [Ah]
        self.max_cell_voltage = 4.2       # [V]
        self.min_cell_voltage = 3.0       # [V]
        self.voltage_tolerance = 0.02     # [V] for optimal balancing
        self.soc_tolerance = 0.02         # [fraction] for optimal balancing
        self.simulation_timestep = 1      # [s]
        self.energy_balancing_rate = 1    # [Ah per hour] maximum energy balancing rate
        self.conversion_efficiency = 0.9  # System efficiency factor
        self.pack_voltage_nominal = 350   # [V]
        
        # Vehicle dynamics parameters.
        self.vehicle_mass = 1600          # [kg]
        self.wheel_radius = 0.3           # [m]
        self.drag_coefficient = 0.3
        self.frontal_area = 2.4           # [m^2]
        self.rolling_resistance = 0.008

        # SOC-to-OCV mapping via experimental Li-NMC data.
        self.soc_ocv_data = np.array([
            [0.00, 3.00], [0.05, 3.42], [0.10, 3.55], [0.20, 3.62],
            [0.30, 3.68], [0.40, 3.72], [0.50, 3.77], [0.60, 3.82],
            [0.70, 3.87], [0.80, 3.93], [0.90, 4.00], [0.95, 4.10],
            [1.00, 4.20]
        ])
        self.ocv_function = interpolate.interp1d(
            self.soc_ocv_data[:, 0],
            self.soc_ocv_data[:, 1],
            kind='linear',
            fill_value="extrapolate"
        )
        
        # Cell-level properties.
        self.cell_capacities = np.full(self.cell_count, self.cell_nominal_capacity)
        self.cell_internal_resistances = np.clip(np.random.normal(0.005, 0.005, size=self.cell_count),
                                                 0.001, 0.01)
        self.cell_charge_levels = None  # Dynamic charge levels for each cell.
        self.cell_voltage_levels = None
        self.load_current_profile = None  # Discharge current profile based on driving cycle.
        self.current_simulation_step = 0
        self.initial_cell_charges = None
        
        # Driving cycle data processing.
        self.driving_cycle_data = pd.read_csv(driving_cycle_filepath, skipinitialspace=True)
        self.driving_cycle_data.columns = self.driving_cycle_data.columns.str.strip()
        self.total_simulation_steps = len(self.driving_cycle_data)
        self._generate_driving_cycle()
        
        # Noise settings.
        self.noise_variance = noise_variance
        self.include_noise = include_noise
        
        # Observation history settings.
        self.history_length = history_length
        self.voltage_history = None
        self.soc_history = None
        
        # Action smoothing settings.
        self.previous_action = np.zeros(self.cell_count - 1)
        self.smoothing_factor = smoothing_factor
        
        # Define observation space (flattened history of voltages, SOCs, differences, and current load).
        obs_dimension = (self.history_length * self.cell_count * 2 + (self.cell_count - 1) * 4 + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dimension,), dtype=np.float32)
        # Define a discrete multi-action space for adjacent cell energy transfers.
        self.action_space = spaces.MultiDiscrete([7] * (self.cell_count - 1))
        
        # Early stopping thresholds.
        self.voltage_std_threshold = 0.5
        self.soc_std_threshold = 0.2
        # Define discrete energy transfer intensity levels.
        self.energy_transfer_levels = {0:0, 1: 0.3, 2: 0.6, 3: 1.0,4: 0.3, 5: 0.6, 6: 1.0}

    # Process and generate a driving cycle from input data.
    def _generate_driving_cycle(self):
        speed_mph = self.driving_cycle_data['Speed'].values
        time_array = self.driving_cycle_data['Test Time'].values
        speed_ms = speed_mph * 0.44704  # Convert mph to m/s
        acceleration = np.gradient(speed_ms, time_array)
        drag_power = 0.5 * 1.225 * self.drag_coefficient * self.frontal_area * (speed_ms ** 3)
        accel_power = self.vehicle_mass * acceleration * speed_ms
        rolling_power = self.rolling_resistance * self.vehicle_mass * 9.81 * speed_ms
        total_mechanical_power = drag_power + accel_power + rolling_power
        # Compute load current using nominal pack voltage and assumed system efficiency factor (0.9).
        discharge_current = total_mechanical_power / (self.pack_voltage_nominal * 0.9)
        self.load_current_profile = np.clip(discharge_current, 0, 150)

    # Reset the environment state for a new simulation episode.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        base_soc = 0.7
        soc_variation = np.linspace(0, 0.03, self.cell_count)
        self.cell_charge_levels = (base_soc + soc_variation) * self.cell_capacities
        self.cell_charge_levels = np.clip(self.cell_charge_levels, 0, self.cell_capacities)
        self.initial_cell_charges = self.cell_charge_levels.copy()
        
        # Optionally add Gaussian noise to the current profile.
        if self.include_noise:
            noise = np.random.normal(0, self.noise_variance, size=self.load_current_profile.shape)
            self.load_current_profile += noise
            self.load_current_profile = np.clip(self.load_current_profile, 0, 150)
        
        self.current_simulation_step = 0
        self.cell_voltage_levels = self._compute_cell_voltages()
        initial_voltages = self.cell_voltage_levels
        initial_socs = np.clip(self.cell_charge_levels / self.cell_capacities, 0.0, 1.0)
        self.voltage_history = np.tile(initial_voltages, (self.history_length, 1))
        self.soc_history = np.tile(initial_socs, (self.history_length, 1))
        self.previous_voltage_std = np.std(initial_voltages)
        self.previous_soc_std = np.std(initial_socs)
        return self._assemble_observation(), {}

    # Execute one simulation step with a given energy transfer action.
    def step(self, action):
        self._execute_energy_transfer(action)
        # Retrieve the current load current for this timestep.
        current_load = self.load_current_profile[self.current_simulation_step] if \
            self.current_simulation_step < len(self.load_current_profile) else 0.0
        delta_hours = self.simulation_timestep / 3600
        # Update cell charge levels due to load current consumption.
        self.cell_charge_levels += -current_load * delta_hours / self.cell_count
        self.cell_voltage_levels = self._compute_cell_voltages()
        state_of_charge = np.clip(self.cell_charge_levels / self.cell_capacities, 0.0, 1.0)
        adjacent_soc_differences = np.diff(state_of_charge)
        
        # Compute rewards based on SOC differences and applied actions.
        balancing_reward_components = []
        for act, soc_diff in zip(action, adjacent_soc_differences):
            abs_difference = np.abs(soc_diff)
            # Sample reward conditions based on deviation and transfer direction.
            if (abs_difference > 0.005) and (soc_diff > 0) and (act == 1):
                component_reward = 0.1
            elif (abs_difference > 0.005) and (soc_diff < 0) and (act == 2):
                component_reward = 0.1
            elif (abs_difference <= 0.005) and (act == 0):
                component_reward = 0.2
            else:
                component_reward = 0.0
            balancing_reward_components.append(component_reward)
        total_balancing_reward = sum(balancing_reward_components)
        
        adjacent_soc_rewards = []
        total_adjacent_diff = np.sum(np.abs(adjacent_soc_differences))
        for soc_diff in adjacent_soc_differences:
            abs_diff = np.abs(soc_diff)
            if abs_diff > 0.005:
                reward_value = - total_adjacent_diff * 10
            elif abs_diff < 0.001:
                reward_value = total_adjacent_diff * 100
            else:
                reward_value = 0.0
            adjacent_soc_rewards.append(reward_value)
        total_adjacent_soc_reward = sum(adjacent_soc_rewards)
        
        # Additional reward component based on overall SOC difference sum.
        total_abs_soc_diff = np.sum(np.abs(adjacent_soc_differences))
        if total_abs_soc_diff <= 0.03:
            soc_diff_reward = total_abs_soc_diff * 10
        elif total_abs_soc_diff <= 0.025:
            soc_diff_reward = total_abs_soc_diff * 20
        elif total_abs_soc_diff <= 0.01:
            soc_diff_reward = total_abs_soc_diff * 30
        elif total_abs_soc_diff <= 0.008:
            soc_diff_reward = total_abs_soc_diff * 100
        else:
            soc_diff_reward = -total_abs_soc_diff
        
        # Final reward is a sum of the three reward components.
        reward = total_balancing_reward + total_adjacent_soc_reward + soc_diff_reward
        
        # Evaluate current variability for potential early stopping.
        soc_std = np.std(state_of_charge)
        voltage_std = np.std(self.cell_voltage_levels)
        total_diff = np.sum(np.abs(adjacent_soc_differences))
        done = ((self.current_simulation_step >= self.total_simulation_steps) or 
                np.all(state_of_charge <= 0.01) or 
                (self.previous_voltage_std > self.voltage_std_threshold) or 
                (self.previous_soc_std >= self.soc_std_threshold) or 
                (total_diff > 0.2))
        reward /= 1  # Normalization factor if required.
        self.previous_voltage_std = voltage_std
        self.previous_soc_std = soc_std
        self.current_simulation_step += 1
        observation = self._assemble_observation()
        info = {
            "soc_std": soc_std,
            "voltage_std": voltage_std,
            "total_adjacent_soc_reward": total_adjacent_soc_reward,
            "soc_diff_abs_sum": np.sum(np.abs(adjacent_soc_differences)),
            "soc_differences": adjacent_soc_differences,
        }
        return observation, reward, done, False, info

    # Execute energy transfer between adjacent cells based on the action vector.
    def _execute_energy_transfer(self, action_vector):
        max_transfer = self.energy_balancing_rate * (self.simulation_timestep / 3600)
        discharge_requests = []
        charge_requests = []
        for idx in range(self.cell_count - 1):
            discrete_command = action_vector[idx]
            intensity = self.energy_transfer_levels[discrete_command]
            transfer_amount = intensity * max_transfer
            # Transfer from right cell to left cell.
            if discrete_command in [0, 1, 2, 3]:
                discharge_requests.append((idx + 1, transfer_amount, intensity))
                charge_requests.append((idx, transfer_amount, intensity))
            # Transfers energy from left cell to right cell.
            elif discrete_command in [4, 5, 6]:
                discharge_requests.append((idx, transfer_amount, intensity))
                charge_requests.append((idx + 1, transfer_amount, intensity))
        # Prioritize requests by transfer intensity.
        discharge_requests.sort(key=lambda x: x[2], reverse=True)
        charge_requests.sort(key=lambda x: x[2], reverse=True)
        total_discharged_energy = 0.0
        # Process discharging requests ensuring energy does not fall below zero.
        for cell_idx, amount, _ in discharge_requests:
            actual_transfer = min(amount, self.cell_charge_levels[cell_idx])
            self.cell_charge_levels[cell_idx] -= actual_transfer
            total_discharged_energy += actual_transfer
        # Apply efficiency factor and distribute energy to charging requests.
        available_energy = total_discharged_energy * self.conversion_efficiency
        for cell_idx, amount, _ in charge_requests:
            if available_energy <= 0:
                break
            actual_charge = min(amount, available_energy, self.cell_capacities[cell_idx] - self.cell_charge_levels[cell_idx])
            self.cell_charge_levels[cell_idx] += actual_charge
            available_energy -= actual_charge

    # Compute cell voltages using SOC-to-OCV mapping and internal resistance drop.
    def _compute_cell_voltages(self):
        state_of_charge = np.clip(self.cell_charge_levels / self.cell_capacities, 0.0, 1.0)
        ocv_values = self.ocv_function(state_of_charge)
        current_load = self.load_current_profile[self.current_simulation_step] if \
            self.current_simulation_step < len(self.load_current_profile) else 0.0
        voltage_drop = current_load * self.cell_internal_resistances / self.cell_count
        return ocv_values - voltage_drop

    # Assemble the observation vector using historical voltages, SOCs, differences and the current load.
    def _assemble_observation(self):
        index = min(self.current_simulation_step, self.total_simulation_steps - 1)
        current_load = self.load_current_profile[index]
        current_voltages = self._compute_cell_voltages()
        current_socs = np.clip(self.cell_charge_levels / self.cell_capacities, 0.0, 1.0)
        # Update historical records using a rolling window.
        self.voltage_history = np.roll(self.voltage_history, shift=-1, axis=0)
        self.voltage_history[-1, :] = current_voltages
        self.soc_history = np.roll(self.soc_history, shift=-1, axis=0)
        self.soc_history[-1, :] = current_socs
        
        # Compute instantaneous differences.
        voltage_differences = np.diff(current_voltages)
        soc_differences = np.diff(current_socs)
        voltage_rate = (current_voltages - self.voltage_history[-2, :]) if self.current_simulation_step > 0 else np.zeros(self.cell_count)
        soc_rate = (current_socs - self.soc_history[-2, :]) if self.current_simulation_step > 0 else np.zeros(self.cell_count)
        
        # Construct the observation vector.
        obs_vector = []
        obs_vector.extend(self.voltage_history.flatten())
        obs_vector.extend(self.soc_history.flatten())
        obs_vector.extend(voltage_differences)
        obs_vector.extend(soc_differences)
        obs_vector.extend(voltage_rate[:-1])
        obs_vector.extend(soc_rate[:-1])
        obs_vector.append(current_load)
        return np.array(obs_vector, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        pass