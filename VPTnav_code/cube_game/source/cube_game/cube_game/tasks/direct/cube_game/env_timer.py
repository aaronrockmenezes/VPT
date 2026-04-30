import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class EnvTimer:
    num_envs: int
    slot_to_env_id: Dict[int, int]
    task_keys: List[str]
    verbose: bool = True
    
    # Internal storage
    _data: Dict[int, Dict[str, Any]] = field(init=False)
    _start_times: Dict[str, float] = field(default_factory=dict)
    _global_start_time: float = field(init=False)

    def __post_init__(self):
        self._global_start_time = time.time()
        
        # Initialize the master dictionary structure
        self._data = {
            self.slot_to_env_id[i]: {
                "attempts": 0,
                "completion_status": "incomplete",
                "total_time": 0,
                "final_setup_time": 0,
                "last_attempt_setup_time": 0,
                # Initialize empty sub-dicts for all tracked tasks
                **{key: {} for key in self.task_keys}
            }
            for i in range(self.num_envs)
        }

    def start_timer(self, key: str) -> None:
        """Start the clock for a specific task section."""
        self._start_times[key] = time.time()

    def stop_timer(self, key: str, attempt: int, retry_mask: torch.Tensor) -> None:
        """
        Stop clock, calculate duration, and log 'F' or Time based on mask.
        """
        end_time = time.time()
        start_time = self._start_times.get(key)
        
        if start_time is None:
            print(f"⚠️ Timer Warning: Key '{key}' was stopped without starting.")
            return

        duration = end_time - start_time
        
        for idx in range(self.num_envs):
            slot_id = self.slot_to_env_id[idx]
            env_data = self._data[slot_id]
            
            # Skip if environment is already fully complete from previous attempts
            # Note: We check if it WAS complete before this step started
            if env_data["completion_status"] == "complete":
                # Check if we should log a success time for the final pass
                # (Optional: logic here depends if you want to log the final success step)
                continue

            needs_retry = retry_mask[idx].item()
            
            if needs_retry:
                env_data[key][attempt] = "F"
            else:
                env_data[key][attempt] = np.round(duration, 3)

    def update_status(self, attempt: int, retry_mask: torch.Tensor) -> None:
        """Updates attempt counters and marks environments as complete."""
        current_time = time.time()
        
        for idx in range(self.num_envs):
            slot_id = self.slot_to_env_id[idx]
            env_data = self._data[slot_id]
            
            if env_data["completion_status"] == "complete":
                continue

            needs_retry = retry_mask[idx].item()

            if needs_retry:
                env_data["attempts"] = attempt
                env_data["completion_status"] = "incomplete"
            else:
                # Mark as complete
                env_data["completion_status"] = "complete"
                
                # 1. Calculate total time from the very beginning of reset
                if env_data["total_time"] == 0:
                    total_duration = current_time - self._global_start_time
                    env_data["total_time"] = np.round(total_duration, 3)
                    
                # 2. Sum up the times for just this successful attempt
                self._finalize_env_stats(slot_id)

    def _finalize_env_stats(self, slot_id: int):
        """Helper to sum up the times for the successful attempt."""
        env_data = self._data[slot_id]
        total_setup = 0.0
        
        for key in self.task_keys:
            # Get the last logged value
            if key in env_data and isinstance(env_data[key], dict):
                values = list(env_data[key].values())
                if values:
                    last_val = values[-1]
                    if isinstance(last_val, (int, float)):
                        total_setup += last_val
        
        env_data["final_setup_time"] = np.round(total_setup, 3)
        env_data["last_attempt_setup_time"] = np.round(total_setup, 3)

    def _round_recursive(self, obj):
        """Helper to recursively round floats in nested dictionaries/lists."""
        if isinstance(obj, float):
            return np.round(obj, 3)
        elif isinstance(obj, dict):
            return {k: self._round_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._round_recursive(v) for v in obj]
        return obj

    def print_summary(self, attempt: int) -> None:
        """Prints the full nested dictionary state."""
        if not self.verbose: return
        
        # Round values for clean printing (without modifying actual data if you prefer, 
        # or modify in place. Here we modify in place for consistency).
        for slot_id in self._data:
            self._data[slot_id] = self._round_recursive(self._data[slot_id])

        print(f"\n--- Spawn Attempt {attempt} ---")
        print(f"-" * 50)
        print(f"Env dict below:")
        
        # Sort keys for consistent output
        sorted_ids = sorted(self._data.keys())
        
        for slot_id in sorted_ids:
            value = self._data[slot_id]
            print(f"Env {slot_id}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
                    
        print(f"-" * 50)
        elapsed = time.time() - self._global_start_time
        print(f"Total reset time for spawn attempts = {attempt}: {elapsed:.3f} seconds")
        print("\n")