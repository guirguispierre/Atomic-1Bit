import psutil
import time
import os
import torch
import sys

class ThermalMonitor:
    def __init__(self, high_threshold=80.0, resume_threshold=70.0, check_interval_steps=50):
        """
        Monitors system temperature to prevent overheating.
        
        Args:
            high_threshold: Pause training if ANY sensor exceeds this (°C).
            resume_threshold: Resume training only when ALL sensors are below this (°C).
            check_interval_steps: How often to check (in training steps).
        """
        self.high = high_threshold
        self.resume = resume_threshold
        self.interval = check_interval_steps
        self.enabled = True
        
        # Initial check for sensor availability
        if not hasattr(psutil, "sensors_temperatures"):
            print("ThermalMonitor: psutil.sensors_temperatures not available. Monitoring disabled.")
            self.enabled = False
            return
            
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                print("ThermalMonitor: No temperature sensors found (common on macOS/WSL). Monitoring disabled.")
                self.enabled = False
        except Exception as e:
            print(f"ThermalMonitor: Error initializing sensors ({e}). Monitoring disabled.")
            self.enabled = False
            
        if self.enabled:
            print(f"ThermalMonitor: Active (Pause > {self.high}°C, Resume < {self.resume}°C)")

    def _get_max_temp(self):
        """Returns the maximum temperature reported by any sensor."""
        try:
            temps = psutil.sensors_temperatures()
            max_t = 0.0
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current > max_t:
                        max_t = entry.current
            return max_t
        except:
            return 0.0

    def check_and_pause(self, step, model=None, optimizer=None, save_path=None):
        """
        Checks temperature. If too high, pauses execution until cool.
        
        Args:
            step: Current training step (used to respect interval).
            model: Optional model to save checkpoint before pausing.
            optimizer: Optional optimizer to save state.
            save_path: Path to save the safety checkpoint.
        """
        if not self.enabled:
            return
            
        # Only check periodically to save overhead
        if step % self.interval != 0:
            return

        current_temp = self._get_max_temp()
        
        # If overheating
        if current_temp >= self.high:
            print(f"\n[THERMAL CRITICAL] System Temp: {current_temp}°C (Threshold: {self.high}°C)")
            print("Training PAUSED to allow cooldown.")
            
            # Save Safety Checkpoint
            if model and save_path:
                print(f"Saving safety checkpoint to {save_path}...")
                try:
                    payload = {"step": step, "model_state_dict": model.state_dict()}
                    if optimizer:
                        payload["optimizer_state_dict"] = optimizer.state_dict()
                    torch.save(payload, save_path)
                    print("Safety checkpoint saved.")
                except Exception as e:
                    print(f"Failed to save safety checkpoint: {e}")

            # Cooldown Loop
            start_pause = time.time()
            while True:
                time.sleep(10) # Sleep 10s
                t = self._get_max_temp()
                sys.stdout.write(f"\r[Cooling] Current: {t:.1f}°C / Resume Target: {self.resume}°C   ")
                sys.stdout.flush()
                
                if t < self.resume:
                    break
            
            duration = (time.time() - start_pause) / 60.0
            print(f"\nResuming training. (Paused for {duration:.1f} minutes)")
