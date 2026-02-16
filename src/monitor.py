import time
import threading
import psutil
import pynvml
from statistics import mean

#__init__: It stores the data and initialize the nvidia drivers. 
#_monitor_loop: Internal loop to poll hardware stats.
#_calculate_aggregates: Compiles raw lists into meaningful peaks and averages.


class ResourceMonitor:
    def __init__(self, sample_interval=0.1, gpu_index=0):
        self.sample_interval = sample_interval
        self.gpu_index = gpu_index
        self.running = False
        self.thread = None
        
        self.stats = {
            "timestamps": [],
            "cpu_ram_usage_mb": [],
            "gpu_vram_usage_mb": [],
            "gpu_power_watts": []
        }
        
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.has_gpu = True
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
            print(f"[Monitor] GPU Detected: {gpu_name}")
        except pynvml.NVMLError as e:
            print(f"[Monitor] Warning: NVIDIA GPU not detected or driver error. {e}")
            self.has_gpu = False

    def _monitor_loop(self):
        while self.running:
            current_time = time.time()
            
            ram = psutil.virtual_memory()
            ram_mb = ram.used / (1024 * 1024)
            
            vram_mb = 0
            power_w = 0
            
            if self.has_gpu:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    vram_mb = mem_info.used / (1024 * 1024)
                    
                    power_mW = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    power_w = power_mW / 1000.0
                except pynvml.NVMLError:
                    pass 
            
            self.stats["timestamps"].append(current_time)
            self.stats["cpu_ram_usage_mb"].append(ram_mb)
            self.stats["gpu_vram_usage_mb"].append(vram_mb)
            self.stats["gpu_power_watts"].append(power_w)
            
            time.sleep(self.sample_interval)

    def start(self):
        self.running = True
        self.stats = {k: [] for k in self.stats} 
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        
        return self._calculate_aggregates()

    def _calculate_aggregates(self):
        if not self.stats["timestamps"]:
            return {}

        return {
            "peak_ram_mb": max(self.stats["cpu_ram_usage_mb"]),
            "avg_ram_mb": mean(self.stats["cpu_ram_usage_mb"]),
            "peak_vram_mb": max(self.stats["gpu_vram_usage_mb"]),
            "avg_vram_mb": mean(self.stats["gpu_vram_usage_mb"]),
            "avg_power_watts": mean(self.stats["gpu_power_watts"]),
            "peak_power_watts": max(self.stats["gpu_power_watts"])
        }

    def cleanup(self):
        if self.has_gpu:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


if __name__ == "__main__":
    print("Testing Monitor for 3 seconds...")
    mon = ResourceMonitor()
    mon.start()

    time.sleep(3)
    
    results = mon.stop()
    print("\n--- Test Results ---")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")
    mon.cleanup()