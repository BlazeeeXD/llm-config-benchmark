import subprocess
import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.monitor import ResourceMonitor

#BenchmarkConfig: Holds the specific parameters for ONE test run.
#run_test: Executes a single benchmark run with the given configuration.

@dataclass
class BenchmarkConfig:
    model_path: str
    cli_path: str
    n_gpu_layers: int
    threads: int
    ctx_size: int
    n_predict: int = 20       
    prompt: str = "Write a short story about a space engineer fixing a broken thruster."
    batch_size: int = 512

@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    success: bool
    monitor_stats: Dict[str, float]
    raw_output: str
    error_log: str = ""
    duration_seconds: float = 0.0

class BenchmarkRunner:
    def __init__(self, gpu_index=0):
        self.monitor = ResourceMonitor(gpu_index=gpu_index)

    def run_test(self, config: BenchmarkConfig) -> BenchmarkResult:

        if not os.path.exists(config.cli_path):
            return BenchmarkResult(config, False, {}, "", f"CLI not found: {config.cli_path}")
        if not os.path.exists(config.model_path):
            return BenchmarkResult(config, False, {}, "", f"Model not found: {config.model_path}")

        cmd = [
            config.cli_path,
            "-m", config.model_path,
            "-n", str(config.n_predict),
            "--threads", str(config.threads),
            "-ngl", str(config.n_gpu_layers),
            "-c", str(config.ctx_size),
            "-b", str(config.batch_size),
            "--temp", "0",
            "-no-cnv",  
            "-p", config.prompt
        ]

        print(f"--> Running: ngl={config.n_gpu_layers} threads={config.threads} ctx={config.ctx_size}")

        self.monitor.start()
        start_time = time.time()
        
        success = False
        error_msg = ""
        stdout_data = ""
        stderr_data = ""

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,  
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            stdout_data, stderr_data = process.communicate(input="/exit\n", timeout=120)

            if process.returncode == 0:
                success = True
            else:
                success = False
                error_msg = f"Process exited with code {process.returncode}\n{stderr_data}"

        except subprocess.TimeoutExpired:
            process.kill()
            success = False
            error_msg = "Benchmark Timed Out (Process Hung)"
        except Exception as e:
            process.kill()
            success = False
            error_msg = str(e)
            print(f"    [!] Unexpected Error: {e}")

        duration = time.time() - start_time
        hw_stats = self.monitor.stop()
        
        full_log = f"--- STDOUT ---\n{stdout_data}\n--- STDERR ---\n{stderr_data}"

        return BenchmarkResult(
            config=config,
            success=success,
            monitor_stats=hw_stats,
            raw_output=full_log,
            error_log=error_msg,
            duration_seconds=duration
        )

""""
# --- Quick Test Block ---
# just a simple test nothing more
if __name__ == "__main__":
    TEST_CLI = r"B:\Desktop\Blaze\Llama CPP\llama.cpp\build\bin\Release\llama-cli.exe"
    TEST_MODEL = r"B:\Desktop\Blaze\do not touch\Ml\GITHUB\DeepSeek-R1-Distill-Qwen-1.5B-BF16.gguf"
    

    if "path/to" in TEST_CLI:
        print("Please edit the TEST_CLI and TEST_MODEL paths in runner.py main block to run the test.")
    else:
        conf = BenchmarkConfig(
            model_path=TEST_MODEL,
            cli_path=TEST_CLI,
            n_gpu_layers=10,
            threads=4,
            ctx_size=2048,
            n_predict=30 # Generate 30 tokens then stop
        )
        
        runner = BenchmarkRunner()
        res = runner.run_test(conf)
        
        if res.success:
            print("\nSUCCESS!")
            print(f"Duration: {res.duration_seconds:.2f}s")
            # Parse crude speed from output just to check
            if "tokens per second" in res.raw_output or "t/s" in res.raw_output:
                print("Speed stats found in logs.")
            else:
                print("Warning: Could not find speed stats in logs (might be in stderr).")
        else:
            print("\nFAILED")
            print(res.error_log)
"""