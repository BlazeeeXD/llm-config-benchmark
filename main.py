import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime

from src.runner import BenchmarkRunner, BenchmarkConfig
from src.parser import parse_log


CLI_PATH = r"B:\Desktop\Blaze\Llama CPP\llama.cpp\build\bin\Release\llama-cli.exe"
MODEL_PATH = r"B:\Desktop\Blaze\do not touch\Ml\GITHUB\DeepSeek-R1-Distill-Qwen-1.5B-BF16.gguf"


N_GPU_LAYERS_TO_TEST = [0, 10, 20, 28, 35] 
THREADS_TO_TEST = [4, 8, 12]               
CONTEXT_SIZE = 2048                        
GENERATE_TOKENS = 100                      

OUTPUT_FILE = "benchmark_results.csv"


def main():
    print("--- Starting LLM Config Benchmark ---")
    print(f"Model: {os.path.basename(MODEL_PATH)}")
    
    runner = BenchmarkRunner(gpu_index=0)
    
    results_data = []

    total_runs = len(N_GPU_LAYERS_TO_TEST) * len(THREADS_TO_TEST)
    pbar = tqdm(total=total_runs, desc="Benchmarking")

    for ngl in N_GPU_LAYERS_TO_TEST:
        for threads in THREADS_TO_TEST:
            
            conf = BenchmarkConfig(
                model_path=MODEL_PATH,
                cli_path=CLI_PATH,
                n_gpu_layers=ngl,
                threads=threads,
                ctx_size=CONTEXT_SIZE,
                n_predict=GENERATE_TOKENS,
                prompt="Write a python script to sort a list using quicksort." 
            )

            result = runner.run_test(conf)

            parsed = parse_log(result.raw_output)

            row = {
                "n_gpu_layers": ngl,
                "threads": threads,
                "ctx_size": CONTEXT_SIZE,
                "success": result.success,
                "tokens_per_second": parsed.generation_tokens_per_sec,
                "prompt_tokens_per_second": parsed.prompt_tokens_per_sec,
                "peak_vram_mb": result.monitor_stats.get("peak_vram_mb", 0),
                "avg_power_watts": result.monitor_stats.get("avg_power_watts", 0),
                "peak_ram_mb": result.monitor_stats.get("peak_ram_mb", 0),
                "duration_s": result.duration_seconds
            }
            results_data.append(row)

            df = pd.DataFrame(results_data)
            df.to_csv(OUTPUT_FILE, index=False)
            
            pbar.update(1)

    pbar.close()
    print(f"\nBenchmark Complete! Results saved to {OUTPUT_FILE}")

    if not df.empty:
        best_run = df.loc[df['tokens_per_second'].idxmax()]
        print("\n--- WINNER CONFIGURATION ---")
        print(f"Speed:   {best_run['tokens_per_second']} t/s")
        print(f"Config:  Layers={best_run['n_gpu_layers']}, Threads={best_run['threads']}")
        print(f"VRAM:    {best_run['peak_vram_mb']:.2f} MB")

if __name__ == "__main__":
    main()