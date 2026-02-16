import re
from dataclasses import dataclass

#parse_log: Scans the llama.cpp stdout/stderr for performance data.

@dataclass
class BenchmarkMetrics:
    prompt_tokens_per_sec: float = 0.0
    generation_tokens_per_sec: float = 0.0
    total_duration_ms: float = 0.0

def parse_log(log_text: str) -> BenchmarkMetrics:
    metrics = BenchmarkMetrics()

    gen_short_match = re.search(r"Generation:\s*([\d\.]+)\s*t/s", log_text)
    if gen_short_match:
        metrics.generation_tokens_per_sec = float(gen_short_match.group(1))

    prompt_short_match = re.search(r"Prompt:\s*([\d\.]+)\s*t/s", log_text)
    if prompt_short_match:
        metrics.prompt_tokens_per_sec = float(prompt_short_match.group(1))
    
    if metrics.generation_tokens_per_sec == 0.0:
        g_match = re.search(
            r"eval time.*,\s*([\d\.]+)\s*tokens per second", 
            log_text, re.DOTALL
        )
        if g_match:
            metrics.generation_tokens_per_sec = float(g_match.group(1))

    if metrics.prompt_tokens_per_sec == 0.0:
        p_match = re.search(
            r"prompt eval time.*,\s*([\d\.]+)\s*tokens per second", 
            log_text, re.DOTALL
        )
        if p_match:
            metrics.prompt_tokens_per_sec = float(p_match.group(1))

    return metrics