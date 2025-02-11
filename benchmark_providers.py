import os
import sys
import asyncio
import httpx
import json
import time
from statistics import mean, stdev
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BenchmarkMetrics:
    ttft: float  # Time to first token
    total_tokens: int
    total_time: float
    token_times: List[float]
    output: str
    tpot: Optional[float] = None  # Time per token
    tpot_stdev: Optional[float] = None
    error: Optional[str] = None
    char_count: Optional[int] = None
    chars_per_token: Optional[float] = None
    success: bool = True

async def try_provider_http_stream(api_url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> BenchmarkMetrics:
    """Stream version that measures TTFT and TPOT with comprehensive error handling"""
    metrics = BenchmarkMetrics(
        ttft=0,
        total_tokens=0,
        total_time=0,
        token_times=[],
        output=""
    )
    
    start_time = time.time()
    first_token_time = None
    last_token_time = None

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", api_url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise Exception(f"HTTP {response.status_code}: {error_body.decode()}")
                
                print("\nStreaming output:", flush=True)
                print("-" * 50, flush=True)
                
                async for chunk in response.aiter_text():
                    for line in chunk.splitlines():
                        line = line.strip()
                        if line.startswith("data:"):
                            line = line[len("data:"):].strip()
                        if not line or line == "[DONE]":
                            continue
                            
                        try:
                            delta = json.loads(line)
                            for choice in delta.get("choices", []):
                                if token := choice.get("delta", {}).get("content", ""):
                                    current_time = time.time()
                                    
                                    if first_token_time is None:
                                        first_token_time = current_time
                                        metrics.ttft = first_token_time - start_time
                                    
                                    metrics.total_tokens += 1
                                    last_token_time = current_time
                                    metrics.token_times.append(current_time)
                                    
                                    print(token, end="", flush=True)
                                    metrics.output += token
                                    
                        except json.JSONDecodeError as e:
                            print(f"\nWarning: JSON decode error: {e}")
                            continue

    except Exception as e:
        print(f"\nError during streaming: {e}")
        metrics.error = str(e)
        metrics.success = False
        return metrics

    print("\n" + "-" * 50)

    if metrics.total_tokens > 0 and len(metrics.token_times) > 1:
        # Calculate TPOT using time differences between consecutive tokens
        token_intervals = []
        for i in range(1, len(metrics.token_times)):
            interval = metrics.token_times[i] - metrics.token_times[i-1]
            token_intervals.append(interval)
        
        metrics.tpot = mean(token_intervals)
        metrics.tpot_stdev = stdev(token_intervals) if len(token_intervals) > 1 else 0
    
    metrics.total_time = last_token_time - start_time if last_token_time else 0
    metrics.char_count = len(metrics.output)
    metrics.chars_per_token = metrics.char_count / metrics.total_tokens if metrics.total_tokens > 0 else 0
    
    return metrics

async def benchmark_provider(name: str, api_url: str, payload: Dict[str, Any], headers: Dict[str, str], prompt: str) -> Optional[BenchmarkMetrics]:
    print(f"\nTesting provider: {name}")
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    try:
        # Update payload with the current prompt
        payload["messages"][0]["content"] = prompt
        
        metrics = await try_provider_http_stream(api_url, payload, headers)
        
        if metrics.success:
            print("\nMetrics:")
            print("-" * 50)
            print(f"TTFT: {metrics.ttft:.3f}s")
            print(f"Total tokens: {metrics.total_tokens}")
            print(f"Total time: {metrics.total_time:.3f}s")
            if metrics.tpot is not None:
                print(f"Average TPOT: {metrics.tpot*1000:.2f}ms ± {metrics.tpot_stdev*1000:.2f}ms")
            print(f"Average throughput: {metrics.total_tokens/metrics.total_time:.1f} tokens/sec")
            print(f"Character count: {metrics.char_count}")
            print(f"Chars per token: {metrics.chars_per_token:.1f}")
        else:
            print(f"\nBenchmark failed: {metrics.error}")
        
        return metrics
    except Exception as e:
        print(f"Provider failed: {e}")
        return None

def save_benchmark_results(results: List[Dict[str, Any]], prompts: List[str]):
    """Save benchmark results to a JSON file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "prompts": prompts,
        "results": results
    }
    
    filename = f"benchmark_results_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {filename}")

def print_comparison_summary(results: List[Dict[str, Any]], prompts: List[str]):
    """Print a comprehensive comparison summary of benchmark results"""
    print("\nComparison Summary:")
    print("=" * 80)
    
    # Group results by prompt
    for prompt_idx, prompt in enumerate(prompts):
        prompt_results = [r for r in results if r["prompt_index"] == prompt_idx]
        
        print(f"\nPrompt {prompt_idx + 1}: {prompt}")
        print("-" * 80)
        
        metrics = ["ttft", "total_tokens", "total_time", "chars_per_token"]
        metric_names = {
            "ttft": "Time to First Token (s)",
            "total_tokens": "Total Tokens",
            "total_time": "Total Time (s)",
            "chars_per_token": "Characters per Token"
        }
        
        for metric in metrics:
            values = [(r["provider"], r["metrics"][metric]) 
                     for r in prompt_results 
                     if r["metrics"].get(metric) is not None]
            
            if values:
                print(f"\n{metric_names[metric]}:")
                values.sort(key=lambda x: x[1])
                for provider, value in values:
                    print(f"  {provider}: {value:.3f}")
                
                best_provider, best_value = values[0]
                print(f"  Winner: {best_provider} ({best_value:.3f})")

async def main():
    # Load environment variables
    API_TOKEN = os.environ.get("API_TOKEN")
    ARK_API_KEY = os.environ.get("ARK_API_KEY")
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    
    if not all([API_TOKEN, ARK_API_KEY, DEEPSEEK_API_KEY]):
        print("Error: Required environment variables not set")
        sys.exit(1)

    # Test prompts
    prompts = [
        "写一首绝句，不要做任何解释",
        "用简单的语言解释量子纠缠",
        "分析中美贸易关系的历史发展",
        "设计一个简单的购物车系统，包括类的定义和主要方法",
    ]
    
    # Configure providers
    providers = [
        {
            "name": "Doubao 1.5 Lite 32K",
            "url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            "payload": {
                "model": "ep-20250206211726-ctqtj",
                "messages": [{"role": "user", "content": ""}],
                "stream": True,
                "max_tokens": 4096,
            },
            "headers": {
                "Authorization": f"Bearer {ARK_API_KEY}",
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Doubao 1.5 Pro 32K",
            "url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            "payload": {
                "model": "ep-20250206203431-bql9h",
                "messages": [{"role": "user", "content": ""}],
                "stream": True,
                "max_tokens": 4096,
            },
            "headers": {
                "Authorization": f"Bearer {ARK_API_KEY}",
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Doubao DeepSeek V3",
            "url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            "payload": {
                "model": "ep-20250206212003-d6k2m",
                "messages": [{"role": "user", "content": ""}],
                "stream": True,
                "max_tokens": 4096,
            },
            "headers": {
                "Authorization": f"Bearer {ARK_API_KEY}",
                "Content-Type": "application/json"
            }
        },
        {
            "name": "SiliconFlow DeepSeek V3",
            "url": "https://api.siliconflow.cn/v1/chat/completions",
            "payload": {
                "model": "deepseek-ai/DeepSeek-V3",
                "messages": [{"role": "user", "content": ""}],
                "stream": True,
                "max_tokens": 4096,
            },
            "headers": {
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            }
        },
        {
            "name": "DeepSeek R1 Official",
            "url": "https://api.deepseek.com/v1/chat/completions",
            "payload": {
                "model": "deepseek-reasoner",
                "messages": [{"role": "user", "content": ""}],
                "stream": True,
                "max_tokens": 8192,
            },
            "headers": {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
        }
    ]

    print("Starting benchmark...")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Number of providers: {len(providers)}")
    
    results = []
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nTesting Prompt {prompt_idx + 1}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        print("=" * 80)
        
        for provider in providers:
            metrics = await benchmark_provider(
                provider["name"],
                provider["url"],
                provider["payload"],
                provider["headers"],
                prompt
            )
            if metrics:
                results.append({
                    "provider": provider["name"],
                    "prompt_index": prompt_idx,
                    "metrics": {
                        "ttft": metrics.ttft,
                        "total_tokens": metrics.total_tokens,
                        "total_time": metrics.total_time,
                        "tpot": metrics.tpot,
                        "tpot_stdev": metrics.tpot_stdev,
                        "char_count": metrics.char_count,
                        "chars_per_token": metrics.chars_per_token,
                        "success": metrics.success,
                        "error": metrics.error
                    },
                    "output": metrics.output
                })
    
    # Print comparison summary and save results
    if results:
        print_comparison_summary(results, prompts)
        save_benchmark_results(results, prompts)

if __name__ == "__main__":
    asyncio.run(main())
