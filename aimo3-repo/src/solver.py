"""
AIMO3 — AI Mathematical Olympiad Progress Prize 3
==================================================
Competition : AI Mathematical Olympiad Progress Prize 3 (Kaggle, April 2026)
Author      : Haseeb Ahmad (haseb_ahmad@yahoo.com)
Result      : 42 / 50 — Bronze Medal 🥉 | GPT-OSS-120B on single H100
Kaggle      : https://kaggle.com/hasib007

Architecture
------------
Model       : GPT-OSS-120B served via vLLM (fp8_e4m3 KV cache)
Inference   : Multi-attempt parallel reasoning with early stopping
Tool Use    : Persistent Jupyter sandbox (Python code execution per attempt)
Selection   : Entropy-weighted voting across attempts
Prompting   : IMO-level chain-of-thought + structured verification + \\boxed{} output

Key Design Decisions
--------------------
1. Parallel attempts (8 workers) with shared entropy-weighted voting
2. Early stopping when 4+ attempts agree on the same answer
3. Per-attempt adaptive seed for diversity
4. Entropy-based confidence scoring via top-k logprobs
5. Dynamic time budgeting across problems (total 17,400s notebook limit)
6. Persistent Jupyter kernels reused across turns (no cold-start overhead)
7. FP8 KV cache for memory efficiency on single H100
"""

import gc
import os
import re
import sys
import math
import time
import queue
import threading
import subprocess
import contextlib
from typing import Optional
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from jupyter_client import KernelManager

import pandas as pd
import polars as pl

from openai import OpenAI
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
    Conversation,
)
from transformers import set_seed
import kaggle_evaluation.aimo_3_inference_server


# ── Environment Setup ─────────────────────────────────────────────────────────

def setup_environment(
    input_archive: str = "/kaggle/input/aimo-3-utils/wheels.tar.gz",
    temp_dir: str = "/kaggle/tmp/setup",
) -> None:
    """Extract offline wheels and install dependencies."""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        subprocess.run(["tar", "-xzf", input_archive, "-C", temp_dir], check=True)

    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", f"{temp_dir}/wheels",
        "unsloth", "trl", "vllm", "openai_harmony",
    ], check=True)


def set_cuda_env() -> None:
    """Configure environment variables for deterministic CUDA execution."""
    os.environ["TRANSFORMERS_NO_TF"]      = "1"
    os.environ["TRANSFORMERS_NO_FLAX"]    = "1"
    os.environ["CUDA_VISIBLE_DEVICES"]    = "0"
    os.environ["TOKENIZERS_PARALLELISM"]  = "false"
    os.environ["TRITON_PTXAS_PATH"]       = "/usr/local/cuda/bin/ptxas"
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = "/kaggle/tmp/setup/tiktoken_encodings"


# ── Config ────────────────────────────────────────────────────────────────────

class CFG:
    """
    Central configuration for the AIMO3 solver.

    Tunable parameters (from 7 experiment iterations):
    - temperature / min_p  : controls sampling diversity
    - attempts / workers   : parallelism budget
    - early_stop           : consensus threshold
    - context_tokens       : 65536 for 120B model
    - gpu_memory_utilization : 0.96 for single H100
    """

    # ── Prompts ──
    system_prompt = (
        "You are an elite mathematical problem solver with expertise at the International "
        "Mathematical Olympiad (IMO) level. Your goal is to find the correct answer through "
        "rigorous mathematical reasoning.\n\n"
        "# Problem-Solving Approach:\n"
        "1. UNDERSTAND: Carefully read and rephrase the problem in your own words. "
        "Identify what is given, what needs to be found, and any constraints.\n"
        "2. EXPLORE: Consider multiple solution strategies. Think about relevant theorems, "
        "techniques, patterns, or analogous problems. Don't commit to one approach immediately.\n"
        "3. PLAN: Select the most promising approach and outline key steps before executing.\n"
        "4. EXECUTE: Work through your solution methodically. Show all reasoning steps clearly.\n"
        "5. VERIFY: Check your answer by substituting back, testing edge cases, or using "
        "alternative methods. Ensure logical consistency throughout.\n\n"
        "# Mathematical Reasoning Principles:\n"
        "- Break complex problems into smaller, manageable sub-problems\n"
        "- Look for patterns, symmetries, and special cases that provide insight\n"
        "- Use concrete examples to build intuition before generalizing\n"
        "- Consider extreme cases and boundary conditions\n"
        "- If stuck, try working backwards from the desired result\n"
        "- Be willing to restart with a different approach if needed\n\n"
        "# Verification Requirements:\n"
        "- Cross-check arithmetic and algebraic manipulations\n"
        "- Verify that your solution satisfies all problem constraints\n"
        "- Test your answer with simple cases or special values when possible\n"
        "- Ensure dimensional consistency and reasonableness of the result\n\n"
        "# Output Format:\n"
        "The final answer must be a non-negative integer between 0 and 99999.\n"
        "Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n"
        "Think step-by-step and show your complete reasoning process. Quality of reasoning "
        "is as important as the final answer."
    )

    tool_prompt = (
        "Use this tool to execute Python code for:\n"
        "- Complex calculations that would be error-prone by hand\n"
        "- Numerical verification of analytical results\n"
        "- Generating examples or testing conjectures\n"
        "- Visualizing problem structure when helpful\n"
        "- Brute-force verification for small cases\n\n"
        "The environment is a stateful Jupyter notebook. Code persists between executions.\n"
        "Always use print() to display results. Write clear, well-commented code.\n\n"
        "Remember: Code should support your mathematical reasoning, not replace it. "
        "Explain what you're computing and why before running code."
    )

    preference_prompt = (
        "You have access to `math`, `numpy`, and `sympy` for:\n\n"
        "# Symbolic Computation (sympy):\n"
        "- Algebraic manipulation and simplification\n"
        "- Solving equations and systems of equations\n"
        "- Number theory functions (primes, divisors, modular arithmetic)\n"
        "- Polynomial operations and factorization\n\n"
        "# Numerical Computation (numpy):\n"
        "- Array operations and linear algebra\n"
        "- Matrix operations and eigenvalue problems\n\n"
        "Best Practices:\n"
        "- Use sympy for exact symbolic answers when possible\n"
        "- Combine symbolic and numerical approaches: derive symbolically, verify numerically\n"
        "- Validate computational results against known cases or theoretical bounds"
    )

    # ── Model ──
    served_model_name      = "gpt-oss"
    model_path             = "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1"
    kv_cache_dtype         = "fp8_e4m3"   # FP8 for H100 memory efficiency
    dtype                  = "auto"

    # ── Sampling (best from experiment grid) ──
    temperature            = 0.7
    min_p                  = 0.1

    # ── Timeouts ──
    high_problem_timeout   = 900    # 15 min for hard problems
    base_problem_timeout   = 300    # 5 min reserved per remaining problem
    notebook_limit         = 17400  # total notebook budget (seconds)
    server_timeout         = 180
    session_timeout        = 960
    jupyter_timeout        = 6
    sandbox_timeout        = 3

    # ── Inference ──
    stream_interval        = 200
    context_tokens         = 65536
    buffer_tokens          = 512
    search_tokens          = 32
    top_logprobs           = 5
    batch_size             = 256

    # ── Parallelism ──
    early_stop             = 4      # consensus threshold
    attempts               = 8      # parallel reasoning attempts
    workers                = 16
    turns                  = 128    # max conversation turns per attempt
    seed                   = 42

    # ── vLLM ──
    gpu_memory_utilization = 0.96


# ── Jupyter Sandbox ───────────────────────────────────────────────────────────

class AIMO3Sandbox:
    """
    Persistent Jupyter kernel for stateful Python code execution.
    Reused across turns to avoid kernel startup overhead.
    """

    def __init__(self, timeout: int = 6):
        self.timeout   = timeout
        self.km        = KernelManager()
        self.km.start_kernel()
        self.kc        = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=30)

    def execute(self, code: str) -> str:
        """Execute code and return stdout + stderr output."""
        msg_id = self.kc.execute(code)
        outputs = []

        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=self.timeout)
            except Exception:
                break

            msg_type = msg["msg_type"]
            content  = msg.get("content", {})

            if msg_type == "stream":
                outputs.append(content.get("text", ""))
            elif msg_type == "error":
                tb = "\n".join(content.get("traceback", []))
                outputs.append(f"[ERROR] {content.get('ename')}: {content.get('evalue')}\n{tb}")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        return "".join(outputs)

    def reset(self) -> None:
        """Clear kernel state between problems."""
        self.execute("%reset -f")

    def close(self) -> None:
        self.kc.stop_channels()
        self.km.shutdown_kernel()


# ── Tool Interface ────────────────────────────────────────────────────────────

class AIMO3Tool:
    """Wraps the Jupyter sandbox as an LLM tool for code execution."""

    def __init__(self, local_jupyter_timeout: int, tool_prompt: str, sandbox: AIMO3Sandbox):
        self.sandbox     = sandbox
        self.tool_prompt = tool_prompt
        self.tool_config = ToolNamespaceConfig(
            namespace   = "python",
            description = tool_prompt,
            author      = Author.TOOL,
        )

    def process_sync_plus(self, message: Message) -> list:
        """Execute code from LLM tool call and return result message."""
        code    = message.content[0].text if message.content else ""
        output  = self.sandbox.execute(code)
        result  = output[:4096] if len(output) > 4096 else output  # truncate long outputs

        return [Message(
            role    = Role.TOOL,
            channel = "python",
            content = [TextContent(text=result or "[No output]")],
        )]


# ── vLLM Server ───────────────────────────────────────────────────────────────

class VLLMServer:
    """Manages the vLLM OpenAI-compatible inference server process."""

    def __init__(self, cfg: CFG, port: int = 8000):
        self.cfg  = cfg
        self.port = port
        self._preload_weights()
        self.server_process = self._start_server()
        self.client         = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
        self._wait_for_server()

    def _preload_weights(self) -> None:
        """Pre-load model weights into OS page cache for faster cold start."""
        print("Pre-loading model weights...")
        start      = time.time()
        files_list = []
        total_size = 0

        for root, _, files in os.walk(self.cfg.model_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                if os.path.isfile(fpath):
                    files_list.append(fpath)
                    total_size += os.path.getsize(fpath)

        def _read(path: str) -> None:
            with open(path, "rb") as f:
                while f.read(1024 * 1024 * 1024):
                    pass

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            list(ex.map(_read, files_list))

        elapsed = time.time() - start
        print(f"Pre-loaded {len(files_list)} files ({total_size/1e9:.2f} GB) in {elapsed:.2f}s\n")

    def _start_server(self) -> subprocess.Popen:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--seed",                   str(self.cfg.seed),
            "--model",                  self.cfg.model_path,
            "--served-model-name",      self.cfg.served_model_name,
            "--tensor-parallel-size",   "1",
            "--max-num-seqs",           str(self.cfg.batch_size),
            "--gpu-memory-utilization", str(self.cfg.gpu_memory_utilization),
            "--host",                   "0.0.0.0",
            "--port",                   str(self.port),
            "--dtype",                  self.cfg.dtype,
            "--kv-cache-dtype",         self.cfg.kv_cache_dtype,
            "--max-model-len",          str(self.cfg.context_tokens),
            "--stream-interval",        str(self.cfg.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
            "--enable-prefix-caching",
        ]
        self.log_file = open("vllm_server.log", "w")
        return subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT,
                                start_new_session=True)

    def _wait_for_server(self) -> None:
        print("Waiting for vLLM server...")
        start = time.time()
        for _ in range(self.cfg.server_timeout):
            rc = self.server_process.poll()
            if rc is not None:
                self.log_file.flush()
                logs = open("vllm_server.log").read()
                raise RuntimeError(f"Server died (code={rc}):\n{logs}")
            try:
                self.client.models.list()
                print(f"Server ready in {time.time()-start:.2f}s\n")
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("Server startup timeout.")

    def shutdown(self) -> None:
        self.server_process.terminate()
        self.server_process.wait()
        self.log_file.close()


# ── AIMO3 Solver ──────────────────────────────────────────────────────────────

class AIMO3Solver:
    """
    Main solver: manages parallel attempts, entropy-weighted voting, early stopping.

    Per-problem flow
    ----------------
    1. Compute time budget based on remaining problems
    2. Launch cfg.attempts parallel reasoning threads
    3. Each attempt: multi-turn LLM + tool loop → extract \\boxed{} answer
    4. Early stop when cfg.early_stop attempts agree
    5. Select final answer by entropy-weighted vote
    """

    def __init__(self, cfg: CFG):
        self.cfg                = cfg
        self.notebook_start_time = time.time()
        self.problems_remaining  = 50  # AIMO3 has 50 problems

        set_cuda_env()

        print("Setting up vLLM server...")
        self.vllm         = VLLMServer(cfg)
        self.client       = self.vllm.client
        self.encoding     = load_harmony_encoding(HarmonyEncodingName.GPT_OSS)
        self.stop_token_ids = self.encoding.stop_token_ids
        self.template     = self.encoding.chat_template

        print(f"Initializing {cfg.workers} Jupyter kernels...")
        self.sandbox_pool = queue.Queue()
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            futures = [ex.submit(AIMO3Sandbox, cfg.jupyter_timeout) for _ in range(cfg.workers)]
            for f in as_completed(futures):
                self.sandbox_pool.put(f.result())
        print("Kernels ready.\n")

    # ── Answer Extraction ──

    def _scan_for_answer(self, text: str) -> int | None:
        """Extract integer from \\boxed{N} or 'final answer is N' patterns."""
        for pattern in [
            r"\\boxed\s*\{\s*([0-9,]+)\s*\}",
            r"final\s+answer\s+is\s*([0-9,]+)",
        ]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    val = int(matches[-1].replace(",", ""))
                    if 0 <= val <= 99999:
                        return val
                except ValueError:
                    pass
        return None

    # ── Entropy Scoring ──

    def _compute_mean_entropy(self, logprobs_buffer: list) -> float:
        """
        Mean token entropy from top-k logprobs.
        Lower entropy = more confident → higher vote weight.
        """
        if not logprobs_buffer:
            return float("inf")
        total, count = 0.0, 0
        for top_dict in logprobs_buffer:
            if not isinstance(top_dict, dict) or not top_dict:
                continue
            h = sum(-math.exp(lp) * lp for lp in top_dict.values() if lp > -1e9)
            total += h
            count += 1
        return total / count if count > 0 else float("inf")

    # ── Single Attempt ──

    def _process_attempt(
        self,
        problem: str,
        system_prompt: str,
        attempt_index: int,
        stop_event: threading.Event,
        deadline: float,
    ) -> dict:
        """
        Run one reasoning attempt: multi-turn LLM conversation with Python tool.
        Returns dict with Answer, Entropy, Python Calls, Python Errors, Response Length.
        """
        if stop_event.is_set() or time.time() > deadline:
            return {"Attempt": attempt_index + 1, "Answer": None,
                    "Python Calls": 0, "Python Errors": 0,
                    "Response Length": 0, "Entropy": float("inf")}

        sandbox       = None
        local_tool    = None
        python_calls  = python_errors = total_tokens = 0
        final_answer  = None
        logprobs_buf  = []
        attempt_seed  = int(math.pow(self.cfg.seed + attempt_index, 2))

        try:
            sandbox    = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(self.cfg.jupyter_timeout, self.cfg.tool_prompt, sandbox)

            messages     = self.template.apply_chat_template(
                system_prompt, problem, local_tool.tool_config)
            conversation = Conversation.from_messages(messages)
            encoding     = self.encoding

            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = encoding.render_conversation_for_completion(
                    conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
                if max_tokens < self.cfg.buffer_tokens:
                    break

                stream = self.client.completions.create(
                    model       = self.cfg.served_model_name,
                    temperature = self.cfg.temperature,
                    logprobs    = self.cfg.top_logprobs,
                    max_tokens  = max_tokens,
                    prompt      = prompt_ids,
                    seed        = attempt_seed,
                    stream      = True,
                    extra_body  = {
                        "min_p":           self.cfg.min_p,
                        "stop_token_ids":  self.stop_token_ids,
                        "return_token_ids": True,
                    },
                )

                try:
                    token_buf, text_chunks = [], []
                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break
                        new_tokens = chunk.choices[0].token_ids
                        new_text   = chunk.choices[0].text
                        if new_tokens:
                            token_buf.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                            lp = chunk.choices[0].logprobs
                            if lp and lp.top_logprobs:
                                logprobs_buf.extend(lp.top_logprobs)
                        if "}" in new_text:
                            search = "".join(text_chunks[-self.cfg.search_tokens:])
                            ans    = self._scan_for_answer(search)
                            if ans is not None:
                                final_answer = ans
                                break
                finally:
                    stream.close()

                if final_answer is not None or not token_buf:
                    break

                new_msgs = encoding.parse_messages_from_completion_tokens(
                    token_buf, Role.ASSISTANT)
                conversation.messages.extend(new_msgs)
                last_msg = new_msgs[-1]

                if last_msg.channel == "final":
                    final_answer = self._scan_for_answer(last_msg.content[0].text)
                    break

                if last_msg.recipient == "python":
                    python_calls += 1
                    tool_resp = local_tool.process_sync_plus(last_msg)
                    resp_text = tool_resp[0].content[0].text
                    if any(e in resp_text for e in ["[ERROR]", "Traceback", "Error:"]):
                        python_errors += 1
                    conversation.messages.extend(tool_resp)

        except Exception:
            python_errors += 1
        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        return {
            "Attempt":         attempt_index + 1,
            "Response Length": total_tokens,
            "Python Calls":    python_calls,
            "Python Errors":   python_errors,
            "Entropy":         self._compute_mean_entropy(logprobs_buf),
            "Answer":          final_answer,
        }

    # ── Answer Selection ──

    def _select_answer(self, results: list) -> int:
        """
        Entropy-weighted voting: weight = 1 / entropy.
        Displays vote table, returns top-scored answer.
        """
        weights = defaultdict(float)
        votes   = defaultdict(int)
        for r in results:
            if r["Answer"] is not None:
                w = 1.0 / max(r["Entropy"], 1e-9)
                weights[r["Answer"]] += w
                votes[r["Answer"]]   += 1

        scored = sorted(
            [{"answer": a, "votes": votes[a], "score": w} for a, w in weights.items()],
            key=lambda x: x["score"], reverse=True,
        )

        df = pd.DataFrame([(s["answer"], s["votes"], round(s["score"], 3)) for s in scored],
                          columns=["Answer", "Votes", "Score"])
        display(df)

        final = scored[0]["answer"] if scored else 0
        print(f"\nFinal Answer: {final}\n")
        return final

    # ── Solve ──

    def solve_problem(self, problem: str) -> int:
        """
        Full solve pipeline for one problem.
        Manages time budget, launches parallel attempts, aggregates results.
        """
        print(f"\nProblem: {problem}\n")
        user_input = f"{problem} {self.cfg.preference_prompt}"

        # Dynamic time budget
        elapsed    = time.time() - self.notebook_start_time
        time_left  = self.cfg.notebook_limit - elapsed
        reserved   = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
        budget     = min(max(time_left - reserved, self.cfg.base_problem_timeout),
                         self.cfg.high_problem_timeout)
        deadline   = time.time() + budget
        print(f"Budget: {budget:.2f}s | Problems remaining: {self.problems_remaining}\n")

        tasks = [(self.cfg.system_prompt, i) for i in range(self.cfg.attempts)]
        results, valid_answers = [], []
        stop_event = threading.Event()

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [
                executor.submit(self._process_attempt, user_input, sp, idx, stop_event, deadline)
                for sp, idx in tasks
            ]
            try:
                for future in as_completed(futures):
                    try:
                        r = future.result()
                        results.append(r)
                        if r["Answer"] is not None:
                            valid_answers.append(r["Answer"])
                        # Early stop on consensus
                        top = Counter(valid_answers).most_common(1)
                        if top and top[0][1] >= self.cfg.early_stop:
                            stop_event.set()
                            break
                    except Exception as e:
                        print(f"Attempt failed: {e}")
            finally:
                stop_event.set()

        self.problems_remaining = max(0, self.problems_remaining - 1)

        if results:
            df = pd.DataFrame(results)
            df["Entropy"] = df["Entropy"].round(3)
            df["Answer"]  = df["Answer"].astype("Int64")
            display(df)

        if not valid_answers:
            print("\nNo valid answers found. Returning 0.\n")
            return 0

        return self._select_answer(results)

    def __del__(self):
        if hasattr(self, "vllm"):
            self.vllm.shutdown()
        if hasattr(self, "sandbox_pool"):
            while not self.sandbox_pool.empty():
                try:
                    self.sandbox_pool.get_nowait().close()
                except Exception:
                    pass


# ── Kaggle Inference Interface ────────────────────────────────────────────────

solver = AIMO3Solver(CFG)


def predict(
    id_: pl.DataFrame,
    question: pl.DataFrame,
    answer: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Kaggle evaluation server entry point."""
    id_value      = id_.item(0)
    question_text = question.item(0)

    gc.disable()
    final_answer = solver.solve_problem(question_text)
    gc.enable()
    gc.collect()

    return pl.DataFrame({"id": id_value, "answer": final_answer})


if __name__ == "__main__":
    inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway((
            "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",
        ))
