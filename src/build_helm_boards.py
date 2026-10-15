"""Build one board per HELM Capabilities benchmark from the compact per-run files of helm_fetch.py.

Each model is one system. Its per-task outcome is the benchmark's own per-instance score (pass/fail for GPQA, MMLU-Pro and
Omni-MATH, a fraction of instructions followed for IFEval). Its per-task cost is the priced token usage of its request,

    cost = prompt_tokens * input_price + completion_tokens * output_price,

with ONE price table (data/helm/model_prices.json, list prices in dollars per token, source key recorded per model). HELM
records token counts, not bills, so a model enters a board only when its recorded counts are usable as billed counts:

  * the benchmark must record prompt tokens (WildBench does not, so it is left out);
  * the model must record completion tokens (the Google and DeepSeek-V3 runs record none);
  * the model must not hide reasoning tokens (the OpenAI o-series and GPT-5 runs, the Claude extended-thinking runs, Grok 4
    and Grok 3 mini record the visible answer only);
  * the model must have a price under the pricing rule below: first-party for OpenAI and Anthropic, Bedrock for Amazon, the
    Together AI list price for open-weight models (OLMo, Marin, Granite, Palmyra, Grok 3, Mixtral 8x22B, Llama 4 Maverick, the
    Qwen3 235B runs and the Mistral Large and Small runs have none and are left out).

The arrival date of a system is the date of the HELM release (v1.0.0 to v1.15.0) in which it first appears.

Writes data/boards/helm_<scenario>.json (honestboard board format, outcomes as rates), data/boards/helm_<scenario>_costs.json
({system: {instance: cost}}), data/boards/helm_summary.json and data/helm/model_prices.json.

usage: python build_helm_boards.py
"""
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HELM = ROOT / "data" / "helm"
OUT = ROOT / "data" / "boards"
HAL_PRICES = json.loads((ROOT / "data" / "hal" / "model_prices.json").read_text(encoding="utf-8"))

# helm model -> (source table, key). "hal": the price table of the HAL audit (first-party prices of OpenAI and Anthropic);
# "ll": LiteLLM's public price list, fetched 2026-10-14. The pricing rule: OpenAI and Anthropic models at their first-party price,
# Amazon Nova at its Bedrock price, and every open-weight model at the Together AI list price (HELM ran them through Together);
# a model with no such price is left out rather than priced at some other host.
SOURCES = {
    "amazon_nova-lite-v1:0": ("ll", "amazon.nova-lite-v1:0"),
    "amazon_nova-micro-v1:0": ("ll", "amazon.nova-micro-v1:0"),
    "amazon_nova-pro-v1:0": ("ll", "amazon.nova-pro-v1:0"),
    "amazon_nova-premier-v1:0": ("ll", "amazon.nova-premier-v1:0"),
    "anthropic_claude-3-5-haiku-20241022": ("hal", "claude-3-5-haiku-20241022"),
    "anthropic_claude-3-5-sonnet-20241022": ("hal", "claude-3-5-sonnet-20241022"),
    "anthropic_claude-3-7-sonnet-20250219": ("hal", "claude-3-7-sonnet-20250219"),
    "anthropic_claude-opus-4-20250514": ("hal", "claude-opus-4-20250514"),
    "anthropic_claude-sonnet-4-20250514": ("hal", "claude-sonnet-4-20250514"),
    "anthropic_claude-sonnet-4-5-20250929": ("hal", "claude-sonnet-4-5-20250929"),
    "anthropic_claude-haiku-4-5-20251001": ("ll", "claude-haiku-4-5-20251001"),
    "openai_gpt-4o-2024-11-20": ("hal", "gpt-4o-2024-11-20"),
    "openai_gpt-4o-mini-2024-07-18": ("hal", "gpt-4o-mini-2024-07-18"),
    "openai_gpt-4.1-2025-04-14": ("hal", "gpt-4.1-2025-04-14"),
    "openai_gpt-4.1-mini-2025-04-14": ("hal", "gpt-4.1-mini-2025-04-14"),
    "openai_gpt-4.1-nano-2025-04-14": ("hal", "gpt-4.1-nano-2025-04-14"),
    "openai_gpt-oss-120b": ("ll", "together_ai/openai/gpt-oss-120b"),
    "openai_gpt-oss-20b": ("ll", "together_ai/openai/gpt-oss-20b"),
    "meta_llama-3.1-405b-instruct-turbo": ("ll", "together_ai/meta-llama/Llama-3.1-405B-Instruct"),
    "meta_llama-3.1-70b-instruct-turbo": ("ll", "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    "meta_llama-3.1-8b-instruct-turbo": ("ll", "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    "meta_llama-4-scout-17b-16e-instruct": ("ll", "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"),
    "mistralai_mistral-7b-instruct-v0.3": ("ll", "together_ai/mistralai/Mistral-7B-Instruct-v0.3"),
    "mistralai_mixtral-8x7b-instruct-v0.1": ("ll", "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"),
    "moonshotai_kimi-k2-instruct": ("ll", "together_ai/moonshotai/Kimi-K2-Instruct"),
    "qwen_qwen2.5-72b-instruct-turbo": ("ll", "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo"),
    "qwen_qwen2.5-7b-instruct-turbo": ("ll", "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo"),
    "zai-org_glm-4.5-air-fp8": ("ll", "together_ai/zai-org/GLM-4.5-Air-FP8"),
}
# the same open-weight models at DeepInfra, where LiteLLM lists a price: the host-sensitivity check (helm_price_sensitivity.py)
ALT_DEEPINFRA = {
    "openai_gpt-oss-120b": "deepinfra/openai/gpt-oss-120b", "openai_gpt-oss-20b": "deepinfra/openai/gpt-oss-20b",
    "meta_llama-3.1-70b-instruct-turbo": "deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct", "meta_llama-3.1-8b-instruct-turbo": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta_llama-4-scout-17b-16e-instruct": "deepinfra/meta-llama/Llama-4-Scout-17B-16E-Instruct", "mistralai_mixtral-8x7b-instruct-v0.1": "deepinfra/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "moonshotai_kimi-k2-instruct": "deepinfra/moonshotai/Kimi-K2-Instruct", "qwen_qwen2.5-72b-instruct-turbo": "deepinfra/Qwen/Qwen2.5-72B-Instruct", "qwen_qwen2.5-7b-instruct-turbo": "deepinfra/Qwen/Qwen2.5-7B-Instruct",
}
HIDDEN_REASONING = ("thinking", "grok-4", "grok-3-mini", "o3-", "o4-mini", "gpt-5")
METRIC = {"gpqa": "chain_of_thought_correctness", "mmlu_pro": "chain_of_thought_correctness", "omni_math": "omni_math_accuracy",
          "ifeval": "ifeval_strict_accuracy", "wildbench": "wildbench_score_rescaled"}
NAMES = {"gpqa": "GPQA", "mmlu_pro": "MMLU-Pro", "omni_math": "Omni-MATH", "ifeval": "IFEval", "wildbench": "WildBench"}


def load_prices():
    ll = json.loads((HELM / "litellm_prices_2026-10-14.json").read_text(encoding="utf-8"))
    out = {}
    for m, (src, key) in SOURCES.items():
        if src == "hal":
            p = HAL_PRICES[key]
            out[m] = {"prompt": p["prompt"], "completion": p["completion"], "source": f"HAL price table: {key}"}
        else:
            p = ll[key]
            out[m] = {"prompt": p["input_cost_per_token"], "completion": p["output_cost_per_token"], "source": f"LiteLLM list: {key}"}
    return out


def main():
    prices = load_prices()
    (HELM / "model_prices.json").write_text(json.dumps(prices, indent=1), encoding="utf-8")
    ll_ = json.loads((HELM / "litellm_prices_2026-10-14.json").read_text(encoding="utf-8"))
    alt = {m: {"prompt": ll_[k]["input_cost_per_token"], "completion": ll_[k]["output_cost_per_token"], "source": f"LiteLLM list: {k}"} for m, k in ALT_DEEPINFRA.items()}
    (HELM / "model_prices_deepinfra.json").write_text(json.dumps(alt, indent=1), encoding="utf-8")
    rel = json.loads((HELM / "releases.json").read_text(encoding="utf-8"))
    runs = defaultdict(dict)
    for f in glob.glob(str(HELM / "runs" / "*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        runs[d["scenario"]][d["model"]] = d
    summary = {}
    for scen, by_model in runs.items():
        if scen == "wildbench":
            summary[scen] = {"benchmark": NAMES[scen], "left_out": "prompt tokens are not recorded for any model", "models": len(by_model)}
            continue
        tasks = sorted(next(iter(by_model.values()))["instances"], key=lambda s: int(s[2:]) if s[2:].isdigit() else s)
        systems, costs, dropped = [], {}, []
        for model, d in sorted(by_model.items(), key=lambda kv: (rel[kv[1]["suite"]], kv[0])):
            ins = d["instances"]
            mean_c = st.mean(r["c"] for r in ins.values())
            if any(h in model for h in HIDDEN_REASONING):
                dropped.append({"system": model, "reason": "hidden reasoning tokens are not recorded"}); continue
            if mean_c < 20:
                dropped.append({"system": model, "reason": "completion tokens are not recorded"}); continue
            if model not in prices:
                dropped.append({"system": model, "reason": "no price under the pricing rule"}); continue
            if set(ins) != set(tasks):
                dropped.append({"system": model, "reason": "does not cover the benchmark's instances"}); continue
            pr = prices[model]
            metric = METRIC[scen]
            rates = {t: float(ins[t]["y"][metric]) for t in tasks}
            costs[model] = {t: ins[t]["p"] * pr["prompt"] + ins[t]["c"] * pr["completion"] for t in tasks}
            systems.append({"name": model, "date": rel[d["suite"]].replace("-", ""), "release": d["suite"], "rates": rates,
                            "accuracy_reported": sum(rates.values()) / len(tasks), "source": d["run"], "cost_source": "per_task_tokens"})
        board = {"benchmark": f"HELM {NAMES[scen]}", "tasks": tasks, "systems": systems, "cost_source": "per_task_tokens"}
        (OUT / f"helm_{scen}.json").write_text(json.dumps(board), encoding="utf-8")
        (OUT / f"helm_{scen}_costs.json").write_text(json.dumps(costs), encoding="utf-8")
        summary[scen] = {"benchmark": NAMES[scen], "tasks": len(tasks), "kept": len(systems), "runs": len(by_model), "dropped": dropped}
        print(f"{NAMES[scen]:10s} {len(tasks):5d} tasks, {len(systems)} systems kept of {len(by_model)}; dropped: "
              + ", ".join(sorted({d['reason'] for d in dropped})))
    (OUT / "helm_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
