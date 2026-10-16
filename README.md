# This is Code is Designed by Anuj Tiwari [26310013] for NeuroAi course Take home assignment 3.

# Designing and Evaluating Cognitive and Language Models

## Layout

```
part1_continuous_agent/      Part 1 - CALM, a continual-learning agent
  environment.py             non-stationary grid world (valence reversal, new pairings, novel objects)
  calm_agent.py              the agent: associative memory, associability, change detector,
                             Hebbian co-occurrence, episodic replay, curiosity-driven action
  run_experiments.py         7 conditions x 5 seeds -> results/part1, figures

part2_llama_ablation/        Part 2 - ablation study of TinyLlama-1.1B (LLaMA-2 architecture)
  ablation.py                hook-based ablations: heads, attention blocks, MLPs, layers, embeddings
  tasks.py                   WikiText-2 perplexity, BLiMP (syntax), SST-2 (sentiment), factual completion
  run_ablation.py            experiments E0-E5 (resumable) -> results/part2/results.json
  plot_results.py            figures and CSV tables
  data/eval_data.json        the exact evaluation items used

results/                     raw results (CSV / JSON)
```

## Running

```bash
python -m venv .venv && .venv/Scripts/activate        # Python 3.11
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

cd part1_continuous_agent && python run_experiments.py    # ~2 min, CPU
cd ../part2_llama_ablation && python run_ablation.py      # GPU
```

Part 2 downloads `TinyLlama/TinyLlama_v1.1` and the WikiText-2, BLiMP and SST-2 datasets from the
Hugging Face Hub on the first run. Weights are stored in fp16 but computed in fp32 (`use_fp32_compute`),
because GTX 16xx GPUs run fp16 matrix multiplies about 5x slower than fp32.

## Status of Part 2

The ablation run finished E0 (baseline), E1 (every layer / attention block / MLP block) and the
importance-ordered part of E2 (head pruning). The random head orders, E3 (layer removal), E4
(embedding corruption) and E5 (generations) did not finish, so the reported results cover E0-E2 only.
`run_ablation.py` resumes from `results/part2/results.json`. `plot_results.py` expects the full set of
experiments, so it only runs end to end once the remaining experiments are complete.
