# Updated logging and analysis workflow

This package contains the updated experiment runner plus an analysis script.

## Files changed

- `BD_Attack_Collusion.py`
  - Keeps the existing terminal output and legacy `.txt` metric files.
  - Adds structured CSV/JSON files inside each experiment result directory.
  - Times every participating client's local training call with `time.perf_counter()`.
  - Records each client as `benign` or `malicious` for the current round.
  - Records per-round average/total benign latency and malicious latency.
  - Records defense retention rates in the same round-level CSV.

- `models/DeepS.py`, `models/Krum.py`, `models/Flame.py`, `models/FLShield.py`, `models/Freqfed.py`, `models/Measa.py`
  - Each defense now writes the selected local indices and retention rates to `args.last_defense_stats`.
  - The main runner reads that object and includes defense retention in `round_metrics_<run>.csv`.
  - The defense-side legacy analysis `.txt` files are still written.

- `analysis/analyze_results.py`
  - Scans new `round_metrics_<run>.csv` and `client_latency_<run>.csv` files.
  - Also reads legacy metric `.txt` files when structured CSV logs do not exist.
  - Produces one merged round-level CSV and summary CSV files for comparing multiple trials.

## New output files per experiment directory

For each `per_run`, the runner now creates:

- `run_config_<run>.json` — copy of run arguments and output location.
- `round_metrics_<run>.csv` — one row per federated round.
- `client_latency_<run>.csv` — one row per selected client per federated round.

The old files are still created, for example:

- `main_loss_<run>.txt`
- `Main_Acc_test_<run>.txt`
- `BD_Acc_test_<run>.txt`
- `bd_loss_<run>.txt`

## Key CSV columns

`round_metrics_<run>.csv` includes:

- `main_loss`
- `main_acc_test`
- `train_acc`
- `backdoor_loss`
- `backdoor_acc_test`
- `round_time_s`
- `benign_latency_mean_s`
- `malicious_latency_mean_s`
- `benign_latency_total_s`
- `malicious_latency_total_s`
- `defense_malicious_retention_rate`
- `defense_benign_retention_rate`
- `selected_client_ids`
- `benign_client_ids`
- `malicious_client_ids`
- `collusion_adjustment_latency_s`

`client_latency_<run>.csv` includes:

- `run`
- `round`
- `client_id`
- `selected_slot`
- `role` (`benign` or `malicious`)
- `num_samples`
- `latency_s`
- `loss`
- `learning_rate`
- experiment identifiers such as dataset, defense, epsilon, clipping bound, and attacker count

## Running experiments

Copy these files into the same locations in your original repository. In particular, replace the main script with `BD_Attack_Collusion.py` and replace the matching files under `models/`.

Then run your experiments the same way as before, for example:

```bash
python BD_Attack_Collusion.py \
  --dataset mnist \
  --lr 0.05 \
  --dp_mechanism MA \
  --dp_epsilon 5 \
  --dp_delta 1e-5 \
  --dp_clip 10 \
  --dp_sample 1 \
  --attack_type Collusion \
  --frac 0.25 \
  --num_attacker 6 \
  --attack \
  --epochs 100 \
  --bs 500 \
  --local_ep 5 \
  --defense Deepsight \
  --gpu 1 \
  --PDR 0.5 \
  --iid qty \
  --re_weight
```

## Summarizing many trials

After running multiple trials, run:

```bash
python analysis/analyze_results.py --roots Results Results_DBA Results_Neuro --out analysis_summary
```

Optional plots:

```bash
python analysis/analyze_results.py --roots Results Results_DBA Results_Neuro --out analysis_summary --plots
```

The summary directory contains:

- `round_metrics_all.csv` — all rounds across all discovered experiments and trials.
- `final_round_by_run.csv` — one final-round row per trial.
- `setup_summary.csv` — averages and standard deviations grouped by setup.
- `client_latency_summary.csv` — role-level latency summary for benign and malicious nodes.
- `manifest.json` — row counts and output list.

## Notes on old runs

The analysis script can read your old `.txt` files and summarize main/backdoor metrics. Old runs did not record client latency, so latency columns will be blank for those runs. Run new experiments with the updated code to populate benign and malicious latency.

## What is counted as latency

`latency_s` is the wall-clock time for the local client training call, measured around `local.train(...)`. For collusion rounds, the extra server-side coordinated noise/update adjustment is logged separately as `collusion_adjustment_latency_s` in the round CSV, rather than being assigned to a single malicious client.
