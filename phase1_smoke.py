"""Tiny deterministic consensus run with persisted audit data; no downloads."""

import json
import numpy as np
import torch

from evaluation_harness import (
    TopKStabilityConfig, _new_topk_run_dir, run_topk_stability_experiment,
    save_per_query_csv, save_topk_stability_csv,
)


def main():
    cfg = TopKStabilityConfig(n_agents=4, proj_dim=2, k=2, n_hops=2,
                              noise_std=0., n_eval_queries=1,
                              lm_checkpoint="offline-numerical-fixture", run_kind="smoke")
    cfg.run_dir = _new_topk_run_dir()
    records = [{"relevance": np.array([1., 0., 1., 0.])}]
    shards = [[{"e_q": torch.tensor([1., 0.]),
                "e_doc": torch.nn.functional.normalize(torch.tensor([x, .5]), dim=0)}]
              for x in [1., .7, .2, -.3]]
    states = [{"weight": torch.eye(2) * (1 + j * .01)} for j in range(4)]
    rows = []
    history = run_topk_stability_experiment(cfg, records, shards, states, rows)
    for row in rows:
        assert row["per_document_bound_holds"]
        assert row["global_certified"] and row["pairwise_certified"] and row["ordered_certified"]
        assert not row["topk_changed"] and not row["ordering_changed"]
    save_topk_stability_csv(history, cfg.run_dir)
    save_per_query_csv(rows, cfg.run_dir)
    print(json.dumps({"run_dir": cfg.run_dir, "hops": len(history),
                      "all_three_certified": True, "violations": 0}))


if __name__ == "__main__":
    main()
