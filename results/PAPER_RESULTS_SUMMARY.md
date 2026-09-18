# PAPER RESULTS SUMMARY

Analysis-only report over the already-completed 3x3x5x5 sweep, candidate-pool
ablation, and per-query certificate data. No experiments were re-run; the core
method and the preserved SIR/Fisher pipeline are untouched.

## Experiment scale

- Main sweep: 3 datasets x 3 retrievers x 5 alphas x 5 hops = 225 aggregate cells
- Per-query decisions logged: 222,525 (query x hop x alpha x dataset x retriever)
- Total query evaluations: 222,525 (each row is one query scored at one hop)
- Candidate-pool ablation: 9 (dataset x candidate-count) cells, 200-query stress-test subset

## Total certified decisions and violations

- **global_certified**: 335 certified, 0 violations, rule-of-three 95% upper bound on violation probability = 8.96e-03
- **pairwise_certified**: 3,724 certified, 0 violations, rule-of-three 95% upper bound on violation probability = 8.06e-04
- **ordered_certified**: 923 certified, 0 violations, rule-of-three 95% upper bound on violation probability = 3.25e-03
- Combined certified decisions across all three certificates: 4,982
- Combined violations: 0

## Best / worst configurations (mean pairwise certification rate)

- Best: 2wikimultihopqa x MiniLM (3.13% mean pairwise rate)
- Worst: 2wikimultihopqa x E5 (0.48% mean pairwise rate)
- Single best cell: 2wikimultihopqa x MiniLM, alpha=0.8, hop=5 (60.1% pairwise rate)

## Retriever differences

| retriever_short   |   margin_k_mean |   margin_k_median |   b_max_mean |   b_max_median |   empirical_delta_mean |   empirical_delta_median |   ratio_mean |   pairwise_rate |
|:------------------|----------------:|------------------:|-------------:|---------------:|-----------------------:|-------------------------:|-------------:|----------------:|
| BGE               |          1.6036 |            1.1147 |      34.4697 |        17.1833 |                 1.4828 |                   0.7790 |       0.0408 |          0.0155 |
| E5                |          0.7347 |            0.5316 |      29.0267 |        14.4681 |                 1.5496 |                   0.8304 |       0.0493 |          0.0055 |
| MiniLM            |          0.5780 |            0.3838 |       8.8672 |         4.3760 |                 0.2910 |                   0.1489 |       0.0322 |          0.0292 |

- Theta (W, deltaW) trajectory spread across retrievers at matched (alpha,hop): 0.000000 (confirms W/deltaW are architecturally identical across retrievers; differences below are entirely embedding-driven)
- Matched-cell ratios: {'E5_vs_MiniLM_bmax_ratio': 3.439519290564312, 'E5_vs_BGE_bmax_ratio': 0.8421154736441052, 'E5_vs_MiniLM_margin_ratio': 1.300406975911315, 'E5_vs_BGE_margin_ratio': 0.45805695497315335}

## Communication-strength findings

- disagreement_vs_pairwise_rate: Spearman rho=-0.5314, p=8.43e-18 (n=225)
- disagreement_vs_mean_slack: Spearman rho=-0.5944, p=6.89e-23 (n=225)
- alpha_vs_pairwise_rate: Spearman rho=0.3090, p=2.31e-06 (n=225)
- hop_vs_pairwise_rate: Spearman rho=0.5065, p=4.63e-16 (n=225)

## Candidate-pool findings (200-query stress-test subset)

| dataset         |   n_candidates |   ndcg_at_5 |   pairwise_certificate_rate |   ordered_certificate_rate |   topk_changed_rate |   global_certificate_violations |   pairwise_certificate_violations |   ordered_certificate_violations |
|:----------------|---------------:|------------:|----------------------------:|---------------------------:|--------------------:|--------------------------------:|----------------------------------:|---------------------------------:|
| 2wikimultihopqa |             10 |      0.5986 |                      0.0600 |                     0.0000 |              0.0150 |                               0 |                                 0 |                                0 |
| 2wikimultihopqa |             20 |      0.5688 |                      0.0200 |                     0.0000 |              0.0000 |                               0 |                                 0 |                                0 |
| 2wikimultihopqa |             50 |      0.4927 |                      0.0050 |                     0.0000 |              0.0250 |                               0 |                                 0 |                                0 |
| hotpotqa        |             10 |      0.7025 |                      0.0300 |                     0.0000 |              0.0150 |                               0 |                                 0 |                                0 |
| hotpotqa        |             20 |      0.6991 |                      0.0200 |                     0.0000 |              0.0050 |                               0 |                                 0 |                                0 |
| hotpotqa        |             50 |      0.6924 |                      0.0200 |                     0.0000 |              0.0050 |                               0 |                                 0 |                                0 |
| musique         |             10 |      0.7205 |                      0.0750 |                     0.0000 |              0.0100 |                               0 |                                 0 |                                0 |
| musique         |             20 |      0.6005 |                      0.0300 |                     0.0000 |              0.0250 |                               0 |                                 0 |                                0 |
| musique         |             50 |      0.5820 |                      0.0050 |                     0.0000 |              0.0450 |                               0 |                                 0 |                                0 |

Note: n=20/50 candidate pools use cross-query distractor padding where a dataset's
native context is smaller than the requested pool size; these are not native benchmark pools.

## Bound tightness findings

| dataset         | retriever   |   mean |   median |    p25 |    p75 |    p90 |
|:----------------|:------------|-------:|---------:|-------:|-------:|-------:|
| 2wikimultihopqa | BGE         | 0.0407 |   0.0382 | 0.0239 | 0.0558 | 0.0693 |
| 2wikimultihopqa | E5          | 0.0495 |   0.0483 | 0.0249 | 0.0727 | 0.0867 |
| 2wikimultihopqa | MiniLM      | 0.0312 |   0.0288 | 0.0193 | 0.0404 | 0.0527 |
| hotpotqa        | BGE         | 0.0421 |   0.0398 | 0.0243 | 0.0583 | 0.0715 |
| hotpotqa        | E5          | 0.0498 |   0.0484 | 0.0246 | 0.0735 | 0.0878 |
| hotpotqa        | MiniLM      | 0.0345 |   0.0317 | 0.0211 | 0.0454 | 0.0584 |
| musique         | BGE         | 0.0397 |   0.0374 | 0.0235 | 0.0546 | 0.0671 |
| musique         | E5          | 0.0486 |   0.0471 | 0.0248 | 0.0712 | 0.0850 |
| musique         | MiniLM      | 0.0309 |   0.0286 | 0.0195 | 0.0399 | 0.0513 |

Ratios are all << 1: the certificate is conservative (loose) by roughly 20-30x on average,
not incorrect -- a valid sufficient condition is expected to be conservative.

## Claim validation

**Claim A** (SUPPORTED): Pairwise certification implies unchanged top-k membership.
  Evidence: 0 violations out of 3724 pairwise-certified query-hops.

**Claim B** (SUPPORTED): Ordered certification implies unchanged exact top-k ordering.
  Evidence: 0 violations out of 923 ordered-certified query-hops.

**Claim C** (SUPPORTED): Pairwise certification has greater or equal coverage than the global certificate.
  Evidence: 0 of 225 (dataset,retriever,alpha,hop) cells violate pairwise>=global.

**Claim D** (SUPPORTED): Certification coverage differs meaningfully across retrievers.
  Evidence: Pairwise-certified rate by retriever: {'BGE': 0.0155, 'E5': 0.0055, 'MiniLM': 0.0292} (max/min ratio=5.32x, MiniLM vs E5).

**Claim E** (SUPPORTED): Larger candidate pools reduce certification coverage.
  Evidence: [{'dataset': '2wikimultihopqa', 'n_candidates': 10, 'pairwise_certificate_rate': 0.06}, {'dataset': '2wikimultihopqa', 'n_candidates': 20, 'pairwise_certificate_rate': 0.02}, {'dataset': '2wikimultihopqa', 'n_candidates': 50, 'pairwise_certificate_rate': 0.005}, {'dataset': 'hotpotqa', 'n_candidates': 10, 'pairwise_certificate_rate': 0.03}, {'dataset': 'hotpotqa', 'n_candidates': 20, 'pairwise_certificate_rate': 0.02}, {'dataset': 'hotpotqa', 'n_candidates': 50, 'pairwise_certificate_rate': 0.02}, {'dataset': 'musique', 'n_candidates': 10, 'pairwise_certificate_rate': 0.075}, {'dataset': 'musique', 'n_candidates': 20, 'pairwise_certificate_rate': 0.03}, {'dataset': 'musique', 'n_candidates': 50, 'pairwise_certificate_rate': 0.005}]

**Claim F** (SUPPORTED): Communication convergence tends to increase certification coverage.
  Evidence: Spearman(disagreement, pairwise_rate)=-0.5314 (p=8.43e-18); Spearman(alpha, pairwise_rate)=0.3090; Spearman(hop, pairwise_rate)=0.5065.

## 5 strongest empirical claims safe to make in the paper

1. Across 222,525 certified-or-not query-hop decisions spanning 3 datasets, 3 retrievers,
   5 alphas and 5 hops, the pairwise and ordered certificates never once certified a query
   whose top-k (or ordered top-k) actually changed -- 0 violations out of 4,647 certified decisions.
2. The pairwise certificate is provably and empirically never less permissive than the
   global certificate (confirmed on all 225 cells), and certifies about 11.1x as many query-hops overall.
3. Certificate slack C shows a monotone dose-response relationship with actual top-k
   instability: P(top-k changed) falls from 42.6% in the most
   negative slack bin to 0.8% just below the certification
   boundary, and to exactly 0% once C>0.
4. Certification coverage rises with communication convergence: Spearman rho between
   agent parameter disagreement and pairwise certification rate is -0.53 (p=8.4e-18),
   i.e. certification becomes easier precisely as agents approach consensus.
5. Certification generalizes across all 3 evaluated datasets with the same qualitative
   pattern (retriever ranking MiniLM > BGE > E5 holds in every dataset), evidence the
   effect is a property of the certificate/retriever interaction, not a dataset artifact.

## 5 claims we should NOT make

1. Do NOT claim the certificate 'usually' or 'typically' certifies queries -- averaged
   over the full sweep, pairwise coverage is only 1.67%; 
   meaningful coverage only appears at high alpha / late hops.
2. Do NOT claim E5 is a 'worse' retriever -- its lower certification rate reflects a
   mix of larger embedding-driven bounds (vs MiniLM) and smaller score margins (vs BGE),
   not retrieval quality (E5's own NDCG@5 is competitive).
3. Do NOT claim the candidate-pool ablation reflects native benchmark difficulty at
   n=20/50 -- those pools are padded with cross-query distractors.
4. Do NOT claim the certificate covers the noisy/robustness setting -- all these results
   use noise_std=0, the only regime the proof covers.
5. Do NOT present the global certificate's near-zero coverage as a weakness of the theory
   -- it is a strictly more conservative, still-valid corollary; the pairwise form is the
   intended primary certificate.

## Recommended main-table numbers

Use Table 1 (mean over the full alpha/hop sweep) as the headline generality table, and
cite the peak cell (2wikimultihopqa x MiniLM, alpha=0.8, hop=5, 60.1% pairwise) as the best-case, not the average-case, result.

## Recommended figure order

1. Fig 1 (certification vs hop, per alpha) -- establishes the core dynamic.
2. Fig 3 (certification by retriever) -- generality / retriever dependence.
3. Fig 2 (global vs pairwise vs ordered) -- justifies the pairwise certificate as primary.
4. Fig 4 (slack vs P(change)) -- validates the certificate is a meaningful, monotone signal.
5. Fig 6 (bound vs empirical) -- honest conservatism disclosure.
6. Fig 5 (candidate-pool ablation) -- scalability caveat, clearly labeled as a stress test.
