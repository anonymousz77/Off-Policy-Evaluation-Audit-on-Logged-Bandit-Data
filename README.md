# Off-Policy Evaluation Audit on Logged Bandit Data

I take a logged bandit dataset where the true value of the evaluation policy is actually known, hide it, ask four off-policy estimators to recover it from logs collected under a different policy, and measure how far each one lands from the answer — so the question "which OPE estimator should I have trusted here?" gets a measured answer instead of an argument from theory.

---

<!-- ------------------------------------------------------------------ -->
<!-- ATTRIBUTION NOTICE — PASTE HERE                                      -->
<!-- ------------------------------------------------------------------ -->

<!-- ------------------------------------------------------------------ -->

---

## What this adds

Run five off-policy estimators on one batch of logged bandit data and you get five different numbers. IPS is unbiased when the logging propensities are known and the evaluation policy's support is covered, but a handful of large importance weights can dominate the sum and send the variance up. DM has low variance and inherits whatever bias its reward model carries; if the model is misspecified in the region the evaluation policy favors, DM is confidently wrong. DR combines the two and is unbiased if either component is right, but it does not escape the weight problem. SNIPS self-normalizes to trade a little bias for stability. Switch-DR cuts over to the reward model wherever the weight exceeds a threshold, and the threshold is a knob nobody can set from the logs alone.

These are all defensible. They disagree. The standard way to break the tie is to reason about which assumption is least likely violated — which is a judgment call, not a measurement, because the quantity that would settle it is the true value of the evaluation policy, and that is exactly the thing you cannot observe. That is the whole reason you were doing off-policy evaluation in the first place.

The Open Bandit Dataset is unusual in a way that dissolves this. ZOZO ran two policies concurrently on the same platform over the same period: uniform random, and Bernoulli Thompson Sampling. That means the logs contain a policy's own on-policy performance *and* a separate batch of logs collected under a different policy. So I can construct an answer key.

The audit does this:

1. Take the BTS logs and average the observed reward. That is an on-policy estimate of V(BTS) — a real measured quantity from a real deployment, not a simulation.
2. Set it aside. Load the **random-policy** logs, and treat BTS as the evaluation policy.
3. Ask each estimator to recover V(BTS) from the random logs alone.
4. Compare each estimate to the number from step 1.

What comes out is a relative estimation error per estimator, plus a 95% bootstrap confidence interval on each estimate. The error is the headline; the interval is what makes it usable.

Two things make the resulting number worth more than a synthetic benchmark score.

**The propensities are exact.** The logging policy is uniform random, so the pscore is known by construction, not estimated. Every assumption IPS needs about propensities holds. Whatever error IPS shows is therefore variance and finite-sample noise — not propensity misspecification quietly contaminating the comparison. That isolates the bias/variance tradeoff, which is the thing actually under test.

**The evaluation policy is reconstructible.** BTS is not a black box here; `obp` reproduces the exact ZOZOTOWN prior that was deployed, so the evaluation policy in the audit is the policy that actually ran, not a stand-in.

I report bootstrap intervals alongside the point estimates because ranking estimators on point error alone is misleading. An estimator can sit close to ground truth on this particular draw and still be unusable, if its interval is wide enough to span the decision you would make with it — say, whether the new policy beats the incumbent. The converse also happens: a visibly biased estimator with a tight interval can still order two policies correctly. Point error tells you accuracy. The interval tells you whether the accuracy would have survived a different week of logs.

A caveat I want stated plainly rather than buried. The dataset bundled in `obd/` is the 10,000-record-per-condition sample that ships with the upstream repo, not the full release. The ground truth in step 1 is itself an average over 10,000 records, so it carries its own sampling error, and the estimator errors measured against it inherit that. The audit runs end to end on the sample and the ranking it produces is real, but tight quantitative claims need the full dataset from [research.zozo.com/data.html](https://research.zozo.com/data.html) dropped in at the same path.

## Method

The ground truth comes from `OpenBanditDataset.calc_on_policy_policy_value_estimate`, called with `behavior_policy="bts"` and `campaign="all"`. This is the mean factual reward over the BTS logs — what the policy actually earned when it ran.

The logged data is a separate `OpenBanditDataset` with `behavior_policy="random"`, same campaign. `obtain_batch_bandit_feedback()` returns the context, action, reward, position and pscore arrays. Because logging was uniform random, the pscore is exact.

The evaluation policy is `BernoulliTS` with `is_zozotown_prior=True` and `campaign="all"`, which loads the production prior from `obp/policy/conf/prior_bts.yaml`. Thompson Sampling has no closed-form action-selection probability, so I get the action distribution by Monte Carlo: `compute_batch_action_dist(n_sim=100000)`.

DM and DR need a reward model. I use `RegressionModel` wrapping `sklearn`'s `LogisticRegression`, fit through `fit_predict` with `n_folds=3`. The cross-fitting matters — without it the reward model is fit on the same rows its predictions are evaluated on, which biases DM optimistically and breaks DR's guarantee.

The estimators are passed to `OffPolicyEvaluation` as a list:

- `InverseProbabilityWeighting`
- `SelfNormalizedInverseProbabilityWeighting`
- `DirectMethod`
- `DoublyRobust`

Results come out of three calls. `summarize_estimators_comparison(ground_truth_policy_value=...)` produces the relative-error table against ground truth. `summarize_off_policy_estimates(n_bootstrap_samples=100)` produces the estimates and their 95% intervals. `visualize_off_policy_estimates(is_relative=True)` writes `ope_audit.png`. Everything is seeded with `random_state=12345`, including the reward model, the Monte Carlo action distribution, and the bootstrap.

The current run covers four estimators. The upstream library implements more that belong in this comparison and that I have not yet added: `SwitchDoublyRobust` and `DoublyRobustWithShrinkage` (`obp/ope/estimators.py`), `SelfNormalizedDoublyRobust`, and the hyperparameter-tuned variants in `obp/ope/estimators_tuning.py` — `SwitchDoublyRobustTuning`, `DoublyRobustWithShrinkageTuning`. Adding one is a line in the `ope_estimators` list; the tuned variants additionally take a candidate list for the threshold or shrinkage parameter, which is the honest way to handle a knob you cannot set a priori.

## Reproducing

Python 3.11.

```
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements-working.txt
```

`requirements-working.txt` was written by `pip freeze` under PowerShell and is UTF-16 encoded, which some pip versions reject. If the install errors on the first line, re-save it as UTF-8:

```
python -c "open('requirements-working.txt','w',encoding='utf-8').write(open('requirements-working.txt',encoding='utf-16').read())"
```

Then run the audit from the repo root:

```
python audit/run_audit.py
```

It reads `obd/` in place, prints the ground truth, the relative-error table and the bootstrap intervals to stdout, and writes `ope_audit.png`. On the bundled sample it is a few minutes, dominated by the 100,000-simulation action distribution and the 3-fold reward model. Nothing needs a GPU.

To run it against the full dataset instead of the sample, download from [research.zozo.com/data.html](https://research.zozo.com/data.html) and replace `obd/` with the same directory structure. The script takes the path through `data_path="obd"` and needs no other change.

## Citation

The dataset and the pipeline are from Saito et al. Cite the paper when using either:

```bibtex
@article{saito2020open,
  title={Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation},
  author={Saito, Yuta and Aihara, Shunsuke and Matsutani, Megumi and Narita, Yusuke},
  journal={arXiv preprint arXiv:2008.07146},
  year={2020}
}
```
