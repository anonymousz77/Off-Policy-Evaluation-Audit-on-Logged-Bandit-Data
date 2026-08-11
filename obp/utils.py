import sys, types

class _T:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return self

def _mk(name):
    m = types.ModuleType(name)
    m.__getattr__ = lambda n: _T
    return m

_t = _mk("torch")
_t.Tensor = _T
_t.tensor = _T
_t.nn = _mk("torch.nn")
_t.nn.Module = _T
_t.nn.functional = _mk("torch.nn.functional")
_t.optim = _mk("torch.optim")
_t.utils = _mk("torch.utils")
_t.utils.data = _mk("torch.utils.data")
_t.distributions = _mk("torch.distributions")

for _k, _v in {
    "torch": _t,
    "torch.nn": _t.nn,
    "torch.nn.functional": _t.nn.functional,
    "torch.optim": _t.optim,
    "torch.utils": _t.utils,
    "torch.utils.data": _t.utils.data,
    "torch.distributions": _t.distributions,
}.items():
    sys.modules[_k] = _v

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from obp.dataset import OpenBanditDataset
from obp.policy import BernoulliTS
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting,
    SelfNormalizedInverseProbabilityWeighting,
    DirectMethod,
    DoublyRobust,
)

CAMPAIGN = "all"
N_BOOT = 100
SEED = 12345

ground_truth = OpenBanditDataset.calc_on_policy_policy_value_estimate(
    behavior_policy="bts", campaign=CAMPAIGN, data_path="obd"
)
print(f"\nGround truth V(BTS) = {ground_truth:.6f}\n")

rand = OpenBanditDataset(behavior_policy="random", campaign=CAMPAIGN, data_path="obd")
bandit_feedback = rand.obtain_batch_bandit_feedback()
print(f"Logged rounds (random policy): {rand.n_rounds}, actions: {rand.n_actions}\n")

policy = BernoulliTS(
    n_actions=rand.n_actions,
    len_list=rand.len_list,
    is_zozotown_prior=True,
    campaign=CAMPAIGN,
    random_state=SEED,
)
action_dist = policy.compute_batch_action_dist(n_sim=100000)

reg = RegressionModel(
    n_actions=rand.n_actions,
    len_list=rand.len_list,
    base_model=LogisticRegression(max_iter=1000, random_state=SEED),
)
est_rewards = reg.fit_predict(
    context=bandit_feedback["context"],
    action=bandit_feedback["action"],
    reward=bandit_feedback["reward"],
    position=bandit_feedback["position"],
    pscore=bandit_feedback["pscore"],
    n_folds=3,
    random_state=SEED,
)

ope = OffPolicyEvaluation(
    bandit_feedback=bandit_feedback,
    ope_estimators=[
        InverseProbabilityWeighting(),
        SelfNormalizedInverseProbabilityWeighting(),
        DirectMethod(),
        DoublyRobust(),
    ],
)

print("=" * 60)
print("RELATIVE ESTIMATION ERROR vs GROUND TRUTH")
print("=" * 60)
print(ope.summarize_estimators_comparison(
    ground_truth_policy_value=ground_truth,
    action_dist=action_dist,
    estimated_rewards_by_reg_model=est_rewards,
))

print("\n" + "=" * 60)
print("ESTIMATES WITH 95% BOOTSTRAP CIs")
print("=" * 60)
ci = ope.summarize_off_policy_estimates(
    action_dist=action_dist,
    estimated_rewards_by_reg_model=est_rewards,
    n_bootstrap_samples=N_BOOT,
    random_state=SEED,
)
print(ci[0])
print()
print(ci[1])

ope.visualize_off_policy_estimates(
    action_dist=action_dist,
    estimated_rewards_by_reg_model=est_rewards,
    n_bootstrap_samples=N_BOOT,
    random_state=SEED,
    is_relative=True,
    fig_dir=".",
    fig_name="ope_audit.png",
)
print("\nSaved ope_audit.png")