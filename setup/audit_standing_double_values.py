"""CPU-only float64 value replay with the original audit tolerances unchanged.

At update 73, float32 CPU inference differed from the saved CUDA critic value
by 4.88e-4; float64 replay reduced the worst error to 9.98e-5. Actor/log-prob,
rewards, GAE and policy KL still use the original independent audit paths.
The physical training code and saved weights are never modified.
"""
from unittest.mock import patch

import numpy as np
import torch

from pgmt.train.policy import Stage1Policy
from setup.analyze_standing_learning import audit_run as original_audit_run
from setup.audit_captured_rollout import independent_gae
from pgmt.train.train_stage1 import _write_metrics


class DoubleValueAuditPolicy(Stage1Policy):
    def __init__(self):
        super().__init__()
        # Keep the independent reference outside the registered module tree:
        # loading/checking the training state_dict must retain its exact keys.
        object.__setattr__(self,'double_reference',Stage1Policy().double().eval())

    def load_state_dict(self,state_dict,strict=True,assign=False):
        result=super().load_state_dict(state_dict,strict=strict,assign=assign)
        self.double_reference.load_state_dict(state_dict,strict=strict)
        return result

    def value(self,observations):
        return self.double_reference.value({k:v.double() for k,v in observations.items()}).float()

    def evaluate_actions(self,observations,actions,*,latent_actions=None):
        result=super().evaluate_actions(observations,actions,latent_actions=latent_actions)
        return result._replace(values=self.value(observations))


def precision_matched_gae(rewards,values,next_values,terminated,timeouts,gamma,lam,*,records=None):
    """Independent NumPy FP32 recurrence plus an FP64 forward-error check.

    Near-zero advantages can result from cancellation of values above 500.
    Matching operation precision tests the executed algorithm without widening
    the old comparison tolerance. A separate FP64 oracle bounds roundoff by
    operand magnitudes through the finite, masked recurrence.
    """
    ideal=independent_gae(rewards,values,next_values,terminated,timeouts,gamma,lam)
    r,v,n=[np.asarray(x,dtype=np.float32) for x in (rewards,values,next_values)]
    result=np.zeros_like(r);carry=np.zeros_like(r[0]);bound=np.zeros_like(rewards,dtype=float)
    carried_bound=np.zeros_like(rewards[0],dtype=float);ideal_carry=np.zeros_like(rewards[0],dtype=float)
    eps=np.finfo(np.float32).eps;g=np.float32(gamma);gl=np.float32(gamma*lam)
    for t in range(len(r)-1,-1,-1):
        boot=(~terminated[t])[:,None];cont=(~(terminated[t]|timeouts[t]))[:,None]
        delta=r[t]+g*boot*n[t]-v[t]
        carry=delta+gl*cont*carry;result[t]=carry
        exact_delta=rewards[t]+gamma*boot*next_values[t]-values[t]
        delta_bound=4*eps*(abs(rewards[t])+gamma*boot*abs(next_values[t])+abs(values[t]))
        carried_bound=(delta_bound+gamma*lam*cont*carried_bound+
            4*eps*(abs(exact_delta)+gamma*lam*cont*abs(ideal_carry)))
        bound[t]=carried_bound;ideal_carry=ideal[t]
    difference=abs(result.astype(float)-ideal)
    if (difference>bound+1e-12).any():raise ValueError('GAE discrepancy exceeds operand-based FP32 roundoff bound')
    if records is not None:records.append({'max_float32_float64_difference':float(difference.max()),
        'max_forward_error_bound':float(bound.max()),'max_error_to_bound_ratio':float((difference/(bound+1e-12)).max())})
    return result


def audit_run(folder,output):
    records=[]
    def gae(*args):return precision_matched_gae(*args,records=records)
    with patch('setup.analyze_standing_learning.Stage1Policy',DoubleValueAuditPolicy), \
         patch('setup.analyze_standing_learning.independent_gae',gae):
        result=original_audit_run(folder,output)
    result['gae_float64_roundoff_checks']=records
    result['replay_precision']='CPU float64 critic; independent NumPy float32 GAE with additional float64 forward-error bound; original tolerances retained'
    _write_metrics(output/'summary.json',result)
    return result
