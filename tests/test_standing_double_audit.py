"""Audit precision changes value replay only, never actor evidence or training."""
from pathlib import Path
import torch
import numpy as np
import pytest
from pgmt.train.policy import Stage1Policy
from setup.audit_standing_double_values import DoubleValueAuditPolicy,precision_matched_gae

CAPTURE=Path('runs/standing_multiseed_20260922/physics_v1/seed0_control/captures/update_73_before.pt')


@pytest.mark.skipif(not CAPTURE.exists(),reason='optional recorded numerical regression')
def test_double_value_reference_preserves_action_logprob_and_state_keys():
    b=torch.load(CAPTURE,map_location='cpu',weights_only=False);s=b['storage']
    obs={k:v.flatten(0,1) for k,v in s['observations'].items()}
    p=Stage1Policy().eval();a=DoubleValueAuditPolicy().eval()
    p.load_state_dict(b['policy_before']);a.load_state_dict(b['policy_before'])
    assert p.state_dict().keys()==a.state_dict().keys()
    with torch.no_grad():
        args=(obs,s['actions'].flatten(0,1));kw={'latent_actions':s['latent_actions'].flatten(0,1)}
        original=p.evaluate_actions(*args,**kw);audited=a.evaluate_actions(*args,**kw)
    assert torch.equal(original.actions,audited.actions)
    assert torch.equal(original.log_probs,audited.log_probs)
    torch.testing.assert_close(audited.values,s['values'].flatten(0,1),atol=2e-4,rtol=2e-5)
    assert (audited.values-s['values'].flatten(0,1)).abs().max() < (original.values-s['values'].flatten(0,1)).abs().max()


def test_precision_gae_distinguishes_timeout_bootstrap_and_termination():
    reward=np.ones((2,2,1));values=np.zeros_like(reward);next_values=np.full_like(reward,10.)
    terminated=np.array([[False,True],[False,False]]);timeouts=np.array([[True,False],[False,False]])
    records=[];a=precision_matched_gae(reward,values,next_values,terminated,timeouts,.9,.8,records=records)
    np.testing.assert_allclose(a[0,:,0],[10.,1.],atol=1e-6)
    assert records[0]['max_error_to_bound_ratio']<=1


@pytest.mark.skipif(not CAPTURE.exists(),reason='optional recorded numerical regression')
def test_fp32_gae_matches_recorded_cancellation_without_relaxing_tolerance():
    path=CAPTURE.parent.parent.parent/'seed0_candidate/captures/update_98_before.pt'
    if not path.exists():pytest.skip('candidate capture unavailable')
    s=torch.load(path,map_location='cpu',weights_only=False)['storage']
    m=__import__('json').loads((path.parent.parent/'metrics.json').read_text())['ppo_config']
    a=precision_matched_gae(*(s[k].double().numpy() for k in ('rewards','values','next_values')),
        s['terminated'].numpy(),s['timeouts'].numpy(),m['gamma'],m['lam'])
    torch.testing.assert_close(torch.from_numpy(a),s['advantages'],atol=2e-4,rtol=2e-5)
    corrupted=s['advantages'].clone();corrupted[3,11,2]+=.01
    with pytest.raises(AssertionError):torch.testing.assert_close(torch.from_numpy(a),corrupted,atol=2e-4,rtol=2e-5)
