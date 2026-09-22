"""Re-query flat reference contacts after nominal vertical placement, on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np

from pgmt.train.fixed_clip_diagnostic import load_manifest, sha256
from pgmt.train.train_stage1 import _write_metrics
from setup.audit_reference_control import SupportGeometry


def run(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = load_manifest(a.manifest, a.reference_data, a.urdf, a.asset)
    geometry = SupportGeometry(a.urdf)
    rows = []
    for clip in manifest['clips']:
        with np.load(a.reference_data / (clip['sequence']+'.npz')) as z:
            start = clip['start_frame']
            stop = start + round(clip['horizon_s']/float(z['frame_time']))
            sl = slice(start, stop)
            rp, q, rq, labels = (z[k][sl].copy() for k in ('root_pos', 'qpos', 'root_rot', 'contacts'))
        # Production ordinary nominal reset places the reference root at
        # default_root_pos. Yaw/xy placement leaves flat-plane clearance intact.
        shift = a.root_height-rp[0, 2]
        rp[:, 2] += shift
        g = geometry.compute(q, rp, rq)
        contacts = (g['clearance'] <= .02).any(-1)
        rows.append({'clip_id': clip['clip_id'], 'nominal_reference_z_shift_m': float(shift),
            'source_contact_fraction': labels.mean(0).tolist(),
            'placed_reference_foot_contact_fraction': contacts.mean(0).tolist(),
            'source_label_mismatch_fraction': float((contacts != labels).mean()),
            'first_frame_foot_clearance_m': g['clearance'][0].min(-1).tolist()})
    _write_metrics(a.output, {'status': 'completed', 'scope': 'CPU nominal reference placement vs unchanged flat floor',
        'root_height_m': a.root_height, 'source_sha256': sha256(__file__),
        'manifest_sha256': sha256(a.manifest), 'urdf_sha256': sha256(a.urdf), 'rows': rows,
        'notes': ['This is the placed reference, not the actual nominal robot state with q=0.',
                  'No reference_state reset collision lift is applied.',
                  'These contact labels enter Stage 2 terrain-contact reward; this does not explain Stage 1 failure.',
                  'A consistent floor/placement convention is needed; do not silently relabel the old corpus.']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'reference-data', 'urdf', 'asset', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--root-height', type=float, default=.793)
    run(p.parse_args())
