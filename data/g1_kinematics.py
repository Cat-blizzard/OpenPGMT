"""Explicit URDF geometry for offline G1 FK, IK and static-load diagnostics.

No global skeleton mutation. The legacy XML model remains independently usable.
"""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from data.bvh import quat_mul, quat_rot_vec
from data.retarget_lafan1 import G1_JOINT_NAMES


class G1Kinematics:
    def __init__(self, urdf):
        self.path = Path(urdf)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        tree = ET.parse(self.path).getroot()
        self.links = {link.get('name'): link for link in tree.findall('link')}
        self.joints = {}
        self.actuated = {}
        for joint in tree.findall('joint'):
            parent, child = (joint.find(k).get('link') for k in ('parent', 'child'))
            typ = joint.get('type')
            if typ not in ('fixed', 'revolute'):
                raise ValueError(f'unsupported G1 joint type: {typ}')
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            quat = Rotation.from_euler('xyz', rpy).as_quat()[[3, 0, 1, 2]]
            axis_node = joint.find('axis')
            axis = np.fromstring(axis_node.get('xyz', '1 0 0'), sep=' ') if axis_node is not None else np.array([1., 0, 0])
            axis /= np.linalg.norm(axis)
            name = joint.get('name').removesuffix('_joint')
            self.joints[child] = (parent, name, typ, xyz, quat, axis)
            if typ != 'fixed':
                if name not in G1_JOINT_NAMES:
                    raise ValueError(f'unknown actuator: {name}')
                self.actuated[name] = child
        roots = set(self.links)-set(self.joints)
        if roots != {'pelvis'} or set(self.actuated) != set(G1_JOINT_NAMES):
            raise ValueError('URDF does not match the G1 joint/root contract')
        self.order = ['pelvis']
        pending = set(self.joints)
        while pending:
            ready = sorted(n for n in pending if self.joints[n][0] in self.order)
            if not ready:
                raise ValueError('disconnected URDF')
            self.order.extend(ready)
            pending.difference_update(ready)
        self.ancestors = {}
        for body in self.order:
            chain = []
            cursor = body
            while cursor != 'pelvis':
                chain.append(cursor)
                cursor = self.joints[cursor][0]
            self.ancestors[body] = chain
        p, q = self.forward(np.zeros((1, 29)))
        self.rest = {n: (p[n][0], q[n][0]) for n in self.order}
        foot_bottoms = []
        for name in ('left_ankle_roll_link', 'right_ankle_roll_link'):
            for collision in self.links[name].findall('collision'):
                sphere = collision.find('geometry/sphere')
                if sphere is None:
                    raise ValueError('expected collision-sphere G1 feet')
                center = np.fromstring(collision.find('origin').get('xyz'), sep=' ')
                point = p[name][0]+quat_rot_vec(q[name][0], center)
                foot_bottoms.append(point[2]-float(sphere.get('radius')))
        self.neutral_height = -min(foot_bottoms)

    def forward(self, qpos, root_pos=None, root_quat=None):
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 29 or not np.isfinite(qpos).all():
            raise ValueError('expected finite (frames,29) joint positions')
        n = len(qpos)
        p = {'pelvis': np.zeros((n, 3)) if root_pos is None else np.asarray(root_pos, dtype=np.float64).copy()}
        q = {'pelvis': np.tile([1., 0, 0, 0], (n, 1)) if root_quat is None else np.asarray(root_quat, dtype=np.float64).copy()}
        q['pelvis'] /= np.linalg.norm(q['pelvis'], axis=-1, keepdims=True)
        for child in self.order[1:]:
            parent, name, typ, xyz, quat, axis = self.joints[child]
            local = np.broadcast_to(quat, (n, 4))
            if typ != 'fixed':
                angle = qpos[:, G1_JOINT_NAMES.index(name)]/2
                rot = np.concatenate((np.cos(angle)[:, None], np.sin(angle)[:, None]*axis), -1)
                local = quat_mul(local, rot)
            p[child] = p[parent]+quat_rot_vec(q[parent], xyz)
            q[child] = quat_mul(q[parent], local)
        return p, q

    def jacobian(self, pos, quat, bodies, joint_names=None, local_points=None):
        names = G1_JOINT_NAMES if joint_names is None else joint_names
        n = len(pos['pelvis'])
        result = np.zeros((n, 3*len(bodies), len(names)))
        for k, body in enumerate(bodies):
            point = pos[body]
            if local_points is not None:
                point = point+quat_rot_vec(quat[body], np.asarray(local_points[k]))
            for j, name in enumerate(names):
                child = self.actuated[name]
                if child not in self.ancestors[body]:
                    continue
                axis = quat_rot_vec(quat[child], self.joints[child][-1])
                result[:, 3*k:3*k+3, j] = np.cross(axis, point-pos[child])
        return result

    def finalize_reference(self, data):
        """One rigid full-sequence height shift, using actual collision support.

        Never lower individual frames or relabel actual rollout contacts.
        """
        import json
        import torch
        from pgmt.envs.reset_geometry import CollisionFloor
        from data.build_mesh_contacts import ReferenceMesh, sphere_geometry, contact_labels, CONTACT_SOURCE
        floor = CollisionFloor(self.path, self.order)
        n = len(data['qpos'])
        clearance = []
        for start in range(0, n, 128):
            sl = slice(start, start+128)
            p, q = self.forward(data['qpos'][sl], data['root_pos'][sl], data['root_rot'][sl])
            bp = torch.tensor(np.stack([p[k] for k in self.order], 1), dtype=torch.float32)
            bq = torch.tensor(np.stack([q[k] for k in self.order], 1), dtype=torch.float32)
            clearance.append(floor.min_height(bp, bq))
        shift = -float(torch.cat(clearance).min())
        result = dict(data)
        result['root_pos'] = data['root_pos'].copy()
        result['root_pos'][:, 2] += shift
        mesh, geometry = ReferenceMesh(), sphere_geometry(self.path)
        contacts = []
        for start in range(0, n, 512):
            sl = slice(start, start+512)
            p, q = self.forward(data['qpos'][sl], result['root_pos'][sl], data['root_rot'][sl])
            bp = torch.tensor(np.stack([p[k] for k in self.order], 1), dtype=torch.float32)
            bq = torch.tensor(np.stack([q[k] for k in self.order], 1), dtype=torch.float32)
            contacts.append(contact_labels(bp, bq, self.order, geometry, mesh))
        result.update(contacts=torch.cat(contacts).numpy(), contact_source=np.array(CONTACT_SOURCE),
            reference_frame_contract=np.array('flat_ground_v1'), reference_ground_z=np.array(0., np.float32),
            kinematic_urdf_sha256=np.array(self.sha256),
            full_sequence_height_shift_m=np.array(shift),
            contact_provenance=np.array(json.dumps({'source': CONTACT_SOURCE, 'mesh': mesh.provenance,
                'tolerance_m': .02, 'urdf_sha256': self.sha256, 'height_anchor': 'full_sequence_collision_minimum'})))
        return result
