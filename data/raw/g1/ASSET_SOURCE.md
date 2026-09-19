# G1 asset source policy

The checked-in MJCF files are **metadata-only** descriptions: their 36 STL
references are intentionally not vendored.  A complete local candidate is
available in the adjacent ProtoMotions checkout (when that checkout is
licensed for the user):

```text
/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets/
├── urdf/g1.urdf
├── mjcf/g1.xml
├── usd/g1.usd
└── mesh/G1/*.stl
```

The candidate has 29 revolute joints in the project order, 29 MJCF motors, and
all mesh paths resolve.  Verify the copy or mount on each machine instead of
committing these binaries:

```bash
python setup/check_g1_asset.py \
  --urdf /path/to/protomotions/data/assets/urdf/g1.urdf \
  --mjcf /path/to/protomotions/data/assets/mjcf/g1.xml \
  --usd /path/to/protomotions/data/assets/usd/g1.usd \
  --json /tmp/g1_asset_manifest.json
```

The ProtoMotions checkout contains the NVIDIA `LICENSE` at its project root.
The G1 mesh directory contains Unitree's BSD-3-Clause `LICENSE` and a README
that identifies the original Unitree source.  Keep both notices with any
private deployment and review the NVIDIA license's non-commercial research or
evaluation restriction before redistributing or using the asset commercially.

The USD check in the script only verifies presence and hash.  Joint and drive
names must still be inspected by an Isaac Sim runtime probe before training;
the normal Python environment does not expose `pxr` without bootstrapping
Isaac Sim.
