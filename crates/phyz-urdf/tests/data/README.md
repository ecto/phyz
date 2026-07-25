# Vendored test fixtures

Real robot descriptions, used to validate the URDF importer against models that
were not written for phyz.

| File | Source | License |
| --- | --- | --- |
| `panda.urdf` | [bulletphysics/bullet3](https://github.com/bulletphysics/bullet3) `examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf`, generated from `franka_ros`' `panda_arm_hand.urdf.xacro` | Apache-2.0 (franka_ros) |
| `kuka_iiwa.urdf` | [bulletphysics/bullet3](https://github.com/bulletphysics/bullet3) `examples/pybullet/gym/pybullet_data/kuka_iiwa/model.urdf` | Zlib (bullet3) |

Both are unmodified. Mesh files are *not* vendored — the importer only records
mesh references (see `UrdfModel::mesh_refs`), so the `.urdf` files are
self-sufficient for these tests.
