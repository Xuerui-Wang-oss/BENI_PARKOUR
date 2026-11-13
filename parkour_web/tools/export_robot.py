"""Export the BENI physics model that mjlab compiles for Beni-Parkour-Env2, for MuJoCo WASM, and check it.

    cd open-beni/python
    PYTHONPATH=. python ../parkour_web/tools/export_robot.py

Writes, under open-beni/parkour_web:
    public/robot/beni.xml     the robot with mjlab's edits (motors, joint ref 0, armature, frictionloss, wheel priority,
                              thigh-wheel exclusion, contact sensors) and the simulation options, without the visual
                              meshes. Every number is written at full precision from the compiled model, not from
                              MuJoCo's XML writer (which keeps 6 digits). ``<!--TERRAIN-->`` marks where the lane goes.
    tools/ref/env2_lanes.json every env2 lane (seed 0): its obstacles and its terrain boxes relative to the lane origin,
                              the reference for the JavaScript lane builder.

Then it compiles beni.xml with the terrain of the reference model and compares every numeric field of the two models,
and steps both from the same state with the same torques.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

torch.set_num_threads(2)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from parkour_rl import ENV2_TASK_ID  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
TERRAIN_MARK = "<!--TERRAIN-->"
GEOM_TYPES = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder"}
INTEGRATORS = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}
SOLVERS = {0: "PGS", 1: "CG", 2: "Newton"}
CONES = {0: "pyramidal", 1: "elliptic"}
JACOBIANS = {0: "dense", 1: "sparse", 2: "auto"}


def num(values) -> str:
    """Numbers as text that parses back to the same doubles."""
    return " ".join(repr(float(v)) for v in np.atleast_1d(values))


def flag(value) -> str:
    return "true" if value else "false"


def geom_xml(m: mujoco.MjModel, g: int, name: str | None = None) -> str:
    """One collision geom of ``m`` with every contact parameter written out. Its mass is 0: the robot bodies carry
    explicit inertias, and mjlab's terrain body has none."""
    kind = GEOM_TYPES[int(m.geom_type[g])]
    size = m.geom_size[g][:2] if kind == "cylinder" else m.geom_size[g]
    name = m.geom(g).name if name is None else name
    return (f'<geom{f" name=\"{name}\"" if name else ""} type="{kind}" size="{num(size)}" pos="{num(m.geom_pos[g])}" '
            f'quat="{num(m.geom_quat[g])}" group="{int(m.geom_group[g])}" contype="{int(m.geom_contype[g])}" '
            f'conaffinity="{int(m.geom_conaffinity[g])}" condim="{int(m.geom_condim[g])}" '
            f'priority="{int(m.geom_priority[g])}" friction="{num(m.geom_friction[g])}" '
            f'solref="{num(m.geom_solref[g])}" solimp="{num(m.geom_solimp[g])}" solmix="{num(m.geom_solmix[g])}" '
            f'margin="{num(m.geom_margin[g])}" gap="{num(m.geom_gap[g])}" mass="0"/>')


def body_xml(m: mujoco.MjModel, b: int, indent: str) -> list[str]:
    """Body ``b`` of ``m`` and its subtree: inertia, joints, collision geoms, sites, cameras."""
    lines = [f'{indent}<body name="{m.body(b).name}" pos="{num(m.body_pos[b])}" quat="{num(m.body_quat[b])}">',
             f'{indent}  <inertial pos="{num(m.body_ipos[b])}" quat="{num(m.body_iquat[b])}" '
             f'mass="{num(m.body_mass[b])}" diaginertia="{num(m.body_inertia[b])}"/>']
    for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
        name = m.joint(j).name
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            lines.append(f'{indent}  <joint name="{name}" type="free"/>')
            continue
        assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        d = m.jnt_dofadr[j]
        lines.append(
            f'{indent}  <joint name="{name}" type="hinge" pos="{num(m.jnt_pos[j])}" axis="{num(m.jnt_axis[j])}" '
            f'limited="{flag(m.jnt_limited[j])}" range="{num(m.jnt_range[j])}" '
            f'actuatorfrclimited="{flag(m.jnt_actfrclimited[j])}" actuatorfrcrange="{num(m.jnt_actfrcrange[j])}" '
            f'armature="{num(m.dof_armature[d])}" frictionloss="{num(m.dof_frictionloss[d])}" '
            f'damping="{num(m.dof_damping[d])}" stiffness="{num(m.jnt_stiffness[j])}" ref="0" '
            f'solreflimit="{num(m.jnt_solref[j])}" solimplimit="{num(m.jnt_solimp[j])}" '
            f'solreffriction="{num(m.dof_solref[d])}" solimpfriction="{num(m.dof_solimp[d])}" '
            f'margin="{num(m.jnt_margin[j])}"/>')
    for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:  # The meshes are visual only: no contact, no mass.
            lines.append(indent + "  " + geom_xml(m, g))
    for s in range(m.nsite):
        if m.site_bodyid[s] == b:
            lines.append(f'{indent}  <site name="{m.site(s).name}" pos="{num(m.site_pos[s])}" '
                         f'quat="{num(m.site_quat[s])}" size="0.01"/>')
    for c in range(m.ncam):
        if m.cam_bodyid[c] == b:
            lines.append(f'{indent}  <camera name="{m.camera(c).name}" pos="{num(m.cam_pos[c])}" '
                         f'quat="{num(m.cam_quat[c])}" fovy="{num(m.cam_fovy[c])}"/>')
    for child in range(m.nbody):
        if m.body_parentid[child] == b and child != b:
            lines += body_xml(m, child, indent + "  ")
    lines.append(f"{indent}</body>")
    return lines


def robot_xml(m: mujoco.MjModel, sensors: str) -> str:
    """The physics model of ``m`` as MJCF, with ``TERRAIN_MARK`` in place of the terrain geoms."""
    o = m.opt
    assert o.disableflags == 0 and o.enableflags == 0 and m.neq == 0 and m.ntendon == 0 and m.npair == 0
    lines = [
        '<mujoco model="beni parkour">',
        '  <compiler angle="radian" autolimits="false"/>',
        f'  <option timestep="{num(o.timestep)}" integrator="{INTEGRATORS[int(o.integrator)]}" '
        f'cone="{CONES[int(o.cone)]}" solver="{SOLVERS[int(o.solver)]}" jacobian="{JACOBIANS[int(o.jacobian)]}" '
        f'iterations="{o.iterations}" ls_iterations="{o.ls_iterations}" tolerance="{num(o.tolerance)}" '
        f'ls_tolerance="{num(o.ls_tolerance)}" impratio="{num(o.impratio)}" ccd_iterations="{o.ccd_iterations}" '
        f'ccd_tolerance="{num(o.ccd_tolerance)}" noslip_iterations="{o.noslip_iterations}" '
        f'gravity="{num(o.gravity)}"/>',
        "  <worldbody>",
        f'    <body name="terrain">{TERRAIN_MARK}</body>',
    ]
    terrain = m.body("terrain").id
    for b in range(1, m.nbody):
        if m.body_parentid[b] == 0 and b != terrain:
            lines += body_xml(m, b, "    ")
    lines += ["  </worldbody>", "  <contact>"]
    for signature in m.exclude_signature:
        lines.append(f'    <exclude body1="{m.body(int(signature) >> 16).name}" '
                     f'body2="{m.body(int(signature) & 0xFFFF).name}"/>')
    lines += ["  </contact>", "  <actuator>"]
    for a in range(m.nu):
        assert m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_dyntype[a] == 0
        assert m.actuator_gainprm[a][0] == 1 and not m.actuator_biasprm[a].any() and m.actuator_gear[a][0] == 1
        lines.append(f'    <general name="{m.actuator(a).name}" joint="{m.joint(m.actuator_trnid[a][0]).name}" '
                     f'ctrllimited="{flag(m.actuator_ctrllimited[a])}" ctrlrange="{num(m.actuator_ctrlrange[a])}" '
                     f'forcelimited="{flag(m.actuator_forcelimited[a])}" '
                     f'forcerange="{num(m.actuator_forcerange[a])}"/>')
    lines += ["  </actuator>", sensors, "</mujoco>", ""]
    return "\n".join(lines)


def terrain_geoms_xml(m: mujoco.MjModel) -> str:
    terrain = m.body("terrain").id
    return "".join(geom_xml(m, g) for g in range(m.ngeom) if m.geom_bodyid[g] == terrain)


# Fields that differ only because the exported model has no visual meshes, or that hold names, rendering or sizes.
SKIP_PREFIXES = ("mesh_", "tex", "mat_", "name", "skin", "hfield", "light_", "vis", "key_", "text_", "tuple_",
                 "numeric_", "plugin", "flex", "bvh_", "oct_", "stat", "site_size", "site_rgba", "site_matid",
                 "site_group", "cam_", "body_bvh", "body_geom", "geom_dataid", "geom_matid", "geom_rgba", "geom_bvh",
                 "paths", "sensor_plugin", "actuator_plugin", "body_plugin", "geom_plugin", "wrap_", "eq_", "tendon_",
                 "pair_", "exclude_signature", "body_user", "jnt_user", "geom_user", "site_user", "sensor_user",
                 "actuator_user", "cam_user", "user",
                 # Length scale of each rotational dof, taken from the bounding boxes of the geoms below it, visual
                 # meshes included. MuJoCo reads it only for the sleep tolerance, and sleeping is disabled.
                 "dof_length")


def compare(ref: mujoco.MjModel, new: mujoco.MjModel) -> int:
    """Print every numeric field that differs between the two models; returns how many differ."""
    keep = np.array([t != mujoco.mjtGeom.mjGEOM_MESH for t in ref.geom_type])
    ref_geom_index = np.cumsum(keep) - 1  # Geom id in ``new`` of every kept geom of ``ref``.
    differing, compared = 0, 0
    for name in sorted(dir(ref)):
        if name.startswith("_") or name.startswith(SKIP_PREFIXES):
            continue
        a = getattr(ref, name)
        if not isinstance(a, np.ndarray) or a.dtype.kind not in "fiub":
            continue
        b = getattr(new, name)
        if name.startswith("geom_"):
            a = a[keep]
        if name in ("sensor_objid", "sensor_refid"):  # Site and body ids are equal; geom ids would shift.
            pass
        if a.shape != b.shape:
            print(f"  {name}: shape {a.shape} vs {b.shape}")
            differing += 1
            continue
        compared += 1
        error = np.abs(a.astype(float) - b.astype(float)).max() if a.size else 0.
        if error > 1e-12:
            print(f"  {name}: max difference {error:.3g}")
            differing += 1
    del ref_geom_index
    for name in ("timestep", "integrator", "cone", "solver", "jacobian", "iterations", "ls_iterations", "tolerance",
                 "ls_tolerance", "impratio", "ccd_iterations", "ccd_tolerance", "noslip_iterations", "noslip_tolerance",
                 "gravity", "wind", "density", "viscosity", "o_margin", "disableflags", "enableflags",
                 "sdf_iterations", "sdf_initpoints", "magnetic"):
        a, b = np.asarray(getattr(ref.opt, name), float), np.asarray(getattr(new.opt, name), float)
        compared += 1
        if np.abs(a - b).max() > 0:
            print(f"  opt.{name}: {a} vs {b}")
            differing += 1
    for size in ("nq", "nv", "nu", "nbody", "njnt", "nsite", "ncam", "nsensor", "nsensordata", "nexclude"):
        compared += 1
        if getattr(ref, size) != getattr(new, size):
            print(f"  {size}: {getattr(ref, size)} vs {getattr(new, size)}")
            differing += 1
    assert sorted(ref.exclude_signature) == sorted(new.exclude_signature)
    for c in range(ref.ncam):
        assert np.array_equal(ref.cam_pos[c], new.cam_pos[c]) and np.array_equal(ref.cam_quat[c], new.cam_quat[c])
        assert ref.cam_fovy[c] == new.cam_fovy[c] and ref.cam_bodyid[c] == new.cam_bodyid[c]
    print(f"compared {compared} fields, {differing} differ")
    return differing


def step_both(ref: mujoco.MjModel, new: mujoco.MjModel, origin: np.ndarray, steps: int = 500) -> float:
    """Step both models from the spawn pose on a lane with the same random torques; returns the largest qpos gap."""
    rng = np.random.default_rng(0)
    datas = [mujoco.MjData(m) for m in (ref, new)]
    for m, d in zip((ref, new), datas):
        mujoco.mj_resetDataKeyframe(ref, d, 0) if m is ref else d.qpos.__setitem__(slice(None), ref.key_qpos[0])
        d.qpos[:3] += origin
        mujoco.mj_forward(m, d)
    worst = 0.
    for _ in range(steps):
        ctrl = rng.uniform(-.3, .3, ref.nu)
        for m, d in zip((ref, new), datas):
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)
        worst = max(worst, float(np.abs(datas[0].qpos - datas[1].qpos).max()))
    return worst


def lanes_reference(env: ManagerBasedRlEnv) -> list[dict]:
    """Every lane: row, column, origin, obstacles, and its terrain boxes relative to the origin."""
    m, terrain = env.sim.mj_model, env.scene.terrain
    generator = terrain.cfg.terrain_generator
    body = m.body("terrain").id
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
    centres = m.geom_pos[geoms]
    length, width = generator.size
    lanes = []
    for column, lane in enumerate(generator.sub_terrains.values()):
        for row in range(generator.num_rows):
            origin = terrain.terrain_origins[row, column].numpy().astype(float)
            corner = origin - np.array((1., width / 2, 0.))  # parkour_terrain.SPAWN_X = 1.
            inside = ((centres[:, 0] > corner[0]) & (centres[:, 0] < corner[0] + length)
                      & (centres[:, 1] > corner[1]) & (centres[:, 1] < corner[1] + width))
            lanes.append(dict(
                row=row, column=column, origin=origin.tolist(),
                obstacles=[dict(kind=o["kind"], value=float(o["value"]), x_start=float(o["x_start"]),
                                x_end=float(o["x_end"])) for o in lane.layouts[row]],
                geoms=[dict(pos=(m.geom_pos[g] - origin).tolist(), size=m.geom_size[g].tolist(),
                            quat=m.geom_quat[g].tolist()) for g, hit in zip(geoms, inside) if hit]))
    assert sum(len(lane["geoms"]) for lane in lanes) == len(geoms), "a terrain geom belongs to no lane or to two"
    return lanes


def main() -> None:
    env = ManagerBasedRlEnv(load_env_cfg(ENV2_TASK_ID, play=True), device="cpu")
    ref = env.sim.mj_model
    spec_xml = env.scene.spec.to_xml()
    sensors = spec_xml[spec_xml.index("  <sensor>"):spec_xml.index("</sensor>") + len("</sensor>")]

    xml = robot_xml(ref, sensors)
    (OUT / "public/robot").mkdir(parents=True, exist_ok=True)
    (OUT / "public/robot/beni.xml").write_text(xml)
    print(f"wrote public/robot/beni.xml ({len(xml)} characters)")

    lanes = lanes_reference(env)
    (OUT / "tools/ref").mkdir(parents=True, exist_ok=True)
    (OUT / "tools/ref/env2_lanes.json").write_text(json.dumps(lanes))
    print(f"wrote tools/ref/env2_lanes.json ({len(lanes)} lanes, {sum(len(lane['geoms']) for lane in lanes)} boxes)")

    new = mujoco.MjModel.from_xml_string(xml.replace(TERRAIN_MARK, terrain_geoms_xml(ref)))
    differing = compare(ref, new)
    gap = step_both(ref, new, env.scene.env_origins[0].numpy().astype(float))
    print(f"500 steps with the same torques: largest qpos difference {gap:.3g}")
    env.close()
    if differing or gap > 1e-9:
        raise SystemExit("the exported model differs from the reference")
    print("exported model equals the reference")


if __name__ == "__main__":
    main()
