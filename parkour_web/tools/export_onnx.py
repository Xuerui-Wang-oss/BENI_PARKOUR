"""Export the four networks of the parkour hierarchy to ONNX and write test vectors for them.

    cd open-beni/python
    PYTHONPATH=. python ../parkour_web/tools/export_onnx.py [high-level checkpoint]

Writes, under open-beni/parkour_web:
    public/models/jump.onnx, crawl.onnx   the frozen low-level actors: obs (1, 40), h_in (1, 1, 512) ->
                                          actions (1, 7), h_out (1, 1, 512). The input normaliser is inside.
    public/models/policy.onnx             the high-level actor: obs (1, 464) = 24 state numbers then the 440 depth
                                          features, h_in (1, 1, 256) -> actions (1, 3) (the mean), h_out.
    public/models/defm.onnx               the frozen DeFM depth encoder with its preprocessing: depth (1, 60, 80) in
                                          metres, 0 = no return -> features (1, 440).
    tools/ref/onnx_vectors.json           inputs and the PyTorch outputs for each network; check_onnx.mjs runs the
                                          ONNX files on them with the runtime the browser uses.

The default checkpoint is model_1000.pt of the env2 run, the one chosen for the demo.
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from rsl_rl.models import RNNModel
from tensordict import TensorDict
from torch import nn

from parkour_rl.depth import DEFM_MEAN, DEFM_STD, DEPTH_SIZE, FEATURE_DIM, UPSAMPLE, DepthEncoder
from parkour_rl.hierarchy import CRAWL_POLICY, JUMP_POLICY, load_low_level_policy

OUT = Path(__file__).resolve().parents[1]
HL_CHECKPOINT = "logs/rsl_rl/beni_parkour_env2/2026-11-10_23-34-36/model_1000.pt"
HL_STATE_DIM = 24
# The actor of Beni-Parkour (parkour_rl/rl_cfg.py), as arguments of rsl_rl's RNNModel.
HL_MODEL = dict(hidden_dims=(256, 128), activation="elu", obs_normalization=True,
                rnn_type="gru", rnn_hidden_dim=256, rnn_num_layers=1)


def load_high_level_policy(path: str) -> RNNModel:
    """The actor of the high-level checkpoint ``path``; its input is hl_actor (24) followed by depth (440)."""
    policy = RNNModel(
        obs=TensorDict({"hl_actor": torch.zeros(1, HL_STATE_DIM), "depth": torch.zeros(1, FEATURE_DIM)},
                       batch_size=[1]),
        obs_groups={"actor": ["hl_actor", "depth"]}, obs_set="actor", output_dim=3,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"}, **HL_MODEL)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    return policy.eval().requires_grad_(False)


class DepthEncoderForExport(nn.Module):
    """``DepthEncoder`` for one image, with ``preprocess_depth_batch`` written in operations that ONNX has.

    The input holds no NaN or infinity (the browser writes 0 for a ray that hits nothing), so ``nan_to_num`` is left
    out; the resize is ``F.interpolate``, which equals torchvision's bilinear ``Resize`` when it enlarges.
    """

    def __init__(self, encoder: DepthEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("mean", torch.tensor(DEFM_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(DEFM_STD).view(1, 3, 1, 1))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        height, width = DEPTH_SIZE
        log_depth = torch.log(1. + depth.view(1, 1, height, width).clamp(0., 100.))
        c1 = log_depth / math.log1p(100.)
        c2 = (log_depth / math.log1p(9.)).clamp(0., 1.)
        low, high = log_depth.min(), log_depth.max()
        spread = high - low
        c3 = torch.where(spread > 0., (log_depth - low) / torch.where(spread > 0., spread, torch.ones_like(spread)),
                         torch.zeros_like(log_depth))
        x = torch.cat([c1, c2, c3], dim=1)
        x = F.interpolate(x, size=(UPSAMPLE * height, UPSAMPLE * width), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        features = self.encoder.backbone(x)["c5"]
        return self.encoder.norm_cls(self.encoder.global_pool(features).flatten(1))


def export_recurrent(policy: RNNModel, path: Path) -> nn.Module:
    """Export ``policy`` as rsl_rl's ONNX wrapper (as mjlab's export_policy_to_onnx does); returns the wrapper."""
    wrapper = policy.as_onnx(verbose=False).to("cpu").eval()
    torch.onnx.export(wrapper, wrapper.get_dummy_inputs(), str(path), export_params=True, opset_version=18,
                      input_names=wrapper.input_names, output_names=wrapper.output_names, dynamic_axes={},
                      dynamo=False)
    return wrapper


def recurrent_vectors(wrapper: nn.Module, observations: torch.Tensor) -> dict:
    """Run ``wrapper`` over the sequence ``observations`` (T, D) from a zero memory; inputs and outputs as lists."""
    hidden = torch.zeros(1, 1, wrapper.hidden_size)
    actions = []
    with torch.inference_mode():
        for observation in observations:
            action, hidden = wrapper(observation[None], hidden)[:2]
            actions.append(action[0].tolist())
    return dict(obs=observations.tolist(), actions=actions, hidden_size=wrapper.hidden_size)


def synthetic_depth(generator: torch.Generator) -> torch.Tensor:
    """Depth images (4, 60, 80) like the robot's: flat ground below the horizon, a step, sky (0) above."""
    rows, columns = DEPTH_SIZE
    v = (torch.arange(rows) + .5 - rows / 2) / 31.875  # tan of the angle below the optical axis, per row.
    ground = torch.where(v > .02, .094 / v.clamp(min=.02), torch.zeros(rows))  # Camera 9.4 cm above the ground.
    images = ground[None, :, None].repeat(4, 1, columns)
    images[1] = images[1].clamp(max=1.5)  # A wall 1.5 m ahead: also covers the sky rows.
    images[1][images[1] == 0] = 1.5
    images[2] = images[2] * (1. + .2 * torch.rand(rows, columns, generator=generator))
    images[3] = 0.  # Nothing in view: the min-max channel has no spread.
    return images.float()


def main() -> None:
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else HL_CHECKPOINT
    models = OUT / "public/models"
    models.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)
    vectors = {}

    for name, path in (("jump", JUMP_POLICY), ("crawl", CRAWL_POLICY)):
        wrapper = export_recurrent(load_low_level_policy(path, torch.device("cpu")), models / f"{name}.onnx")
        observations = torch.randn(30, 40, generator=generator)
        observations[:, 7] = 0.  # The lateral velocity command is always 0; its normaliser std is 0.
        vectors[name] = recurrent_vectors(wrapper, observations)

    wrapper = export_recurrent(load_high_level_policy(checkpoint), models / "policy.onnx")
    vectors["policy"] = recurrent_vectors(wrapper, torch.randn(30, HL_STATE_DIM + FEATURE_DIM, generator=generator))

    encoder = DepthEncoder()
    export = DepthEncoderForExport(encoder).eval()
    depth = synthetic_depth(generator)
    with torch.inference_mode():
        reference = encoder(depth)  # The training code path.
        exported = torch.cat([export(image[None]) for image in depth])
    gap = (reference - exported).abs().max().item()
    print(f"depth encoder: export module against the training module, largest feature difference {gap:.2e} "
          f"(feature scale {reference.abs().mean().item():.2f})")
    assert gap < 1e-3
    torch.onnx.export(export, depth[:1], str(models / "defm.onnx"), export_params=True, opset_version=18,
                      input_names=["depth"], output_names=["features"], dynamic_axes={}, dynamo=False)
    vectors["defm"] = dict(depth=depth.flatten(1).tolist(), features=reference.tolist())

    (OUT / "tools/ref").mkdir(parents=True, exist_ok=True)
    (OUT / "tools/ref/onnx_vectors.json").write_text(json.dumps(vectors))
    for file in sorted(models.glob("*.onnx")):
        print(f"{file.name}: {file.stat().st_size / 1e6:.1f} MB")
    print(f"high-level checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
