import torch
from omegaconf import OmegaConf

from rlinf.models import get_model, register_model


class _TinyLoraPolicy(torch.nn.Module):
    """Small policy exposing the same target-module names used by StarVLA."""

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.fc1 = torch.nn.Linear(4, 2)

    def forward(self, inputs):
        return self.fc1(self.q_proj(inputs))


def _build_tiny_lora_policy(_cfg, _torch_dtype):
    return _TinyLoraPolicy()


def test_v03_full_state_checkpoint_round_trip_preserves_lora_weights():
    """RLinf v0.3 can rebuild PEFT and strictly load its full state dict."""
    register_model(
        "crl_lora_v03_test",
        _build_tiny_lora_policy,
        force=True,
    )
    cfg = OmegaConf.create(
        {
            "model_type": "crl_lora_v03_test",
            "precision": "fp32",
            "is_lora": True,
            "lora_rank": 2,
            "lora_path": None,
        }
    )

    first = get_model(cfg)
    trainable_names = [
        name for name, param in first.named_parameters() if param.requires_grad
    ]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)

    with torch.no_grad():
        for name, param in first.named_parameters():
            if "lora_B" in name:
                param.fill_(0.125)

    saved_state = first.state_dict()
    assert any("base_layer.weight" in name for name in saved_state)
    assert any("lora_B" in name for name in saved_state)
    second = get_model(cfg)
    second.load_state_dict(saved_state, strict=True)

    for name, expected in saved_state.items():
        torch.testing.assert_close(second.state_dict()[name], expected)
