import unittest
import tempfile
import io
from pathlib import Path

import torch

from features.dissonance_adapter import DissonanceFeatureAdapter, FEATURE_VERSION
from llama.llama_adapter import LLaMA_adapter
from llama.dissonance_modules import (
    DSTemporalEncoder,
    GatedResidualFusion,
    TemporalGatedAttentionFusion,
    build_ds_encoder,
    build_ds_temporal_encoder,
)
from util.config import load_config


class DissonanceModuleTest(unittest.TestCase):
    def test_disabled_adapter_never_loads_cache(self):
        adapter = DissonanceFeatureAdapter({"enabled": False})
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            adapter.load("sample", "missing.wav")

    def test_all_encoder_and_fusion_variants_backward(self):
        maximum_encoder_parameters = 0
        for encoder_type in ("cnn_small", "mlp_pool"):
            for bins in (288, 576):
                encoder = build_ds_encoder({
                    "type": encoder_type,
                    "embedding_dim": 1024,
                    "hidden_dim": 256,
                    "dropout": 0.0,
                }, bins)
                spectrum = torch.randn(2, bins, 32)
                mask = torch.ones(2, 32, dtype=torch.bool)
                embedding = encoder(spectrum, mask)
                self.assertEqual(tuple(embedding.shape), (2, 1024))
                embedding.square().mean().backward()
                self.assertTrue(any(
                    parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                    for parameter in encoder.parameters()
                ))
                maximum_encoder_parameters = max(
                    maximum_encoder_parameters, sum(parameter.numel() for parameter in encoder.parameters())
                )
        detached_embedding = torch.randn(1, 1024)
        for position, base_dim in (("pre_proj", 1024), ("post_proj", 4096)):
            for gate in ("scalar", "channel"):
                fusion = GatedResidualFusion(
                    base_dim=base_dim, ds_dim=1024, hidden_dim=256,
                    gate=gate, gate_bias_init=-2.0,
                )
                base = torch.randn(1, base_dim)
                fused, stats = fusion(base, detached_embedding, torch.ones(1, dtype=torch.bool))
                self.assertEqual(tuple(fused.shape), tuple(base.shape))
                self.assertTrue(0.0 < stats["gate_mean"].item() < 1.0)
                fused.square().mean().backward()
                self.assertTrue(any(
                    parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                    for parameter in fusion.parameters()
                ))
                self.assertLess(
                    maximum_encoder_parameters + sum(parameter.numel() for parameter in fusion.parameters()),
                    5_300_000,
                )

    def test_temporal_disabled_keeps_legacy_tensor_output(self):
        encoder = build_ds_encoder({"type": "cnn_small", "embedding_dim": 32,
                                    "hidden_dim": 16, "dropout": 0.0}, 24)
        spectrum = torch.randn(2, 24, 19)
        mask = torch.ones(2, 19, dtype=torch.bool)
        self.assertIsNone(build_ds_temporal_encoder({"enabled": False}))
        self.assertEqual(tuple(encoder(spectrum, mask).shape), (2, 32))

    def test_temporal_order_padding_and_attention_mask(self):
        encoder = build_ds_encoder({"type": "cnn_small", "embedding_dim": 32,
                                    "hidden_dim": 16, "dropout": 0.0}, 24)
        temporal = DSTemporalEncoder(input_dim=128, token_dim=32, conv_kernel=5, dropout=0.0)
        spectrum = torch.randn(2, 24, 32)
        mask = torch.tensor([[True] * 20 + [False] * 12, [True] * 32])
        clean = spectrum.clone()
        dirty = spectrum.clone()
        dirty[0, :, 20:] = torch.randn_like(dirty[0, :, 20:]) * 1000
        tokens_clean, token_mask = temporal(encoder.extract_feature_map(
            clean * mask[:, None, :]
        ), mask)
        tokens_dirty, dirty_mask = temporal(encoder.extract_feature_map(
            dirty * mask[:, None, :]
        ), mask)
        self.assertTrue(torch.equal(token_mask, dirty_mask))
        self.assertTrue(torch.allclose(tokens_clean[0], tokens_dirty[0], atol=1e-5))
        reversed_tokens, _ = temporal(encoder.extract_feature_map(clean.flip(-1)), mask.flip(-1))
        self.assertFalse(torch.allclose(tokens_clean[1], reversed_tokens[1]))

        fusion = TemporalGatedAttentionFusion(base_dim=64, token_dim=32, attention_dim=16)
        base = torch.randn(2, 64)
        padded_tokens = tokens_clean.clone()
        padded_tokens[0, ~token_mask[0]] = 9999
        fused_a, stats = fusion(base, tokens_clean, token_mask)
        fused_b, _ = fusion(base, padded_tokens, token_mask)
        self.assertTrue(torch.allclose(fused_a, base, atol=1e-7))
        self.assertTrue(torch.allclose(fused_a, fused_b, atol=1e-7))
        self.assertIn("attention_entropy", stats)
        self.assertLess(sum(p.numel() for p in temporal.parameters()), 1_000_000)

        buffer = io.BytesIO()
        torch.save({"temporal": temporal.state_dict(), "fusion": fusion.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, map_location="cpu")
        temporal.load_state_dict(checkpoint["temporal"], strict=True)
        fusion.load_state_dict(checkpoint["fusion"], strict=True)
        fusion.eval()
        self.assertEqual(tuple(fusion(base, tokens_clean, token_mask)[0].shape), (2, 64))
        configured_temporal = DSTemporalEncoder(input_dim=128, token_dim=256)
        configured_post_bridge = TemporalGatedAttentionFusion(
            base_dim=4096, token_dim=256, attention_dim=256
        )
        self.assertLess(
            sum(p.numel() for p in configured_temporal.parameters())
            + sum(p.numel() for p in configured_post_bridge.parameters()),
            5_000_000,
        )

    def test_stage2_trainability(self):
        class MinimalModel(torch.nn.Module):
            get_trainable_params = LLaMA_adapter.get_trainable_params
            set_default_trainability = LLaMA_adapter.set_default_trainability
            set_stage2_training_stage = LLaMA_adapter.set_stage2_training_stage

            def __init__(self):
                super().__init__()
                self.ds_encoder = torch.nn.Linear(4, 4)
                self.ds_temporal = torch.nn.Linear(4, 4)
                self.ds_fusion = torch.nn.Linear(4, 4)
                self.prefix_query = torch.nn.Embedding(2, 4)
                self.mu_mert_norm_1 = torch.nn.LayerNorm(4)
                self.mu_mert_norm_2 = torch.nn.LayerNorm(4)
                self.mu_mert_norm_3 = torch.nn.LayerNorm(4)
                self.mu_mert_proj = torch.nn.Linear(4, 4)
                self.mu_mert_f1_1 = torch.nn.Linear(4, 4)
                self.llama = torch.nn.Linear(4, 4)
                self.phase = "finetune"
                self.trainable_mode = "ds_stage2_minimal"
                self.training_stage = 1
                self.dissonance_enabled = True

        model = MinimalModel()
        model.set_stage2_training_stage(1)
        stage1 = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(all(name.startswith(("ds_encoder.", "ds_temporal.", "ds_fusion."))
                            for name in stage1))
        model.set_stage2_training_stage(2)
        stage2 = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(any(name.startswith("prefix_query.") for name in stage2))
        self.assertTrue(any(name.startswith("mu_mert_norm_1.") for name in stage2))
        self.assertFalse(any(name.startswith(("mu_mert_proj.", "mu_mert_f1_1.", "llama."))
                             for name in stage2))

    def test_stage2_configs_and_cqt_cache_contract(self):
        config_root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
        for number in range(9):
            path = next(config_root.glob(f"{number:02d}_*.yaml"))
            config = load_config(path)
            self.assertNotEqual(config.get("training", {}).get("trainable_mode"), "ds_stage2_minimal")
        for name, temporal_enabled, position, feature in (
            ("09_ds_staged_global.yaml", False, "pre_proj", "dissonance_spectrum"),
            ("10_ds_temporal_pre_proj.yaml", True, "pre_proj", "dissonance_spectrum"),
            ("11_ds_temporal_post_bridge.yaml", True, "post_bridge", "dissonance_spectrum"),
            ("12_cqt_temporal_post_bridge.yaml", True, "post_bridge", "processed_cqt"),
        ):
            config = load_config(config_root / name)
            ds = config["model"]["dissonance"]
            self.assertEqual(bool(ds["temporal"]["enabled"]), temporal_enabled)
            self.assertEqual(ds["fusion"]["position"], position)
            self.assertEqual(ds["input_feature"], feature)

        budget_root = config_root / "stage2_core_budget"
        for name in (
            "00_baseline_peft.yaml", "01_ds_default.yaml", "09_ds_staged_global.yaml",
            "10_ds_temporal_pre_proj.yaml", "11_ds_temporal_post_bridge.yaml",
            "12_cqt_temporal_post_bridge.yaml",
        ):
            budget = load_config(budget_root / name)
            self.assertEqual(budget["training"]["epochs"], 6)
            self.assertEqual(budget["training"]["early_stopping"]["patience"], 2)
            self.assertEqual(budget["training"]["save_every"], 6)

        common = {"enabled": True, "cache_root": tempfile.mkdtemp(), "feature": {
            "n_octaves": 2, "bins_per_octave": 12,
        }}
        ds_adapter = DissonanceFeatureAdapter(common)
        cqt_adapter = DissonanceFeatureAdapter({**common, "input_feature": "processed_cqt"})
        self.assertNotEqual(ds_adapter.config_hash, cqt_adapter.config_hash)
        tensors = torch.randn(24, 11)
        payload = {
            "dissonance": tensors,
            "processed_cqt": tensors.clone(),
            "metadata": {"feature_version": FEATURE_VERSION, "config_hash": cqt_adapter.config_hash},
        }
        cqt_adapter.validate(payload)
        self.assertEqual(payload["dissonance"].shape, payload["processed_cqt"].shape)


if __name__ == "__main__":
    unittest.main()
