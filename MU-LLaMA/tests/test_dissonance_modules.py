import unittest

import torch

from features.dissonance_adapter import DissonanceFeatureAdapter
from llama.dissonance_modules import GatedResidualFusion, build_ds_encoder


class DissonanceModuleTest(unittest.TestCase):
    def test_disabled_adapter_never_loads_cache(self):
        adapter = DissonanceFeatureAdapter({"enabled": False})
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            adapter.load("sample", "missing.wav")

    def test_all_encoder_and_fusion_variants_backward(self):
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
                for position, base_dim in (("pre_proj", 1024), ("post_proj", 4096)):
                    for gate in ("scalar", "channel"):
                        fusion = GatedResidualFusion(
                            base_dim=base_dim, ds_dim=1024, hidden_dim=256,
                            gate=gate, gate_bias_init=-2.0,
                        )
                        base = torch.randn(2, base_dim)
                        fused, stats = fusion(base, embedding.detach(), torch.ones(2, dtype=torch.bool))
                        self.assertEqual(tuple(fused.shape), tuple(base.shape))
                        self.assertTrue(0.0 < stats["gate_mean"].item() < 1.0)
                        loss = fused.square().mean()
                        loss.backward()
                        self.assertTrue(any(
                            parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                            for parameter in fusion.parameters()
                        ))
                        new_parameters = sum(parameter.numel() for parameter in encoder.parameters()) + sum(
                            parameter.numel() for parameter in fusion.parameters()
                        )
                        self.assertLess(new_parameters, 5_300_000)


if __name__ == "__main__":
    unittest.main()
