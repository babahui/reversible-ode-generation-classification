import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from unified_transport.checkpoint import (
    ExponentialMovingAverage,
    load_for_inference,
    save_checkpoint,
)
from unified_transport.interpolant import (
    MultimarginalInterpolant,
    TransportSampler,
    differentiable_edge_integrate,
)
from unified_transport.generation_only import (
    ConditionalFlowMatching,
    GenerationOnlySampler,
)
from unified_transport.latent import OrthogonalGaussianLatent
from unified_transport.model import MarginalUNet
from unified_transport.ot_pairing import pair_latents_by_class, path_diagnostics


class ConstantMarginals(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.num_marginals = len(values)
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, x, alpha, previous=None, recent_velocity=None):
        return self.values[None, :, None, None, None].expand(
            x.shape[0], -1, *x.shape[1:]
        )


class CoreTests(unittest.TestCase):
    def test_generation_only_objective_and_sampler(self):
        model = MarginalUNet(
            1, 2, 8, (1, 2), False, True, output_marginals=False
        )
        objective = ConditionalFlowMatching(model)
        images = torch.randn(3, 1, 8, 8)
        noise = torch.randn_like(images)
        loss, _ = objective(images, noise)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        generated = GenerationOnlySampler(model).integrate(noise, steps=2)
        self.assertEqual(generated.shape, images.shape)

    def test_latent_centers_are_orthogonal_and_classifiable(self):
        latent = OrthogonalGaussianLatent(5, (1, 4, 4), center_scale=3.0, sigma=0.1)
        flat = latent.centers.flatten(1)
        gram = flat @ flat.T
        self.assertTrue(torch.allclose(gram, torch.eye(5) * 9, atol=1e-5))
        self.assertTrue(torch.equal(latent.classify(latent.centers), torch.arange(5)))
        labels = torch.arange(5)
        self.assertEqual(latent.endpoint_projection_loss(latent.centers, labels), 0)
        self.assertGreater(
            latent.endpoint_projection_loss(torch.zeros_like(latent.centers), labels), 0
        )

    def test_coded_low_frequency_centers_are_redundant_and_orthogonal(self):
        latent = OrthogonalGaussianLatent(
            10,
            (3, 32, 32),
            center_scale=12.0,
            sigma=0.5,
            center_mode="coded_low_frequency",
        )
        flat = latent.centers.flatten(1)
        gram = flat @ flat.T
        self.assertTrue(torch.allclose(gram, torch.eye(10) * 144, atol=1e-4))
        self.assertTrue(torch.equal(latent.classify(latent.centers), torch.arange(10)))

    def test_generation_only_ode_class_loss_backpropagates(self):
        model = MarginalUNet(
            1, 2, 8, (1, 2), False, True, output_marginals=False
        )
        latent = OrthogonalGaussianLatent(
            2, (1, 8, 8), center_scale=3.0, sigma=0.1
        )
        objective = ConditionalFlowMatching(
            model, latent=latent, ode_class_weight=0.1,
            ode_class_steps=2, ode_class_batch=2,
        )
        images = torch.randn(3, 1, 8, 8)
        labels = torch.tensor([0, 1, 0])
        noise = latent.sample(labels)
        loss, metrics = objective(images, noise, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("ode_class_loss", metrics)

    def test_sinkhorn_pairing_preserves_class_components(self):
        latent = OrthogonalGaussianLatent(
            2, (1, 8, 8), center_scale=3.0, sigma=0.1, center_mode="low_frequency"
        )
        images = torch.randn(4, 1, 8, 8)
        labels = torch.tensor([0, 0, 1, 1])
        paired, metrics = pair_latents_by_class(
            images, labels, latent, epsilon=0.1, iterations=20
        )
        self.assertEqual(paired.shape, images.shape)
        self.assertEqual(metrics["ot_classes_in_batch"], 2)
        self.assertTrue(torch.equal(latent.classify(paired), labels))

    def test_path_diagnostics_are_time_binned(self):
        state = torch.randn(8, 1, 8, 8)
        target = torch.randn_like(state)
        predicted = target + 0.1 * torch.randn_like(target)
        time = torch.linspace(0.01, 0.99, 8)
        result = path_diagnostics(state, target, predicted, time, bins=4, neighbors=2)
        self.assertEqual(len(result["path_diagnostics"]), 4)
        for row in result["path_diagnostics"]:
            self.assertTrue(torch.isfinite(torch.tensor(row["local_velocity_variance"])))

    def test_objective_backpropagates(self):
        model = MarginalUNet(
            image_channels=1,
            num_marginals=3,
            base_channels=8,
            channel_mults=(1, 2),
            use_history=True,
        )
        objective = MultimarginalInterpolant(model, simplex_probability=0.5)
        marginals = torch.randn(4, 3, 1, 8, 8)
        loss, metrics = objective(marginals)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["history_speed"], 0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_alpha_condition_can_be_removed(self):
        model = MarginalUNet(
            image_channels=1,
            num_marginals=2,
            base_channels=8,
            channel_mults=(1, 2),
            use_history=False,
            condition_on_alpha=False,
        ).eval()
        x = torch.randn(2, 1, 8, 8)
        source = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        with torch.no_grad():
            at_source = model(x, source)
            at_target = model(x, target)
        self.assertTrue(torch.equal(at_source, at_target))

    def test_sampler_uses_signed_edge_velocity(self):
        model = ConstantMarginals([0.0, 2.0, -1.0])
        sampler = TransportSampler(model)
        initial = torch.zeros(2, 1, 4, 4)
        forward = sampler.integrate(initial, 0, 1, steps=4, method="heun")
        reverse = sampler.integrate(forward, 1, 0, steps=4, method="euler")
        self.assertTrue(torch.allclose(forward, torch.full_like(forward, 2.0)))
        self.assertTrue(torch.allclose(reverse, initial))

        alpha_path = torch.tensor(
            [[1.0, 0.0, 0.0], [0.2, 0.6, 0.2], [0.0, 0.0, 1.0]]
        )
        via_interior = sampler.integrate_path(initial, alpha_path, method="heun")
        self.assertTrue(torch.allclose(via_interior, torch.full_like(initial, -1.0)))
        training_result = differentiable_edge_integrate(model, initial, 0, 1, 4)
        self.assertTrue(torch.allclose(training_result, forward))

    def test_checkpoint_round_trip(self):
        config = {
            "mode": "cifar10",
            "image_channels": 1,
            "num_marginals": 2,
            "base_channels": 8,
            "channel_mults": [1, 2],
            "use_history": True,
            "num_classes": 3,
            "image_size": 8,
            "center_scale": 2.0,
            "latent_sigma": 0.2,
            "seed": 7,
        }
        model = MarginalUNet(1, 2, 8, (1, 2), True)
        latent = OrthogonalGaussianLatent(3, (1, 8, 8), 2.0, 0.2, 7)
        ema = ExponentialMovingAverage(model)
        optimizer = torch.optim.Adam(model.parameters())
        ema.update(model)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            save_checkpoint(path, config, model, ema, optimizer, 12, latent)
            loaded_model, loaded_latent, checkpoint = load_for_inference(
                path, torch.device("cpu")
            )
        self.assertEqual(checkpoint["step"], 12)
        self.assertEqual(loaded_model.num_marginals, 2)
        self.assertTrue(torch.equal(loaded_latent.centers, latent.centers))


if __name__ == "__main__":
    unittest.main()
