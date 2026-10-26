"""Gaussian diffusion over VAE latents: noise schedule, training loss and DDIM sampling."""

import math

import torch
import torch.nn.functional as F


def make_betas(schedule, timesteps):
    if schedule == 'linear':
        # The "scaled linear" schedule of Latent Diffusion / Stable Diffusion
        return torch.linspace(0.00085 ** .5, 0.012 ** .5, timesteps, dtype=torch.float64).square()
    if schedule == 'cosine':
        # Nichol & Dhariwal 2021
        t = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alpha_bar = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
        return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=.999)
    raise ValueError(f'Unknown noise schedule {schedule!r}')


class GaussianDiffusion:
    # prediction='eps' regresses the added noise; 'v' regresses v = sqrt(a)*noise - sqrt(1-a)*z0
    # (Salimans & Ho 2022), which stays well conditioned at the high-noise end.
    def __init__(self, timesteps=1000, schedule='linear', prediction='v'):
        if prediction not in ('eps', 'v'):
            raise ValueError("prediction must be 'eps' or 'v'")
        self.timesteps, self.prediction = timesteps, prediction
        self.alpha_bar = torch.cumprod(1 - make_betas(schedule, timesteps), 0).float()

    def coefficients(self, t):
        # sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t), shaped to broadcast over [B, C, H, W]
        alpha_bar = self.alpha_bar.to(t.device)[t].view(-1, 1, 1, 1)
        return alpha_bar.sqrt(), (1 - alpha_bar).sqrt()

    def loss(self, model, z0, cond, t, noise):
        a, s = self.coefficients(t)
        prediction = model(a * z0 + s * noise, t, cond)
        target = noise if self.prediction == 'eps' else a * noise - s * z0
        return F.mse_loss(prediction.float(), target)

    def predict(self, z, t, output):
        # Model output at noisy latent z -> estimates of the clean latent and of the noise
        a, s = self.coefficients(t)
        output = output.float()
        if self.prediction == 'eps':
            return (z - s * output) / a, output
        return a * z - s * output, s * z + a * output

    @torch.no_grad()
    def ddim_sample(self, model, cond, steps=50, generator=None, noise=None):
        # Deterministic DDIM (Song et al. 2021) from pure noise at t = T-1 down to t = 0;
        # `noise` fixes the starting point, otherwise it is drawn with `generator`
        if not 1 <= steps <= self.timesteps:
            raise ValueError(f'steps must be in [1, {self.timesteps}]')
        times = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        z = torch.randn(cond.shape, generator=generator, device=cond.device) if noise is None else noise
        for i, t in enumerate(times):
            t_batch = torch.full((len(cond),), t, dtype=torch.long, device=cond.device)
            z0, noise = self.predict(z, t_batch, model(z, t_batch, cond))
            alpha_prev = self.alpha_bar[times[i + 1]].item() if i + 1 < steps else 1.
            z = math.sqrt(alpha_prev) * z0 + math.sqrt(1 - alpha_prev) * noise
        return z
