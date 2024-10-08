import torch
import tqdm
import logging
from diffusion.scheduling import *

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, objective='ddpm', schedule='linear', device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.objective = objective
        self.beta = self.prepare_noise_schedule(schedule, beta_start, beta_end).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self, schedule, beta_start, beta_end):
        if schedule == 'linear':
            return linear_beta_schedule(self.noise_steps, beta_start, beta_end)
        else:
            return cosine_beta_schedule(self.noise_steps)
    
    def noise_images(self, x, t, z=None):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        if z == None:
            noise = torch.randn_like(x)
        else:
            noise = z
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        t = torch.randint(low=1, high=self.noise_steps, size=(n,))
        return t
    def sample_timesteps2(self, n):
        t = torch.randint(low=300, high=800, size=(n,))
        return t

    def tensor_to_image(self, x):
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def sample(self, model, n, num_classes = None, channels = None):
        # reverse process
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)
            classes = torch.randint(
                low=0, high= num_classes, size=(n,), device='cuda'
            )
            for i in tqdm.tqdm(reversed(range(1, self.noise_steps))):
                t = (torch.ones(n, dtype=torch.long) * i).to(self.device)

                alpha = self.alpha[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                alpha_hat_prev = self.alpha_hat[t-1][:, None, None, None]
                beta_tilde = beta * (1 - alpha_hat_prev) / (1 - alpha_hat) # similar to beta

                predicted_noise, _, _, _ = model(x, x, t, classes, classes)

                noise = torch.randn_like(x)

                if self.objective == 'ddpm':
                    predict_x0 = 0
                    direction_point = 1 / torch.sqrt(alpha) * (x - (beta / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
                    random_noise = torch.sqrt(beta_tilde) * noise

                    x = predict_x0 + direction_point + random_noise
                else:
                    predict_x0 = torch.sqrt(alpha_hat_prev) * (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                    direction_point = torch.sqrt(1 - alpha_hat_prev) * predicted_noise
                    random_noise = 0

                    x = predict_x0 + direction_point + random_noise

        model.train()
        return torch.clamp(x, -1.0, 1.0), classes
