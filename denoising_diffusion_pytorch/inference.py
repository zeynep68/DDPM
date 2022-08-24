import torch
import os
import argparse
import matplotlib.pyplot as plt

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, num_to_groups
from main import get_args_parser


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Available GPUs:", torch.cuda.device_count())
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).cuda()
    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Initialized UNet model with {num_model_params:,} parameters.')

    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,  # number of steps, 1000
        sampling_timesteps=args.sampling_timesteps,  # 250
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type=args.loss_type  # L1 or L2
    ).cuda()
    print(f'Initialized diffusion with {args.timesteps} timesteps.')

    trainer = Trainer(diffusion, args)

    trainer.load("69")

    batches = num_to_groups(args.inference_num_samples, args.inference_bs)
    samples = list(map(lambda n: trainer.ema.ema_model.sample(batch_size=n), batches))
    for img in samples:
        img = img[0].permute(1,2,0).detach().cpu().numpy()
        plt.imshow(img)
        plt.show()
        plt.close()
