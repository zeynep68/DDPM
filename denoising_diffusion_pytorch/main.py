import torch
import argparse
import os

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--image_size', type=int, default=32, help="""image size of in-distibution data. 
                negative samples are first resized to image_size and then inflated to vit_image_size. This
                ensures that aux samples have same resolution as in-dist samples""")

    # Training/Optimization parameters
    parser.add_argument("--train_lr", default=1e-4, type=float, help="""Learning rate at the end of
            linear warmup (highest LR used during training). The learning rate is linearly scaled
            with the batch size, and specified here for a reference batch size of 256""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU')

    parser.add_argument('--inference_bs', default=1, type=int, help='Batch size for sampling/inference')
    parser.add_argument('--inference_num_samples', default=1, type=int, help='Number of samples to save during inference')

    parser.add_argument('--timesteps', type=int, default=1000, help='Number of noise levels')
    parser.add_argument('--sampling_timesteps', type=int, default=1000, help='Number of noise levels for sampling')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2'], help='Loss type')
    parser.add_argument('--train_num_steps', type=int, default=1000, help='Number of total training steps')
    parser.add_argument("--ema_decay", default=0.995, type=float, help="""EMA decay on parameters for inference""")
    parser.add_argument('--gradient_accumulate_every', type=int, default=2, help='Number of steps to accumulate gradients')
    parser.add_argument('--augment_horizontal_flip', type=bool, default=True, help='Augment with horizontal flip')
    parser.add_argument('--ema_update_every', type=int, default=10, help='Number of steps to update EMA')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.99), help='Adam betas')
    parser.add_argument('--save_and_sample_every', type=int, default=1000, help='Number of steps to save and sample')
    parser.add_argument('--num_samples', type=int, default=25, help='Number of samples to save during training')
    parser.add_argument('--results_folder', type=str, default='./results', help='Folder to save results')
    parser.add_argument('--amp', type=bool, default=False, help='Whether to use amp')
    parser.add_argument('--fp16', type=bool, default=False, help='Whether to use fp16')
    parser.add_argument('--split_batches', type=bool, default=True, help='Whether to split batches')


    # Inference


    # Misc
    # parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
    #                     help='Please specify path to the ImageNet training data.')
    parser.add_argument('--num_gpus', default=0, type=int, help='Number of GPUs. If 0, use all available ones.')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    print("Available GPUs:", torch.cuda.device_count())
    torch.backends.cudnn.benchmark = True
    os.environ["WANDB_API_KEY"] = "a9efb3b6cddc090dbf125d4c5d0dff12b178eb36"

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

    if args.num_gpus == 0:
        num_gpus = torch.cuda.device_count()
        print(f"Use all available GPUs ({torch.cuda.device_count()}).")
    else:
        num_gpus = args.num_gpus

    trainer = Trainer(diffusion, args)

    trainer.train()
