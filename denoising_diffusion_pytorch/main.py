import torch
import argparse
import os

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from pathlib import Path
from torchvision import transforms as T, utils


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--vit_image_size', type=int, default=64,
                        help="""image size that enters vit; 
                    must match with patch_size: num_patches = (
                    vit_image_size/patch_size)**2""")
    parser.add_argument('--image_size', type=int, default=56, help="""image 
    size of in-distibution data. 
                    negative samples are first resized to image_size and then 
                    inflated to vit_image_size. This
                    ensures that aux samples have same resolution as in-dist 
                    samples""")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help="""Name of 
                            architecture to train. For quick experiments with 
                            ViTs, we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=4, type=int, help="""Size in 
    pixels
                of input square patches - default 16 (for 16x16 patches). 
                Using smaller
                values leads to better performance but requires more memory. 
                Applies only
                for ViTs (vit_tiny, vit_small and vit_base). If <16, 
                we recommend disabling
                mixed precision training (--use_fp16 false) to avoid 
                unstabilities.""")
    parser.add_argument('--out_dim', default=1, type=int,
                        help="""Dimensionality of
                the DINO head output. For complex and large datasets large 
                values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=False, help="""Whether or not to 
                        weight normalize the last layer of the DINO head.
                Not normalizing leads to better performance but can make the 
                training unstable.
                In our experiments, we typically set this paramater to False 
                with vit_small and True with vit_base.""")

    parser.add_argument('--use_bn_in_head', default=False,
                        help="Whether to use batch normalizations in "
                             "projection head (Default: False)")

    # Training/Optimization parameters
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Number of epochs for the linear learning-rate "
                             "warm up.")
    parser.add_argument('--drop_path_rate', type=float, default=0,
                        help="stochastic depth rate")  # todo was 0.1
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume training from this epoch')
    # Multi-crop parameters

    ############################################################################
    parser.add_argument("--train_lr", default=1e-4, type=float,
                        help="""Learning rate at the end of
                linear warmup (highest LR used during training). The learning 
                rate is linearly scaled
                with the batch size, and specified here for a reference batch 
                size of 256""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='Per-GPU batch-size : number of distinct images '
                             'loaded on one GPU')

    parser.add_argument('--inference_bs', default=1, type=int,
                        help='Batch size for sampling/inference')
    parser.add_argument('--inference_num_samples', default=1, type=int,
                        help='Number of samples to save during inference')

    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of noise levels')
    parser.add_argument('--sampling_timesteps', type=int, default=1000,
                        help='Number of noise levels for sampling')
    parser.add_argument('--loss_type', type=str, default='l1',
                        choices=['l1', 'l2'], help='Loss type')
    parser.add_argument('--train_num_steps', type=int, default=1000,
                        help='Number of total training steps')
    parser.add_argument("--ema_decay", default=0.995, type=float,
                        help="""EMA decay on parameters for inference""")
    parser.add_argument('--gradient_accumulate_every', type=int, default=2,
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--augment_horizontal_flip', type=bool, default=True,
                        help='Augment with horizontal flip')
    parser.add_argument('--ema_update_every', type=int, default=10,
                        help='Number of steps to update EMA')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.99),
                        help='Adam betas')
    parser.add_argument('--save_and_sample_every', type=int, default=1000,
                        help='Number of steps to save and sample')
    parser.add_argument('--num_samples', type=int, default=25,
                        help='Number of samples to save during training')
    parser.add_argument('--results_folder', type=str, default='./results',
                        help='Folder to save results')
    parser.add_argument('--amp', type=bool, default=False,
                        help='Whether to use amp')
    parser.add_argument('--fp16', type=bool, default=False,
                        help='Whether to use fp16')
    parser.add_argument('--split_batches', type=bool, default=True,
                        help='Whether to split batches')

    return parser


def get_args_parser2():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--image_size', type=int, default=32, help="""image 
    size of in-distibution data. 
                negative samples are first resized to image_size and then 
                inflated to vit_image_size. This
                ensures that aux samples have same resolution as in-dist 
                samples""")

    # Training/Optimization parameters
    parser.add_argument("--train_lr", default=1e-4, type=float,
                        help="""Learning rate at the end of
            linear warmup (highest LR used during training). The learning 
            rate is linearly scaled
            with the batch size, and specified here for a reference batch 
            size of 256""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='Per-GPU batch-size : number of distinct images '
                             'loaded on one GPU')

    parser.add_argument('--inference_bs', default=1, type=int,
                        help='Batch size for sampling/inference')
    parser.add_argument('--inference_num_samples', default=1, type=int,
                        help='Number of samples to save during inference')

    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of noise levels')
    parser.add_argument('--sampling_timesteps', type=int, default=1000,
                        help='Number of noise levels for sampling')
    parser.add_argument('--loss_type', type=str, default='l1',
                        choices=['l1', 'l2'], help='Loss type')
    parser.add_argument('--train_num_steps', type=int, default=1000,
                        help='Number of total training steps')
    parser.add_argument("--ema_decay", default=0.995, type=float,
                        help="""EMA decay on parameters for inference""")
    parser.add_argument('--gradient_accumulate_every', type=int, default=2,
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--augment_horizontal_flip', type=bool, default=True,
                        help='Augment with horizontal flip')
    parser.add_argument('--ema_update_every', type=int, default=10,
                        help='Number of steps to update EMA')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.99),
                        help='Adam betas')
    parser.add_argument('--save_and_sample_every', type=int, default=1000,
                        help='Number of steps to save and sample')
    parser.add_argument('--num_samples', type=int, default=25,
                        help='Number of samples to save during training')
    parser.add_argument('--results_folder', type=str, default='./results',
                        help='Folder to save results')
    parser.add_argument('--amp', type=bool, default=False,
                        help='Whether to use amp')
    parser.add_argument('--fp16', type=bool, default=False,
                        help='Whether to use fp16')
    parser.add_argument('--split_batches', type=bool, default=True,
                        help='Whether to split batches')

    # Inference

    # Misc
    # parser.add_argument('--data_path', default='/path/to/imagenet/train/',
    # type=str,
    #                     help='Please specify path to the ImageNet training
    #                     data.')
    parser.add_argument('--num_gpus', default=0, type=int,
                        help='Number of GPUs. If 0, use all available ones.')
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")
    return parser


if __name__ == '__main__':

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    print("Available GPUs:", torch.cuda.device_count())
    torch.backends.cudnn.benchmark = True
    os.environ["WANDB_API_KEY"] = "2e0fdcd07ddbc7559b7b097fbb4e066126c06d7e"

    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    args.num_gpus = 1
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print('test:', args.image_size)
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()
    num_model_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Initialized UNet model with {num_model_params:,} parameters.')

    diffusion = GaussianDiffusion(model, image_size=args.image_size,
        timesteps=args.timesteps,  # number of steps, 1000
        sampling_timesteps=args.sampling_timesteps,  # 250
        # number of sampling timesteps (using ddim for faster inference [see
                                  # citation for ddim paper])
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
