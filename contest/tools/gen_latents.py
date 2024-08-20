import jittor as jt
import argparse

parser = argparse.ArgumentParser(description="Simple example of a Dreambooth training script.")
parser.add_argument(
    "--seed",
    type=str,
    default=None,
    required=True,
)
args=parser.parse_args()

jt.flags.use_cuda = 1
jt.set_global_seed(int(args.seed),False)
import os
os.makedirs("latents_new",exist_ok=True)
jt.save(jt.randn((1, 4,96,96),dtype=jt.float32), f"latents_new/{args.seed}.pth")