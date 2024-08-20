from utils import *
import json, os,  torch
from JDiffusion.pipelines import StableDiffusionPipeline
import jittor as jt
import copy
import argparse
from config import *
import yaml
style_dict = {
    "00":"Neon Wireframe, Retro-Futuristic",
    "01":"Impasto Impressionistic Painting",
    "02":"Clay Cculpture Realism",
    "03":"Ornamental Line Art",
    "04":"Surreal Sculpting",
    "05":"Vintage Tattoo Illustration",
    "06":"Red Papercut",
    "07":"Pastel Urban Illustration",
    "08":"Paper Sculpture Realism",
    "09":"Soft Watercolor Realism",
    "10":"Retro Pop Art",
    "11":"Cartoon Sticker Art",
    "12":"Pixel Block Modeling",
    "13":"Ink Wash",
    "14":"Ink Art",
    "15":"Neon Cyberpunk Illustration",
    "16":"Pixellated Minimalism",
    "17":"Watercolor, Realism",
    "18":"Cute Sketch",
    "19":"Pixelated Block",
    "20":"Fantasy Ink Sketch",
    "21":"Layered Paper Cut",
    "22":"Vintage Colored Sketch",
    "23":"Ink Illustration",
    "24":"Gelatin Sculptures",
    "25":"Pixel Art, Retro",
    "26":"Origami Realism",
    "27":"Neon Retro-Futurism",
}

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Dreambooth training script.")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--weight",
        type=str,
        default='style',
    )
    parser.add_argument(
        "--use_id",
        action='store_true',)
    parser.add_argument(
        "--adain",
        action='store_true',)
    parser.add_argument(
        "--al",
        action='store_true',)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=50,
        )
    parser.add_argument(
        "--output",
        type=str,
        default='output',
        )
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        )
    parser.add_argument(
        "--similarity",
        type=str,
        default="config/similarity.json",
        )
    parser.add_argument(
        "--substitute",
        type=str,
        default="config/substitute.json",
        )
    parser.add_argument(
        "--negative",
        type=str,
        default="config/negative.json",
        )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args
args=parse_args()
import random
import numpy as np
random.seed(args.seed)
jt.set_seed(args.seed)
np.random.seed(args.seed)
try:
    import cupy
    cupy.random.seed(args.seed)
except:
    pass
rank=args.rank
dataset_root = "B"
input_root=args.input_root
model_id="stabilityai/stable-diffusion-2-1"
num_infer=40
device = 'cuda'
if args.use_id:
    dtype = jt.float16
else:
    dtype = jt.float32
if args.substitute is not None:
    substitute_dict=json.load(open(args.substitute))
if args.negative is not None:
    negative_dict=json.load(open(args.negative))
similar_dict=json.load(open(args.similarity))
with torch.no_grad():
        taskid = "{:0>2d}".format(rank)
        style_prompt = f"{style_dict[taskid]}"
        os.makedirs(f"./{args.output}/{taskid}", exist_ok=True)
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
        inv_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder='scheduler')
        pipe=StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=dtype).to("cuda")
        wrapper=apply_config(pipe.unet,load_args(f"weights/{args.weight}/{taskid}/config.yaml"))
        pipe.unet=get_unet(wrapper,f"weights/{args.weight}/{taskid}/unet.pth")
        scheduler.set_timesteps(num_infer)
        inv_scheduler.set_timesteps(num_infer)
        initial_latents=jt.load(f"latents/{args.seed}.pth")
        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            prompts = json.load(file)
        config=json.load(open(f"config/config.json"))
        for id, prompt in prompts.items():
            if str(args.seed) not in str(config[taskid][prompt]["seed"]):
                continue
            if args.weight not in config[taskid][prompt]["weight"] and args.use_id==False:
                continue
            if args.use_id and config[taskid][prompt]["level"]!="a":
                continue
            if args.adain and config[taskid][prompt]["level"]!="adain":
                continue
            now_latent=copy.deepcopy(initial_latents)
            now_prompt=f"a {prompt} in the style of {style_prompt}"
            if args.use_id:
                if os.path.exists(f"{args.input_root}/{taskid}/{prompt}.png"):
                    refer_image=jt.array(load_image(f"{dataset_root}/{taskid}/images/{similar_dict[taskid][prompt]}",target_size=(768,768)))
                    input_image=jt.array(load_image(os.path.join(args.input_root,taskid,prompt+".png"),target_size=(768,768)))
                    image=generate(now_latent,pipe,inv_scheduler,scheduler,num_infer,now_prompt,True,input_image,refer_image)
                else:
                    continue
            else:
                if args.adain:
                    if os.path.exists(f"{args.input_root}/{taskid}/{prompt}.png"):
                        refer_image=jt.array(load_image(f"{dataset_root}/{taskid}/images/{similar_dict[taskid][prompt]}",target_size=(768,768)))
                        input_image=jt.array(load_image(f"{args.input_root}/{taskid}/{prompt}.png",target_size=(768,768)))
                        os.makedirs(f"{dataset_root}/{taskid}/latents_42",exist_ok=True)
                        image = adain_gen(pipe,inv_scheduler,scheduler,50,now_prompt,input_image,refer_image=refer_image,save_dir=f"{dataset_root}/{taskid}/latents_42/{similar_dict[taskid][prompt][:-4]}.pth")
                    else:
                        continue
                else:
                    if args.substitute is not None:
                        if prompt in substitute_dict[taskid]:
                            now_prompt=f"a {substitute_dict[taskid][prompt]} in the style of {style_prompt}"
                    if args.negative is not None:
                        if prompt in negative_dict[taskid]:
                            negative_prompt=negative_dict[taskid][prompt]
                        else:
                            negative_prompt=None
                    print(now_prompt,negative_prompt)
                    pipe.scheduler=scheduler
                    image = pipe(now_prompt, num_inference_steps=args.total_steps, width=768, height=768,latents=now_latent,negative_prompt=negative_prompt).images[0]
            image.save(f"./{args.output}/{taskid}/{prompt}.png")
            print(f"Saved to ./{args.output}/{taskid}/{prompt}.png")