import yaml
import argparse
from peft import OFTModel, OFTConfig, LoraModel, LoraConfig, AdaLoraModel, AdaLoraConfig
import jittor as jt
TARGET=["to_q", "to_v", "to_k", "query", "value", "key", "to_out.0", "add_k_proj", "add_v_proj"]
def load_args(path):
    # Load the YAML file
    with open(path, 'r') as config_file:
        config_dict = yaml.safe_load(config_file)

    # Convert the dictionary to argparse.Namespace
    args = argparse.Namespace(**config_dict)
    return args

def apply_config(unet,args):
    if args.lora_type.lower() == "oft":
        unet_lora_config = OFTConfig(
            target_modules=TARGET,
            r=args.r,
        )
        unet=OFTModel(unet,unet_lora_config,'default')
    elif args.lora_type.lower() == "dora":
        unet_lora_config = LoraConfig(
            target_modules=TARGET,
            r=args.r,
            use_dora=True,
            use_rslora=True,
        )
        unet=LoraModel(unet,unet_lora_config,'default')
    else:
        raise ValueError("Unknown lora_type")
    return unet

def get_unet(unet,state_dict):
    state=jt.load(state_dict)
    unet.load_state_dict(state)
    merged=unet.merge_and_unload()
    return merged
