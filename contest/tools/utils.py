from typing import Union, Tuple, Optional
import copy
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler,  DDIMScheduler
import jittor.transform as tvt
import jittor as jt
import jtorch
# Diffusers attention code for getting query, key, value and attention map
def attention_op(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query=None, key=None, value=None, attention_probs=None, temperature=1.0):
    residual = hidden_states
    
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temperature)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    if query is None:
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    if key is None:
        key = attn.to_k(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
    if value is None:
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)

    
    if key.shape[0] != query.shape[0]:
        key, value = key[:query.shape[0]], value[:query.shape[0]]

    # apply temperature scaling
    query = query * temperature # same as applying it on qk matrix

    if attention_probs is None:
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

    batch_heads, img_len, txt_len = attention_probs.shape
    
    # h = w = int(img_len ** 0.5)
    # attention_probs_return = attention_probs.reshape(batch_heads // attn.heads, attn.heads, h, w, txt_len)
    
    hidden_states = jtorch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor
    
    return attention_probs, query, key, value, hidden_states
def get_unet_layers(unet):
    
    layer_num = [i for i in range(12)]
    resnet_layers = []
    attn_layers = []
    
    for idx, ln in enumerate(layer_num):
        up_block_idx = idx // 3
        layer_idx = idx % 3
        
        resnet_layers.append(getattr(unet, 'up_blocks')[up_block_idx].resnets[layer_idx])
        if up_block_idx > 0:
            attn_layers.append(getattr(unet, 'up_blocks')[up_block_idx].attentions[layer_idx])
        else:
            attn_layers.append(None)
        
    return resnet_layers, attn_layers

def make_callback(wrapper=None,sceduler=None):

    def callback_on_step_end(pipeline: StableDiffusionPipeline, i: int, t, callback_kwargs):
        wrapper.cur_t = sceduler.timesteps[max(len(sceduler.timesteps)-2-i,0)]
        return {}
    return  callback_on_step_end


def img_to_latents(x, vae):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents

class style_transfer_module():
           
    def __init__(self,
        unet, style_transfer_params = None,al=False
    ):  
        self.al=al
        style_transfer_params_default = {
            'gamma': 0.75,
            'tau': 1.5,
            'injection_layers': [7, 8, 9, 10, 11]
        }
        if style_transfer_params is not None:
            style_transfer_params_default.update(style_transfer_params)
        self.style_transfer_params = style_transfer_params_default
        
        self.unet = unet # SD unet

        self.attn_features = {} # where to save key value (attention block feature)
        self.attn_features_modify = {} # where to save key value to modify (attention block feature)

        self.cur_t = None
        
        # Get residual and attention block in decoder
        # [0 ~ 11], total 12 layers
        resnet, attn = get_unet_layers(unet)
        
        # where to inject key and value
        qkv_injection_layer_num = self.style_transfer_params['injection_layers']

        
        for i in qkv_injection_layer_num:
            self.attn_features["layer{}_attn".format(i)] = {}
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__get_query_key_value("layer{}_attn".format(i)))
               
        # triggers for obtaining or modifying features
        
        self.trigger_get_qkv = False # if set True --> save attn qkv in self.attn_features
        self.trigger_modify_qkv = False # if set True --> save attn qkv by self.attn_features_modify
        
        self.modify_num = None # ignore
        self.modify_num_sa = None # ignore
    
    def reset_hook(self):
        resnet, attn = get_unet_layers(self.unet)
        qkv_injection_layer_num = self.style_transfer_params['injection_layers']

        
        for i in qkv_injection_layer_num:
            self.attn_features["layer{}_attn".format(i)] = {}
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__get_query_key_value("layer{}_attn".format(i)))
    
    def modify_hook(self):
        qkv_injection_layer_num = self.style_transfer_params['injection_layers']
        resnet, attn = get_unet_layers(self.unet)
        if self.al:
            for i in qkv_injection_layer_num:
                attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__modify_self_attn_qkv_al("layer{}_attn".format(i)))
        else:
            for i in qkv_injection_layer_num:
                attn[i].transformer_blocks[0].attn1.register_forward_hook(self.__modify_self_attn_qkv("layer{}_attn".format(i)))

    def remove_hook(self):
        qkv_injection_layer_num = self.style_transfer_params['injection_layers']
        resnet, attn = get_unet_layers(self.unet)
        for i in qkv_injection_layer_num:
            attn[i].transformer_blocks[0].attn1.register_forward_hook(self.empty())

        
    # ============================ hook operations ===============================

    def empty(self):
        def hook(model, input, output,args):
            pass
        return hook
    
    # save key value in self.original_kv[name]
    def __get_query_key_value(self, name):
        def hook(model, input, output,args):
            if self.trigger_get_qkv:
                    
                _, query, key, value, _ = attention_op(model, input[0])

                
                self.attn_features[name][int(self.cur_t)] = (query.detach(), key.detach(), value.detach())
            
        return hook

    
    def __modify_self_attn_qkv(self, name):
        def hook(model, input, output,args):
        
            if self.trigger_modify_qkv:
                
                _, q_cs, k_cs, v_cs, _ = attention_op(model, input[0])
                
                q_c, k_s, v_s = self.attn_features_modify[name][int(self.cur_t)]
                
                # style injection
                q_hat_cs = q_c * self.style_transfer_params['gamma'] + q_cs * (1 - self.style_transfer_params['gamma'])
                k_cs, v_cs = k_s, v_s
                
                # Replace query key and value
                _, _, _, _, modified_output = attention_op(model, input[0], key=k_cs, value=v_cs, query=q_hat_cs, temperature=self.style_transfer_params['tau'])
                
                return modified_output
        
        return hook
    
    def __modify_self_attn_qkv_al(self, name):
        def hook(model, input, output,args):
        
            if self.trigger_modify_qkv:
                
                _, q_cs, k_cs, v_cs, _= attention_op(model, input[0])
                
                q_c, k_s, v_s = self.attn_features_modify[name][int(self.cur_t)]
                
                # style injection
                q_hat_cs = adain2(q_cs, q_c,-1)
                k_cs, v_cs =adain2(k_cs, k_s,-1), v_cs 

                kk=jt.concat([k_s,k_cs],dim=0)
                vv=jt.concat([v_s,v_cs],dim=0)
                
                # Replace query key and value
                _, _, _, _, modified_output = attention_op(model, input[0], key=kk, value=vv, query=q_hat_cs, temperature=self.style_transfer_params['tau'])
                
                return modified_output
        
        return hook
    
def adain2(zT_content, zT_style,dim=-1):
    return (zT_content - zT_content.mean(dim=dim, keepdim=True)) / (jittor_std(zT_content,dim=dim, keepdim=True) + 1e-4) * jittor_std(zT_style,dim=dim, keepdim=True) + zT_style.mean(dim=dim, keepdim=True)

def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None):
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]

def jittor_std(x, dims=None, keepdim=False,dim=None):
    if dim is not None:
        return jt.sqrt(((x - x.mean(dim=dim, keepdim=True)) ** 2).mean(dim=dim, keepdim=True))
    mean = x.mean(dims=dims, keepdim=keepdim)
    variance = ((x - mean) ** 2).mean(dims=dims, keepdim=keepdim)
    return jt.sqrt(variance)


def adain(zT_content, zT_style):
    return (zT_content - zT_content.mean(dims=(2, 3), keepdim=True)) / (jittor_std(zT_content,dims=(2, 3), keepdim=True) + 1e-4) * jittor_std(zT_style,dims=(2, 3), keepdim=True) + zT_style.mean(dims=(2, 3), keepdim=True)

MIDLLE_SIZE=(768,768)

def generate(initial_latents, pipe,inv_scheduler,scheduler,total_steps,prompt,from_image,input_image,refer_image):
    inv_scheduler.set_timesteps(total_steps)
    scheduler.set_timesteps(total_steps)
    #setup
    unet_wrapper=style_transfer_module(pipe.unet)
    callback=make_callback(unet_wrapper,scheduler)
    unet_wrapper.trigger_get_qkv = True
    unet_wrapper.trigger_modify_qkv = False
    #if from_image, reverse the image, else use pipe to generate the image
    unet_wrapper.cur_t = scheduler.timesteps[-1].item()
    if from_image:
        content_latents = img_to_latents(input_image,pipe.vae)
        pipe.scheduler=inv_scheduler
        zts0=pipe(prompt=prompt,num_inference_steps=total_steps,  width=content_latents.shape[-1], height=content_latents.shape[-2],guidance_scale=0.,
                            output_type='latent', return_dict=False,latents=content_latents,callback_on_step_end=callback)[0]
    else:
        pipe.scheduler=scheduler
        pipe(prompt=prompt,num_inference_steps=total_steps,latents=initial_latents,width=MIDLLE_SIZE[0],height=MIDLLE_SIZE[1],guidance_scale=7.5,
                            return_dict=False,callback_on_step_end=callback)
        zts0=initial_latents
    content_features = copy.deepcopy(unet_wrapper.attn_features)
    refer_latents = img_to_latents(refer_image,pipe.vae)
    pipe.scheduler=inv_scheduler
    unet_wrapper.cur_t = scheduler.timesteps[-1].item()
    zts1=pipe(prompt=prompt,num_inference_steps=total_steps,  width=refer_latents.shape[-1], height=refer_latents.shape[-2],guidance_scale=0.,
                            output_type='latent', return_dict=False,latents=refer_latents,callback_on_step_end=callback)[0]
    style_features = copy.deepcopy(unet_wrapper.attn_features)
    unet_wrapper.attn_features = {}
    for layer_name in style_features.keys():
        unet_wrapper.attn_features_modify[layer_name] = {}
        for t in scheduler.timesteps:
            t = t.item()
            unet_wrapper.attn_features_modify[layer_name][t] = (content_features[layer_name][t][0], style_features[layer_name][t][1], style_features[layer_name][t][2])
    unet_wrapper.modify_hook()
    unet_wrapper.trigger_get_qkv = False
    unet_wrapper.trigger_modify_qkv = True
    zT_content=zts0
    zT_style=zts1
    latent_cs = adain(zT_content, zT_style)
    unet_wrapper.cur_t = scheduler.timesteps[0]
    pipe.scheduler=scheduler
    image = pipe(prompt=prompt,num_inference_steps=total_steps,latents=latent_cs,callback_on_step_end=callback,guidance_scale=1, width=MIDLLE_SIZE[0], height=MIDLLE_SIZE[1]).images[0]
    unet_wrapper.remove_hook()
    return image

import os
def adain_gen(pipe,inv_scheduler,scheduler,total_steps,prompt,input_image,refer_image=None,refer_latents=None,save_dir=None):
    inv_scheduler.set_timesteps(total_steps)
    scheduler.set_timesteps(total_steps)
    content_latents = img_to_latents(input_image,pipe.vae)
    pipe.scheduler=inv_scheduler
    zts0=pipe(prompt=prompt,num_inference_steps=total_steps,  width=content_latents.shape[-1], height=content_latents.shape[-2],guidance_scale=0.,
                        output_type='latent', return_dict=False,latents=content_latents)[0]
    if os.path.exists(save_dir):
        zts1=jt.load(save_dir)
    elif refer_latents is None:
        refer_latents = img_to_latents(refer_image,pipe.vae)
        pipe.scheduler=inv_scheduler
        zts1=pipe(prompt=prompt,num_inference_steps=total_steps,  width=refer_latents.shape[-1], height=refer_latents.shape[-2],guidance_scale=0.,
                                output_type='latent', return_dict=False,latents=refer_latents)[0]
        if save_dir is not None:
            jt.save(zts1,save_dir)
            print(f"save refer_latents to {save_dir}")
    else:
        zts1 = refer_latents
    zT_content=zts0
    zT_style=zts1
    latent_cs = adain(zT_content, zT_style)
    pipe.scheduler=scheduler
    image = pipe(prompt=prompt,num_inference_steps=total_steps,latents=latent_cs,guidance_scale=1, width=MIDLLE_SIZE[0], height=MIDLLE_SIZE[1]).images[0]
    return image
