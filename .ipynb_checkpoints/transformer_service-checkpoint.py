import uvicorn
import torch
import os
import math
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import QwenImageTransformer2DModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from optimum.quanto import freeze, qint8, quantize
from safetensors.torch import safe_open
from common import b64_to_tensor, tensor_to_b64

# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "/root/Qwen-Image/models/Qwen-Image")
LORA_PATH = os.getenv("LORA_PATH", None) # e.g., "/path/to/your/lora.safetensors"
DEVICE_ID = int(os.getenv("WORKER_GPU", 0))
DEVICE = f"cuda:{DEVICE_ID}"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

torch.cuda.set_device(DEVICE_ID)
print(f"Starting Transformer Service on device: {DEVICE}")

# --- LoRA Loading and Merging Functions ---
def load_and_merge_lora(model, lora_path):
    """Loads LoRA weights from a safetensors file and merges them into the model."""
    print(f"Loading and merging LoRA from {lora_path}...")
    lora_state_dict = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state_dict[key] = f.get_tensor(key)

    is_native = any("diffusion_model." in key for key in lora_state_dict)
    
    for key, value in model.named_parameters():
        lora_down_key = "diffusion_model." if is_native else ""
        lora_down_key += key.replace(".weight", ".lora_down.weight")
        
        if lora_down_key in lora_state_dict:
            lora_up_key = lora_down_key.replace("lora_down.weight", "lora_up.weight")
            lora_alpha_key = lora_down_key.replace("lora_down.weight", "alpha")

            lora_down = lora_state_dict[lora_down_key].to(device=DEVICE, dtype=torch.float32)
            lora_up = lora_state_dict[lora_up_key].to(device=DEVICE, dtype=torch.float32)
            lora_alpha = float(lora_state_dict[lora_alpha_key]) if lora_alpha_key in lora_state_dict else lora_down.shape[0]
            
            rank = lora_down.shape[0]
            scale = lora_alpha / rank
            
            delta_W = torch.matmul(lora_up, lora_down) * scale
            value.data += delta_W.to(device=DEVICE, dtype=value.dtype)
    
    print("LoRA weights merged successfully.")
    return model

# --- Model and Scheduler Loading ---
transformer = QwenImageTransformer2DModel.from_pretrained(
    MODEL_PATH, subfolder="transformer", torch_dtype=DTYPE
)

if LORA_PATH and os.path.exists(LORA_PATH):
    transformer = load_and_merge_lora(transformer, LORA_PATH)
    # Use the special scheduler for the lightning LoRA
    scheduler_config = QwenImageTransformer2DModel.load_config(MODEL_PATH, subfolder="transformer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    print("Using FlowMatchEulerDiscreteScheduler for LoRA.")
else:
    # Use the default scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    print("Using default FlowMatchEulerDiscreteScheduler.")

# Quantize the transformer to INT8 for performance and move to device
print("Quantizing transformer to INT8...")
quantize(transformer, weights=qint8)
freeze(transformer)
transformer.to(DEVICE)
transformer.eval()
print("Transformer loaded, quantized, and moved to device.")

# --- FastAPI App ---
app = FastAPI(title=f"Qwen-Transformer-Service-GPU-{DEVICE_ID}")

# --- Pydantic Models ---
class DenoiseRequest(BaseModel):
    prompt_embeds: str
    prompt_embeds_mask: str
    negative_prompt_embeds: str
    negative_prompt_embeds_mask: str
    latents: str
    height: int
    width: int
    num_inference_steps: int = 20
    true_cfg_scale: float = 4.0

class DenoiseResponse(BaseModel):
    final_latents: str

# --- Helper Functions ---
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b

# --- API Endpoint ---
@app.post("/denoise", response_model=DenoiseResponse)
@torch.no_grad()
def denoise(req: DenoiseRequest):
    """Performs the main denoising loop."""
    print(f"Received denoise request for {req.num_inference_steps} steps.")
    
    # Deserialize inputs to tensors on the correct device
    prompt_embeds = b64_to_tensor(req.prompt_embeds, DTYPE, DEVICE)
    prompt_embeds_mask = b64_to_tensor(req.prompt_embeds_mask, torch.bool, DEVICE)
    neg_prompt_embeds = b64_to_tensor(req.negative_prompt_embeds, DTYPE, DEVICE)
    neg_prompt_embeds_mask = b64_to_tensor(req.negative_prompt_embeds_mask, torch.bool, DEVICE)
    latents = b64_to_tensor(req.latents, DTYPE, DEVICE)

    # Prepare timesteps
    mu = calculate_shift(latents.shape[1])
    timesteps, _ = retrieve_timesteps(scheduler, req.num_inference_steps, DEVICE, mu=mu)
    
    img_shapes = [(1, req.height // 16, req.width // 16)] # VAE scale factor is 8, but latents are 2x downsampled
    
    # Denoising loop
    for t in timesteps:
        timestep = t.expand(latents.shape[0])
        
        # Conditional prediction
        noise_pred_cond = transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
            return_dict=False
        )[0]

        # Unconditional prediction
        noise_pred_uncond = transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            encoder_hidden_states=neg_prompt_embeds,
            encoder_hidden_states_mask=neg_prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=neg_prompt_embeds_mask.sum(dim=1).tolist(),
            return_dict=False
        )[0]
        
        # Perform True-CFG with norm correction
        comb_pred = noise_pred_uncond + req.true_cfg_scale * (noise_pred_cond - noise_pred_uncond)
        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        noise_pred = comb_pred * (cond_norm / noise_norm)

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    print("Denoising complete.")
    return DenoiseResponse(final_latents=tensor_to_b64(latents))

if __name__ == "__main__":
    port = 8001 + DEVICE_ID
    print(f"Launching transformer worker on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
