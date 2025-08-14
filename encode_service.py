import uvicorn
import torch
import os
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import QwenImagePipeline
from diffusers.models import QwenImageTransformer2DModel
from common import tensor_to_b64, pil_to_b64, b64_to_tensor

# --- Configuration ---
MODEL_PATH = os.getenv("MODEL_PATH", "/root/Qwen-Image/models/Qwen-Image")
DEVICE = os.getenv("ENC_DEVICE", "cuda:0")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Starting Encode/Decode Service on device: {DEVICE}")

# --- Model Loading ---
# Load the main pipeline but exclude the transformer to save memory.
pipe = QwenImagePipeline.from_pretrained(
    MODEL_PATH,
    transformer=None,  # Explicitly do not load the transformer
    torch_dtype=DTYPE
)
pipe.text_encoder.to(DEVICE)
pipe.vae.to(DEVICE)
print("Text Encoder and VAE loaded successfully.")

# We need the transformer's config to know the latent channel dimensions,
# but we don't need to load the whole model.
transformer_config = QwenImageTransformer2DModel.load_config(MODEL_PATH, subfolder="transformer")
NUM_CHANNELS_LATENTS = transformer_config['in_channels'] // 4

# --- FastAPI App ---
app = FastAPI(title="Qwen-Encode-Decode-Service")

# --- Pydantic Models for API validation ---
class EncodeRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    seed: int = 42
    max_sequence_length: int = 512

class EncodeResponse(BaseModel):
    prompt_embeds: str
    prompt_embeds_mask: str
    negative_prompt_embeds: str
    negative_prompt_embeds_mask: str
    init_latents: str

class DecodeRequest(BaseModel):
    latents_b64: str
    height: int
    width: int

class DecodeResponse(BaseModel):
    image_b64: str

# --- API Endpoints ---
@app.post("/encode", response_model=EncodeResponse)
def encode_prompt_and_latents(req: EncodeRequest):
    """Encodes prompts and prepares initial latent variables."""
    print(f"Encoding prompt: '{req.prompt[:50]}...' with seed {req.seed}")
    with torch.no_grad():
        generator = torch.Generator(device=DEVICE).manual_seed(req.seed)

        # Encode positive prompt
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=req.prompt,
            device=DEVICE,
            num_images_per_prompt=1,
            max_sequence_length=req.max_sequence_length
        )

        # Encode negative prompt
        neg_prompt_embeds, neg_prompt_embeds_mask = pipe.encode_prompt(
            prompt=req.negative_prompt,
            device=DEVICE,
            num_images_per_prompt=1,
            max_sequence_length=req.max_sequence_length
        )

        # Prepare initial noisy latents
        latents, _ = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=NUM_CHANNELS_LATENTS,
            height=req.height,
            width=req.width,
            dtype=prompt_embeds.dtype,
            device=DEVICE,
            generator=generator,
        )

        return EncodeResponse(
            prompt_embeds=tensor_to_b64(prompt_embeds),
            prompt_embeds_mask=tensor_to_b64(prompt_embeds_mask),
            negative_prompt_embeds=tensor_to_b64(neg_prompt_embeds),
            negative_prompt_embeds_mask=tensor_to_b64(neg_prompt_embeds_mask),
            init_latents=tensor_to_b64(latents),
        )

@app.post("/decode", response_model=DecodeResponse)
def decode_latents_to_image(req: DecodeRequest):
    """Decodes final latents into a PIL image."""
    print("Decoding final latents to image.")
    with torch.no_grad():
        latents = b64_to_tensor(req.latents_b64, dtype=pipe.vae.dtype, device=DEVICE)
        
        # The VAE expects latents in a specific shape, _unpack_latents handles this.
        latents = pipe._unpack_latents(latents, req.height, req.width, pipe.vae_scale_factor)

        # The VAE decoding process requires specific mean/std scaling.
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        # Decode latents and post-process to get the final image
        # --- FIX --- Restored the `[:, :, 0]` slice to correctly handle the 5D VAE output.
        image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]

        return DecodeResponse(image_b64=pil_to_b64(image))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
