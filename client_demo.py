import requests
import random
import time
import os
import common

# --- Configuration ---
ENCODE_DECODE_HOST = os.getenv("ENCODE_HOST", "http://localhost:8000")
# Assumes workers are running on ports 8001, 8002, ...
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2)) # Set this to the number of GPUs running the transformer service
WORKER_HOSTS = [f"http://localhost:{8001+i}" for i in range(NUM_WORKERS)]

# --- Generation Parameters ---
PROMPT = "A bald-headed PhD student from Zhejiang University's Eagle Lab is coding, photorealistic, cinematic lighting"
NEGATIVE_PROMPT = "low quality, blurry, watermark, text, signature, ugly, deformed"
WIDTH = 1024
HEIGHT = 1024
SEED = 42
# Use 8 steps and CFG 1.0 for the lightning LoRA, otherwise use ~20 steps and CFG 4.0
INFERENCE_STEPS = 20
CFG_SCALE = 2.5

def generate_image():
    """Orchestrates the remote services to generate an image."""
    print(f"Sending request with prompt: '{PROMPT}'")
    
    # 1. Call Encoder Service
    # -----------------------
    print(f"Step 1: Calling encoder at {ENCODE_DECODE_HOST}/encode...")
    start_time = time.time()
    encode_payload = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "width": WIDTH,
        "height": HEIGHT,
        "seed": SEED,
    }
    try:
        encode_resp = requests.post(f"{ENCODE_DECODE_HOST}/encode", json=encode_payload, timeout=30)
        encode_resp.raise_for_status()
        encode_data = encode_resp.json()
        print(f"   ...Encode complete in {time.time() - start_time:.2f}s")
    except requests.exceptions.RequestException as e:
        print(f"Error calling encoder service: {e}")
        return

    # 2. Call Transformer (Denoise) Service
    # -------------------------------------
    worker_url = random.choice(WORKER_HOSTS)
    print(f"Step 2: Calling random denoise worker at {worker_url}/denoise...")
    start_time = time.time()
    denoise_payload = {
        "prompt_embeds": encode_data["prompt_embeds"],
        "prompt_embeds_mask": encode_data["prompt_embeds_mask"],
        "negative_prompt_embeds": encode_data["negative_prompt_embeds"],
        "negative_prompt_embeds_mask": encode_data["negative_prompt_embeds_mask"],
        "latents": encode_data["init_latents"],
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": INFERENCE_STEPS,
        "true_cfg_scale": CFG_SCALE,
    }
    try:
        denoise_resp = requests.post(f"{worker_url}/denoise", json=denoise_payload, timeout=120) # Longer timeout for denoising
        denoise_resp.raise_for_status()
        denoise_data = denoise_resp.json()
        print(f"   ...Denoise complete in {time.time() - start_time:.2f}s")
    except requests.exceptions.RequestException as e:
        print(f"Error calling denoise worker at {worker_url}: {e}")
        return

    # 3. Call Decoder Service
    # -----------------------
    print(f"Step 3: Calling decoder at {ENCODE_DECODE_HOST}/decode...")
    start_time = time.time()
    decode_payload = {
        "latents_b64": denoise_data["final_latents"],
        "height": HEIGHT,
        "width": WIDTH,
    }
    try:
        decode_resp = requests.post(f"{ENCODE_DECODE_HOST}/decode", json=decode_payload, timeout=30)
        decode_resp.raise_for_status()
        decode_data = decode_resp.json()
        print(f"   ...Decode complete in {time.time() - start_time:.2f}s")
    except requests.exceptions.RequestException as e:
        print(f"Error calling decoder service: {e}")
        return

    # 4. Save the final image
    # -----------------------
    final_image = common.b64_to_pil(decode_data["image_b64"])
    output_filename = f"result_{int(time.time())}.png"
    final_image.save(output_filename)
    print(f"\nSuccess! Image saved as {output_filename}")

if __name__ == "__main__":
    generate_image()
