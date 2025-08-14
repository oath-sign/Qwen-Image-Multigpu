import torch
import numpy as np
import base64
import io
from PIL import Image

def tensor_to_b64(t: torch.Tensor) -> str:
    """Serializes a torch.Tensor into a base64 string."""
    buf = io.BytesIO()
    # Convert tensor to float32 before converting to numpy, as numpy doesn't support bfloat16
    np.save(buf, t.detach().cpu().to(torch.float32).numpy())
    # Encode the buffer's content to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def b64_to_tensor(b: str, dtype: torch.dtype = torch.float16, device: str = "cuda") -> torch.Tensor:
    """Deserializes a base64 string into a torch.Tensor."""
    # Decode the base64 string to bytes
    buf = io.BytesIO(base64.b64decode(b))
    # Load the numpy array from the buffer
    arr = np.load(buf)
    # Convert the numpy array to a torch tensor and move it to the specified device and dtype
    return torch.from_numpy(arr).to(dtype=dtype, device=device)

def pil_to_b64(img: Image.Image) -> str:
    """Serializes a PIL Image into a base64 string."""
    buf = io.BytesIO()
    # Save the image to the buffer in PNG format
    img.save(buf, format="PNG")
    # Encode the buffer's content to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def b64_to_pil(b: str) -> Image.Image:
    """Deserializes a base64 string into a PIL Image."""
    # Decode the base64 string and open it as a PIL Image
    return Image.open(io.BytesIO(base64.b64decode(b)))
