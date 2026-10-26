"""Restore masked image regions with the original Task2 inpainting pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "Task2" / "ddpm-NASA-inpainting-512"
IMAGE_SIZE = 512


def resolve_checkpoint(checkpoint: str | Path = "checkpoint-5") -> Path:
    """Resolve a saved checkpoint name or an explicit adapter directory."""
    path = Path(checkpoint).expanduser()
    if path.parent == Path(".") and path.name.startswith("checkpoint-"):
        path = CHECKPOINT_ROOT / path / "unet"
    path = path.resolve()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Missing adapter file: {path / filename}")
    return path


def generate_random_square_mask(
    image_size: int = IMAGE_SIZE,
    num_squares: int = 3,
    seed: int | None = None,
    mask_size_range: tuple[int, int] = (128, 128),
) -> Image.Image:
    """Reproduce the square masks used by the original notebook demo."""
    import random

    import numpy as np
    from PIL import Image

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    mask_array = np.array(Image.new("L", (image_size, image_size), 0))
    for i in range(num_squares):
        if seed is not None:
            random.seed(seed + i * 1000)
            np.random.seed(seed + i * 1000)
        mask_size = random.randint(mask_size_range[0], mask_size_range[1])
        x = random.randint(0, max(1, image_size - mask_size))
        y = random.randint(0, max(1, image_size - mask_size))
        mask_array[y:y + mask_size, x:x + mask_size] = 255
    return Image.fromarray(mask_array)


class InpaintingGenerator:
    """Load one LoRA adapter and reuse the pipeline across restoration calls."""

    image_size = IMAGE_SIZE

    def __init__(
        self, checkpoint: str | Path = "checkpoint-5", device: str | None = None
    ) -> None:
        self.checkpoint_path = resolve_checkpoint(checkpoint)

        import torch
        from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
        from peft import PeftModel

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Use --device cpu or install CUDA PyTorch.")
        model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            BASE_MODEL_ID, torch_dtype=model_dtype
        )
        base_unet = UNet2DConditionModel.from_pretrained(
            BASE_MODEL_ID, subfolder="unet", torch_dtype=model_dtype
        )
        trained_unet = PeftModel.from_pretrained(base_unet, self.checkpoint_path)
        trained_unet = trained_unet.merge_and_unload()
        trained_unet = trained_unet.to(dtype=model_dtype)
        trained_unet.eval()
        pipe.unet = trained_unet.to(self.device)
        self.pipeline = pipe.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        num_inference_steps: int = 50,
        seed: int | None = None,
    ) -> Image.Image:
        """Restore white mask regions; None continues the current torch RNG stream."""
        import torch
        from PIL import Image

        if image.size != mask.size:
            raise ValueError("The input image and mask must have the same dimensions.")
        if not 1 <= num_inference_steps <= 1000:
            raise ValueError("The number of inference steps must be between 1 and 1000.")
        image = image.convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        mask = mask.convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            result = self.pipeline(
                prompt="", image=image, mask_image=mask,
                num_inference_steps=num_inference_steps, guidance_scale=7.5,
            )
        return result.images[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Restore a single image using an explicit mask file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--mask", type=Path, required=True, help="Mask path; white marks regions to restore.")
    parser.add_argument(
        "--checkpoint", default="checkpoint-5",
        help="Saved checkpoint name or path to an adapter's unet directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps (1-1000).")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Default: CUDA if available.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG or JPEG path.")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be between 1 and 1000.")
    if not -(2**63) <= args.seed <= 2**64 - 1:
        parser.error("--seed must be between -2**63 and 2**64 - 1.")
    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        parser.error("--output must use a .png, .jpg or .jpeg extension.")

    from PIL import Image

    try:
        checkpoint_path = resolve_checkpoint(args.checkpoint)
        with Image.open(args.image.expanduser()) as source:
            image = source.convert("RGB")
        with Image.open(args.mask.expanduser()) as source:
            mask = source.convert("L")
        if image.size != mask.size:
            raise ValueError("The input image and mask must have the same dimensions.")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    generator = InpaintingGenerator(checkpoint_path, args.device)
    print(f"Using device: {generator.device}")
    print(f"Loading checkpoint from {generator.checkpoint_path}")
    result = generator.inpaint(image, mask, args.steps, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved image to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
