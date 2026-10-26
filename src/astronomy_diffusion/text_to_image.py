"""Generate images with the original Task1 DDPM sampling procedure."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "Task1" / "ddpm-NASA-512"


def resolve_checkpoint(checkpoint: str | Path = "checkpoint-2") -> Path:
    """Resolve a saved checkpoint name or an explicit adapter directory."""
    path = Path(checkpoint).expanduser()
    if path.parent == Path(".") and path.name.startswith("checkpoint-"):
        path = CHECKPOINT_ROOT / path / "unet"
    path = path.resolve()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Missing adapter file: {path / filename}")
    return path


class TextToImageGenerator:
    """Load one LoRA adapter and reuse the models across generation calls."""

    def __init__(
        self, checkpoint: str | Path = "checkpoint-2", device: str | None = None
    ) -> None:
        self.checkpoint_path = resolve_checkpoint(checkpoint)

        import torch
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from peft import PeftModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Use --device cpu or install CUDA PyTorch.")

        base_model = UNet2DConditionModel.from_pretrained(
            BASE_MODEL_ID, subfolder="unet"
        ).to(self.device)
        self.model = PeftModel.from_pretrained(base_model, self.checkpoint_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()

        self.vae = AutoencoderKL.from_pretrained(
            BASE_MODEL_ID, subfolder="vae"
        ).to(self.device)
        self.vae.eval()
        self.text_encoder = CLIPTextModel.from_pretrained(
            BASE_MODEL_ID, subfolder="text_encoder"
        ).to(self.device)
        self.text_encoder.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(
            BASE_MODEL_ID, subfolder="tokenizer"
        )
        self.scheduler = DDPMScheduler.from_pretrained(
            BASE_MODEL_ID, subfolder="scheduler"
        )

    def generate(
        self, prompts: Sequence[str], num_inference_steps: int = 50, seed: int = 42
    ) -> list[Image.Image]:
        """Generate 512x512 images, using seed + prompt index as in Task1."""
        import torch
        from PIL import Image
        from tqdm.auto import tqdm

        if isinstance(prompts, str):
            raise TypeError("Pass a sequence of prompts, for example [prompt].")
        if not 1 <= num_inference_steps <= self.scheduler.config.num_train_timesteps:
            raise ValueError("The number of inference steps must be between 1 and 1000.")

        self.model.eval()
        self.text_encoder.eval()
        self.vae.eval()
        images = []
        for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
            torch.manual_seed(seed + i)
            with torch.no_grad():
                text_input = self.tokenizer(
                    [prompt], padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt",
                )
                input_ids = text_input.input_ids.to(self.device)
                encoder_hidden_states = self.text_encoder(input_ids).last_hidden_state
                latents = torch.randn((1, 4, 64, 64), device=self.device)
                self.scheduler.set_timesteps(num_inference_steps)
                for t in self.scheduler.timesteps:
                    noise_pred = self.model(
                        latents, timestep=t,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )[0]
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                latents = latents / self.vae.config.scaling_factor
                image = self.vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (image * 255).round().astype("uint8")
                images.append(Image.fromarray(image))
                del latents, image, encoder_hidden_states, input_ids, text_input
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return images


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a single image from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Text describing the image.")
    parser.add_argument(
        "--checkpoint", default="checkpoint-2",
        help="Saved checkpoint name or path to an adapter's unet directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps (1-1000).")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Default: CUDA if available.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG or JPEG path.")
    args = parser.parse_args(argv)

    if not args.prompt.strip():
        parser.error("--prompt must contain non-whitespace text.")
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be between 1 and 1000.")
    if not -(2**63) <= args.seed <= 2**64 - 1:
        parser.error("--seed must be between -2**63 and 2**64 - 1.")
    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        parser.error("--output must use a .png, .jpg or .jpeg extension.")
    try:
        checkpoint_path = resolve_checkpoint(args.checkpoint)
    except FileNotFoundError as error:
        parser.error(str(error))

    generator = TextToImageGenerator(checkpoint_path, args.device)
    print(f"Using device: {generator.device}")
    print(f"Loading checkpoint from {generator.checkpoint_path}")
    image = generator.generate([args.prompt], args.steps, args.seed)[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
