# Image inpainting

Use Python 3.10. Install PyTorch for your device first. For NVIDIA GPUs with a compatible driver, the tested CUDA 12.6 build is:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

From the repository root, install the package in editable mode:

```sh
python -m pip install -e .
```

The first run downloads the base inpainting model from Hugging Face. The CSV and training caches are not needed.

## Prepare an example

Run this Python code from the repository root to create an input and mask from an included image:

```python
from pathlib import Path
from PIL import Image
from astronomy_diffusion.inpainting import generate_random_square_mask

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
image = Image.open("Task2/ddpm-NASA-inpainting-512/img/1452899771.jpg").convert("RGB")
image = image.resize((512, 512), Image.Resampling.LANCZOS)
image.save(output_dir / "inpainting_input.png")
generate_random_square_mask(seed=0).save(output_dir / "inpainting_mask.png")
```

You can use your own image and mask instead. They must have the same original dimensions. White mask pixels mark regions to regenerate; black pixels provide the surrounding context. Both inputs are resized to 512x512, using nearest-neighbor interpolation for the mask.

## Command line

```sh
astronomy-inpaint --image outputs/inpainting_input.png --mask outputs/inpainting_mask.png --checkpoint checkpoint-5 --seed 42 --steps 50 --output outputs/inpainted.png
```

You can also use `python -m astronomy_diffusion.inpainting` with the same arguments. After editable installation, both commands work from any directory. Relative input, mask, and output paths are resolved against the current working directory.

`--checkpoint` accepts `checkpoint-4`, `checkpoint-5`, or a path to a directory containing `adapter_config.json` and `adapter_model.safetensors`. Saved names resolve to this checkout's `Task2/ddpm-NASA-inpainting-512` directory. `--device cpu` or `--device cuda` overrides automatic device selection.

The mask helper's seed controls square positions. The CLI's `--seed` controls diffusion sampling. The pipeline uses an empty prompt, guidance scale 7.5, and FP16 on CUDA or FP32 on CPU, matching the original notebook.

## Python

```python
from PIL import Image
from astronomy_diffusion.inpainting import InpaintingGenerator

generator = InpaintingGenerator(checkpoint="checkpoint-5")
image = Image.open("outputs/inpainting_input.png")
mask = Image.open("outputs/inpainting_mask.png")
result = generator.inpaint(image, mask, num_inference_steps=50, seed=42)
result.save("outputs/inpainted.png")
```

In the Python API, `seed=None` continues the current torch random-number stream, as the original notebook did across its three examples.

## Notebook

For plots, install `python -m pip install -r Task2/requirements-inference.txt` and select the same Python environment in your notebook editor. Run the last two cells of `iigenerator.ipynb`. They call the installed module; no training cells are required.
