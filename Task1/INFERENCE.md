# Text-to-image inference

Use Python 3.10. Install PyTorch for your device first. For NVIDIA GPUs with a compatible driver, the tested CUDA 12.6 build is:

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

From the repository root, install the package in editable mode:

```sh
python -m pip install -e .
```

The first run downloads the base SD 1.5 components from Hugging Face. The CSV and training caches are not needed.

## Command line

```sh
astronomy-generate --prompt "A colorful nebula with glowing gas clouds and distant stars" --checkpoint checkpoint-2 --seed 42 --steps 50 --output outputs/nebula.png
```

You can also use `python -m astronomy_diffusion.text_to_image` with the same arguments. After editable installation, both commands work from any directory. Relative output paths are resolved against the current working directory.

`--checkpoint` accepts `checkpoint-2`, `checkpoint-3`, `checkpoint-10`, or a path to a directory containing `adapter_config.json` and `adapter_model.safetensors`. Saved names resolve to the adapters in this checkout's `Task1/ddpm-NASA-512` directory. `--device cpu` or `--device cuda` overrides automatic device selection.

## Python

```python
from astronomy_diffusion.text_to_image import TextToImageGenerator

generator = TextToImageGenerator(checkpoint="checkpoint-2")
images = generator.generate(["A colorful nebula in space"], num_inference_steps=50, seed=42)
images[0].save("nebula.png")
```

The module preserves the notebook's DDPM sampling, FP32 model loading, 512x512 resolution, and per-prompt seed offset. It does not add classifier-free guidance or change the trained adapter.

## Notebook

For plots, install `python -m pip install -r Task1/requirements-inference.txt` and select the same Python environment in your notebook editor. Run the last five cells of `nasa512.ipynb`. Those cells call the installed module; no training cells are required.
