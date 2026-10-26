Task1: astronomy image generation with Stable Diffusion 1.5 and LoRA

Setup and command-line examples: see INFERENCE.md.

Use Python 3.10. Install PyTorch for your device first. The tested NVIDIA CUDA 12.6 build is:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

From the repository root, install the package and notebook dependencies:
python -m pip install -e ".[notebooks]"

For notebook inference, select that Python environment and run the last five cells of nasa512.ipynb, starting with "Run this cell and the cells below for inference". Training cells and the CSV are not required. After editable installation, inference works from the repository root or Task1 directory.

Change checkpoint_name in the first inference cell to select checkpoint-2, checkpoint-3, or checkpoint-10. Change the prompts in the final example cells to generate your own images.

The original experiment notes recommend checkpoints 2 and 3. Checkpoint 10 is included as an example of degraded generation. This is a qualitative observation, not a controlled quality benchmark.

The first run downloads the base SD 1.5 components from:
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
The local checkpoint directories contain LoRA adapters only.

The original training run used Kaggle with two T4 GPUs. Local training requires adapting the configuration.

Original training log:
https://www.kaggle.com/code/vladimirtoporkov/nasa256
