Task2: astronomy image inpainting with Stable Diffusion and LoRA

Setup and command-line examples: see INFERENCE.md.

Use Python 3.10. Install PyTorch for your device first. The tested NVIDIA CUDA 12.6 build is:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

From the repository root, install the package and notebook dependencies:
python -m pip install -e .
python -m pip install -r Task2/requirements-inference.txt

For notebook inference, select that Python environment and run the last two cells of iigenerator.ipynb, starting with "Run this cell and the cell below for inference". Training cells and the CSV are not required. After editable installation, the demo works from the repository root or Task2 directory.

Change checkpoint_name in the first inference cell to select checkpoint-5 (default) or checkpoint-4. The final cell restores three included images with generated square masks. You can change its input paths to use your own images.

The command-line interface accepts an image, a mask of the same dimensions, a checkpoint, a seed, and a number of steps. White mask pixels mark regions to regenerate. See INFERENCE.md for a complete example using an included image.

The first run downloads the base inpainting model from:
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting
The local checkpoint directories contain LoRA adapters only.

The original training run used Kaggle with two T4 GPUs. Local training requires adapting the configuration.
