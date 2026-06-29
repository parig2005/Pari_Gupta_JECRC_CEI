# MNIST Denoising Autoencoder

Project overview
----------------

A minimal PyTorch convolutional denoising autoencoder that learns to remove
additive Gaussian noise from MNIST PNG images. The repo includes data loaders,
model definition, training and prediction scripts, and produces example
visualizations after training.

Why this project
-----------------

- Demonstrates a compact convolutional autoencoder for image denoising.
- Small, easy-to-read PyTorch code suitable for learning and quick experiments.

Quick start (recommended)
-------------------------

1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Prepare the dataset: place the PNG-MNIST files under `dataset/mnist_png`
     with `training/` and `testing/` subfolders. Expected structure:

```
dataset/mnist_png/
    training/
        0/
        1/
        ...
    testing/
        0/
        1/
        ...
```

4. Run training and prediction from the `src` folder:

```bash
cd src
python train.py    # trains and writes ../outputs/model.pth
python predict.py  # creates ../outputs/results.png
```

Files and purpose
-----------------

- `src/dataset.py`: dataset loader using `torchvision.datasets.ImageFolder`.
- `src/model.py`: `DenoisingAutoEncoder` model (small Conv encoder/decoder).
- `src/train.py`: training loop (adds Gaussian noise, saves model to `../outputs/model.pth`).
- `src/predict.py`: loads model, denoises a test batch, and saves `../outputs/results.png`.
- `requirements.txt`: Python dependencies.

Default hyperparameters (current code)
-------------------------------------

- Epochs: 10 (hard-coded in `train.py` loop)
- Learning rate: 1e-3 (`torch.optim.Adam`)
- Train batch size: 128 (default in `get_loaders`)
- Test batch size: 10
- Noise: Gaussian with std 0.5 added to inputs via `torch.randn_like`

Model summary
-------------

Encoder:
- Conv2d(1,16,3,padding=1) -> ReLU -> MaxPool2d(2)
- Conv2d(16,8,3,padding=1) -> ReLU -> MaxPool2d(2)

Decoder:
- ConvTranspose2d(8,16,2,stride=2) -> ReLU
- ConvTranspose2d(16,1,2,stride=2) -> Sigmoid

Outputs
-------

- `outputs/model.pth`: saved model state dict after training.
- `outputs/results.png`: a 3x5 grid showing Original / Noisy / Denoised images.

Tips & troubleshooting
----------------------

- Always run `train.py` and `predict.py` from the `src` directory because the
    scripts use relative paths (`../dataset/mnist_png` and `../outputs`).
- If you get a `FileNotFoundError`, verify the dataset folder and subfolders
    are present and you gave the correct path.
- To force CPU (if CUDA is misdetected), set the environment variable
    `CUDA_VISIBLE_DEVICES=""` before running the script.

Suggested improvements (I can implement any of these)
---------------------------------------------------

- Add CLI flags for dataset path, epochs, batch size, and learning rate.
- Add a small data-setup script to download/extract MNIST PNG automatically.
- Add unit tests and an example notebook that walks through training.

Contributing
------------

PRs welcome — open an issue for discussion before large changes. Keep code
style consistent with the existing files and add tests for new functionality.

License
-------

This project is provided as-is for educational use.
