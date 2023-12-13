# 🌴 Installation

1. Prepare conda environment: `conda create -n smoe python=3.11` (If your environment name is not `smoe`, you may need to change environment in launching scripts)
2. Add correct environment variables in `~/.bashrc` (`gcc` is set to newer version for installing `flash-attn`). e.g.:
    ```bash
    export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH
    export PATH=/mnt/petrelfs/share/gcc-10.1.0/bin:$PATH
    export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc-10.1.0/lib64:$LD_LIBRARY_PATH
    ```
3. Take the variables into effect: `source ~/.bashrc`
4. Install PyTorch (CUDA-11.8): `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
5. Install dependencies: `pip install -r requirements.txt`
6. Install `flash-attn`: `pip install flash-attn==2.0.1 --no-build-isolation`. You may need to follow the [flash-attn installation instructions](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) to avoid some errors.
7. Install the latest Git: `conda install git`
8. Clone the repo: `git clone git@github.com:pjlab-sys4nlp/train-moe.git` (If you don't setup the ssh key to GitHub, you may not able to clone through ssh. Check the [docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) about it.)
9. Change current directory: `cd train-moe`
10. Install `smoe` in [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e): `pip install -e .[dev]`
11. Setup `pre-commit` hooks: `pre-commit install`
