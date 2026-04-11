"""Download script for SD v1.5 and ADC weights."""
import sys
import os

# Try to install huggingface_hub if not available
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    import subprocess
    import shutil
    pip_cmd = ["uv", "pip", "install"] if shutil.which("uv") else [sys.executable, "-m", "pip", "install"]
    subprocess.check_call(pip_cmd + ["huggingface_hub"])
    from huggingface_hub import hf_hub_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def download_sd15():
    dest = os.path.join(BASE_DIR, "stable-diffusion-v1-5")
    os.makedirs(dest, exist_ok=True)
    ckpt_path = os.path.join(dest, "v1-5-pruned.ckpt")
    if os.path.exists(ckpt_path):
        print(f"SD v1.5 already exists at: {ckpt_path}")
        return ckpt_path
    print("Downloading SD v1.5 v1-5-pruned.ckpt (~7.7 GB)...")
    path = hf_hub_download(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        filename="v1-5-pruned.ckpt",
        local_dir=dest,
    )
    print(f"Done: {path}")
    return path

def download_adc_weights():
    dest = os.path.join(BASE_DIR, "adc_weights")
    os.makedirs(dest, exist_ok=True)
    ckpt_path = os.path.join(dest, "merged_pytorch_model.pth")
    if os.path.exists(ckpt_path):
        print(f"ADC weights already exist at: {ckpt_path}")
        return ckpt_path
    print("Downloading ADC merged_pytorch_model.pth (~9.6 GB)...")
    path = hf_hub_download(
        repo_id="SylarQ/ADC",
        filename="merged_pytorch_model.pth",
        local_dir=dest,
    )
    print(f"Done: {path}")
    return path

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sd15", action="store_true", help="Download SD v1.5")
    p.add_argument("--adc", action="store_true", help="Download ADC weights")
    p.add_argument("--all", action="store_true", help="Download everything")
    args = p.parse_args()

    if args.all or args.sd15:
        download_sd15()
    if args.all or args.adc:
        download_adc_weights()
    if not (args.sd15 or args.adc or args.all):
        print("Usage: python download_weights.py [--sd15] [--adc] [--all]")
