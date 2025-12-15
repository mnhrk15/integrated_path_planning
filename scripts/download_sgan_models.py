#!/usr/bin/env python3
"""
Download pretrained Social-GAN models from official repository.
Reference: https://github.com/agrimgupta92/sgan
"""

import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for download"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path, desc: str = "Downloading"):
    """Download file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=desc) as t:
        urlretrieve(url, filename=str(output_path), 
                   reporthook=t.update_to)


def download_sgan_models(models_dir: Path, download_pooling: bool = False):
    """
    Download Social-GAN pretrained models
    
    Args:
        models_dir: Directory to save models
        download_pooling: Whether to download pooling models too
    """
    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download main models
    print("\n" + "="*60)
    print("Downloading Social-GAN Pretrained Models")
    print("="*60 + "\n")
    
    models_url = "https://www.dropbox.com/s/h8q5z4axfgzx9eb/models.zip?dl=1"
    zip_path = models_dir / "models.zip"
    
    print(f"Source: {models_url}")
    print(f"Target: {models_dir / 'sgan-models'}\n")
    
    try:
        # Download
        download_url(models_url, zip_path, "Downloading main models")
        
        # Extract
        print("\nExtracting models...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        # Verify
        sgan_models_dir = models_dir / "sgan-models"
        if sgan_models_dir.exists():
            print(f"\n✓ Models extracted successfully!")
            print(f"\nAvailable models in {sgan_models_dir}:")
            for model_file in sorted(sgan_models_dir.glob("*.pt")):
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"  - {model_file.name} ({size_mb:.1f} MB)")
        else:
            print("\n✗ Extraction failed")
            return False
        
        # Cleanup
        zip_path.unlink()
        print(f"\n✓ Cleanup complete")
        
    except Exception as e:
        print(f"\n✗ Error downloading main models: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False
    
    # Download pooling models if requested
    if download_pooling:
        print("\n" + "="*60)
        print("Downloading Pooling Models (SGAN-P)")
        print("="*60 + "\n")
        
        pooling_url = "https://www.dropbox.com/s/ppfw8xfb95jxlvl/sgan-p-models.zip?dl=1"
        zip_path_p = models_dir / "sgan-p-models.zip"
        
        try:
            download_url(pooling_url, zip_path_p, "Downloading pooling models")
            
            print("\nExtracting pooling models...")
            with zipfile.ZipFile(zip_path_p, 'r') as zip_ref:
                zip_ref.extractall(models_dir)
            
            sgan_p_models_dir = models_dir / "sgan-p-models"
            if sgan_p_models_dir.exists():
                print(f"\n✓ Pooling models extracted successfully!")
                print(f"\nAvailable pooling models in {sgan_p_models_dir}:")
                for model_file in sorted(sgan_p_models_dir.glob("*.pt")):
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    print(f"  - {model_file.name} ({size_mb:.1f} MB)")
            
            zip_path_p.unlink()
            
        except Exception as e:
            print(f"\n✗ Error downloading pooling models: {e}")
            if zip_path_p.exists():
                zip_path_p.unlink()
    
    print("\n" + "="*60)
    print("Download Complete")
    print("="*60 + "\n")
    
    print("Usage in configuration file:")
    print('  sgan_model_path: "models/sgan-models/eth_8.pt"')
    print("\nAvailable datasets: eth, hotel, univ, zara1, zara2")
    print("Available prediction lengths: 8, 12")
    print()
    
    return True


def main():
    """Main function"""
    # Get project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    models_dir = project_root / "models"
    
    # Parse arguments
    download_pooling = False
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-p", "--pooling"]:
            download_pooling = True
        elif sys.argv[1] in ["-h", "--help"]:
            print("Usage: python download_sgan_models.py [-p|--pooling]")
            print()
            print("Options:")
            print("  -p, --pooling    Also download pooling models (SGAN-P)")
            print("  -h, --help       Show this help message")
            return
    
    # Download models
    success = download_sgan_models(models_dir, download_pooling)
    
    if success:
        print("✓ All downloads completed successfully!")
    else:
        print("✗ Download failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
