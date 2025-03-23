import os
import argparse
import zipfile
import shutil
from pathlib import Path
import urllib.request

def download_file(url, output_path):
    """
    Download a file from URL to the specified path.
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
    """
    print(f"Downloading from {url}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a progress tracker
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        if total_size > 0:
            print(f"\rProgress: {percent:.1f}% ({downloaded} / {total_size} bytes)", end="")
    
    # Download the file
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def extract_zip(zip_path, extract_dir):
    """
    Extract a zip file to the specified directory.
    
    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract to
    """
    print(f"Extracting {zip_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete!")
        return True
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False

def download_and_setup_models(output_dir, models_to_download):
    """
    Download and set up pretrained models.
    
    Args:
        output_dir: Base directory to save models to
        models_to_download: List of model types to download ('classifier', 'separator', or 'both')
    """
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Placeholder for model download URLs
    # In a real-world scenario, you would host these models on a cloud service
    model_urls = {
        "classifier": "https://example.com/pretrained_instrument_classifier.zip",
        "separator": "https://example.com/pretrained_instrument_separator.zip",
    }
    
    # Download requested models
    for model_type in models_to_download:
        if model_type not in model_urls:
            print(f"Unknown model type: {model_type}")
            continue
        
        url = model_urls[model_type]
        zip_path = os.path.join(output_dir, f"{model_type}_models.zip")
        extract_dir = os.path.join(output_dir, f"{model_type}_models")
        
        # Download and extract
        if download_file(url, zip_path):
            if extract_zip(zip_path, extract_dir):
                # Clean up zip file
                os.remove(zip_path)
            
            # Set up models in the correct structure
            if model_type == "classifier":
                # Create an 'ensemble' directory and move models there
                ensemble_dir = os.path.join(output_dir, "ensemble")
                os.makedirs(ensemble_dir, exist_ok=True)
                
                # Move classifier models to ensemble directory
                for file in os.listdir(extract_dir):
                    if file.endswith('.pth'):
                        src = os.path.join(extract_dir, file)
                        dst = os.path.join(ensemble_dir, file)
                        shutil.move(src, dst)
                
                print(f"Classifier models set up in {ensemble_dir}")
                
            elif model_type == "separator":
                # Move separator models to the separation_models directory
                sep_dir = os.path.join(output_dir, "separation_models")
                os.makedirs(sep_dir, exist_ok=True)
                
                # Move separator models
                for file in os.listdir(extract_dir):
                    if file.endswith('.pth'):
                        src = os.path.join(extract_dir, file)
                        dst = os.path.join(sep_dir, file)
                        shutil.move(src, dst)
                
                print(f"Separator models set up in {sep_dir}")
            
            # Clean up extraction directory
            shutil.rmtree(extract_dir)

def main():
    parser = argparse.ArgumentParser(description='Download pretrained instrument models')
    parser.add_argument('--output_dir', type=str, default='./pretrained',
                        help='Directory to save the pretrained models')
    parser.add_argument('--models', type=str, choices=['classifier', 'separator', 'both'], 
                        default='both', help='Which models to download')
    args = parser.parse_args()
    
    models_to_download = ['classifier', 'separator'] if args.models == 'both' else [args.models]
    
    print(f"Downloading {'both models' if args.models == 'both' else args.models} to {args.output_dir}")
    download_and_setup_models(args.output_dir, models_to_download)
    
    # Provide usage instructions
    print("\nDownload complete! To use the pretrained models:")
    if 'classifier' in models_to_download and 'separator' in models_to_download:
        print("\nTo extract instruments from an audio file:")
        print(f"python extract_instruments.py --input path/to/audio.wav --output_dir ./extracted --classifier_dir {args.output_dir} --separation_dir {args.output_dir}/separation_models --format mp3")
    elif 'classifier' in models_to_download:
        print("\nClassifier models downloaded. You still need to train separation models:")
        print(f"python train_separation.py --classifier_dir {args.output_dir} --output_dir ./separation_models")
    elif 'separator' in models_to_download:
        print("\nSeparator models downloaded. You'll need classifier models to use them:")
        print("python train.py")
        print("Or download the classifier models with:")
        print(f"python download_pretrained.py --models classifier --output_dir {args.output_dir}")
    
    print("\n⚠️ Note: The download URLs in this script are placeholders. In a real implementation, you would need to host the models on a cloud service and update the URLs.")

if __name__ == "__main__":
    main() 