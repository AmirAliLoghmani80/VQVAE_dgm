#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import warnings
from data import get_dataloader
from train_model import train_model_vqvae

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Deep Learning Project")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the configuration file (YAML format)."
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from a YAML file."""
    import yaml
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
#     config = load_config(args.config)

    # Step 1: Load data
#     batch_size = config.get("batch_size", 32)
    dataloader_train, dataloader_val = get_dataloader(batch_size)

    # Step 2: Train model
#     epochs = config.get("epochs", 10)
#     lr = config.get("learning_rate", 0.001)
    model = train_model_vqvae(dataloader_train, 32, 0.001)

    # Save the trained model
#     model_save_path = config.get("model_save_path", "trained_model.pth")
    model_save_path = "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as '{model_save_path}'")

if __name__ == "__main__":
    # Suppress warnings if needed
    warnings.filterwarnings("ignore")
    main()

