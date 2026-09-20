import os
import psutil
import GPUtil
import yaml

def get_hardware_specs():
    # 1. Detect CPU Cores
    cpu_cores = os.cpu_count() or 4
    
    # 2. Detect Total RAM (Rounded to nearest GB)
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3))
    
    # 3. Detect GPU & VRAM
    try:
        gpus = GPUtil.getGPUs()
        gpu_count = len(gpus)
        # Sum up VRAM if multiple, or map per GPU. 
        # Converting MB to GB (rounded)
        vram_per_gpu = round(gpus[0].memoryTotal / 1024) if gpu_count > 0 else 0
    except Exception:
        # Fallback if no NVIDIA GPU or drivers are missing
        gpu_count = 0
        vram_per_gpu = 0

    return {
        "name": "desktop-steeve",
        "cpu": cpu_cores,
        "ram": total_ram_gb,
        "gpu": gpu_count,
        "vram": vram_per_gpu
    }

def update_config_yaml(target_path):
    specs = get_hardware_specs()
    
    # Structure the YAML format exactly as required
    yaml_data = {
        "configurations": [specs]
    }
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Write directly to Steeve's target path
    with open(target_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
    print(f"✅ Configuration successfully written to: {target_path}")
    print(yaml.dump(yaml_data, default_flow_style=False, sort_keys=False))

if __name__ == "__main__":
    # Target path from your instructions
    CONFIG_PATH = r"c:\Users\Steeve\Sources\github.com\SteeveGL\ollama-modelfiles\config.yaml"
    update_config_yaml(CONFIG_PATH)
