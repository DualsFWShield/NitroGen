
# NitroGen v2

NitroGen v2 is a tweaked version made by DualsFWShield of the official NitroGen made originaly by NVIDIA.
Only works on windows. Tested with these exact files and versions.

# Installation
## Setup

Install this repo:
```bash
git clone https://github.com/DualsFWShield/NitroGen.git
```
Install dependencies:

Install python and ViGEm Bus Driver
Don't forget to check the boxes to add them to the PATH.
(Given in the repository but avaible on the official website)
ViGEmBus : https://vigembusdriver.com/download/
Python : https://www.python.org/downloads/release/python-31311/

Install torch
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
``` 
Install HuggingFace
```bash
pip install -U "huggingface_hub"
```

Download NitroGen checkpoint from [HuggingFace](https://huggingface.co/nvidia/NitroGen):
```bash
hf download nvidia/NitroGen ng.pt
```

Install NitroGen:
```bash
cd NitroGen
pip install -e .
```

# Getting Started

Use the StartNitroGen.bat file to start the agent.
Enter your path to NitroGen and the path to ng.pt (if not in the same directory) in the StartNitroGen.bat file.
Modify the configuration in the StartNitroGen.bat file to your liking :

| Parameter | Description |
| :--- | :--- |
| `PROJECT_DIR` | Path to NitroGen |
| `MODEL_FILE` | Path to ng.pt |
| `TIMESTEPS` | Number of timesteps |
| `ARGS` | Arguments for the agent |
| `--compile` | Compile the model |
| `--ctx` | Number of contexts |
| `--cfg` | CFG scale |
| `--no-cache` | Disable cache |
| `timesteps` | Number of timesteps (default: 12), more timesteps = more actions |

Official project : https://github.com/nvidia/NitroGen
<div align="center">
  <p style="font-size: 1.2em;">
    <a href="https://nitrogen.minedojo.org/"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/nvidia/NitroGen"><strong>Model</strong></a>
  </p>
</div>