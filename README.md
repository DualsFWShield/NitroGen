
# NitroGen v2

NitroGen v2 is a tweaked version made by DualsFWShield of the official NitroGen made originaly by NVIDIA.
Only works on windows. Tested with these exact files and versions.

# Installation
## Setup

Clone this repo:
```bash
git clone https://github.com/DualsFWShield/NitroGen.git
```
## Install dependencies:

- Install python and ViGEm Bus Driver (Given in the repository but avaible on the official website).
- Don't forget to check the boxes to add them to the PATH.

ViGEmBus : 
```bash
https://vigembusdriver.com/download/
```

Python : 
```bash
https://www.python.org/downloads/release/python-31311/
```

### Install torch
```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
``` 
### Install HuggingFace
```bash
pip install -U "huggingface_hub"
```

### Download NitroGen checkpoint from [HuggingFace](https://huggingface.co/nvidia/NitroGen):
```bash
hf download nvidia/NitroGen ng.pt
```

### Install NitroGen:
```bash
cd NitroGen
pip install -e .
```

# Getting Started

- Use the StartNitroGen.bat file to start the agent.
- Enter your path to NitroGen and the path to ng.pt (if not in the same directory) in the StartNitroGen.bat file.
- Modify the configuration in the StartNitroGen.bat file to your liking :

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

# Tips
- You can use any game that has controller support (i don't know if those with steam input are supported).
- The agent will crash if the game is not in fenetre mode, the whole game needs to be in fenetre mode and fully visible on the screen.
- You can use Cheat Engine to change the game's speed to 0.5 it may help with the agent's performance on some quick and tricky games.
- The agent fail to find the game if multiple exe with the same name are open.

- You can tag me if this version helped you but you are not forced to, i uploaded it the 30/12/2025 and probably will not update it.
- Please note that this version is not the same as the official one, it is a tweaked version.
- Refer to the original project for more information and to get the latest version and updates as the project is still in development and will probably be updated in the future with new features.

Official project : https://github.com/MineDojo/NitroGen
Special Thanks to : https://github.com/Tybost for the help and some tweak

<div align="center">
  <p style="font-size: 1.2em;">
    <a href="https://nitrogen.minedojo.org/"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/nvidia/NitroGen"><strong>Model</strong></a>
  </p>
</div>