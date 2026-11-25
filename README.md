# Spellcast3D

A free, open-source app for making game-ready 3D props **on your own PC**, in a browser.</br>
You need 8GB vram NVIDIA gpu and 16GB ram.

Its idea is to develop several 3D props simultaneously, before moving between stages.</br>
We load each neural net once per stage, and all 3D props just re-use it.</br>
So, the more props you create at once, the more time savings you get.

Using asynchronous loading and Qwen Image 2.1 + Pixal3D + TriFlow/LATO.2/InstantMeshes code.</br>
Highly optimized by Claude Fable 5.2 over many iterations.

| Stage | What it does | Tool |
|---|---|---|
| Images | render or import images or character sheets | **Qwen-Image 2.1** (built in), optionally FLUX.3 Klein or Forge Neo |
| 3D | picture (or several views) -> textured high-poly model | [Pixal3D](https://github.com/IgorAherne/Pixal3D), optionally [Hi3D](http://hi3d.ai/) for the shape |
| Omni-Texture | optional: repaints the model's colours, sharper, from 8 sides | Qwen-Image 2.1</br>custom-trained Marigold Albedo LoRA |
| Retopo | clean low-poly mesh with UVs | [TriFlow](https://github.com/IgorAherne/triflow), [LATO.2](https://github.com/IgorAherne/LATO2-stableprojectorz) or [Instant Meshes](https://github.com/IgorAherne/instant-meshes-py) |
| Projection | bakes the detailed textures onto the clean mesh -> GLB, FBX, OBJ + maps | Spellcast3D's own baker |

## Requirements

- Windows 10 or 11, 64-bit
- NVIDIA graphics card with 8 GB of video memory or more, GTX 10xx series or newer (RTX 30xx and newer are
  the fastest)
- 16 GB of RAM
- About 92 GB of free disk space: downloads libraries, AI models

Don't worry about Python, git or CUDA: they are inside the zip.

## Install

1. Download `spellcast3d-<version>-windows.zip` from the
   [Releases page](https://github.com/IgorAherne/spellcast3d/releases)
2. Unzip it to a **short** folder with only English letters, for example `C:\spellcast3d`.
3. Double-click `run-browser\run-spellcast3d.bat`.

Initial startup takes a few minutes and opens your browser.</br>
The AI tools then install in the background (30 to 90 minutes, mostly downloads).

## Image generators

Pick one per prop on the Images page, under *Model*:

| Model | Speed (1024 x 1024, RTX 3060) | Notes |
|---|---|---|
| **Qwen-Image 2.1 Turbo** | about 15 s | the default; installed at the first start |
| **Qwen-Image 2.1 Full** | about 31 s | sometimes finer detail; a 4.2 GB download, on request |
| **FLUX.3 Klein** | about 30 s | follows *long prompts* best, no transparent background; a 6.8 GB download, on request |

All three take your words and up to 10 example pictures, and can redraw the picture in the viewport with
marks you paint over it. Resolutions: 1K, 1.5K and 2K.

**Forge Neo** is also supported. Open `spellcast3d-config.ini` with Notepad before the
first start and switch the backend:

```ini
; backend = qwen
backend = forge
```

The same file sets the size of the built-in models (`quality`, `text_encoder`): a bigger size is a
little finer and needs more memory. The file explains each setting.

## Models and disk space

AI models (neural nets) live in the `models` folder. 
Updates and reinstalls never touch it, and models you already have are not downloaded again if you put them there first (see [`models/README.txt`](models/README.txt)).

| Folder | What | Size |
|---|---|---|
| `models\qwen-image` | Qwen-Image 2.1 | about 11 GB (+4.2 GB for Full) |
| `models\flux3-klein` | FLUX.3 Klein, optional | 6.8 GB |
| `models\checkpoints` | your image models for Forge Neo | yours |
| `models\pixal3d` | Pixal3D | about 27 GB, plus up to 14 GB of faster-loading copies |
| `models\pixal3d` | multi-view models, optional | 22 GB, plus up to 14 GB of copies |
| `models\hi3d` | Hi3D, optional | 11.4 GB |
| `models\lato2` | LATO.2 | 4.8 GB |
| `models\triflow` | TriFlow | 1.3 GB |

Finished props are saved into the `exports` folder, one folder per prop.

## Updating

Close Spellcast3D and run `tools\update.bat`. It downloads the newest version and reinstalls only what
changed. Your projects, your settings and the `models` folder are kept.

## Licences

Each part keeps its own licence. Read them before selling anything you make: some parts allow
non-commercial use only. [`README - LICENSE.txt`](README%20-%20LICENSE.txt) lists them all, with the
notices some of them ask for.

| Part | Licence |
|---|---|
| Spellcast3D, including its baker | [AGPL-3.0](LICENSE) |
| Qwen-Image 2.1 | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE) (renders/generations can be used commercially) |
| FLUX.3 Klein (optional) | [FLUX Kommunity License](https://huggingface.co/unsloth/flux-3-image-klein-GGUF/blob/12ef13ce920d53f45fe805093910c899668b9e76/LICENSE.md) |
| Forge Neo (optional) | [AGPL-3.0](https://github.com/IgorAherne/stable-diffusion-webui-forge/blob/neo/LICENSE), and the licence of each image model you use with it |
| Pixal3D | [MIT](https://github.com/IgorAherne/Pixal3D/blob/master/LICENSE); parts it downloads: [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0), [nvdiffrast](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt), [nvdiffrec](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt), [DINOv3](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |
| Hi3D (optional) | [Hi3D Open License Agreement](https://huggingface.co/hi3d-official/Hi3Dv1.0/blob/d502e938cbedc6e87aba8884614c5ca6d177a4b6/LICENSE) |
| TriFlow | [ADPNCL 1.0](https://github.com/IgorAherne/triflow/blob/main/LICENSE); part it downloads: [MeshLib](https://github.com/MeshInspector/MeshLib/blob/master/LICENSE) |
| LATO.2 | [MIT](https://github.com/IgorAherne/LATO2-stableprojectorz/blob/master/LICENSE) |
| Instant Meshes | [BSD-3-Clause](https://github.com/IgorAherne/instant-meshes-py/blob/master/LICENSE.txt) |

## For developers

Building from source, rebuilding the wheels and making a release: [`DEVELOPERS.md`](DEVELOPERS.md).

## Community

- Discord: https://discord.gg/aWbnX2qan2
- Website: https://stableprojectorz.com
