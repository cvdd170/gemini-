# gemini-
gemini模型用于comfyui提示词润色以及英文输出用来解决ltx英文提示词9"Using the Gemini model to refine prompts for ComfyUI and generate English output, specifically to handle the translation and optimization of LTX prompts from Chinese to English."0
# ComfyUI-ZhouhanNode: Cinematic Prompt Director 🎬

A powerful, fully localized custom node for ComfyUI that transforms simple Chinese concepts into highly detailed, cinematic English prompts. It is powered by local **Gemma 4** models and specifically optimized to generate the complex "director-level" camera and lighting terminologies required by advanced video generation models like **LTX-Video**.

## ✨ Why this Node? (The LTX-Video Problem)
Video generation models like LTX-Video have strict prompt requirements. They do not understand simple phrasing well and demand explicit cinematic descriptions (e.g., tracking shots, motion blur, anamorphic lens, specific lighting). 

**Zhouhan Director Prompt** solves this by:
1. **Zero-Barrier Translation:** Type your ideas in simple Chinese.
2. **Cinematic Expansion:** Automatically injects professional cinematography terms.
3. **Strict VRAM Management:** It only loads the LLM into VRAM during prompt generation and **instantly unloads it** afterward. This ensures your GPU memory is 100% free for the heavy lifting of LTX-Video or FLUX generation.

## 🚀 Features
* **Fully Local & Offline:** No API keys required. Uses Google's open-weights Gemma 4 models.
* **Dynamic Model Switching:** Supports 4 different sizes of Gemma models (e2b, e4b, 26b, 31b) directly from a dropdown menu.
* **Auto VRAM Clearing:** Employs aggressive garbage collection and `torch.cuda.empty_cache()` to prevent OutOfMemory errors.
* **Parameter Control:** Built-in temperature and seed controls to tweak the AI's creativity.

## 🛠️ Installation

1. Navigate to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
Clone this repository:

Bash
git clone [https://github.com/cvdd170/gemini-.git](https://github.com/cvdd170/gemini-.git)
(Note: You can rename the cloned folder to ComfyUI-ZhouhanNode for better organization).

Install required Python packages (if missing):

Bash
pip install transformers torch accelerate
📦 Model Setup (Crucial Step)
This node does not download models automatically. You must download the Gemma 4 Instruct weights (safetensors format) and place them in specific folders.

Go to your ComfyUI root directory and create the following path: ComfyUI/models/zh/

Download your preferred Gemma 4 Instruct model(s) from HuggingFace or ModelScope.

Create a subfolder for the model and place all core files (config.json, model.safetensors, tokenizer.json, etc.) inside.

Recommended Setup for Video Generation:

For Ultimate Speed: Download gemma-4-e2b-it and place it in ComfyUI/models/zh/gemma-4-e2b-it

For Epic Cinematic Detail (Recommended): Download gemma-4-e4b-it and place it in ComfyUI/models/zh/gemma-4-e4b-it

🎮 Usage
Restart ComfyUI.

Right-click on the canvas -> Add Node -> Zhouhan Nodes -> 🎬 Zhouhan Director Prompt.

Connect the english_prompt string output to the positive prompt input of your text encoder (e.g., the LTX-Video prompt node).

Type your idea in Chinese, select your downloaded Gemma model, hit "Queue Prompt", and watch the magic happen!
