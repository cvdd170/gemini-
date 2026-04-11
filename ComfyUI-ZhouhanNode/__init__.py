import os
import folder_paths
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 注册并指向您指定的模型文件夹: ComfyUI/models/zh
ZH_MODELS_DIR = os.path.join(folder_paths.models_dir, "zh")
if not os.path.exists(ZH_MODELS_DIR):
    os.makedirs(ZH_MODELS_DIR)

# 动态读取 zh 文件夹下的四个模型目录
def get_gemma_models():
    models = []
    if os.path.exists(ZH_MODELS_DIR):
        for item in os.listdir(ZH_MODELS_DIR):
            item_path = os.path.join(ZH_MODELS_DIR, item)
            # 识别文件夹为可选模型
            if os.path.isdir(item_path):
                models.append(item)
    if not models:
        return ["No models found in models/zh"]
    return sorted(models)

class ZhouhanPromptDirector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "一个赛博朋克风格的武士在霓虹雨中"}),
                "gemma_model": (get_gemma_models(),),
                "max_new_tokens": ("INT", {"default": 150, "min": 50, "max": 500}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("english_prompt",)
    FUNCTION = "optimize_prompt"
    # 将节点分类归属到您的专属菜单下
    CATEGORY = "Zhouhan Nodes"

    def optimize_prompt(self, text, gemma_model, max_new_tokens):
        if gemma_model == "No models found in models/zh":
            return ("Error: 请确保四个Gemma模型已放入 ComfyUI/models/zh 文件夹中",)

        model_path = os.path.join(ZH_MODELS_DIR, gemma_model)
        
        # 导演级 System Prompt 设定
        instruction = f"""You are a master film director and expert cinematographer. 
Your task is to take the user's basic concept (which may be in Chinese) and expand it into a highly detailed, professional image generation prompt in ENGLISH.
- Keep the core semantic meaning strictly intact.
- Add rich details about: camera angles, lighting (e.g., cinematic lighting, volumetric), mood, composition, and lens specifications.
- OUTPUT ONLY THE FINAL PROMPT IN ENGLISH. Do not add any conversational filler, greetings, or explanations.

Concept: {text}
Final Prompt:"""

        messages = [
            {"role": "user", "content": instruction}
        ]

        # 在终端打印带有专属命名的日志
        print(f"[Zhouhan Node] 正在从 {gemma_model} 加载 Gemma 语言模型...")
        
        try:
            # 加载 Tokenizer 和 模型
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype=torch.bfloat16, # Gemma 架构强烈建议使用 bfloat16
            )

            # 格式化输入，自动应用 Gemma 的特殊 token
            text_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text_input], return_tensors="pt").to(self.device)

            # 生成提示词
            print("[Zhouhan Node] 导演级提示词生成中...")
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7, 
                do_sample=True,
                top_p=0.9
            )
            
            # 截取新生成的内容
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 执行显存释放 (对画图节点至关重要)
            print("[Zhouhan Node] 提示词生成完毕，正在卸载 Gemma 模型以释放显存...")
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            return (response.strip(),)
            
        except Exception as e:
            # 捕获异常并清理内存
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[Zhouhan Node] 发生错误: {str(e)}")
            return (f"Error generating prompt: {str(e)}",)

# 注册您的专属节点
NODE_CLASS_MAPPINGS = {
    "ZhouhanPromptDirector": ZhouhanPromptDirector
}

# 设置 UI 面板上显示的节点名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhouhanPromptDirector": "🎬 Zhouhan Director Prompt"
}