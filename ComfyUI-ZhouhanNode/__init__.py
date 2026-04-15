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
                # 接收用户输入的原始概念
                "text": ("STRING", {"multiline": True, "default": "一个赛博朋克风格的武士在霓虹雨中"}),
                
                # 强制接收前置 ZhouhanSkillAdapter 节点传来的数据
                "skill_system_prompt": ("STRING", {"forceInput": True}),
                "skill_strength": ("FLOAT", {"forceInput": True}),
                
                "gemma_model": (get_gemma_models(),),
                "max_new_tokens": ("INT", {"default": 150, "min": 50, "max": 500}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("english_prompt",)
    FUNCTION = "optimize_prompt"
    CATEGORY = "Zhouhan Nodes"

    def optimize_prompt(self, text, skill_system_prompt, skill_strength, gemma_model, max_new_tokens):
        if gemma_model == "No models found in models/zh":
            return ("Error: 请确保Gemma模型已放入 ComfyUI/models/zh 文件夹中，优先排查路径问题。",)

        model_path = os.path.join(ZH_MODELS_DIR, gemma_model)
        
        # --- 融合提示词架构 (Fusion Prompt Engineering) ---
        # 将前置的 Skill 身份与画图指令完美结合
        instruction = f"""{skill_system_prompt}

----------------------------------------
[CRITICAL TASK OVERRIDE]
While keeping the persona, mindset, and aesthetic preferences defined above (Strength: {skill_strength}), your core task is now to act as a Master Cinematographer. 

Take the user's concept and translate it into a highly detailed, professional IMAGE GENERATION PROMPT in strictly ENGLISH.
- Infuse the visual mood, lighting, and grit that matches your injected persona.
- Add rich cinematic details: camera angles, lighting (e.g., cinematic, volumetric), and composition.
- OUTPUT ONLY THE FINAL PROMPT IN ENGLISH. Do not add any conversational filler, greetings, or explanations.

User Concept: {text}
Final English Prompt:"""

        messages = [
            {"role": "user", "content": instruction}
        ]

        print(f"[Zhouhan Node] 正在从 {gemma_model} 加载 Gemma 语言模型...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype=torch.bfloat16,
            )

            text_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text_input], return_tensors="pt").to(self.device)

            print("[Zhouhan Node] 导演级提示词生成中 (已注入 Skill)...")
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7, 
                do_sample=True,
                top_p=0.9
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 严格执行显存释放，防止多轮运行后 OOM
            print("[Zhouhan Node] 提示词生成完毕，正在卸载 Gemma 模型以释放显存...")
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            return (response.strip(),)
            
        except Exception as e:
            # 异常时也要确保显存被清理
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[Zhouhan Node] 发生错误: {str(e)}。请优先排查环境路径与显存溢出 (OutOfMemory) 问题。")
            return (f"Error generating prompt: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "ZhouhanPromptDirector": ZhouhanPromptDirector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhouhanPromptDirector": "🎬 Zhouhan Director Prompt"
}