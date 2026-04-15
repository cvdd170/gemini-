import os
import folder_paths

# 1. 定义专属文件夹路径: ComfyUI/models/zh
ZH_DIR = os.path.join(folder_paths.models_dir, "zh")

# 如果文件夹不存在，启动时自动创建它，防止报错
if not os.path.exists(ZH_DIR):
    os.makedirs(ZH_DIR)

class ZhouSkillAdapter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # 扫描 ZH_DIR 下的所有子文件夹
        skill_folders = []
        if os.path.exists(ZH_DIR):
            for item in os.listdir(ZH_DIR):
                item_path = os.path.join(ZH_DIR, item)
                # 只识别内部包含 SKILL.md 文件的文件夹
                if os.path.isdir(item_path) and "SKILL.md" in os.listdir(item_path):
                    skill_folders.append(item)
        
        # 如果没有找到任何符合条件的 skill 文件夹，提供一个默认选项防止 UI 崩溃
        if not skill_folders:
            skill_folders = ["none_found"]

        return {
            "required": {
                # 下拉菜单：显示所有有效的 Skill 文件夹名称
                "skill_module": (skill_folders, ),
                
                # 强度调整：控制这个设定的极性或权重
                "skill_strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 5.0, 
                    "step": 0.05,
                    "display": "slider"
                }),
            }
        }

    # 设定输出给下游 Gemma 节点的数据类型
    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("formatted_system_prompt", "strength")
    
    FUNCTION = "inject_skill"
    CATEGORY = "Zhou Nodes" # 在 ComfyUI 右键菜单中显示的分类

    def inject_skill(self, skill_module, skill_strength):
        # 如果没有选择任何有效的 skill，输出默认的安全提示词
        if skill_module == "none_found":
            print("[Zhou Node] 警告: 未在 models/zh 目录下找到有效的 Skill 文件夹。")
            return ("You are a helpful AI assistant.", skill_strength)

        # 定位到该 skill 目录下的具体 SKILL.md 文件
        skill_file_path = os.path.join(ZH_DIR, skill_module, "SKILL.md")
        skill_content = ""
        
        try:
            with open(skill_file_path, 'r', encoding='utf-8') as f:
                skill_content = f.read()
        except Exception as e:
            print(f"[Zhou Node] 错误: 读取 SKILL.md 失败 -> {e}")
            return ("You are a helpful AI assistant.", skill_strength)

        # --- Gemma 专属格式化处理与注入 ---
        # 使用明确的 XML 标签隔离背景设定，防止模型产生幻觉，并根据 strength 设定严格度
        formatted_prompt = f"""<persona_definition>
{skill_content}
</persona_definition>

严格遵守上述 <persona_definition> 中的身份卡、核心心智模型和表达DNA进行回复。
如果你遇到了错误或需要排查问题，请优先检查环境变量路径以及显存 (OutOfMemory) 问题。
如果设定的 strength (当前强度: {skill_strength}) 大于 1.0，你必须更加极端地展现该角色的性格特征；
如果涉及角色未公开表态的领域（如商业决策），必须明确说明局限性。"""

        # 将处理好的系统提示词和强度系数传递给下一个节点
        return (formatted_prompt, skill_strength)

# --- 节点注册 ---
# ComfyUI 必须通过这两个字典来识别和渲染你的自定义节点
NODE_CLASS_MAPPINGS = {
    "ZhouSkillAdapter": ZhouSkillAdapter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhouSkillAdapter": "Zhou Skill Adapter"
}