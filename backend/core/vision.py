import base64
import os

from core.openai_client import get_vision_client
from config import get_settings


_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

# 截图场景：要的是「忠实转写」，而不是泛泛描述——用户是靠图里的原文检索的。
SCREENSHOT_PROMPT = """你是一个图片/截图内容提取助手。请把图片中的内容完整、忠实地转写为 Markdown：
- 完整保留所有文字，按原有阅读顺序排列，不要遗漏、不要总结、不要改写。
- 保留结构：标题用 #，列表用 - 或数字，表格用 Markdown 表格。
- 界面元素（按钮、菜单、输入框、标签等）标注其文字内容。
- 代码或命令用代码块包裹。
- 如有图表 / 流程图 / 示意图等非文字信息，在末尾用一段「图示说明：…」简要描述其含义与关键数据。
只输出 Markdown 正文，不要添加额外解释、开场白或结束语。"""


def _mime_for(path: str) -> str:
    return _MIME.get(os.path.splitext(path)[1].lower(), "image/png")


async def image_to_markdown(filepath: str) -> str:
    """调用多模态模型把图片转写成 Markdown 文本，供后续编辑与分块入库。"""
    s = get_settings()
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:{_mime_for(filepath)};base64,{b64}"

    resp = await get_vision_client().chat.completions.create(
        model=s.vision_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": SCREENSHOT_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        temperature=0,
        max_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip()
