import base64
from typing import Dict, List, Optional, Tuple, Any, Union
from openai import OpenAI

class ExpertTemplate():
    def __init__(
        self,
        model_name: str = "qwen-vl-max-1119",
        cache_dir: Optional[str] = None,
        api_key: str = "sk-",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        system_prompt: str = "You are an expert in chest X-rays."
    ) -> None:
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.messages = []
        self.messages.append({
                "role": "system",
                "content": self.system_prompt
            })
    def __call__(
        self,
        message: Optional[str] = None,
        image_paths: Optional[Union[str, List[str]]] = None,  # 支持 str 或 List[str]
        max_new_tokens: int = 300,
        messages: Optional[List[Dict]] = None,
        role: str = "user"
    ) -> Any:
        # 初始化 messages
        if messages is None:
            messages = self.messages.copy()
        else:
            messages = self.messages.copy() + messages.copy()

        # 处理图片（支持单张或多张）
        if image_paths:
            image_contents = []

            # 如果是字符串，转为单元素列表
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            # 遍历所有图片路径
            for image_path in image_paths:
                with open(image_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                })

            # 添加图片消息
            messages.append({
                "role": "user",
                "content": image_contents
            })

        # 处理文本（和之前一样）
        if message:
            if messages and "content" in messages[-1] and isinstance(messages[-1]["content"], list):
                messages[-1]["content"].append({"type": "text", "text": message})
            else:
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": message}]
                })

        # 检查输入
        if not messages:
            raise ValueError("必须提供至少 message 或 image_paths 中的一个参数")

        # 发送请求
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens
        )
        return response.choices[0].message.content
