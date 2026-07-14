from typing import Dict, Tuple, Type, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests
import os

# 1. 定义输入模型
class ChexagentVQAInput(BaseModel):
    image_path: str = Field(..., description="path to chest X-ray images to analyze")
    question: str = Field(..., description="question about the chest X-ray images")

# 2. 正确定义工具类
class ChexagentVQATool(BaseTool):
    name: str = "chexagent_vqa"
    description: str = (
        "An intelligent chest X-ray assistant and Professional QA generation tool. You can ask any question you like. "
        "Input should be paths to X-ray images "
        "and a natural language prompt describing the analysis needed."
    )
    args_schema: Type[BaseModel] = ChexagentVQAInput
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    base_url: str = "http://localhost:8001"  # 设置默认值

    def __init__(self, base_url: str = "http://localhost:8001", **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip('/')

    def _run(self, image_path: str, question: str) -> Tuple[Dict, Dict]:
        # Check if image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read image file
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Prepare request data
        files = {'file': (os.path.basename(image_path), image_bytes)}
        data = {
            'prompt': question if question else "Describe this medical image",
        }
        response = requests.post(f"{self.base_url}/analyze", files=files, data=data)
        response.raise_for_status()

        # 解析并打印结果 (服务器返回的是'analysis'字段)
        if response.status_code == 200:
            result = response.json()['analysis']
        else:
            result = f"Tool request failed: {response.status_code} - {response.text}"
        
        # 核心业务结果（传递给大模型）
        output = {
            "original_question": question,
            "response": result
        }

        # 元数据（用于日志/调试）
        metadata = {
            "status": "success",
            "model": "chexagent"
        }

        return output, metadata  # 必须返回二元组

    async def _arun(self, image_path: str, question: str) -> Tuple[Dict, Dict]:
        """异步实现（直接复用同步逻辑）"""
        return self._run(image_path, question)