from typing import Dict, Tuple, Type, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests
import os

# 1. 定义输入模型
class LLavaRadQAInput(BaseModel):
    image_path: str = Field(..., description="path to chest X-ray images to analyze")
    question: str = Field(..., description="question about the chest X-ray images")

# 2. 正确定义工具类
class LLavaRadVQATool(BaseTool):
    name: str = "llava_rad_report"
    description: str = (
        "An intelligent chest X-ray assistant and Professional report generation tool. You can ask any question you like. "
        "Input should be paths to X-ray images "
        "and a natural language prompt describing the analysis needed."
    )
    args_schema: Type[BaseModel] = LLavaRadQAInput
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    base_url: str = "http://localhost:8001"  # 设置默认值

    def __init__(self, base_url: str = "http://localhost:8001", **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        
    def _run(self, image_path: str, question: str) -> Tuple[Dict, Dict]:
        url = f"{self.base_url}/analyze"
        # Prepare request data
        data = {}
        if question:
            data['query'] = question

        # Send request (file handle properly closed by with-statement before post)
        with open(image_path, 'rb') as f:
            file_content = f.read()
        response = requests.post(
            url,
            files={'image': (os.path.basename(image_path), file_content)},
            data=data,
        )

        if response.status_code == 200:
            result = response.json()['result']
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
            "model": "llava_rad"
        }

        return output, metadata  # 必须返回二元组

    async def _arun(self, image_path: str, question: str) -> Tuple[Dict, Dict]:
        """异步实现（直接复用同步逻辑）"""
        return self._run(image_path, question)