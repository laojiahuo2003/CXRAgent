from typing import Dict, Tuple, Type, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests

# 1. 定义输入模型
class MedgemmaQAInput(BaseModel):
    image_path: str = Field(..., description="path to chest X-ray images to analyze")
    question: str = Field(..., description="Question or instruction about the chest X-ray images")

# 2. 正确定义工具类
class GemmaVQATool(BaseTool):
    name: str = "med_gemma_assistant"
    description: str = (
        "A professional QA Tool for chest X-rays. You can also have a conversation with it."
        "Input should be paths to X-ray images "
        "and a natural language prompt describing the analysis needed."
    )
    args_schema: Type[BaseModel] = MedgemmaQAInput
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    base_url: str = "http://localhost:8000"  # 设置默认值

    def __init__(self, base_url: str = "http://localhost:8000", **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        
    def _run(self, image_path: str, question: str) -> Tuple[Dict, Dict]:
        """同步执行逻辑"""
        url = f"{self.base_url}/describe"
        
        # 准备请求数据
        files = {'file': open(image_path, 'rb')}
        data = {}
        if question:
            data['query'] = question
        
        # 发送请求
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()['description']
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
            "model": "medGemma",
            "response_format": "structured"
        }

        return output, metadata  # 必须返回二元组

    async def _arun(self, image_path: str, question: str) -> Tuple[Dict, Dict]:
        """异步实现（直接复用同步逻辑）"""
        return self._run(image_path, question)