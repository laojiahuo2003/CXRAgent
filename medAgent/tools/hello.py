from typing import Dict, Tuple, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


# 1. 定义输入模型
class CustomToolInputModel(BaseModel):
    user_txt: str = Field(..., description="用户输入的文本内容")
    image: str = Field(..., description="用户输入的图片内容")


# 2. 正确定义工具类
class CustomTool(BaseTool):
    name: str = "medbook"  # 添加类型注解
    description: str = "这是一本医学书籍,你可以查询器官的相关病变,以及病变的诊断方式"  # 添加类型注解
    args_schema: Type[BaseModel] = CustomToolInputModel  # 关联输入模型
    # Optional: Add tool-specific initialization
    def __init__(self):
        super().__init__()
        
    def _run(self, user_txt: str,image: str) -> Tuple[Dict, Dict]:
        """同步执行逻辑"""
        print(f"收到输入: {user_txt}")

        # 核心业务结果（传递给大模型）
        output = {
            "optimized_text": "哈哈哈哈你被假的工具骗啦",
            "original_text": user_txt
        }

        # 元数据（用于日志/调试）
        metadata = {
            "status": "success",
            "transform_type": "humor"
        }

        return output, metadata  # 必须返回二元组

    async def _arun(self, user_txt: str,image: str) -> Tuple[Dict, Dict]:
        """异步实现（直接复用同步逻辑）"""
        return self._run(user_txt)