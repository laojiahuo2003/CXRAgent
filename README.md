# 🩺 CXRAgent：医学影像智能对话代理

## 📋 项目简介

CXRAgent 是一个专为医学影像分析设计的可扩展智能对话代理（AI Agent）框架。它主要面向胸部X光（Chest X-ray）应用场景，提供全面的影像分析、报告生成、视觉问答（VQA）等功能，并支持与多种外部医学工具无缝集成。

### 🚀 核心功能

* ✅ 多模型支持（OpenAI / Qwen / GPT / LLaVA / Gemma 等）
* ✅ 灵活的工具注册与调用机制
* ✅ 模块化设计，便于扩展和定制
* ✅ 兼容多种医学影像任务（CXR诊断 / 报告生成 / 视觉问答）
* ✅ 支持自定义工具接入
* ✅ 可配置任意OpenAI兼容接口

## 💻 环境安装

### 系统要求
- Python 3.9 或更高版本
- 建议使用CUDA支持的GPU以获得最佳性能

### 安装步骤

1. 克隆项目仓库（如果尚未克隆）
   ```bash
   git clone <项目仓库地址>
   ```

2. 安装依赖包
   ```bash
   pip install -e .
   ```
   
   该命令会以开发模式安装当前项目及所有必需的依赖库。

## 🔧 配置说明

### API 设置

在运行项目前，需要配置API密钥与接口地址。以下是配置示例：

```python
# 配置API密钥和基础URL
openai_kwargs = {}
openai_kwargs["api_key"] = "sk-xxx"  # 替换为你的API密钥
openai_kwargs["base_url"] = "https://your-api-endpoint.com/v1"  # 替换为你的API端点

# 初始化代理和工具
gent, tools_dict = init_agent(
    model_dir="model-weights",  # 模型下载存储位置
    model_name="gpt-4o-mini",  # 可选模型：gpt-4o-mini、qwen-vl-max-latest、gpt-4o等
    temp_dir="temp",  # 临时文件存储目录
    device="cuda",  # 运行设备：cuda或cpu
    temperature=0.7,  # 生成温度参数
    top_p=0.95,  # 采样参数
    openai_kwargs=openai_kwargs,  # API配置参数
)
```

### 工具配置

项目支持自定义工具集，可根据需求加载多种功能模块。以下是工具配置示例：

```python
# 工具配置示例
tool_dict = {
    # 可以在这里添加预定义工具或自定义工具
    # "工具名称": 工具实例
}
```

## 🎯 使用指南

### 运行主程序

配置完成后，可直接运行主程序启动交互式对话代理：

```bash
python chat.py
```

启动后，你可以：
- 上传胸部X光图像进行分析
- 向代理提问关于医学影像的问题
- 生成详细的医学报告
- 测试和使用各种注册的工具

## 📊 数据说明

项目的`data`文件夹包含以下元数据文件：

- `mimic_report_400.json`: MIMIC数据集的报告元数据，包含400个胸部X光报告样本
- `vqa_data_1000.json`: Medical-CXR-VQA数据集的元数据，包含1000个视觉问答样本

这些数据可用于模型测试和评估。

## 🔍 自定义工具开发

### 工具开发指南

你可以通过参考`medAgent/tools/hello.py`中的模板代码来开发自定义工具。主要开发步骤：

1. 创建新的Python文件，继承基础工具类
2. 实现必要的方法，如`run`、`get_name`、`get_description`等
3. 注册工具到代理系统中

### API服务方式（推荐）

如果遇到环境冲突或需要部署为服务，建议采用API方式调用工具。以下是使用FastAPI创建医学图像描述服务的示例：

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MedGemma Image Description API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型和处理器全局变量
model = None
processor = None

# 请求模型定义
class DescriptionRequest(BaseModel):
    prompt: Optional[str] = "Describe this X-ray"
    system_prompt: Optional[str] = "You are an expert radiologist."

# 启动时加载模型
@app.on_event("startup")
async def load_model():
    """在启动时加载MedGemma模型"""
    global model, processor
    
    model_id = "google/medgemma-4b-it"
    logger.info(f"Loading model {model_id}...")
    
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

# 图像描述端点
@app.post("/describe")
async def describe_image(
    file: UploadFile = File(...),
    request: DescriptionRequest = None
):
    """接收X光图像并返回专业描述"""
    if request is None:
        request = DescriptionRequest()
    
    try:
        # 读取上传的图像
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # 准备聊天消息
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": request.system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # 处理输入并生成描述
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=400, do_sample=False)
            generation = generation[0][input_len:]
        
        decoded = processor.decode(generation, skip_special_tokens=True)
        
        return {"description": decoded}
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 主程序入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

## 🛠️ 项目结构

```
medAgent/
├── agent/         # 代理核心代码
├── llava/         # LLaVA模型相关代码
├── llms/          # 大语言模型接口
├── tools/         # 工具实现目录
│   ├── chexagent.py      # ChexAgent工具
│   ├── classification.py # 分类工具
│   ├── grounding.py      # 图像定位工具
│   ├── llava_med.py      # 医学LLaVA工具
│   ├── report_generation.py # 报告生成工具
│   └── ...
└── utils/         # 工具函数
```

## 💡 常见问题

1. **Q: 如何添加新的医学分析工具？**
   A: 参考`tools/tool_template.py`创建新工具类，实现必要方法后注册到代理系统。

2. **Q: 支持哪些类型的医学影像？**
   A: 目前主要针对胸部X光图像优化，部分工具可能支持其他类型医学影像。

3. **Q: 是否需要GPU运行？**
   A: 虽然可以在CPU上运行，但强烈建议使用支持CUDA的GPU以获得更好性能。