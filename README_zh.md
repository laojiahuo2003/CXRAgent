# 🩺 CXRAgent：基于Director-协调的胸部X光多阶段智能诊断代理

<p align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-DBEDFA"></a>
  <a href="./README_zh.md"><img alt="简体中文版自述文件" src="https://img.shields.io/badge/简体中文-DFE0E5"></a>
</p>

## 📋 项目简介

CXRAgent 是一个创新的胸部X光（Chest X-ray）智能诊断代理框架，采用Director-协调的多阶段架构设计，旨在解决现有医疗AI模型在适应新诊断任务和复杂推理场景中的局限性。CXRAgent通过工具调用和协调、EDV校验机制、多步骤推理和团队协作，显著提升了胸部X光影像分析的可靠性和适应性。

### 🎯 研究背景

胸部X光在临床诊断中扮演着关键角色，尽管已有多种任务特定和基础模型用于自动X光解释，但这些模型往往难以适应新诊断任务和复杂推理场景。现有的基于LLM的代理模型虽然通过工具协调、多步推理等方式增强了模型能力，但通常依赖单一诊断流程，缺乏评估工具输出可靠性的机制，限制了其适应性和可信度。

### 🔍 技术亮点

* ✅ **证据驱动验证器（EDV）**：为工具诊断输出提供视觉证据支持，确保诊断可靠性
* ✅ **专家团队协作机制**：根据任务需求动态组建专家团队，实现自适应协作推理
* ✅ **多阶段诊断流程**：工具调用、诊断规划、协作决策三大阶段有机结合
* ✅ **多模型支持**：兼容OpenAI、Qwen、GPT、LLaVA、Gemma等多种模型
* ✅ **灵活的工具生态**：可扩展的工具注册与调用机制，支持多种医学影像任务
* ✅ **记忆管理机制**：集成专家团队见解与上下文记忆，合成有证据支持的诊断结论

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
- `CheXbench_metadata.jsonl`: CheXbench数据集的元数据，包含2500个问题
  - CheXbench数据集链接：[StanfordAIMI/chexbench](https://huggingface.co/datasets/StanfordAIMI/chexbench)

这些数据可用于模型测试和评估。

## 🔍 自定义工具开发

### 工具开发指南

你可以通过参考`medAgent/tools/tool_template.py`中的模板代码来开发自定义工具。主要开发步骤：

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

## 📁 项目架构

CXRAgent采用模块化的架构设计，主要组件包括：

```
medAgent/
├── agent/         # 代理核心代码，包含Director和多阶段协调逻辑
│   ├── __init__.py
│   └── agent.py   # 核心代理实现，包含Director和三阶段推理逻辑
├── llava/         # LLaVA视觉语言模型集成
│   ├── model/     # 模型定义和加载
│   └── serve/     # 服务部署相关代码
├── llms/          # 大语言模型接口和专家模板
├── tools/         # 工具实现目录，包含各种胸部X光分析工具
│   ├── chexagent.py      # ChexAgent诊断工具
│   ├── classification.py # 病变分类工具
│   ├── grounding.py      # 视觉证据定位工具（EDV核心组件）
│   ├── llava_med.py      # 医学LLaVA分析工具
│   ├── tool_template.py  # 工具开发模板
│   └── ...
└── utils/         # 辅助工具函数
```

## 💡 常见问题

1. **Q: 如何添加新的医学分析工具？**
   A: 参考`tools/tool_template.py`创建新工具类，实现`run`、`get_name`、`get_description`等必要方法后注册到代理系统。对于需要视觉证据支持的工具，建议实现与EDV兼容的输出格式。

2. **Q: 支持哪些类型的医学影像？**
   A: CXRAgent主要针对胸部X光（Chest X-ray）图像进行了优化和评估，部分工具可能支持其他类型医学影像，但性能可能有所差异。

3. **Q: 是否需要GPU运行？**
   A: 是的，强烈建议使用支持CUDA的GPU以获得最佳性能，特别是在处理高分辨率医学影像和运行多个模型时。


## Citation
如果你认为我们的工作有用, 请支持我们的工作:
```bibtex
@misc{lou2025cxragentdirectororchestratedmultistagereasoning,
      title={CXRAgent: Director-Orchestrated Multi-Stage Reasoning for Chest X-Ray Interpretation}, 
      author={Jinhui Lou and Yan Yang and Zhou Yu and Zhenqi Fu and Weidong Han and Qingming Huang and Jun Yu},
      year={2025},
      eprint={2510.21324},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.21324}, 
}
```