# 🩺 CXRAgent: Director-Coordinated Multi-Stage Intelligent Diagnostic Agent for Chest X-rays

<p align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-DBEDFA"></a>
  <a href="./README_zh.md"><img alt="简体中文版自述文件" src="https://img.shields.io/badge/简体中文-DFE0E5"></a>
</p>

## 📋 Project Overview

CXRAgent is an innovative chest X-ray intelligent diagnostic agent framework that employs a Director-coordinated multi-stage architecture. It aims to address the limitations of existing medical AI models in adapting to new diagnostic tasks and complex reasoning scenarios. By leveraging tool calling and coordination, EDV validation mechanism, multi-step reasoning, and team collaboration, CXRAgent significantly enhances the reliability and adaptability of chest X-ray image analysis.

### 🎯 Research Background

Chest X-rays play a crucial role in clinical diagnosis. While various task-specific and foundation models exist for automatic X-ray interpretation, these models often struggle to adapt to new diagnostic tasks and complex reasoning scenarios. Existing LLM-based agent models, though enhanced through tool coordination and multi-step reasoning, typically rely on a single diagnostic process and lack mechanisms to evaluate the reliability of tool outputs, limiting their adaptability and credibility.

### 🔍 Technical Highlights

* ✅ **Evidence-Driven Validator (EDV)**: Provides visual evidence support for tool diagnostic outputs, ensuring diagnostic reliability
* ✅ **Expert Team Collaboration Mechanism**: Dynamically forms expert teams based on task requirements, enabling adaptive collaborative reasoning
* ✅ **Multi-Stage Diagnostic Process**: Seamlessly integrates three key stages: tool calling, diagnostic planning, and collaborative decision-making
* ✅ **Multi-Model Support**: Compatible with various models including OpenAI, Qwen, GPT, LLaVA, Gemma, etc.
* ✅ **Flexible Tool Ecosystem**: Extensible tool registration and calling mechanism supporting multiple medical imaging tasks
* ✅ **Memory Management**: Integrates expert team insights and contextual memory to synthesize evidence-supported diagnostic conclusions

## 💻 Environment Setup

### System Requirements
- Python 3.9 or higher
- CUDA-supported GPU recommended for optimal performance

### Installation Steps

1. Clone the project repository (if not already cloned)
   ```bash
   git clone <project_repository_url>
   ```

2. Install dependencies
   ```bash
   pip install -e .
   ```
   
   This command installs the project in development mode along with all required dependencies.

## 🔧 Configuration

### API Setup

Before running the project, configure your API keys and endpoint addresses. Here's a configuration example:

```python
# Configure API key and base URL
openai_kwargs = {}
openai_kwargs["api_key"] = "sk-xxx"  # Replace with your API key
openai_kwargs["base_url"] = "https://your-api-endpoint.com/v1"  # Replace with your API endpoint

# Initialize agent and tools
agent, tools_dict = init_agent(
    model_dir="model-weights",  # Model download storage location
    model_name="gpt-4o-mini",  # Available models: gpt-4o-mini, qwen-vl-max-latest, gpt-4o, etc.
    temp_dir="temp",  # Temporary file storage directory
    device="cuda",  # Running device: cuda or cpu
    temperature=0.7,  # Generation temperature parameter
    top_p=0.95,  # Sampling parameter
    openai_kwargs=openai_kwargs,  # API configuration parameters
)
```

### Tool Configuration

The project supports custom tool sets, allowing you to load various functional modules according to your needs. Here's a tool configuration example:

```python
# Tool configuration example
tool_dict = {
    # You can add predefined tools or custom tools here
    # "tool_name": tool_instance
}
```

## 🎯 Usage Guide

### Run the Main Program

After configuration, run the main program to start the interactive dialogue agent:

```bash
python chat.py
```

Once started, you can:
- Upload chest X-ray images for analysis
- Ask the agent questions about medical images
- Generate detailed medical reports
- Test and use various registered tools

## 📊 Data Description

The project's `data` folder contains the following metadata files:

- `refs-3858-all.json`: Report metadata from the MIMIC dataset, containing 3858 chest X-ray report samples
- `vqa_data_1000.json`: Metadata from the Medical-CXR-VQA dataset, containing 1000 visual question-answer samples
- `CheXbench_metadata.jsonl`: Metadata from the CheXbench dataset, containing 2500 questions
  - CheXbench dataset link: [StanfordAIMI/chexbench](https://huggingface.co/datasets/StanfordAIMI/chexbench)

These data can be used for model testing and evaluation.

## 🔍 Custom Tool Development

### Tool Development Guide

You can develop custom tools by referring to the template code in `medAgent/tools/tool_template.py`. The main development steps are:

1. Create a new Python file inheriting from the base tool class
2. Implement necessary methods such as `run`, `get_name`, `get_description`, etc.
3. Register the tool in the agent system

### API Service Method (Recommended)

If you encounter environment conflicts or need to deploy as a service, it's recommended to call tools via API. Here's an example of using FastAPI to create a medical image description service:

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and processor global variables
model = None
processor = None

# Request model definition
class DescriptionRequest(BaseModel):
    prompt: Optional[str] = "Describe this X-ray"
    system_prompt: Optional[str] = "You are an expert radiologist."

# Load model on startup
@app.on_event("startup")
async def load_model():
    """Load MedGemma model on startup"""
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

# Image description endpoint
@app.post("/describe")
async def describe_image(
    file: UploadFile = File(...),
    request: DescriptionRequest = None
):
    """Receive X-ray image and return professional description"""
    if request is None:
        request = DescriptionRequest()
    
    try:
        # Read uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Prepare chat messages
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
        
        # Process input and generate description
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

# Main program entry
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

## 📁 Project Architecture

CXRAgent adopts a modular architecture design with the following main components:

```
medAgent/
├── agent/         # Core agent code with Director and multi-stage coordination logic
│   ├── __init__.py
│   └── agent.py   # Core agent implementation with Director and three-stage reasoning logic
├── llava/         # LLaVA visual language model integration
│   ├── model/     # Model definitions and loading
│   └── serve/     # Service deployment related code
├── llms/          # Large language model interfaces and expert templates
├── tools/         # Tool implementation directory containing various chest X-ray analysis tools
│   ├── chexagent.py      # ChexAgent diagnostic tool
│   ├── classification.py # Lesion classification tool
│   ├── grounding.py      # Visual evidence localization tool (EDV core component)
│   ├── llava_med.py      # Medical LLaVA analysis tool
│   ├── tool_template.py  # Tool development template
│   └── ...
└── utils/         # Utility functions
```

## 💡 Frequently Asked Questions

1. **Q: How to add new medical analysis tools?**
   A: Create a new tool class by referencing `tools/tool_template.py`, implement necessary methods like `run`, `get_name`, `get_description`, etc., then register it to the agent system. For tools requiring visual evidence support, it's recommended to implement EDV-compatible output formats.

2. **Q: What types of medical images are supported?**
   A: CXRAgent is primarily optimized and evaluated for chest X-ray images. Some tools may support other types of medical images, but performance may vary.

3. **Q: Is a GPU required to run?**
   A: Yes, a CUDA-supported GPU is strongly recommended for optimal performance, especially when processing high-resolution medical images and running multiple models.

## Citation
If you find this work useful, please cite our paper:
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

