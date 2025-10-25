from typing import Dict, Tuple, Type, Optional,Any
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch

# 1. MedVLM-R1模型
class MedicalImageQAInput(BaseModel):
    image_path: str = Field(..., description="path to chest X-ray images to analyze")
    question: str = Field(..., description="Question or instruction about the chest X-ray images")

# 2. 定义工具类
class MedicalImageQATool(BaseTool):
    name: str = "medical_image_qa"
    description: str = (
        "A tool that provides you with diagnostic ideas for analyzing chest X-rays. "
        "You need to formulate a question that can trigger self-reflection and provide you with some enlightenment. "
        "Input should be paths to X-ray images "
        "and a natural language prompt describing the analysis needed."
    )
    args_schema: Type[BaseModel] = MedicalImageQAInput
    device: Optional[str] = None
    cache_dir: Optional[str] = None
    
    _model: Any = None
    _processor: Any = None
    _generation_config: Any = None
    # _model: Optional[Qwen2VLForConditionalGeneration] = PrivateAttr()
    # _processor: Optional[AutoProcessor] = PrivateAttr()
    # _generation_config: Optional[GenerationConfig] = PrivateAttr()
    
    def __init__(self,
                cache_dir: Optional[str] = None,
                device: Optional[str] = "cuda",
                **kwargs):
        super().__init__(**kwargs)
        # 初始化模型（仅在工具创建时执行一次）
        MODEL_PATH = 'JZPeterPan/MedVLM-R1'
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        self._generation_config = GenerationConfig(
            max_new_tokens=300,
            do_sample=False,  
            temperature=1, 
            num_return_sequences=1,
            pad_token_id=151643,
        )


    def _run(self, image_path: str, question: str) -> Tuple[Dict, Dict]:
        """同步执行医学图像问答"""
        QUESTION_TEMPLATE = """
        {Question} 
        Your task: 
        1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
        2. Then provide the correct result inside <answer>...</answer>.
        3. No extra information or text outside of these tags.
        """
        
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"}, 
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question)}
            ]
        }]
        
        text = self._processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self._processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        
        generated_ids = self._model.generate(
            **inputs, 
            use_cache=True, 
            max_new_tokens=1024, 
            do_sample=False, 
            generation_config=self._generation_config
        )
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # 返回结构化的结果
        output = {
            "answer": output_text[0],
            "original_question": question,
            "image_path": image_path
        }
        
        metadata = {
            "status": "success",
            "model": "MedVLM-R1",
            "response_format": "structured"
        }
        
        return output, metadata

    async def _arun(self, image_path: str, question: str) -> Tuple[Dict, Dict]:
        """异步实现（直接复用同步逻辑）"""
        return self._run(image_path, question)
    