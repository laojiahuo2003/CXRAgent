import base64
import time
import re
from typing import List, Optional, AsyncGenerator, Tuple
from medAgent.agent import *
from medAgent.tools import *
from medAgent.utils import *
from medAgent.llms import *
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
import os
import warnings
from typing import *

warnings.filterwarnings("ignore")
class ChatCMD:
    def __init__(self, agent, tools_dict):
        self.agent = agent
        self.tools_dict = tools_dict
        self.image_path = ""
        self.display_path = self.image_path  # Can be changed if DICOM conversion needed
        self.current_thread_id = None
        self.chat_history: List[dict] = []

    async def process_message(self, message: str) -> AsyncGenerator[str, None]:
        if not self.current_thread_id:
            self.current_thread_id = str(time.time())
        messages = []
        image_path = self.image_path
        if image_path is not None:
            # Send path for tools
            messages.append({"role": "user", "content": f"image_path: {image_path}"})

            # Load and encode image for multimodal
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        }
                    ],
                }
            )

        if message is not None:
            messages.append({"role": "user", "content": [{"type": "text", "text": message}]})

        for event in self.agent.workflow.stream(
            {"messages": messages},
            {"configurable": {"thread_id": self.current_thread_id}}
        ):
            if isinstance(event, dict):
                if "execute" in event:
                    for message in event["execute"]["messages"]:
                        tool_name = message.name
                        print(message)
                        tool_result = eval(message.content)[0]

                        if tool_result:
                            result_text = " ".join(
                                line.strip() for line in str(tool_result).splitlines()
                                ).strip()
                            self.chat_history.append(
                                    {"role": "assistant", "content": result_text}
                                )
                            yield f"[Tool: {tool_name}] {result_text}"
                elif "response" in event:
                    content = event["response"]["messages"][-1].content
                    yield f"🤖 Assistant（response）: {content}"


    def chat_loop(self):
        print("Type your questions about the X-ray image.\nType 'exit' to quit.\n")

        while True:
            image_path = input("🖼️ Enter the image path (or type 'exit' to quit): ")
            if image_path.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            
            user_input = input("🧑 You: ")
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            agent.set_image_paths([image_path])
            agent.set_user_input(user_input)
            agent.init_info()
            # Update the image path for each input
            self.image_path = image_path

            for response in self._run_async(user_input):
                print(response)


    def _run_async(self, message: str):
        import asyncio
        return asyncio.run(self._collect_async(message))

    async def _collect_async(self, message: str) -> List[str]:
        results = []
        async for output in self.process_message(message):
            results.append(output)
        return results



def init_agent(
        model_dir,
        model_name,
        temp_dir,
        device,
        temperature=0.7,
        top_p=0.95,
        openai_kwargs={}
):
    # 自定义选项
    tool_dict = {
    "ChestXRayClassifierTool": ChestXRayClassifierTool(device=device),
    # "ChestXRaySegmentationTool": ChestXRaySegmentationTool(device=device),
    # "ChestXRayReportGeneratorTool": ChestXRayReportGeneratorTool(
    #         cache_dir=model_dir, device=device),
    # "XRayPhraseGroundingTool": XRayPhraseGroundingTool(
    #         cache_dir=model_dir, temp_dir=temp_dir, device=device,load_in_4bit=True
    # ),

    # "MedGemmaQATool": GemmaVQATool(base_url="http://10.4.121.7:8001"),
    # "LlavaMedTool": LlavaMedTool(cache_dir=model_dir, device=device, load_in_4bit=True),
    # "XRayVQATool": XRayVQATool(cache_dir=model_dir, device=device),
    # "LLavaRadReportTool": LLavaRadVQATool(base_url="http://10.4.121.7:8000"),
    # "MedicalImageQATool": MedicalImageQATool(cache_dir=model_dir,device=device)
    }

    checkpointer = MemorySaver()
    model = ChatOpenAI(model=model_name, temperature=temperature, top_p=top_p, **openai_kwargs)
    agent = Agent(
        model=model,
        tools=list(tool_dict.values()),
        checkpointer=checkpointer,
        model_name=model_name,
        api_key=openai_kwargs.get("api_key", ""),
        base_url=openai_kwargs.get("base_url", "")
    )

    return agent,tool_dict


if __name__ == "__main__":
    openai_kwargs = {}

    openai_kwargs["api_key"] = ""
    openai_kwargs["base_url"] = ""

    agent,tools_dict = init_agent(
        model_dir="model-weights",
        model_name="gpt-4o",  # gpt-4o-mini、qwen-vl-max-latest、gpt-4o
        temp_dir="temp",
        device="cuda",
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs,
    )
chat = ChatCMD(agent, tools_dict)
chat.chat_loop()