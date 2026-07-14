import json
import base64
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional
import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage,HumanMessage,AIMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from medAgent.llms import *
_ = load_dotenv()
# 定义一个名为AgentState的类，继承自TypedDict
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
class Agent:
    def __init__(self,
    model:BaseLanguageModel,
    tools:List[BaseTool],
    checkpointer: Any=None,
    user_input: str=None,
    image_paths: List=None,
    model_name: str = None,
    api_key: str = None,
    base_url: str = None
):
        
        self.tools_explanations = []
        self.tools_messages = []
        self.system_prompt = """
You are an expert medical AI assistant who can answer any medical questions and analyze medical images similar to a doctor.
Make multiple tool calls in parallel or sequence as needed for comprehensive answers.
If you need to look up some information before asking a follow up question, you are allowed to do that.
You need to ask different questions of the tools to gather information.
Use a variety of tools to assist your judgment
"""
        workflow = StateGraph(AgentState)
        workflow.add_node("start",self.init_func)
        workflow.add_node("process",self.process_func)
        workflow.add_node("execute",self.execute_tools)
        workflow.add_node("recruit_judge",self.recruit_judge_func)
        workflow.add_node("Probe",self.Probe_func)
        workflow.add_node("Dispatch",self.Dispatch_func)
        workflow.add_node("response",self.response_func)
        workflow.add_node("Relay",self.Relay_func)
        workflow.add_edge(
            "start",
            "process"
        )
        workflow.add_conditional_edges(
            "process",
            self.has_tools_calls,
            {True:"execute",False: "recruit_judge"})
        workflow.add_edge(
            "execute",
            "process"
        )
        workflow.add_conditional_edges(
            "recruit_judge",
            self.recruit_mode,
            {"1":"Probe","2":"Dispatch","3":"Relay","4":"response"}
        )
        workflow.add_edge(
            "Dispatch",
            "response"
        )
        workflow.add_edge(
            "Relay",
            "response"
        )
        workflow.add_edge(
            "Probe",
            "response"
        )
        workflow.add_edge(
            "response",
            END
        )
        workflow.set_entry_point("process")
        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.experts = []
        self.model = model.bind_tools(tools)
        self.user_input = user_input  # 确保始终存在
        self.image_paths = image_paths or []  # 确保列表存在
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    # 在对话时设置变量
    def init_info(self)->None:
        self.tools_explanations = []
        self.tools_messages = []
        self.teams_responses = []
        self.experts = []
    def set_user_input(self,user_input:str)->None:
        self.user_input = user_input
    def set_image_paths(self,image_paths:list)->None:
        self.image_paths = image_paths
    def set_image_path(self,image_path:str)->None:
        self.image_paths.append(image_path)
    def init_func(self,state:AgentState)->Dict[str,List[AnyMessage]]:
            # 清除上一轮的消息
        state["messages"] = []
        # 返回当前的聊天消息
        # return {"messages": state["messages"]}
        self.tools_explanations = []
        self.tools_messages = []
        self.teams_responses = []
        self.experts = []
        # self.image_paths = []
        
        messages = []
        # 处理多张图片
        for image_path in self.image_paths:
            messages.append(HumanMessage(content=f"image_path: {image_path}"))
            # 读取并编码图片
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            messages.append(HumanMessage(content=[{
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}]))

        # 添加用户文本输入
        messages.append(HumanMessage(content=[{"type": "text", "text": self.user_input}]))
        return {"messages": messages}
    
    def recruit_mode(self,state:AgentState)->str:
        print("招募阶段recruit_mode：")
        sys_prompt = """
        userquery: {user_input}

        1. Probe Strategy
        This strategy enables targeted evidence gathering through question-driven interaction. Particularly suitable for cases with uncertain or ambiguous findings, it allows the system to actively seek clarification and additional information when needed.

        2. Task Dispatch Strategy
        Designed to promote parallelization, this strategy assigns modular subtasks to domain-specialized agents. This architecture improves both efficiency and diagnostic coverage by leveraging the capabilities of different expert agents simultaneously.
        
        3. Relay Strategy
        This strategy supports progressive reasoning through sequential refinement, mimicking real-world diagnostic chains where agents build upon prior outputs. The sequential nature allows for step-by-step verification and improvement of the diagnostic process.

        4. Bypass Strategy
        For cases with non-medical issues or typical textbook-like characteristics that can be immediately identified, the system automatically activates the fast-track strategy, allowing for direct output of results without the need to activate the expert team, ensuring the maximization of the processing efficiency of clear cases.

        Please choose a strategy and output only one strategy number
        """.format(user_input = self.user_input)
        messages = state["messages"].copy()
        if self.system_prompt:
            messages = messages + [SystemMessage(content=sys_prompt)]
        response = self.model.invoke(messages).content
        # response = "3"
        print(f"招募阶段选择的策略: {response}")
        match = re.search(r'\d', response)
        return str(match.group(0)) if match else "1"

    def recruit_judge_func(self,state:AgentState)->Dict[str,List[AnyMessage]]:
        return {}


    def response_func(self,state:AgentState)->Dict[str,List[AnyMessage]]:
        expert = ExpertTemplate(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            system_prompt="""
As the final responder, you need to review the tool call results and team results in the history record. Other information can serve as reference, but you must make the most accurate judgment. Do not mention any tools or other conclusions. Please provide the final result.
        """
        
        )
        messages = self.tools_explanations + self.teams_responses
        expert_response = expert(self.user_input, self.image_paths,max_new_tokens=2048, messages=messages, role="user")
        return {"messages": [AIMessage(content=expert_response)]}
    @staticmethod
    def extract_content(text):
        start_tag = "<Result>"
        end_tag = "</Result>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)
        if start_idx == -1 or end_idx == -1:
            return text  # 如果没有找到标签，返回空字符串
        return text[start_idx + len(start_tag):end_idx]

    def Dispatch_func(self,state: AgentState) -> Dict[str,Any]:
        print("招募阶段Dispatch_func：")
        self.experts = []
        messages_temp = state["messages"].copy()# 取出历史消息
        
        small_question = """
        You are a medical case analysis expert. Your task is to evaluate the provided medical case and formulate 2 - 3 parallel sub - questions for a specialist team. These sub - questions should break down the case into smaller, more specific questions that address distinct clinical aspects of the case in a logical way, rather than making random breakdowns. They should enable independent evaluation by different specialists to support the final decision - making.
Here is the medical instruction:
<Instruction>
{user_input}
</Instruction>
When formulating sub - questions, please follow these rules:
1. Each sub - question should be clear and address a distinct clinical aspect of the case.
2. Do not raise questions related to historical diagnoses as they cannot be answered.
3. Separate each sub - question with $$. Each SubQuestion is a complete sub-question.
4. please provide the sub - questions in the required output format within the <Result> tag.
<Result>
    Here we present the sub-problems separated by $$. Each sub-question is in the form of an interrogative sentence.
</Result>
""".format(user_input=self.user_input)        
        LLM = ExpertTemplate(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            system_prompt=small_question
                    )
        response = LLM("Now, based on the Instruction and the pictures, etc., we will break down these issues into several sub-problems that can help you improve your accuracy, and wrap them with the <Result> tag.",self.image_paths,1000,self.tools_explanations,"user")
        # messages =  messages_temp+[SystemMessage(content=small_question)]
        # messages =  [HumanMessage(content=small_question)]
        # response = self.model.invoke(messages)
        results=""
        print(f"分解子任务:\n {response}")
        if response:
            content = self.extract_content(response)
            for all_description in content.split('$$'):

                if len(all_description.strip()) > 10:
    # 执行代码
                    expert = ExpertTemplate(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
                    system_prompt="You are an expert in chest X - ray analysis. Imitate the language of professional radiologists. Provide a brief and accurate response. You can refer to the conclusions from historical records (the outputs of relevant tools)."
                    )
                    res = expert(all_description,self.image_paths,400,[],"user")
                    # res = expert(all_description,self.image_paths,400,self.tools_explanations,"user")
                    print(all_description+"回答:"+res)
                    results+="question:"+all_description +'\n'+ "answer:"+res+'\n'
                    self.teams_responses.append(
                        {
                            "role": "user",
                            "content": "question:"+all_description +'\n'+ "team_proposal:"+res+'\n'
                        })
                    self.experts.append(expert)
        else:
            self.experts = []

        # self.teams_responses.append(
        #     {
        #         "role": "user",
        #         "content": results,
        #     }
        # )
        return {}
        # return {"messages": [SystemMessage(content=results)]}

    def Relay_func(self,state: AgentState) -> Dict[str,Any]:
        print("招募阶段Relay_func：")
        self.experts = []
        small_question = """
You are a senior radiologist overseeing a stepwise diagnostic workflow. Your task is to analyze the provided medical case and generate 2-3 sequential sub-steps that form a logical diagnostic chain.

Here is the medical instruction:
<Instruction>
{user_input}
</Instruction>

When formulating sub-steps, strictly follow these rules:
1. Sequential Dependency: Each sub-question must logically depend on the answer to the prior question (e.g., Question2 should only be answerable after resolving Question1)
2. Progressive Depth: Move from general findings → specific features → clinical implications
3. Actionable Focus: Later questions should guide concrete management decisions
4. Format: Separate each sub-question with $$ and ensure clinical executability

Example Structure:
1. [Existence Verification] Is there definitive evidence of [key finding]?
2. [Characterization] If present, what are the specific features of [key finding]?
3. [Clinical Correlation] How do these features correlate with [differential diagnosis]?

<Result>
Here is the sequential question chain:
1. [Detection] Is there radiographic evidence of [primary abnormality]? $$
2. [Phenotyping] If confirmed, what morphological features suggest [specific pathology]? $$
3. [Actionability] Based on these features, what is the recommended next diagnostic/therapeutic step?
</Result>
""".format(user_input=self.user_input)
        LLM = ExpertTemplate(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            system_prompt=small_question
                    )
        response = LLM("Now, based on the Instruction and the pictures, etc., we will break down these issues into several sub-problems that can help you improve your accuracy, and wrap them with the <Result> tag.",self.image_paths,1000,self.tools_explanations,"user")
        
        print(f"分解子任务:\n {response}")
        results = ""
        if response:
            content = self.extract_content(response)
            for all_description in content.split('$$'):

                if all_description.strip():
                    expert = ExpertTemplate(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
                    system_prompt="As a professional expert in chest X-ray analysis, you need to provide the most accurate and explanatory answers. If there is any historical information, it should be referred to. The answer should be concise and to the point, with only one paragraph"
                    )
                    res = expert("Others' analysis: {results}".format(results=results) + all_description,self.image_paths,400,self.tools_explanations,"user")
                    print(all_description+"回答:"+res)
                    results+=res+'\n'
                    self.experts.append(expert)
        else:
            self.experts = []
    
        return {"messages": [SystemMessage(content="medical team result:"+results)]}

    def Probe_func(self,state: AgentState) -> Dict[str,Any]:
        print("招募阶段Probe_func：")
        self.experts = []

        recruit_system = """
You are a multidisciplinary medical coordinator tasked with assembling the most appropriate expert team to interpret a chest X-ray or related imaging study.

Your task:
- Based on the imaging focus and clinical context (if provided), assign 1–3 expert roles to analyze different anatomical regions or concerns.

For each role, include the following:
∘ Role title  
∘ Required qualifications  
∘ Analysis focus (specific anatomical region or organ system)  
∘ Specific sub-tasks (the detailed responsibilities this role handles in image interpretation)

Formatting rules:
- Use clear, structured descriptions as in the example format below (for formatting reference only – do not repeat it).
- Separate each role with `$$`  
- Wrap the full output inside `<Result>...</Result>` tags.  
- Do not add any commentary, explanation, or use any tools. Only return the result.

Example format (do not repeat):
Role Title:  
∘ Required qualifications: ...  
∘ Analysis focus: ...  
∘ Specific sub-tasks: ...

"""

        LLM = ExpertTemplate(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            system_prompt=recruit_system
                    )
        res = LLM("Based on the specific circumstances of this user, a team recruitment will be carried out.",self.image_paths,1000,self.tools_explanations,"user")
        print(f"招募结果: {res}")
        results = ""
        if res:
            results = "The following is the result of the team discussion:\n"
            content = self.extract_content(res)
            for expert_description in content.split('$$'):
                if len(expert_description.strip()) < 10: 
                    continue
                if expert_description.strip():
                    expert = ExpertTemplate(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
                    system_prompt="Your identity is as follows\n"+expert_description
                    )
                    self.experts.append(expert)
                    res = expert("Answer concisely and in alignment with your assigned expert role.",self.image_paths,400,self.tools_explanations,"user")
                    print("回答:"+str(res))
                    results+=str(res)+"\n"
            self.teams_responses.append(
            {
                "role": "tool",
                "content": results,
            }
        )
        else:
            self.experts = []
    
        return {}

    def process_func(self,state:AgentState)->Dict[str,List[AnyMessage]]:
        print("处理阶段process_func：")
        messages = state["messages"]# 取出历史消息
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)]+messages
        response = self.model.invoke(messages)
        return {"messages":[response]}
        
    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        print("工具执行阶段execute_tools：")
        tool_call_list = state["messages"][-1].tool_calls
        results = []
        for call in tool_call_list:
            print(f"执行工具{call}")
            if call["name"] not in self.tools:
                print("\n...未找到该工具工具...")
                result = "invalid tool, please retry"
            else:
                try:
                    result = self.tools[call["name"]].invoke(call["args"])
                except Exception as e:
                    print(f"工具调用失败: {e}")
                    result = "tool invocation failed, skipping"
                print(f"工具执行结果: {result}")
            
            # 对工具调用进行解释（仅在成功时）
            if result not in ("invalid tool, please retry", "tool invocation failed, skipping"):
                try:
                    expl_response = self._explain_func(str(result))
                except Exception:
                    expl_response = "Evidence validation skipped due to error."
            else:
                expl_response = "Tool execution failed, no evidence to validate."

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )
            content = """
<Tool conclusion>
{result}
</Tool conclusion>
<Tool explanation>
{explanation}
</Tool explanation>
""".format(result=str(result), explanation=str(expl_response))
            self.tools_explanations.append({
                "role": "user",
                "content": content,
                "tool_call_id": call["id"]
            })

        return {"messages": results}
    def _explain_func(self, result: str) -> str:
        expert = ExpertTemplate(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            system_prompt = (
    "Critically evaluate each individual conclusion made by the tool based on the provided chest X-ray images.\n"
    "Do not group multiple conclusions into one evaluation—analyze each separately.\n"
    "Thoroughly examine the image to identify any subtle or overlooked findings that may support or contradict the conclusion.\n"
    "Be especially attentive to subtle signs.\n"
    "For each conclusion:\n"
    "- Clearly restate the specific conclusion.\n"
    "- Describe visual evidence in the image that supports it.\n"
    "- Describe visual evidence that may contradict or refute it.\n"
    "- Assess the overall consistency of the conclusion with the image findings.\n"
    "- End with a brief one-sentence judgment on its credibility.\n"
    "Format your evaluation using one structured block per conclusion, as follows:\n"
    "[Conclusion] — [Supporting Evidence] — [Refuting Evidence] — [Credibility Evaluation]"
)
        )
        response = expert("Tool result: "+ result, self.image_paths,max_new_tokens=600, messages=[], role="user")
        print("解释: "+response)
        return response

    def has_tools_calls(self,state:AgentState)->bool:
        response = state["messages"][-1]
        return hasattr(response, "tool_calls") and len(response.tool_calls) > 0






