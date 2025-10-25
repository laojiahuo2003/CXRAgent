# 🩺 CXRagent

CXRagent 是一个可扩展的医学影像智能对话代理（AI Agent），用于支持胸部 X 光（Chest X-ray）分析、报告生成、VQA（Visual Question Answering）以及与多种外部医学工具的交互。该项目支持自定义工具接入，并可配置任意 OpenAI 接口。

---

## 🚀 环境安装

首先确保你已安装 Python 3.9+ 环境，然后执行以下命令安装依赖：

```bash
pip install -e .
````

该命令会以开发模式安装当前项目及依赖库。

---

## 🔑 API 设置

在运行项目前，请先配置 API 密钥与接口地址。

```python
openai_kwargs["api_key"] = "sk-xxx"
openai_kwargs["base_url"] = "https://your-api-endpoint.com/v1"
```


---

## 🧩 工具配置

项目支持自定义工具集，可根据需要加载多种功能（如影像分析、报告生成、数据库检索等）。
部分工具需以 API 形式提供服务。

```python
tool_dict = {

}
```

---

## 🧪 运行测试

配置完成后，可直接运行主程序：

```bash
python chat.py
```

程序将启动交互式对话代理，可进行模型调用、工具测试或任务演示。

---

* ✅ 支持多模型（OpenAI / Qwen / GPT / LLaVA / Gemma 等）
* ✅ 灵活的工具注册机制
* ✅ 模块化设计，便于扩展
* ✅ 兼容多种医学影像任务（CXR Interpretation / Report / VQA）