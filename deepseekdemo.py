from langchain_core.messages import HumanMessage, SystemMessage
from deepseek_config import llm

# 2. 基本调用 (Invoke)
print("--- 基本调用 ---")
message = HumanMessage(content="你好，DeepSeek！给我写一首关于春天的短诗。")
response = llm.invoke([message])
print(response.content)

# 3. 加入系统消息
print("\n--- 加入系统消息 ---")
messages = [
    SystemMessage(content="你是一个乐于助人的 AI 助手，请用简洁明了的语言回答。"),
    HumanMessage(content="简单解释一下什么是大型语言模型 (LLM)?"),
]
response = llm.invoke(messages)
print(response.content)

# 4. 流式输出 (Stream)
print("\n--- 流式输出 ---")
try:
    # 注意：流式输出通常需要异步环境或特定处理方式
    # 这里是一个简化的同步示例，逐块打印
    stream = llm.stream("介绍一下 LangChain 框架的主要功能。")
    full_response = ""
    for chunk in stream:
        # chunk 的类型可能是 AIMessageChunk 或类似结构
        if hasattr(chunk, 'content'):
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
    print("\n--- 流式输出结束 ---")
    # print("完整响应:\n", full_response) # 可以取消注释查看完整响应
except NotImplementedError:
    print("当前模型或环境配置可能不支持同步流式输出，请尝试异步或检查库版本。")
except Exception as e:
    print(f"\n流式输出时发生错误: {e}")


# 5. 与 LangChain Expression Language (LCEL) 结合使用
print("\n--- 结合 LCEL ---")
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的翻译助手。"),
    ("human", "将下面的英文句子翻译成中文： {sentence}")
])

# 构建链
chain = prompt | llm

# 调用链
result = chain.invoke({"sentence": "LangChain makes it easy to build applications with LLMs."})
print(result.content)

# 也可以直接获取字符串输出
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result_str = chain.invoke({"sentence": "DeepSeek offers powerful language models."})
print(result_str)
