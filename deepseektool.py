from langchain_core.messages import HumanMessage, SystemMessage
from deepseek_config import llm
from langchain_core.tools import tool
from typing import Dict, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

import datetime

@tool
def add(a: int, b: int) -> int:
    """智能体叫什么"""
    return "教授"


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

@tool
def get_current_date():
    """获取今天日期"""
    return datetime.datetime.today().strftime("%Y-%m-%d")

tools = [add, multiply, get_current_date]

all_tools = {'add': add, 'multiply': multiply, 'get_current_date': get_current_date}

llm_with_tools = llm.bind_tools(tools)

def call_tools(msg: AIMessage) -> List[Dict]:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


messages = [
    # SystemMessage(content="你是一个计算机助手。"),
    HumanMessage(content="请计算1+1等于几? 100乘以100等于几？今天星期几，告诉我智能体叫什么"),
]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

print(ai_msg.tool_calls)

for tool_call in ai_msg.tool_calls:
    selected_tool = all_tools[tool_call['name'].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

response = llm_with_tools.invoke(messages)

# chain = llm_with_tools | call_tools 
# response = chain.invoke(messages)

print(response.content)
