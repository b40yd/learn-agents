import getpass
import os
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv # 导入dotenv库，在本项目中用于加载环境变量文件

load_dotenv()

# LangChain 会自动查找 DEEPSEEK_API_KEY 环境变量
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter your DeepSeek API key: ")


llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)