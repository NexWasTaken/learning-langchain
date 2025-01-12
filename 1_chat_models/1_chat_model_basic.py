import getpass
import os
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
r = model.invoke("Hello, world!")
print(r)