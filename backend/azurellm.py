import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

ENDPOINT = os.getenv("AZURE_GPT5_MINI_ENDPOINT")
API_KEY = os.getenv("AZURE_GPT5_MINI_API_KEY")
API_VERSION = os.getenv("OPENAI_API_VERSION")

llm = AzureChatOpenAI(
    deployment_name="gpt-5-mini",
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
)

response = llm.invoke("Tell me a joke")

print(response)