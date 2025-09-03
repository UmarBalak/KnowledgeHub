import os
from typing import Optional, List, Mapping, Any, Dict
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_openai import AzureChatOpenAI
from requests.exceptions import RequestException

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LLM_MODEL = os.getenv("TOGETHER_MODEL_LLM1")

AZURE_ENDPOINT = os.getenv("AZURE_GPT5_MINI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_GPT5_MINI_API_KEY")
AZURE_API_VERSION = os.getenv("OPENAI_API_VERSION")

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY must be set as an environment variable.")

class LLM():
    """Custom LangChain wrapper for TogetherAI LLM"""

    model_name: str = LLM_MODEL
    together_api_key: str = TOGETHER_API_KEY
    azure_api_key: str = AZURE_API_KEY
    azure_endpoint: str = AZURE_ENDPOINT

    def __init__(self, gpt5: bool = True):
        self.gpt5 = gpt5

    @staticmethod
    def normalize_ai_message(msg) -> Dict[str, Any]:
        """Convert raw AIMessage into a normalized dict for logging/monitoring"""
        return {
            "id": msg.id,
            "content": msg.content,
            "finish_reason": msg.response_metadata.get("finish_reason"),
            "refusal": msg.additional_kwargs.get("refusal"),
            "tokens": {
                "input": msg.usage_metadata.get("input_tokens"),
                "output": msg.usage_metadata.get("output_tokens"),
                "total": msg.usage_metadata.get("total_tokens")
            }
        }
    
    def __azure_llm(self, prompt: str, stop: Optional[List[str]] = None):
        try:
            llm = AzureChatOpenAI(
                deployment_name="gpt-5-mini",
                api_key=self.azure_api_key,
                azure_endpoint=self.azure_endpoint,
                )
            response = llm.invoke(prompt, stop=stop)
            return response
        except Exception as e:
            print(f"Error with Azure LLM: {str(e)}")
            raise RuntimeError(f"Azure LLM error: {str(e)}") from e
        
    def __together_llm(self, prompt: str, stop: Optional[List[str]] = None):
        try:
            llm_model = ChatTogether(
                together_api_key=self.together_api_key,
                model=self.model_name,
                temperature=0.1,
            )
            response = llm_model.invoke(prompt, stop=stop)
            return response
        except RequestException as e:
            if "429" in str(e):  # Rate limit error
                print("Rate limit exceeded after all retries")
                raise RuntimeError("Rate limit exceeded, please try again later") from e
            
        except Exception as e:
            print(f"Unexpected error generating LLM response: {str(e)}")
            raise RuntimeError(f"Unexpected error: {str(e)}") from e


    def invoke(self, prompt: str, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generates response from TogetherAI model.
        """
        if self.gpt5:
            response =  self.__azure_llm(prompt, stop)
            return self.normalize_ai_message(response)
        else:
            response = self.__together_llm(prompt, stop)
            return self.normalize_ai_message(response)
        
