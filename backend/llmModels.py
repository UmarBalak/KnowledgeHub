import os
from typing import Optional, List, Mapping, Any, Dict
from dotenv import load_dotenv
from langchain_together import ChatTogether

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LLM_MODEL = os.getenv("TOGETHER_MODEL_LLM1")

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY must be set as an environment variable.")

class LLM():
    """Custom LangChain wrapper for TogetherAI LLM"""

    model_name: str = LLM_MODEL
    api_key: str = TOGETHER_API_KEY

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


    def invoke(self, prompt: str, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generates response from TogetherAI model.
        """
        try:
            llm_model = ChatTogether(
                together_api_key=self.api_key,
                model=self.model_name,
                temperature=0.1,
            )
            response = llm_model.invoke(prompt, stop=stop)
            return self.normalize_ai_message(response)
        
        except Exception as e:
            print(f"Error generating LLM response with TogetherAI: {e}")
            raise

# Example usage in LangChain pipeline:
llm = LLM()
response = llm.invoke("Hello, how are you?")
print(response["content"])

