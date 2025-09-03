import os
from typing import Optional, List, Mapping, Any, Dict
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from requests.exceptions import RequestException

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LLM_MODEL = os.getenv("TOGETHER_MODEL_LLM1")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_GPT5_MINI_ENDPOINT")
AZURE_AI_FOUNDRY_API_KEY = os.getenv("AZURE_AI_FOUNDRY_API_KEY")
AZURE_API_VERSION = os.getenv("OPENAI_API_VERSION")

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY must be set as an environment variable.")

class LLM():
    """Custom LangChain wrapper for TogetherAI LLM with ConversationBufferMemory"""

    model_name: str = LLM_MODEL
    together_api_key: str = TOGETHER_API_KEY
    AZURE_AI_FOUNDRY_API_KEY: str = AZURE_AI_FOUNDRY_API_KEY
    AZURE_OPENAI_ENDPOINT: str = AZURE_OPENAI_ENDPOINT

    def __init__(self, gpt5: bool = True, return_messages: bool = True):
        self.gpt5 = gpt5
        
        # Initialize the LLM instance for memory
        self._llm_instance = self._get_llm_instance()
        
        # Initialize ConversationBufferMemory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=return_messages,
        )

    def _get_llm_instance(self):
        """Get the appropriate LLM instance for memory operations"""
        if self.gpt5:
            return AzureChatOpenAI(
                deployment_name="gpt-5-nano",
                api_key=self.AZURE_AI_FOUNDRY_API_KEY,
            )
        else:
            return ChatTogether(
                together_api_key=self.together_api_key,
                model=self.model_name,
                temperature=0.1,
            )

    @staticmethod
    def normalize_ai_message(msg) -> Dict[str, Any]:
        """Convert raw AIMessage into a normalized dict for logging/monitoring"""
        return {
            "id": getattr(msg, 'id', None),
            "content": msg.content,
            "finish_reason": getattr(msg, 'response_metadata', {}).get("finish_reason"),
            "refusal": getattr(msg, 'additional_kwargs', {}).get("refusal"),
            "tokens": {
                "input": getattr(msg, 'usage_metadata', {}).get("input_tokens"),
                "output": getattr(msg, 'usage_metadata', {}).get("output_tokens"),
                "total": getattr(msg, 'usage_metadata', {}).get("total_tokens")
            }
        }
    
    def __azure_llm(self, prompt: str, stop: Optional[List[str]] = None):
        try:
            llm = AzureChatOpenAI(
                deployment_name="gpt-5-nano",
                api_key=self.AZURE_AI_FOUNDRY_API_KEY,
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
        Generates response from LLM model with memory context.
        """
        try:
            # Get chat history from memory
            chat_history = self.memory.chat_memory.messages
            
            # Create messages list with history + current prompt
            if isinstance(prompt, str):
                messages = chat_history + [HumanMessage(content=prompt)]
            else:
                messages = chat_history + [prompt]
            
            # Get response from appropriate LLM
            if self.gpt5:
                response = self.__azure_llm(messages, stop)
            else:
                response = self.__together_llm(messages, stop)
            
            # Save the interaction to memory
            self.memory.save_context(
                {"input": prompt if isinstance(prompt, str) else prompt.content},
                {"output": response.content}
            )
            
            return self.normalize_ai_message(response)
            
        except Exception as e:
            print(f"Error in invoke with memory: {str(e)}")
            raise

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get current memory state information"""
        return {
            "message_count": len(self.memory.chat_memory.messages),
            "summary": getattr(self.memory, 'moving_summary_buffer', None),
            "recent_messages": [
                {"type": type(msg).__name__, "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content}
                for msg in self.memory.chat_memory.messages[-5:]  # Last 5 messages
            ]
        }

    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()

if __name__ == "__main__":
    # Initialize LLM with memory (using GPT-5 by default)
    llm_with_memory = LLM(gpt5=True)
    
    # Example conversation
    try:
        # First message
        response1 = llm_with_memory.invoke("Hello, my name is John and I love programming.")
        print("Response 1:", response1["content"])
        
        # Second message (memory will include previous context)
        response2 = llm_with_memory.invoke("What's my name and what do I love?")
        print("Response 2:", response2["content"])
        
        # Check memory state
        memory_info = llm_with_memory.get_memory_summary()
        print("Memory Info:", memory_info)
        
    except Exception as e:
        print(f"Error: {e}")