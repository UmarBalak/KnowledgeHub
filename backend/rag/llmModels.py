import logging
import os
from typing import Optional, List, Mapping, Any, Dict
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from requests.exceptions import RequestException
from pydantic import Field

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LLM_MODEL = os.getenv("TOGETHER_MODEL_LLM1")

AZURE_AI_FOUNDRY_API_KEY = os.getenv("AZURE_AI_FOUNDRY_API_KEY")
AZURE_DEPLOYMENT_MODELS = os.getenv("AZURE_DEPLOYMENT_MODEL_NAMES")
if AZURE_DEPLOYMENT_MODELS:
    AZURE_DEPLOYMENT_MODEL_NAMES = AZURE_DEPLOYMENT_MODELS.split(",")

OPENAI_API_VERSION=os.getenv("OPENAI_API_VERSION")

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY must be set as an environment variable.")

class LLM:
    """Custom LangChain wrapper for TogetherAI LLM with ConversationBufferMemory"""

    model_name: str = LLM_MODEL
    together_api_key: str = TOGETHER_API_KEY
    AZURE_AI_FOUNDRY_API_KEY: str = AZURE_AI_FOUNDRY_API_KEY
    AZURE_MODEL_NAMES: list = AZURE_DEPLOYMENT_MODEL_NAMES
    OPENAI_API_VERSION: str = OPENAI_API_VERSION

    class LimitedBufferMemory(ConversationBufferMemory):
        max_messages: int = Field(default=5)

        def save_context(self, inputs, outputs):
            super().save_context(inputs, outputs)
            if len(self.chat_memory.messages) > self.max_messages:
                self.chat_memory.messages = self.chat_memory.messages[-self.max_messages:]


    def __init__(self, llm_model: str = "gpt", return_messages: bool = True, max_messages: int = 10):
        self.llm_model = llm_model

        # Initialize the LLM instance for memory
        self._llm_instance = self._get_llm_instance()
        
        # Initialize LimitedBufferMemory with max_messages
        self.memory = self.LimitedBufferMemory(
            memory_key="chat_history",
            return_messages=return_messages,
            max_messages=max_messages
        )

    def _get_llm_instance(self):
        """Get the appropriate LLM instance for memory operations"""
        if self.llm_model == "gpt":
            return AzureChatOpenAI(
                deployment_name=self.AZURE_MODEL_NAMES[0],
                api_key=self.AZURE_AI_FOUNDRY_API_KEY,
            )
        elif self.llm_model == "kimi":
            return AzureChatOpenAI(
                deployment_name=self.AZURE_MODEL_NAMES[1],
                api_key=self.AZURE_AI_FOUNDRY_API_KEY,
            )
        else:
            return ChatTogether(
                together_api_key=self.together_api_key,
                model=self.model_name,
                temperature=0.1,
            )
    
    @staticmethod
    def normalize_ai_message(msg) -> dict:
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

    def __azure_gpt5_llm(self, prompt: str, stop: Optional[list] = None):
        try:
            llm = AzureChatOpenAI(
                deployment_name=self.AZURE_MODEL_NAMES[0],
                api_key=self.AZURE_AI_FOUNDRY_API_KEY,
            )
            logging.info("Gpt5-mini LLM initialized.")
            response = llm.invoke(prompt, stop=stop)
            logging.info(f"LLM response: {response}")
            return response
        except Exception as e:
            print(f"Error with Azure LLM: {str(e)}")
            raise RuntimeError(f"Azure LLM error: {str(e)}") from e
        
    def __azure_kimi_llm(self, prompt: str, stop: Optional[list] = None):
        try:
            llm = AzureChatOpenAI(
                deployment_name=self.AZURE_MODEL_NAMES[1],
                api_key=self.AZURE_AI_FOUNDRY_API_KEY,
            )
            logging.info("Kimi-K2-Thinking LLM initialized.")
            response = llm.invoke(prompt, stop=stop)
            logging.info(f"LLM response: {response}")
            return response
        except Exception as e:
            print(f"Error with Azure LLM: {str(e)}")
            raise RuntimeError(f"Azure LLM error: {str(e)}") from e

    def __together_llm(self, prompt: str, stop: Optional[list] = None):
        try:
            llm_model = ChatTogether(
                together_api_key=self.together_api_key,
                model=self.model_name,
                temperature=0.1,
            )
            logging.info("TogetherAI LLM initialized.")
            response = llm_model.invoke(prompt, stop=stop)
            return response
        except RequestException as e:
            if e.response is not None and e.response.status_code == 429:  # Rate limit error
                print("Rate limit exceeded after all retries")
                raise RuntimeError("Rate limit exceeded, please try again later") from e
        except Exception as e:
                print(f"Unexpected error generating LLM response: {str(e)}")
                raise RuntimeError(f"Unexpected error: {str(e)}") from e

    # Required for instant llm testing
    def invoke(self, prompt: str, stop: Optional[list] = None) -> dict:
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
            if self.llm_model == "gpt":
                logging.info("Calling gpt5...")
                response = self.__azure_gpt5_llm(messages, stop)
            elif self.llm_model == "kimi":
                logging.info("Calling kimi...")
                response = self.__azure_kimi_llm(messages, stop)
            else:
                response = self.__together_llm(messages, stop)

            # Save the interaction to memory
            self.memory.save_context(
                {"input": prompt if isinstance(prompt, str) else prompt.content},
                {"output": response.content}
            )
            return self.normalize_ai_message(response)

        except Exception as e:
            logging.error("Error normalizing response")
            print(f"Error in invoke with memory: {str(e)}")
            raise

    def invoke_with_template(self, chat_prompt_template, variables: dict, stop: Optional[list] = None) -> dict:
        """
        Invoke LLM with ChatPromptTemplate while preserving memory
        """
        try:
            # Get chat history from memory
            chat_history = self.memory.chat_memory.messages
            
            # Format the prompt template
            formatted_messages = chat_prompt_template.format_messages(**variables)
            
            # Combine with history
            all_messages = chat_history + formatted_messages
            
            # Get response from appropriate LLM
            if self.llm_model == "gpt":
                response = self.__azure_gpt5_llm(all_messages, stop)
            elif self.llm_model == "kimi":
                response = self.__azure_kimi_llm(all_messages, stop)
            else:
                response = self.__together_llm(all_messages, stop)

            # Save to memory - use the last user message for input
            user_input = formatted_messages[-1].content if formatted_messages else str(variables)
            
            self.memory.save_context(
                {"input": user_input},
                {"output": response.content}
            )
            
            return self.normalize_ai_message(response)
            
        except Exception as e:
            logging.error("Error in invoke_with_template")
            raise
    
    def get_memory_summary(self) -> dict:
        """Get current memory state information"""
        return {
            "message_count": len(self.memory.chat_memory.messages),
            "summary": getattr(self.memory, 'moving_summary_buffer', None),
            "recent_messages": [
                {"type": type(msg).__name__, "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content}
                for msg in self.memory.chat_memory.messages[-5:]  # Last 5 messages to display
            ]
        }

    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()

if __name__ == "__main__":
    # Initialize LLM with memory (using GPT-5 by default)
    llm_with_memory = LLM(llm_model="kimi")
    
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