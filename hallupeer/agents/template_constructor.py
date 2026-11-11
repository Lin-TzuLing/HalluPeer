import asyncio

from .base import BaseAgent


class TemplateConstructor(BaseAgent):
    def __init__(self, model_name="Qwen/Qwen3-32B", max_new_tokens=1024, temperature=0.0, port=8003, **kwargs):
        super().__init__(model_name, max_new_tokens, temperature, port, **kwargs)

        self.think_mode = kwargs.get("think_mode", False)
    
    async def get_query(self, messages):
        """
        Make a single query.
        
        Args:
            messages (list): List of message dictionaries.
            retries (int): Number of retries.
            delay (float): Delay between retries.
            
        Returns:
            str or list: Generated response(s).
        """
        for attempt in range(self.max_retries):
            try:
                if self.model_name == "Qwen/Qwen3-32B":
                    response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": self.think_mode},
                                    },
                    n=self.n,
                    timeout=120,
                    )
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_new_tokens,
                        n=self.n,
                        timeout=120,
                    )
                    
                # Extract usage information
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }    
                    
                if self.n > 1:
                    content = [choice["message"]["content"] for choice in response.choices]
                else:
                    content = response.choices[0].message.content
                    
                return content, usage
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error: {str(e)}, attempt {attempt}")
                    await asyncio.sleep(self.retry_delay)  # wait before retrying
                else:
                    return f"Error: {str(e)}"
    

    