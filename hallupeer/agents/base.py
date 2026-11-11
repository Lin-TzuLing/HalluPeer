import asyncio
from openai import AsyncOpenAI


class BaseAgent:
    def __init__(self, model_name, max_new_tokens, temperature, port, **kwargs):
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.port = port
        
        base_url = f"http://localhost:{self.port}/v1"
        api_key = kwargs.get('api_key', 'abc')  # Use provided key or default

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.n = kwargs.get('n', 1)
        
        self.max_retries = kwargs.get('max_retries', 80)
        self.retry_delay = kwargs.get('retry_delay', 0.1)


    async def predict(self, input_data):
        """
        Generate predictions for the given input data.
        
        Args:
            input_data (list): List of messages for each request.
            
        Returns:
            list: List of generated responses.
        """
        
        result = []
        usages = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "max_total_tokens": 0
        }
        
        # Gather the results concurrently using asyncio
        tasks = [self.get_query(messages) for messages in input_data]

        # Run all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for response in responses:
            if isinstance(response, Exception):
                print(f"An error occurred: {response}")
            else:
                result.append(response[0])
                usage = response[1]
                usages["prompt_tokens"] += usage["prompt_tokens"]
                usages["completion_tokens"] += usage["completion_tokens"]
                usages["total_tokens"] += usage["total_tokens"]
                usages["max_total_tokens"] = max(usages["max_total_tokens"], usage["total_tokens"])

        return result, usages


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


    def __call__(self, input_data):
        """ 
        Generate responses for the given input data.
        
        Args:   
            input_data (list): List of messages for each request.
        
        Returns:
            list: List of generated text responses.
        """
        return self.predict(input_data)
    

    def verbose_print(self):
        """
        Print a message if verbose mode is enabled.
        """
        
        print(f"Model Configuration:")
        print(f"   - Model name: {self.model_name}")
        print(f"   - Max new tokens: {self.max_new_tokens}")
        print(f"   - Temperature: {self.temperature}")