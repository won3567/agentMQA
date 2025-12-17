"""
OpenRouter API client for accessing ChatGPT and other models
"""
import os
import requests
import logging
from typing import Optional, Dict
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Hardened OpenRouter API Client with retries & chunked error recovery"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            self.api_key = "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"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        # Create a robust session
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _post(self, payload, headers):
        """Internal safe POST with chunk-error fallback"""

        try:
            # First attempt
            response = self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=(10, 60),  # connect timeout, read timeout
                stream=False,       # IMPORTANT: avoid chunk streaming issues
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ChunkedEncodingError:
            logger.warning("ChunkedEncodingError — retrying with stream=False fallback")

            # Retry once more with a new clean request
            response = self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=(10, 60),
                stream=False,
            )
            response.raise_for_status()
            return response.json()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Generate response from OpenRouter API with enhanced stability.
        """

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,   # prevent server chunk-streaming (causes most errors)
        }

        try:
            data = self._post(payload, headers)

            return {
                "response": data["choices"][0]["message"]["content"],
                "model": data.get("model"),
                "usage": data.get("usage", {}),
                "finish_reason": data["choices"][0].get("finish_reason"),
            }

        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise


    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Answer a question, optionally with context
        
        Args:
            question: Question to answer
            context: Retrieved context (optional)
            
        Returns:
            Answer string
        """
        system_prompt = """
You are a medical science expert. Choose one correct answer from the given options using correct biomedical reasoning.
Format your response strictly as:
[
"step_by_step_thinking": "Your reasoning",
"answer_choice": "One letter from [A, B, C, D]"
]
"""

        if context:
            system_prompt += "\n\nUse the following context to help think step by step and answer the question:"
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
                
        result = self.generate(prompt, system_prompt=system_prompt)
        return result["response"]
    
    def give_inference_chain(
        self,
        question: str,
    ) -> str:
        """
        Given a question, return the inference chain
        
        Args:
            question: Question to answer
            
        Returns:
            Inference chain string
        """
        system_prompt = """
You are a medical science expert. Give a brief inference chain to answer the question.
Format your response strictly as:
[
"Inference_chain": "Your inference chain",
]
"""
        prompt = f"Question: {question}\n\nAnswer:"
                
        result = self.generate(prompt, system_prompt=system_prompt)
        return result["response"]
     
    def answer_question_with_inference(
        self,
        question: str,
        inference: str
    ) -> str:
        """
        Answer a question with inference hint
        
        Args:
            question: Question to answer
            inference: inference hint to help answer the question
            
        Returns:
            Answer string
        """
        system_prompt = """
You are a medical science expert. Choose one correct answer from the given options, by using the inference as hint and completing the reasoning step by step.
Format your response strictly as:
[
"step_by_step_thinking": "Your reasoning",
"answer_choice": "One letter from [A, B, C, D]"
]
"""
        prompt = f"Question: {question}\n\nInference: {inference}"
        
        result = self.generate(prompt, system_prompt=system_prompt)
        return result["response"]


def test_openrouter():
    """Test OpenRouter client"""
    client = OpenRouterClient()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_openrouter()
