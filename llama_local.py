"""
Local Llama 3.1 8B model with 4-bit quantization
"""
import torch
import logging
from typing import Optional, Dict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)


class LlamaLocal:
    """Local Llama 3.1 8B Instruct model"""
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit: bool = True,
        max_memory: str = "38GB",
        temperature: float = 0.0,
        max_new_tokens: int = 2048
    ):
        """
        Initialize local Llama model
        
        Args:
            model_path: HuggingFace model path
            load_in_4bit: Use 4-bit quantization
            max_memory: Maximum GPU memory to use
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        logger.info(f"Loading Llama model: {model_path}")
        
        # Configure quantization
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory={0: max_memory} if max_memory else None,
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        self.model.eval()
        logger.info("Model loaded successfully")
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> Dict:
        """
        Generate response from Llama
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Override temperature
            max_new_tokens: Override max tokens
            
        Returns:
            Dictionary with response and metadata
        """
        # Format with Llama 3.1 chat template
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Generate
        temp = temperature if temperature is not None else self.temperature
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp if temp > 0 else 1.0,  # Avoid temp=0 for sampling
                do_sample=temp > 0,
                top_p=0.9 if temp > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return {
            "response": generated_text.strip(),
            "model": self.model_path,
            "input_length": inputs['input_ids'].shape[1],
            "output_length": len(outputs[0]) - inputs['input_ids'].shape[1]
        }
    
    def answer_question(
        self,
        question: str,
        context: Optional[str] = None
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
    
    def answer_question_half_reasoning(
        self,
        question: str,
        reasoning: str
    ) -> str:
        """
        Answer a question with half reasoning
        
        Args:
            question: Question to answer
            reasoning: Reasoning to help answer the question
            
        Returns:
            Answer string
        """
        system_prompt = """
You are a medical science expert. Choose one correct answer from the given options, by using the half reasoning as hint and completing the reasoning step by step.
Format your response strictly as:
[
"step_by_step_thinking": "Completed reasoning(Half reasoning + your supplemented reasoning)",
"answer_choice": "One letter from [A, B, C, D]"
]
"""
        prompt = f"Question: {question}\n\nHalf reasoning: {reasoning}"
        
        result = self.generate(prompt, system_prompt=system_prompt)
        return result["response"]


    def choose_reasoning(
        self,
        question: str,
        reasoning_a: str,
        reasoning_b: str,
        label_a: str = "Reasoning A",
        label_b: str = "Reasoning B"
    ) -> Dict:
        """
        Choose between two reasoning processes
        
        Args:
            question: Original question
            reasoning_a: First reasoning process
            reasoning_b: Second reasoning process
            label_a: Label for first reasoning
            label_b: Label for second reasoning
            
        Returns:
            Dictionary with choice and explanation
        """
        system_prompt = """You are evaluating two different reasoning processes for a medical question.
Choose the reasoning that is more correct, logical, and leads to the right answer.
Respond with ONLY "A" or "B" followed by a brief explanation."""
        
        prompt = f"""Question: {question}

{label_a}:
{reasoning_a}

{label_b}:
{reasoning_b}

Which reasoning process is more correct? Respond with "A" or "B" followed by your explanation."""
        
        result = self.generate(prompt, system_prompt=system_prompt)
        response = result["response"]
        
        # Parse choice
        choice = None
        if response.strip().upper().startswith('A'):
            choice = 'A'
        elif response.strip().upper().startswith('B'):
            choice = 'B'
        
        return {
            "choice": choice,
            "explanation": response,
            "full_response": response
        }


def test_llama_local():
    """Test local Llama model"""
    print("Loading Llama 3.1 8B model...")
    model = LlamaLocal()
    
    question = "What is the main function of mitochondria?"
    print(f"\nQuestion: {question}\n")
    
    # Test basic generation
    answer = model.answer_question(question)
    print(f"Answer:\n{answer}\n")
    
    # # Test with reasoning
    # print("\nTesting with reasoning format...")
    # result = model.generate_with_reasoning(question)
    # print(f"Reasoning:\n{result['reasoning']}\n")
    # print(f"Answer:\n{result['answer']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_llama_local()
