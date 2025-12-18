import pandas as pd
import numpy as np
import anthropic
import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from collections import defaultdict
import time

@dataclass
class PromptTemplate:
    """Container for prompt templates"""
    name: str
    template: str
    description: str
    score: float = 0.0
    success_count: int = 0
    total_count: int = 0

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    question: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    prompt_name: str
    confidence: float
    reasoning: str

class PromptOptimizer:
    """
    Automated prompt optimization system for MedQA question answering.
    Uses genetic algorithm-inspired approach with performance-based evolution.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize prompt optimizer.
        
        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        """
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.prompt_templates = self._initialize_prompts()
        self.evaluation_history = []
        
    def _initialize_prompts(self) -> List[PromptTemplate]:
        """Initialize diverse prompt templates for testing."""
        
        templates = [
            PromptTemplate(
                name="baseline",
                description="Simple baseline prompt",
                template="""You are a medical expert. Answer this MedQA question using the provided context.

Context:
{context}

Question: {question}

Answer:"""
            ),
            
            PromptTemplate(
                name="step_by_step",
                description="Chain-of-thought reasoning",
                template="""You are a medical expert answering MedQA questions. Use the context below and think step-by-step.

Context:
{context}

Question: {question}

Please analyze this question step by step:
1. Identify key clinical features
2. Consider differential diagnoses
3. Apply relevant medical knowledge from the context
4. Eliminate incorrect options
5. Select the most appropriate answer

Your answer:"""
            ),
            
            PromptTemplate(
                name="differential_diagnosis",
                description="Focused on differential diagnosis process",
                template="""You are an expert physician answering a MedQA question. Use systematic clinical reasoning.

Reference Material:
{context}

Clinical Question: {question}

Apply the following diagnostic approach:
• Patient presentation: What are the key symptoms/signs?
• Differential diagnosis: What conditions should be considered?
• Context analysis: What relevant information is in the reference material?
• Clinical decision: Which answer is most consistent with the presentation?

Provide your final answer with brief justification:"""
            ),
            
            PromptTemplate(
                name="evidence_based",
                description="Emphasizes evidence-based reasoning with citations",
                template="""You are a medical expert providing evidence-based answers to MedQA questions.

Available Evidence:
{context}

Question: {question}

Instructions:
- Cite specific evidence from the context using [Source X]
- Prioritize high-quality clinical evidence
- Consider epidemiology, pathophysiology, and clinical presentation
- If multiple answers seem plausible, explain why one is most likely

Evidence-based answer:"""
            ),
            
            PromptTemplate(
                name="multiple_choice_focused",
                description="Optimized for multiple choice questions",
                template="""You are answering a MedQA multiple choice question. Use the provided medical context.

Medical Context:
{context}

Question: {question}

Strategy:
1. Read the question stem carefully - identify the clinical scenario
2. Predict the answer before looking at options (if applicable)
3. Eliminate clearly incorrect options
4. Compare remaining options based on:
   - Frequency/epidemiology (common things are common)
   - Specificity of symptoms/signs
   - Temporal relationships
   - Evidence from context
5. Choose the BEST answer (not just a correct answer)

Your answer and brief reasoning:"""
            ),
            
            PromptTemplate(
                name="pattern_recognition",
                description="Uses pattern recognition and clinical pearls",
                template="""You are an experienced clinician answering a MedQA question. Apply pattern recognition.

Reference Information:
{context}

Clinical Scenario: {question}

Think like an expert:
- What classic presentation or pattern does this match?
- Are there "red flag" or pathognomonic features?
- What would be the first-line test/treatment?
- What's the most likely diagnosis given the epidemiology?

Apply clinical pearls and the context to select the best answer:"""
            ),
            
            PromptTemplate(
                name="structured_soap",
                description="SOAP note structure for systematic analysis",
                template="""You are a medical expert using structured clinical reasoning (SOAP format).

Reference Material:
{context}

Question: {question}

Analyze systematically:
S (Subjective): What symptoms are described?
O (Objective): What signs/findings are present?
A (Assessment): What are the most likely diagnoses based on S+O and the reference material?
P (Plan): What answer best addresses the clinical question?

Final answer with justification:"""
            ),
            
            PromptTemplate(
                name="risk_vs_benefit",
                description="Decision-making framework",
                template="""You are a clinical decision-maker answering a MedQA question.

Medical Context:
{context}

Question: {question}

Clinical decision framework:
• What is being asked? (diagnosis, treatment, prognosis, mechanism)
• What does the context tell us about this scenario?
• For each option: What are the benefits vs risks/likelihood?
• Which option provides the best outcome or is most accurate?

Decision and rationale:"""
            ),
            
            PromptTemplate(
                name="teaching_mode",
                description="Educational approach with detailed explanation",
                template="""You are a medical educator answering a MedQA board-style question.

Educational Material:
{context}

Question: {question}

Teaching points:
- Explain the relevant pathophysiology or clinical principle
- Connect the question to the material provided
- Identify why correct answer is correct AND why others are wrong
- Provide a memorable clinical pearl

Answer with educational explanation:"""
            ),
            
            PromptTemplate(
                name="meta_reasoning",
                description="Explicit reasoning about reasoning process",
                template="""You are a medical expert with meta-cognitive awareness answering a MedQA question.

Context:
{context}

Question: {question}

Meta-reasoning approach:
1. What type of question is this? (diagnosis, treatment, mechanism, etc.)
2. What knowledge domains are being tested?
3. What information from context is most relevant?
4. What cognitive biases should I avoid? (anchoring, availability, etc.)
5. What is the most defensible answer given the evidence?

Final answer with meta-commentary:"""
            )
        ]
        
        return templates
    
    def extract_answer(self, response_text: str, question: str) -> str:
        """
        Extract the actual answer from Claude's response.
        Handles various response formats.
        """
        # Look for explicit answer patterns
        patterns = [
            r"(?i)(?:the )?(?:correct )?answer is:?\s*([A-E])[.\s]",
            r"(?i)(?:select|choose|pick)\s*(?:option\s*)?([A-E])[.\s]",
            r"(?i)^([A-E])[.\s)]",  # Starts with letter
            r"(?i)final answer:?\s*([A-E])[.\s]",
            r"(?i)therefore,?\s*([A-E])[.\s]",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1).upper()
        
        # If no explicit pattern, look for most prominent letter (A-E)
        letters = re.findall(r'\b([A-E])\b', response_text)
        if letters:
            # Return most common letter in second half of response
            second_half = response_text[len(response_text)//2:]
            second_half_letters = re.findall(r'\b([A-E])\b', second_half)
            if second_half_letters:
                from collections import Counter
                return Counter(second_half_letters).most_common(1)[0][0]
        
        return "UNKNOWN"
    
    def evaluate_prompt(self, prompt_template: PromptTemplate, 
                       questions_df: pd.DataFrame,
                       context_retrieval_func,
                       sample_size: Optional[int] = None) -> float:
        """
        Evaluate a prompt template on a set of questions.
        
        Args:
            prompt_template: The prompt template to evaluate
            questions_df: DataFrame with columns: 'question', 'answer', 'options'
            context_retrieval_func: Function that takes question and returns context
            sample_size: Number of questions to test (None = all)
        
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if sample_size:
            questions_df = questions_df.sample(n=min(sample_size, len(questions_df)))
        
        correct_count = 0
        results = []
        
        print(f"\n{'='*80}")
        print(f"Evaluating prompt: {prompt_template.name}")
        print(f"Description: {prompt_template.description}")
        print(f"Testing on {len(questions_df)} questions...")
        print(f"{'='*80}\n")
        
        for idx, row in questions_df.iterrows():
            question = row['question']
            correct_answer = row['answer']
            
            # Get context (from RAG system)
            context = context_retrieval_func(question)
            
            # Format prompt
            formatted_prompt = prompt_template.template.format(
                context=context,
                question=question
            )
            
            # Get response from Claude
            try:
                message = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1500,
                    temperature=0.0,  # Deterministic for evaluation
                    messages=[{"role": "user", "content": formatted_prompt}]
                )
                
                response_text = message.content[0].text
                predicted_answer = self.extract_answer(response_text, question)
                is_correct = (predicted_answer == correct_answer)
                
                if is_correct:
                    correct_count += 1
                
                result = EvaluationResult(
                    question=question,
                    correct_answer=correct_answer,
                    predicted_answer=predicted_answer,
                    is_correct=is_correct,
                    prompt_name=prompt_template.name,
                    confidence=0.0,  # Could extract from response
                    reasoning=response_text[:200]
                )
                
                results.append(result)
                
                # Progress update
                if (idx + 1) % 5 == 0:
                    current_acc = correct_count / (idx + 1)
                    print(f"Progress: {idx + 1}/{len(questions_df)} | "
                          f"Current accuracy: {current_acc:.2%}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error on question {idx}: {str(e)}")
                continue
        
        accuracy = correct_count / len(questions_df) if len(questions_df) > 0 else 0.0
        
        # Update template statistics
        prompt_template.score = accuracy
        prompt_template.success_count = correct_count
        prompt_template.total_count = len(questions_df)
        
        # Store results
        self.evaluation_history.extend(results)
        
        print(f"\n{'='*80}")
        print(f"Results for {prompt_template.name}:")
        print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(questions_df)})")
        print(f"{'='*80}\n")
        
        return accuracy
    
    def optimize_prompts(self, questions_df: pd.DataFrame,
                        context_retrieval_func,
                        n_iterations: int = 3,
                        questions_per_iteration: int = 20) -> PromptTemplate:
        """
        Optimize prompts through iterative evaluation and evolution.
        
        Args:
            questions_df: Evaluation dataset
            context_retrieval_func: Function to retrieve context for questions
            n_iterations: Number of optimization iterations
            questions_per_iteration: Questions to test per iteration
        
        Returns:
            Best performing prompt template
        """
        print(f"\n{'#'*80}")
        print(f"# PROMPT OPTIMIZATION - {n_iterations} ITERATIONS")
        print(f"# Testing {questions_per_iteration} questions per prompt")
        print(f"# Total prompts: {len(self.prompt_templates)}")
        print(f"{'#'*80}\n")
        
        for iteration in range(n_iterations):
            print(f"\n{'█'*80}")
            print(f"█ ITERATION {iteration + 1}/{n_iterations}")
            print(f"{'█'*80}\n")
            
            # Evaluate all current prompts
            for template in self.prompt_templates:
                self.evaluate_prompt(
                    template,
                    questions_df,
                    context_retrieval_func,
                    sample_size=questions_per_iteration
                )
            
            # Sort by performance
            self.prompt_templates.sort(key=lambda x: x.score, reverse=True)
            
            # Show iteration results
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1} RESULTS - LEADERBOARD:")
            print(f"{'='*80}")
            for i, template in enumerate(self.prompt_templates, 1):
                print(f"{i}. {template.name:25s} | "
                      f"Accuracy: {template.score:.2%} | "
                      f"({template.success_count}/{template.total_count})")
            print(f"{'='*80}\n")
            
            # Generate new prompts based on top performers (for iterations 2+)
            if iteration < n_iterations - 1:
                print("Generating evolved prompts from top performers...\n")
                new_prompts = self._evolve_prompts(self.prompt_templates[:3])
                self.prompt_templates.extend(new_prompts)
        
        # Final evaluation on all questions
        print(f"\n{'#'*80}")
        print(f"# FINAL EVALUATION ON FULL DATASET")
        print(f"{'#'*80}\n")
        
        best_template = self.prompt_templates[0]
        final_accuracy = self.evaluate_prompt(
            best_template,
            questions_df,
            context_retrieval_func,
            sample_size=None  # Use all questions
        )
        
        print(f"\n{'█'*80}")
        print(f"█ OPTIMIZATION COMPLETE")
        print(f"█ Best Prompt: {best_template.name}")
        print(f"█ Final Accuracy: {final_accuracy:.2%}")
        print(f"{'█'*80}\n")
        
        return best_template
    
    def _evolve_prompts(self, top_prompts: List[PromptTemplate]) -> List[PromptTemplate]:
        """
        Generate new prompt variations by combining elements from top performers.
        Uses Claude to create intelligent variations.
        """
        evolved = []
        
        # Combine top prompt characteristics
        prompt_analysis = "\n\n".join([
            f"Prompt '{p.name}' (Accuracy: {p.score:.2%}):\n{p.template}"
            for p in top_prompts
        ])
        
        evolution_prompt = f"""You are a prompt engineering expert. Analyze these high-performing prompts for medical question answering and create 2 NEW improved prompt variations.

Top performing prompts:
{prompt_analysis}

Create 2 new prompts that:
1. Combine the best elements from the top prompts
2. Introduce novel reasoning strategies
3. Are specifically optimized for medical multiple-choice questions
4. Maintain the variables {{context}} and {{question}}

Return your response in this JSON format:
{{
  "prompts": [
    {{
      "name": "descriptive_name",
      "description": "what makes this prompt unique",
      "template": "the actual prompt template with {{context}} and {{question}} variables"
    }},
    {{
      "name": "another_name",
      "description": "another unique approach",
      "template": "another prompt template"
    }}
  ]
}}"""
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.7,  # Some creativity for variation
                messages=[{"role": "user", "content": evolution_prompt}]
            )
            
            response_text = message.content[0].text
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                for prompt_data in data.get('prompts', []):
                    evolved.append(PromptTemplate(
                        name=f"evolved_{prompt_data['name']}",
                        description=prompt_data['description'],
                        template=prompt_data['template']
                    ))
                    
            print(f"✓ Generated {len(evolved)} evolved prompts\n")
            
        except Exception as e:
            print(f"Warning: Could not generate evolved prompts: {str(e)}\n")
        
        return evolved
    
    def save_results(self, filepath: str = "prompt_optimization_results.json"):
        """Save optimization results to file."""
        results = {
            'prompts': [
                {
                    'name': p.name,
                    'description': p.description,
                    'template': p.template,
                    'score': p.score,
                    'success_count': p.success_count,
                    'total_count': p.total_count
                }
                for p in sorted(self.prompt_templates, key=lambda x: x.score, reverse=True)
            ],
            'evaluation_history': [
                {
                    'question': e.question[:100],
                    'correct_answer': e.correct_answer,
                    'predicted_answer': e.predicted_answer,
                    'is_correct': e.is_correct,
                    'prompt_name': e.prompt_name
                }
                for e in self.evaluation_history
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to {filepath}")
    
    def generate_report(self) -> str:
        """Generate detailed optimization report."""
        report = []
        report.append("\n" + "="*80)
        report.append("PROMPT OPTIMIZATION REPORT")
        report.append("="*80 + "\n")
        
        # Sort prompts by score
        sorted_prompts = sorted(self.prompt_templates, key=lambda x: x.score, reverse=True)
        
        report.append("FINAL RANKINGS:")
        report.append("-"*80)
        for i, template in enumerate(sorted_prompts, 1):
            report.append(f"\n{i}. {template.name}")
            report.append(f"   Description: {template.description}")
            report.append(f"   Accuracy: {template.score:.2%} ({template.success_count}/{template.total_count})")
        
        report.append("\n" + "="*80)
        report.append("BEST PROMPT TEMPLATE:")
        report.append("="*80)
        best = sorted_prompts[0]
        report.append(f"\nName: {best.name}")
        report.append(f"Accuracy: {best.score:.2%}")
        report.append(f"\nTemplate:\n{best.template}")
        
        report.append("\n" + "="*80)
        report.append("RECOMMENDATIONS:")
        report.append("="*80)
        
        # Analyze what works
        top_3 = sorted_prompts[:3]
        common_elements = []
        
        if all('step' in p.template.lower() or 'systematic' in p.template.lower() for p in top_3):
            common_elements.append("• Structured, step-by-step reasoning performs well")
        if all('evidence' in p.template.lower() or 'context' in p.template.lower() for p in top_3):
            common_elements.append("• Explicit reference to context/evidence improves accuracy")
        if all('differential' in p.template.lower() or 'diagnosis' in p.template.lower() for p in top_3):
            common_elements.append("• Differential diagnosis frameworks are effective")
        
        report.append("\nKey success factors identified:")
        report.extend(common_elements if common_elements else ["• Analysis inconclusive - more iterations needed"])
        
        report.append("\n" + "="*80 + "\n")
        
        return "\n".join(report)


# Example usage and integration with RAG system
def example_usage():
    """Demonstrate prompt optimization workflow."""
    
    # Sample evaluation dataset (replace with real MedQA data)
    sample_questions = pd.DataFrame({
        'question': [
            "A 45-year-old woman presents with fatigue and weight gain. Labs show TSH 12 mU/L (normal 0.4-4.0). What is the most likely diagnosis? A) Hyperthyroidism B) Hypothyroidism C) Cushing syndrome D) Addison disease E) Diabetes mellitus",
            "A 65-year-old man with crushing chest pain radiating to left arm. ECG shows ST elevations in leads II, III, aVF. What is the diagnosis? A) Anterior MI B) Inferior MI C) Lateral MI D) Posterior MI E) STEMI",
            # Add more questions...
        ],
        'answer': ['B', 'B'],  # Correct answers
        'options': [
            ['Hyperthyroidism', 'Hypothyroidism', 'Cushing syndrome', 'Addison disease', 'Diabetes mellitus'],
            ['Anterior MI', 'Inferior MI', 'Lateral MI', 'Posterior MI', 'STEMI']
        ]
    })
    
    # Mock context retrieval function (replace with real RAG system)
    def get_context(question):
        # This should call your RAG system's hybrid_search
        # For demo, return mock context
        return """
        Hypothyroidism is characterized by elevated TSH and low thyroid hormones.
        Common symptoms include fatigue, weight gain, cold intolerance, and bradycardia.
        Primary hypothyroidism shows elevated TSH with low T4/T3.
        """
    
    # Initialize optimizer
    optimizer = PromptOptimizer(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Run optimization
    best_prompt = optimizer.optimize_prompts(
        questions_df=sample_questions,
        context_retrieval_func=get_context,
        n_iterations=2,
        questions_per_iteration=10
    )
    
    # Generate report
    report = optimizer.generate_report()
    print(report)
    
    # Save results
    optimizer.save_results("optimization_results.json")
    
    return best_prompt


# Integration with MedQARAGSystem
def integrate_with_rag(rag_system, eval_dataset_path: str, api_key: str):
    """
    Complete integration: optimize prompts for your RAG system.
    
    Args:
        rag_system: Instance of MedQARAGSystem
        eval_dataset_path: Path to evaluation dataset CSV
        api_key: Anthropic API key
    """
    
    # Load evaluation dataset
    eval_df = pd.read_csv(eval_dataset_path)
    
    # Create context retrieval function that uses your RAG
    def get_rag_context(question: str, top_k: int = 5) -> str:
        """Retrieve context using RAG system."""
        retrieved_docs = rag_system.hybrid_search(
            query=question,
            top_k=top_k,
            alpha=0.5,
            use_advanced_semantic=True
        )
        
        # Format context
        context_parts = []
        for i, item in enumerate(retrieved_docs, 1):
            doc = item.doc
            content = doc['contents'][:400]  # Truncate
            context_parts.append(f"[{i}] {doc['title']}: {content}")
        
        return "\n\n".join(context_parts)
    
    # Initialize and run optimizer
    optimizer = PromptOptimizer(api_key=api_key)
    
    print("Starting prompt optimization for RAG system...")
    
    best_prompt = optimizer.optimize_prompts(
        questions_df=eval_df,
        context_retrieval_func=get_rag_context,
        n_iterations=3,
        questions_per_iteration=15
    )
    
    # Save and report
    optimizer.save_results("rag_prompt_optimization.json")
    report = optimizer.generate_report()
    
    with open("optimization_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n✓ Optimization complete!")
    print(f"✓ Best prompt: {best_prompt.name} ({best_prompt.score:.2%} accuracy)")
    print(f"✓ Results saved to: rag_prompt_optimization.json")
    print(f"✓ Report saved to: optimization_report.txt")
    
    return best_prompt


if __name__ == "__main__":
    # Run example
    best_prompt = example_usage()
    
    print("\n" + "="*80)
    print("BEST PROMPT TO USE IN YOUR RAG SYSTEM:")
    print("="*80)
    print(best_prompt.template)
    print("="*80)
