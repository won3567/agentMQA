import os
import json
import time
# import llama_local
import openrouter_client
import re
from rag import MedQARAGSystem


# Augment the question with the chatGPT reasoning
def augment_qa(augment_method):
    # client = llama_local.LlamaLocal()
    client = openrouter_client.OpenRouterClient(model="meta-llama/llama-3.1-8b-instruct")

    QA_dataset = json.load(open("gold/QAmedqa.json"))
    chatGPT_answer = json.load(open("gold/Gpt-4.1-mini.json"))
    chatGPT_answer = chatGPT_answer["result"]
    Llama_answer = json.load(open("gold/Llama-3.1-8B.json"))
    Llama_answer = Llama_answer["result"]
    chains = json.load(open("gold/chain.json"))["result"]
    # rag = MedQARAGSystem(
    #     corpus_path="textbooks/data-00000-of-00001.arrow",
    #     tokenizer_path="models--BAAI--bge-m3", 
    #     llm_model="meta-llama/llama-3.1-8b-instruct"
    # )

    count=0
    correct=0
    threshold=10000
    correct_rates = []
    params = {}
    answers = {}
    timestamp = time.time()
    for qa_id, ra in chains.items():
        if count > threshold:
            break

        # # # if Llama answer is wrong and chatGPT answer is correct
        # if ra["answer"] != QA_dataset[qa_id]['answer'] and chatGPT_answer[qa_id]["answer"] == QA_dataset[qa_id]['answer']:
        q = f'question: {QA_dataset[qa_id]["question"]} \noptions: {QA_dataset[qa_id]["options"]},'
        a = QA_dataset[qa_id]["answer"]

        # augment method
        if augment_method == "no_augment":
            if count == 0:
                params["temperature"] = 0.8
                client = llama_local.LlamaLocal(temperature=params["temperature"])
            llm_a = client.answer_question(q)
        elif augment_method == "by_hint":
            hint = json.load(open("/workspace/answers/gemini3_hints.json"))["result"][qa_id]["hint"]
            llm_a = client.answer_question(q, context=hint)
        elif augment_method == "prompt":
            context = chatGPT_answer[qa_id]["reasoning"]
            llm_a = client.answer_question(q, context=context)
        elif augment_method == "half_reasoning":
            context = chatGPT_answer[qa_id]["reasoning"][:int(len(chatGPT_answer[qa_id]["reasoning"])/2)]
            llm_a = client.answer_question_half_reasoning(q, reasoning=context)
        elif augment_method == "rag":
            llm_a = rag.answer_question(question=q,
                                        top_k=5,
                                        alpha=0.1,
                                        verbose=False)["answer"]
        elif augment_method == "by_chain":
            llm_a = client.answer_question_with_inference(q, inference=ra["inference_chain"])

        count += 1

        reasoning = re.search(r'"step_by_step_thinking"\s*:\s*"([^"]*)"', llm_a)
        answer = re.search(r'"answer_choice"\s*:\s*"([^"]*)"', llm_a)
        answers[qa_id] = {
            "reasoning": reasoning.group(1) if reasoning else None,
            "answer": answer.group(1) if answer else None
        }
        if answers[qa_id]["answer"] == a:
            correct += 1
            correct_rates.append(round(correct/count, 5))
        else:
            correct_rates.append(round(correct/count, 5))
        
        # Save result to JSON
        time_spent = round(time.time() - timestamp, 2)
        params["time_spent"] = time_spent
        params["number_of_questions"] = count
        output = {
            "param": params,
            "correct_rate": correct_rates,
            "result": answers
        }
        filename = f"answers/{augment_method}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)



# augment_method: ["by_hint", "prompt", "half_reasoning", "no_augment", "rag"]
augment_qa(augment_method="by_chain")

