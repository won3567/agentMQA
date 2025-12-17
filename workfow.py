

import os
import json
import time
import openrouter_client
import llama_local
import re

################### Load QA dataset #################
# benchmark = json.load(open("benchmark/MIRAGE.json"))
# # QA datasets: ['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu']
# dataset_name = 'medqa'
# QA_dataset = benchmark[dataset_name]

base = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base, "QAmedqa.json")
QA_dataset = json.load(open(path))




################### Save to JSON #################
def save_to_json(answers, correct_rates, time_spent, custom_name):
    output = {
        "time_spent": time_spent,
        "correct_rate": correct_rates,
        "result": answers
    }
    filename = f"a/{custom_name}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    return



################### Advanced QA #################
def generate_hint(smart_client):
    # chatGPT_answer = json.load(open(os.path.join(base, "Gpt-4.1-mini.json")))
    # chatGPT_answer = chatGPT_answer["result"]
    # Llama_answer = json.load(open(os.path.join(base, "Llama-3.1-8B.json")))
    # Llama_answer = Llama_answer["result"]

    err_ids = json.load(open(os.path.join(base, "err_ids.json")))
    err_ids = err_ids["result"]
    threshold=0
    answers = {}
    timestamp = time.time()
    for qa_id in err_ids:
        # if Llama answer is wrong and chatGPT answer is correct
        q = f'question: {QA_dataset[qa_id]["question"]} \noptions: {QA_dataset[qa_id]["options"]},'
        hint = smart_client.give_hint(q)
        answers[qa_id] = {
            "hint": hint
        }
        time_spent = round(time.time() - timestamp, 2)
        save_to_json(answers, 0, time_spent, "gemini3_hints")
    return


################### QA #################
model_choice = "openai/gpt-4.1-mini"
model_choice = "google/gemini-3-pro-preview"
client = openrouter_client.OpenRouterClient(model=model_choice)
# client = llama_local.LlamaLocal()


generate_hint(client)

count=0
correct=0
threshold=0
times=[]
correct_rates =[]
answers = {}
timestamp = time.time()
for qa_id, qa in QA_dataset.items():
    if count > threshold:
        break

    q = f'question: {qa["question"]} \noptions: {qa["options"]},'
    a = qa['answer']

    llm_a = client.answer_question(q)
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
    
    time_spent = round(time.time() - timestamp, 2)
    save_to_json(answers, correct_rates, time_spent, "Llama-3.1-8B")






