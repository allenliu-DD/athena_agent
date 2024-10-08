# Databricks notebook source
# MAGIC %pip install openai==0.28.1 aiohttp==3.9.3 tiktoken==0.5.2 sentence_transformers datasets

# COMMAND ----------

# MAGIC %run /Workspace/Users/tony.luo@doordash.com/voicebot/voice_prompt_actions/voice_prompts_summary_exp9&10&11

# COMMAND ----------

# MAGIC %run /Workspace/Users/tony.luo@doordash.com/voicebot/voice_prompt_actions/voice_prompts_choose_exp3&5&7

# COMMAND ----------

# import libraries
import openai
import pandas as pd
import re
import random
import numpy as np
import collections
import json
# Set up OpenAI API, load KB and Intent Map
openai_api_key = dbutils.secrets.get(scope="tonyluo-scope", key="openai-api")
openai.api_key = openai_api_key
intent_workflow_map = pd.read_json("/Workspace/Users/tony.luo@doordash.com/voicebot/production_intent_map_Sep9.json")
intent_list_all = list(intent_workflow_map.keys())
new_kb = pd.read_csv("/dbfs/FileStore/tonyluo/voice/production_voice_KB_Sep9.csv")

# COMMAND ----------

# define retrival model
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
# Load Sentence Transformers model
model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_id)
kb_embeddings = model.encode(new_kb['ai_instructions'].tolist(), batch_size=64, show_progress_bar=True, convert_to_tensor=False)
# retrive the top 3 similar documents
nn_model = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='cosine').fit(kb_embeddings)

# COMMAND ----------

out_criteria = {
    "action": "out",
    "criteria": "Choose this action if the message meets none of the criteria of the other actions."
}

def summarize_context_with_prompt(context):
    messages = []
    messages.append({"role": "system", "content": initial_system_prompt})
    messages.append({"role": "user", "content": initial_prompt.replace("{{CHAT_HISTORY}}", context)})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        functions=initial_functions,
        temperature=0,
        seed=1234,
        function_call="auto",
    )
    
    function_json = response['choices'][0]["message"]["function_call"]["arguments"]
    return_dict = json.loads(function_json)
    query = return_dict["query"]
    need_retrieval = return_dict["need_retrieval"]
    return query, need_retrieval
  

def choose_action_with_prompts(query, actions_criteria_str):
    messages = []
    messages.append({"role": "system", "content": choose_system_prompt})
    messages.append({
      "role": "user", 
      "content": choose_prompt
        .replace("{{USER_QUERY}}", query)
        .replace("{{ACTIONS_CRITERIA}}", actions_criteria_str)})

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        functions=functions,
        temperature=0,
        seed=1234,
        function_call="auto",
    )
    if 'function_call' in response['choices'][0]['message']:
      arguments = json.loads(response['choices'][0]['message']['function_call']['arguments'])
      return arguments['action'], arguments['reasoning_steps']
    else:
      return 'None'

# COMMAND ----------

# Enter user utterance test case here:
user_message = "how much I made last year?"

# Summary Phase
chat_history_str = "Assistant: You've reached the DoorDash Support Virtual Phone Concierge, please give me a brief description of what I can help you with\n User: "+user_message
query, need_retrieval = summarize_context_with_prompt(chat_history_str)
print(f"User Message: {user_message}")
print(f"Summary: {query}")
print(f"need_retrieval: {need_retrieval} \n")

# KB Retrieval Phase
embedding = model.encode([query])
distances, nn_idxs = nn_model.kneighbors(embedding, return_distance=True)
similarity = 1 - distances[0]
nn_idxs = nn_idxs[0]
threshold = 0.3
idx_list = []
for score, idx in zip(similarity, nn_idxs):
  if score >= threshold:
    idx_list.append(idx)
actions_list = new_kb['actions'].tolist()
ai_instructions_list = new_kb['ai_instructions'].tolist()
intent_list = [actions_list[idx] for idx in idx_list]
kb_sum_list = [ai_instructions_list[idx] for idx in idx_list]
print(f"KB Retrieval Intent: {intent_list}")
print(f"KB Retrieval Common Verbiage: {kb_sum_list}\n")
intent_summary_dict = collections.defaultdict(list)
for intent, summary in zip(intent_list, kb_sum_list):
  intent_summary_dict[intent].append(summary)
criteria_list_with_KB_example = []
for intent, summary_list in intent_summary_dict.items():
  criteria_list_with_KB_example.append({"action": intent,
                                        "criteria": intent_workflow_map[intent]["criteria"],
                                        "examples": summary_list+intent_workflow_map[intent]["examples"]
                                        })
criteria_list_with_KB_example.append(out_criteria)

# Choose Action Phase
action_chosen, rationale = choose_action_with_prompts(query, str(criteria_list_with_KB_example))
print(f'Action chosen: {action_chosen}')
print(f'Raionale: \n{rationale}')
