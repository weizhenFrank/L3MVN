import retry
import re
import numpy as np
from enum import Enum
import torch
import base64
import os
import json
from openai import OpenAI
import requests
import concurrent.futures
import time
from agents.utils import semantic_prediction

LLM_SYSTEM_PROMPT_POSITIVE = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of scene descriptions numbered 1 to N. 

Your job is to provide guidance about where we should explore next. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide reasoning along with a number identifying where we should explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following scenes while exploring a house:

1. Descriptions of scene 1
2. Descriptions of scene 2
3. Descriptions of scene 3

Where should I search next if I am looking for a knife?

Assistant:
Reasoning: <Explain reason of your choice>.
Answer: <your answer(s)>


Other considerations 


1. Provide reasoning for each cluster before giving the final answer.
2. Feel free to think multiple steps in advance; for example if one room is typically located near another then it is ok to use that information to provide higher scores in that direction.
"""


USER_EXAMPLE_1 = """You see the following clusters of objects:

1. door
2. sofa, plant
3. bed, plant, table

Question: Your goal is to find a toilet. Where should you go next?
"""

AGENT_EXAMPLE_1 = """Reasoning: a bathroom is usually attached to a bedroom so it is likely that if you explore a bedroom you will find a bathroom and thus find a toilet
Answer: 3
"""

USER_EXAMPLE_2 = """You see the following clusters of objects:

1. plant
2. bed, chair, dresser

Question: Your goal is to find a tv. Where should you go next?
"""

AGENT_EXAMPLE_2 = """Reasoning: The tv is not likely to be in a bedroom but a plant does not provide enough information.
Answer: 0
"""


V2_SYSTEM_PROMPT_NEGATIVE = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should not waste time exploring. For example if we are in a house and looking for a tv we should not waste time looking in the bathroom. It is your job to point this out. 

You should always provide reasoning along with a number identifying where we should not explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space after Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I avoid spending time searching if I am looking for a knife?

Assistant:
Reasoning: Cluster 1 contains items that are likely part of an entertainment room. Cluster 2 contains objects that are likely part of an office room and cluster 3 contains items likely found in a kitchen. A knife is not likely to be in an entertainment room or an office room so we should avoid searching those spaces.
Answer: 1,2


Other considerations 

1. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
2. Provide reasoning for each cluster before giving the final answer
"""

V2_SYSTEM_PROMPT_POSITIVE = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should explore next. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide reasoning along with a number identifying where we should explore. If there are multiple right answers you should separate them with commas. Always include Reasoning: <your reasoning> and Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I search next if I am looking for a knife?

Assistant:
Reasoning: Cluster 1 contains items that are likely part of an entertainment room. Cluster 2 contains objects that are likely part of an office room and cluster 3 contains items likely found in a kitchen. Because we are looking for a knife which is typically located in a ktichen we should check cluster 3.
Answer: 3


Other considerations 

1. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
2. Provide reasoning for each cluster before giving the final answer.
3. Feel free to think multiple steps in advance; for example if one room is typically located near another then it is ok to use that information to provide higher scores in that direction.
"""

V2_SYSTEM_PROMPT_NEGATIVE_NO_REASONING = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should not waste time exploring. For example if we are in a house and looking for a tv we should not waste time looking in the bathroom. It is your job to point this out. 

You should always provide a number identifying where we should not explore. If there are multiple right answers you should separate them with commas. Always include Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I avoid spending time searching if I am looking for a knife?

Assistant:
Answer: 1,2


Other considerations 

1. Disregard the frequency of the objects listed on each line. If there are multiple of the same item in  a cluster it will only be listed once in that cluster.
2. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
"""

V2_SYSTEM_PROMPT_POSITIVE_NO_REASONING = """You are a robot exploring an environment for the first time. You will be given an object to look for and should provide guidance of where to explore based on a series of observations. Observations will be given as a list of object clusters numbered 1 to N. 

Your job is to provide guidance about where we should explore next. For example if we are in a house and looking for a tv we should explore areas that typically have tv's such as bedrooms and living rooms.

You should always provide a number identifying where we should explore. If there are multiple right answers you should separate them with commas. Always include Answer: <your answer(s)>. If there are no suitable answers leave the space afters Answer: blank.

Example

User:
I observe the following clusters of objects while exploring a house:

1. sofa, tv, speaker
2. desk, chair, computer
3. sink, microwave, refrigerator

Where should I search next if I am looking for a knife?

Answer: 3


Other considerations 

1. Disregard the frequency of the objects listed on each line. If there are multiple of the same item in  a cluster it will only be listed once in that cluster.
2. You will only be given a list of common items found in the environment. You will not be given room labels. Use your best judgment when determining what room a cluster of objects is likely to be in.
"""

def object_query_constructor(objects):
    """
    Construct a query string based on a list of objects

    Args:
        objects: torch.tensor of object indices contained in a room

    Returns:
        str query describing the room, eg "This is a room containing
            toilets and sinks."
    """
    assert len(objects) > 0
    # query_str = "This room contains "
    query_str = "You see "
    names = []
    for ob in objects:
        names.append(ob)
    if len(names) == 1:
        query_str += names[0]
    elif len(names) == 2:
        query_str += names[0] + " and " + names[1]
    else:
        for name in names[:-1]:
            query_str += name + ", "
        query_str += "and " + names[-1]
    query_str += "."
    return query_str

def find_first_integer(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    else:
        raise ValueError('No integer found in string')

@retry.retry(tries=3, delay=1)
def get_completion(prompt, max_tokens=100, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0, engine="davinci", echo=True):
    client = OpenAI()
    # Save the prompt to a text file in tmp, give the file a random name
    # with open(f"tmp/prompt_{random.random()}.txt", "w+") as f:
    #      f.write(str(prompt))

    response = client.chat.completions.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=5,
        echo=echo)
    return response

@retry.retry(tries=5)
def ask_gpt(goal, object_clusters):
    client = OpenAI()
    system_message = "You are a robot exploring a house. You have access to semantic sensors that can detect objects. You are in the middle of the house with clusters of objects. Your goal is to figure near which cluster to explore next. Always provide reasoning and if there is no clear choice select answer 0" 
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": USER_EXAMPLE_1},
        {"role": "assistant", "content": AGENT_EXAMPLE_1},
        {"role": "user", "content": USER_EXAMPLE_2},
        {"role": "assistant", "content": AGENT_EXAMPLE_2}
    ]
    if len(object_clusters) > 0:
        options = ""
        for i, cluster in enumerate(object_clusters):
            cluser_string = ""
            for ob in cluster:
                cluser_string += ob + ", "
            options += f"{i+1}. {cluser_string}\n"
        messages.append({"role": "user", "content": f"You see the following clusters of objects:\n\n {options}\nQuestion: You goal is to find a {goal}. Where should you go next? If there is not clear choice select answer 0.\n"})
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=messages)
        complete_response = completion.choices[0].message["content"]
        # Make the response all lowercase
        complete_response = complete_response.lower()
        reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
        # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
        answer = int(find_first_integer(complete_response.split("answer")[1]))
        return answer, reasoning
    raise Exception("Object categories must be non-empty")

@retry.retry(tries=5)
def ask_gpts(goal, object_clusters, num_samples=10, model="gpt-4-0125-preview"):
    client = OpenAI()
    system_message = "You are a robot exploring a house. You have access to semantic sensors that can detect objects. You are in the middle of the house with clusters of objects. Your goal is to figure near which cluster to explore next. Always provide reasoning and if there is no clear choice select answer 0" 
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": USER_EXAMPLE_1},
        {"role": "assistant", "content": AGENT_EXAMPLE_1},
        {"role": "user", "content": USER_EXAMPLE_2},
        {"role": "assistant", "content": AGENT_EXAMPLE_2}
    ]
    if len(object_clusters) > 0:
        options = ""
        for i, cluster in enumerate(object_clusters):
            cluser_string = ""
            for ob in cluster:
                cluser_string += ob + ", "
            options += f"{i+1}. {cluser_string}\n"
        messages.append({"role": "user", "content": f"You see the following clusters of objects:\n\n {options}\nQuestion: You goal is to find a {goal}. Where should you go next? If there is not clear choice select answer 0.\n"})
        completion = client.chat.completions.create(
            model=model, temperature=1,
            n=num_samples, messages=messages)
        
        answers = []
        reasonings = []
        for choice in completion.choices:
            try:
                complete_response = choice.message["content"]
                # Make the response all lowercase
                complete_response = complete_response.lower()
                reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
                # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
                answer = int(find_first_integer(complete_response.split("answer")[1]))
                answers.append(answer)
                reasonings.append(reasoning)
            except:
                continue

        unique_answers = list(set(answers))
        # It is possible GPT gives an invalid answer less than zero or greater than 1 plus the number of object clusters. Remove invalid answers
        unique_answers = [x for x in unique_answers if x >= 0 and x <= len(object_clusters)]
        answers = [x for x in answers if x >= 0 and x <= len(object_clusters)]
        # Aggregate into counts and normalize to probabilities
        answer_counts = {x: answers.count(x) / len(answers) for x in unique_answers}
        return answer_counts, reasonings
    raise Exception("Object categories must be non-empty")

@retry.retry(tries=5)
def ask_gpts_v2(goal, object_clusters, env="a house", positives=True, num_samples=5, model="gpt-4-0125-preview", reasoning_enabled=True):
    client = OpenAI()
    if reasoning_enabled:
        if positives:
            system_message = V2_SYSTEM_PROMPT_POSITIVE
        else:
            system_message = V2_SYSTEM_PROMPT_NEGATIVE
    else:
        if positives:
            system_message = V2_SYSTEM_PROMPT_POSITIVE_NO_REASONING
        else:
            system_message = V2_SYSTEM_PROMPT_NEGATIVE_NO_REASONING


    messages=[
        {"role": "system", "content": system_message},
    ]
    if len(object_clusters) > 0:
        options = ""
        for i, cluster in enumerate(object_clusters):
            cluser_string = ""
            for ob in cluster:
                cluser_string += ob + ", "
            options += f"{i+1}. {cluser_string[:-2]}\n"
        if positives:
            messages.append({"role": "user", "content": f"I observe the following clusters of objects while exploring {env}:\n\n {options}\nWhere should I search next if I am looking for {goal}?"})
            print(f"I observe the following objects while exploring {env}:\n\n {options}\nWhere should I search next if I am looking for {goal}?")

        else:
            messages.append({"role": "user", "content": f"I observe the following clusters of objects while exploring {env}:\n\n {options}\nWhere should I avoid spending time searching if I am looking for {goal}?"})

        completion = client.chat.completions.create(
            model=model, temperature=1,
            n=num_samples, messages=messages)
        
        answers = []
        reasonings = []
        for choice in completion.choices:
            try:
                complete_response = choice.message["content"]
                # Make the response all lowercase
                complete_response = complete_response.lower()
                if reasoning_enabled:
                    reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
                else:
                    reasoning = "disabled"
                # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
                if len(complete_response.split("answer:")) > 1:
                    answer = complete_response.split("answer:")[1].split("\n")[0]
                    # Separate the answers by commas
                    answers.append([int(x) for x in answer.split(",")])
                else:
                    answers.append([])
                reasonings.append(reasoning)
            except:
                answers.append([])

        # Flatten answers
        flattened_answers = [item for sublist in answers for item in sublist]
        # It is possible GPT gives an invalid answer less than 1 or greater than 1 plus the number of object clusters. Remove invalid answers
        filtered_flattened_answers = [x for x in flattened_answers if x >= 1 and x <= len(object_clusters)]
        # Aggregate into counts and normalize to probabilities
        answer_counts = {x: filtered_flattened_answers.count(x) / len(answers) for x in set(filtered_flattened_answers)}

        return answer_counts, reasonings
    raise Exception("Object categories must be non-empty")


@retry.retry(tries=5)
def greedy_ask_gpt(goal, object_clusters, model="gpt-4-0125-preview", env="a house"):
    client = OpenAI()
    if len(object_clusters) > 0:
        options = ""
        for i, cluster in enumerate(object_clusters):
            cluser_string = ""
            for ob in cluster:
                cluser_string += ob + ", "
            options += f"{i+1}. {cluser_string[:-2]}\n"
        messages = [{"role": "system", "content": "You are a robot exploring and environment. You have access to semantic sensors that can detect objects.Your goal is to figure near which cluster to explore next. You should pick one from the list and answer by providing the number of the cluster. If there is not clear choice select answer 0.You must provide reasoning before providing your answer. The response format must follow:\n\nreasoning: <your reasoning here>\nanswer: <your answer here, number only>"}]
        messages.append({"role": "user", "content": f"I observe the following clusters of objects while exploring {env}:\n\n {options}\nWhere should I search next if I am looking for {goal}?"})    
        completion = client.chat.completions.create(
        model=model, temperature=1,
        n=1, messages=messages)
        complete_response = completion.choices[0].message["content"]
        reasoning = complete_response.split("reasoning:")[1].split("\n")[0]
        answer = int(complete_response.split("answer:")[1].split("\n")[0])
        return reasoning, answer
    raise Exception("Object categories must be non-empty")

EXAMPLE = """You are a robot exploring a house. You have access to semantic sensors that can detect objects. You are in the middle of the house with clusters of objects. Your goal is to figure near which cluster to explore next. If there is not clear choice select answer 0.

You see the following clusters of objects:

0. no clear option
1. shower
2. sofa, plant, TV
3. bed, plant, table

Question: You goal is to find a toilet. Where should you go next? 
Answer: 1"""

def score_clusters(goal, object_clusters):
    preface = "\n\nYou see the following clusters of objects:\n\n 0. no clear option\n"
    options = ""
    for i, cluster in enumerate(object_clusters):
        cluser_string = ""
        for ob in cluster:
            cluser_string += ob + ", "
        options += f"{i+1}. {cluser_string}\n"
    question = f"Question: Your goal is to find a {goal}. Where should you go next?\nAnswer:"
    prompt = EXAMPLE + preface + options + "\n" + question
    res = get_completion(prompt, max_tokens=1, temperature=0, engine="davinci", echo=False)
    choice = res["choices"][0]
    logprob = choice["logprobs"]["token_logprobs"][0]

    # Compute the probability by getting the logprob scores for each other option 0, 1, 2, 3, ... and normalizing
    top_logprobs = choice["logprobs"]["top_logprobs"][0]
    total_probability_mass = 0
    for key, value in top_logprobs.items():
        # If we can cast the key to an int, do it and check the value
        try:
            key = int(key)
            if 0 <= key < len(object_clusters):
                total_probability_mass += np.exp(value)
        except:
            continue
    print("total_probability_mass", total_probability_mass)
    print("logprob", logprob)
    prob = np.exp(logprob) / total_probability_mass
    return int(choice["text"]), logprob, prob

class LanguageMethod(Enum):
    SINGLE_SAMPLE = 0
    SAMPLING = 1
    SAMPLING_POSTIIVE = 2
    SAMPLING_NEGATIVE = 3
    GREEDY = 4
    VISION_DES = 5

def query_llm(method: LanguageMethod, object_clusters: list, goal: str, reasoning_enabled: bool = True, model="gpt-4-0125-preview", out_path=None, item_mode=False) -> list:
    """
    Query the LLM fore a score and a selected goal. Returns a list of language scores for each target point
    method = SINGLE_SAMPLE uses the naive single sample LLM and binary scores of 0 or 1
    method = SAMPLING uses the sampling based approach and gives scores between 0 and 1
    method = SAMPLING_POSTIIVE uses the sampling based approach and gives scores between 0 and 1 if the agent should explore a cluster
    method = SAMPLING_NEGATIVE uses the sampling based approach and gives scores between 0 and 1 if the agent should not explore a cluster
    method = GREEDY gives etiher 0 or 10000 (a large positive score) that tells the agent to go to a particular cluster
    method = VISION_DES uses the vision based LLM to give scores between 0 and 1
    """
    if method == LanguageMethod.VISION_DES:
        response_lists = combine_response(object_clusters, item_mode)
        
        import time
        start_time = time.time()
        reconstructed = reconstruct_response(response_lists, item_mode=item_mode)
        print("Time to reconstruct:", time.time() - start_time)
        try:
            answer_counts, reasoning = llm_nav(goal, reconstructed, positives=True, reasoning_enabled=reasoning_enabled)
        except Exception as excptn:
            answer_counts, reasoning = {}, "LLMs failed"
            print("GPTs failed:", excptn)
        language_scores = [0] * len(object_clusters)
        # import pdb; pdb.set_trace()
        for key, value in answer_counts.items():
            if key >= 1 and key <= len(object_clusters):
                language_scores[key-1] = value
    else:
        # Convert object clusters to a tuple of tuples so we can hash it and get unique elements
        object_clusters_tuple = [tuple(x) for x in object_clusters]
        # Remove empty clusters and duplicate clusters
        query = list(set(tuple(object_clusters_tuple)) - set({tuple([])}))

        if method == LanguageMethod.SINGLE_SAMPLE:
                try:
                    goal_id, reasoning = ask_gpt(goal, query)
                except Exception as excptn:
                    goal_id, reasoning = 0, "GPT failed"
                    print("GPT failed:", excptn)
                if goal_id != 0:
                    goal_id = np.argmax([1 if x == query[goal_id - 1] else 0 for x in object_clusters_tuple]) + 1
                language_scores = [0] * (len(object_clusters_tuple) + 1)
                language_scores[goal_id] = 1
        elif method == LanguageMethod.SAMPLING:
            try: 
                answer_counts, reasoning = ask_gpts(goal, query)
            except Exception as excptn:
                answer_counts, reasoning = {}, "GPTs failed"
                print("GPTs failed:", excptn)
            language_scores = [0] * (len(object_clusters_tuple) + 1)
            for key, value in answer_counts.items():
                if key != 0:
                    for i, x in enumerate(object_clusters_tuple):
                        if x == query[key - 1]:
                            language_scores[i + 1] = value
                else:
                    language_scores[0] = value
        elif method == LanguageMethod.SAMPLING_POSTIIVE:
            try:
                answer_counts, reasoning = ask_gpts_v2(goal, query, positives=True, reasoning_enabled=reasoning_enabled, model=model)
            except Exception as excptn:
                answer_counts, reasoning = {}, "GPTs failed"
                print("GPTs failed:", excptn)
            language_scores = [0] * len(object_clusters_tuple)
            for key, value in answer_counts.items():
                for i, x in enumerate(object_clusters_tuple):
                    if x == query[key - 1]:
                        language_scores[i] = value

        elif method == LanguageMethod.SAMPLING_NEGATIVE:
            try:
                answer_counts, reasoning = ask_gpts_v2(goal, query, positives=False,  reasoning_enabled=reasoning_enabled, model=model)
            except Exception as excptn:
                answer_counts, reasoning = {}, "GPTs failed"
                print("GPTs failed:", excptn)
            language_scores = [0] * len(object_clusters_tuple)
            for key, value in answer_counts.items():
                for i, x in enumerate(object_clusters_tuple):
                    if x == query[key - 1]:
                        language_scores[i] = value

        elif method == LanguageMethod.GREEDY:
            # This is the language greedy method. We simply ask an LLM where we should go next and directly follow that
            language_scores = [0] * len(object_clusters_tuple)
            reasoning, answer = greedy_ask_gpt(goal, query)
            if answer-1 >= len(query):
                answer = 0
            for i, x in enumerate(object_clusters_tuple):
                if x == query[answer-1]:
                    language_scores[i] = 1000

        else:
            raise Exception("Invalid method")
    
    # The first element of language scores is the scores for uncertain, the last n-1 correspond to the semantic scores for each point
    return language_scores, reasoning

def aggregate_reasoning(reasoning: list):
    # Pass in a list of reasoning strings and aggregate them into a single string
    # Ask GPT to aggregate the reasoning into a single consensus

    # Construct the prompt
    client = OpenAI()
    system_prompt = "You are given a series of explanations regarding where to navigate in order to find an object. You should aggregate the reasoning from multiple agents into a single sentence"
    prompt = ""
    for i, r in enumerate(reasoning):
        prompt += f"Reasoning {i}: {r}\n"

    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=messages)
    complete_response = completion.choices[0].message["content"]
    return complete_response



def score_func(sampling_mathod, frontiers, goal, reasoning_enabled=False, model=None, wp=1, wn=0.5):
    
    if sampling_mathod == 'positive':
        scores, reasoning = query_llm(LanguageMethod.SAMPLING_POSTIIVE, frontiers, goal, reasoning_enabled=reasoning_enabled, model=model)
        return scores, reasoning
    elif sampling_mathod == 'negative':
        scores, reasoning = query_llm(LanguageMethod.SAMPLING_NEGATIVE, frontiers, goal, reasoning_enabled=reasoning_enabled, model=model)
        return [-score for score in scores], reasoning
    elif sampling_mathod == 'greedy':
        scores, reasoning = query_llm(LanguageMethod.GREEDY, frontiers, goal, reasoning_enabled=reasoning_enabled, model=model)
        return scores, reasoning
    elif sampling_mathod == 'single_sample':
        scores, reasoning = query_llm(LanguageMethod.SINGLE_SAMPLE, frontiers, goal, reasoning_enabled=reasoning_enabled, model=model)
        return scores, reasoning
    elif sampling_mathod == 'pn':
        scores, reasoning = query_llm(LanguageMethod.SAMPLING_POSTIIVE, frontiers, goal, reasoning_enabled=reasoning_enabled, model=model)
        n_scores, reasoning = query_llm(LanguageMethod.SAMPLING_NEGATIVE, frontiers, goal, reasoning_enabled=reasoning_enabled, model=model)
        return [wp*scores[i] - wn*n_scores[i] for i in range(len(scores))], reasoning
    else:
        raise Exception("Invalid sampling method")
    
def extract_info_from_key(key, args):
    import os
    import json
    process_rank = key // 10000000
    episode_number = (key % 10000000) // 1000
    timestep = key % 1000
    
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)
    ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(dump_dir, process_rank, episode_number)
    filename = f'{ep_dir}{process_rank}-{episode_number}-Obs-{timestep}.json'
    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            json_data = file.read()
            data = json.loads(json_data)
            choices = data['response']['choices']
            if choices:
                return choices[0]['message']['content']
            else:
                return "No response"
    return "No response"
        
        
@retry.retry(tries=5)
def vision_nav(key_list, goal='toilet', args=None, num_samples=1, model="gpt-4-vision-preview", detail="low", downsampling=8):
    client = OpenAI()
    img_list = decode_img_list(key_list, args)
    if downsampling:
        img_list = sample_images(img_list, downsampling)
    messages=[
        {"role": "system", "content": "You are a robotic home assistant that can find one object. There's several frontiers region in the house you can explore. Frontier index starts from 0. For each frontier, you have observation as images."}
    ]

    user_message =  {
        "role": "user",
        "content": []
        }
    user_message['content'] += create_content(img_list, detail, goal)
    messages.append(user_message)

    completion = client.chat.completions.create(
            model=model, temperature=1,
            n=num_samples, messages=messages, max_tokens=300)
    ans = 0
    if img_list[0][0]:
        
        json_file_path = os.path.splitext(img_list[0][0])[0] + '.json'

    # Save the response to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump({"response": completion}, json_file, indent=4)
        
    try:
        complete_response = completion["choices"][0]["message"]["content"]
        # Make the response all lowercase
        ans = complete_response.lower().split("answer:")[1].split("\n")[0]
        try:
            return int(ans)
        except:
            raise Exception("Invalid answer")
    except:
        return ans


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def create_content(image_list, detail, goal):
    # Getting the base64 string
    contents = []
    for ind, frontier in enumerate(image_list):
        text_promot = {
            "type": "text",
            "text": f"For frontier {ind}, you observed following images"
            }
        contents.append(text_promot)
        for i in frontier:
            base64_image = encode_image(i)
            # base64_image = ""
            sub_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": f"{detail}"
                    }
                    }
            contents.append(sub_content)
    query = {
        "type": "text",
        "text": f"""For the above frontiers, the goal object is {goal}, which frontier should be explored next?\n
         The response MUST follow the format:\n"
         Answer:<your answer here, index of frontier only>"""
        }

    
    contents.append(query)
    return contents

def decode_img_list(frontiers, args):
    img_list = []
    for frontier_keys in frontiers:
        frontier = []
        for key in frontier_keys:
            process_rank = key // 10000000
            episode_number = (key % 10000000) // 1000
            timestep = key % 1000

            ep_dir = f"{args.dump_location}/dump/{args.exp_name}/episodes/thread_{process_rank}/eps_{episode_number}/"
            filename = f'{ep_dir}{process_rank}-{episode_number}-Obs-{timestep}.png'
            frontier.append(filename)
        img_list.append(frontier)
    return img_list

    
def sample_images(image_paths, max_samples=5):
    sampled_images = []
    for path_list in image_paths:
        if len(path_list) <= max_samples:
            sampled_images.append(path_list)
        else:
            interval = len(path_list) / float(max_samples)
            indices = [int(round(i * interval)) for i in range(max_samples)]
            sampled_images.append([path_list[i] for i in indices])
    return sampled_images

@retry.retry(tries=5)
def ask_vision(num_samples=1, model="gpt-4-vision-preview", image_path="obs.jpg", detail="low"):
    import base64
    import requests
    import os
    import json
    client = OpenAI()
    messages=[
        {"role": "system", "content": "You are a robotic home assistant that can help people find objects. You have an observation of the house as images."},
    ]
    # messages = []
    # Getting the base64 string
    base64_image = encode_image(image_path)

    user_message =  {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "So desbribe the image that can provide useful information for object navigation. The description should be informative but concise"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": f"{detail}"
            }
            }
        ]
        }
    
    messages.append(user_message)
    import time
    now = time.time()
    completion = client.chat.completions.create(
            model=model, temperature=1,
            n=num_samples, messages=messages, max_tokens=300)
       
    # answers = []
    # for choice in completion.choices:
    #     try:
    #         complete_response = choice.message["content"]
    #         # Make the response all lowercase
    #         complete_response = complete_response.lower()
    #         answers.append(complete_response)
    #     except:
    #         answers.append([])
    # Define the JSON file path
    json_file_path = os.path.splitext(image_path)[0] + '.json'

    # Save the response to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump({"response": completion}, json_file, indent=4)
    # print(answers)
    # return answers


@retry.retry(tries=5)
def ask_llava(image_path, model="llava-v1.5-7b", item_mode=False, room_mode=False):
    if item_mode:
        system_prompt = """You are MaskRCNN. Please list the items you deteced. Only return the items. 
        EXAMPLE:
        Your answer should only contain the items in the image, the item name choose from:
        
        wall
        door
        ceiling
        floor
        picture
        window
        chair
        cushion
        table
        sofa
        bed
        chest_of_drawers
        cabinet
        plant
        sink
        stairs
        appliances
        toilet
        stool
        towel
        mirror
        tv_monitor
        shower
        column
        bathtub
        counter
        fireplace
        lighting
        beam
        railing
        shelving
        blinds
        gym_equipment
        seating
        board_panel
        furniture
        unlabeled
        clothes
        objects
        misc.
        
        Your answer will be Like: bathtub, shower, tv_monitor, furniture.Note: No sentence, just list of items.
        """
    elif room_mode:
        system_prompt = """You are room annotator. You have observation as image of indoor house. Please give the room label of the image. Only return the name of the room. 
        
        Your answer should only contain the name of the room in the image, the name MUST choose from:
        
        Foyer
        Mudroom
        Hallway
        Living Room
        Family Room
        Dining Room
        Kitchen
        Breakfast Nook
        Sunroom
        Home Office
        Library
        Game Room
        Home Theater
        Bedrooms
        Master Bedroom
        Guest Bedroom
        Nursery
        Bunk Room
        Master Bathroom
        Full Bathroom
        Half Bathroom (Powder Room)
        En Suite Bathroom
        Jack and Jill Bathroom
        Laundry Room
        Pantry
        Closet
        Basement
        Attic
        Patio
        Deck
        Balcony
        Porch
        Garden
        Garage
        Workshop
                
        Note: No sentence. Only return the name from above.
        """
    else:
        system_prompt = "You are home assistant robot and you are in a house. You have the obervation. Please describe the image you observed. The reason why you need to describe the image because I need to find a object. So your description should reflect the information that can help me find the object. The object I need to find is one of: chair, couch, potted plant, bed, toilet, and TV."
    base64_image = encode_image(image_path)
    
    if model == "llava-v1.5-7b":
        api_key = "ddd"
        # Controller endpoint
        base_url = "http://10.230.220.36:8000/api/v1"

        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        import time
        now = time.time()
        response = client.chat.completions.create(
        model="llava-v1.5-7b",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{system_prompt}"},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
                },
            ],
            }
        ],
        max_tokens=300,
        stream=False
        )

        # Extracting 'content' value from reponse
        try:
            content = response.choices[0].message.content
            print(f"detector Content for :{image_path}", content)
        # print(content)
        except:
            content = "No response"
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": model,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                }
            ]
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            content = response.json()['content'][0]['text']
        except Exception as e:
            print("Error in request:", e)
            content = "No response"
            
    json_file_path = os.path.splitext(image_path)[0] + '.json'

    # Save the response to a JSON file
    print("Time to get img caption response:", time.time() - now)
    print("Room label:", content)
    with open(json_file_path, 'w') as json_file:
        json.dump({"response": content}, json_file, indent=4)

def combine_response(img_list, item_mode, detect_func=None):
    response_list = []
    for clu in img_list:
        clu_response = []
        for img in clu:
            json_file_path = os.path.splitext(img)[0] + '.json'
            if not os.path.exists(json_file_path):
                if detect_func:
                    print("Detecting items in image")
                    detect_func(img)
                else:
                    ask_llava(img, item_mode=item_mode, room_mode=True)
            with open(json_file_path) as json_file:
                # Load the JSON data as a dictionary
                data = json.load(json_file)
                clu_response.append(data['response'])
        response_list.append(clu_response)
    return response_list


@retry.retry(tries=5)
def llm_nav(goal, description_clusters, env="a house", positives=True, num_samples=5, model="gpt-3.5-turbo-0125", reasoning_enabled=True, out_path=None):
    client = OpenAI()

    system_message = LLM_SYSTEM_PROMPT_POSITIVE

    messages=[
        {"role": "system", "content": system_message},
    ]
    
    if len(description_clusters) > 0:
        options = ""
        for i, cluster in enumerate(description_clusters):
            options += f"{i+1}. {cluster}\n"
            options += "----------------------------\n"
            
        messages.append({"role": "user", "content": f"""I observe the following clusters while exploring {env}:\n\n {options}\nWhere should I search next if I am looking for {goal}? \n The response MUST follow format:\n 
                        Reasoning: <Explain reason of your choice>.
                        Answer: <your answer(s), NUMBER ONLY >"""})
        
        # print(f"I observe the following objects while exploring {env}:\n\n {options}\nWhere should I search next if I am looking for {goal}?")


        completion = client.chat.completions.create(
            model=model, temperature=1,
            n=num_samples, messages=messages)
        
        answers = []
        reasonings = []
        for choice in completion.choices:
            try:
                complete_response = choice.message.content
                # Make the response all lowercase
                complete_response = complete_response.lower()
                if reasoning_enabled:
                    reasoning = complete_response.split("reasoning: ")[1].split("\n")[0]
                else:
                    reasoning = "disabled"
                # Parse out the first complete integer from the substring after  the text "Answer: ". use regex
                if len(complete_response.split("answer:")) > 1:
                    answer = complete_response.split("answer:")[1].split("\n")[0]
                    # Separate the answers by commas
                    answers.append([int(x) for x in answer.split(",")])
                else:
                    answers.append([])
                reasonings.append(reasoning)
            except:
                answers.append([])

        # Flatten answers
        flattened_answers = [item for sublist in answers for item in sublist]
        # It is possible GPT gives an invalid answer less than 1 or greater than 1 plus the number of object clusters. Remove invalid answers
        filtered_flattened_answers = [x for x in flattened_answers if x >= 1 and x <= len(description_clusters)]
        # Aggregate into counts and normalize to probabilities
        answer_counts = {x: filtered_flattened_answers.count(x) / len(answers) for x in set(filtered_flattened_answers)}

        return answer_counts, reasonings
    raise Exception("Cluster descriptions must be non-empty")



@retry.retry(tries=5)
def process_response_list(response_list, model, system_prompt, item_mode=False):
    if len(response_list) > 0:
        if item_mode:
            query = "Some items may be redundant or conflicting, but based on your reasoning capabilities, you should provide a list of items that are present in the scene."
        else:
            query = "Some of the descriptions may be redundant or conflicting, but based on your reasoning capabilities, you should provide a single description that is complete and accurate. Limit the description to 500 words."
        if 'claude' in model:
            messages = []
        else:
            messages = [
                {"role": "system", "content": system_prompt}
            ]
        prompt = ""
        for i, response in enumerate(response_list):
            prompt += f"For image {i}, your observation is:\n{response}\n\n"

        prompt += query
        user_message = {"role": "user", "content": prompt}
        messages.append(user_message)

        if 'gpt' in model:
            client = OpenAI()
            completion = client.chat.completions.create(
                model=model, temperature=1,
                messages=messages, max_tokens=800
            )
            result = completion.choices[0].message.content
            return result
        if 'claude' in model:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "messages": messages,
                "model": model,
                "max_tokens": 1024,
                "temperature": 0.5,
                "system": system_prompt,
            }
        else:
            url = "http://10.230.220.36:1337/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
            }
            payload = {
                "messages": messages,
                "model": model,
                "stream": False,
                "max_tokens": 1024,
                "temperature": 0.5,
            }
        try:
            if 'claude' in model:
                response = requests.post(url, headers=headers, json=payload)
            else:
                response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            if 'claude' in model:
                result = response.json()['content'][0]['text']
            else:
                result = response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            # Handle request exceptions (e.g., connection errors, timeouts)
            print(f"Error occurred during API request: {e}")
            result = "Error occurred during API request"
        except (KeyError, IndexError) as e:
            # Handle JSON parsing errors or missing keys
            print(f"Error occurred while parsing API response: {e}")
            result = "Error occurred while parsing API response"
        return result
    else:
        return "No response"

@retry.retry(tries=5)
def reconstruct_response(response_lists, model="mistral-ins-7b-q4", max_parallel_requests=3, delay=1, item_mode=False):
    if item_mode:
        system_prompt = "You are given a list of items that are present in the scene. You should aggregate them into a single list for the scene"
    else:
        system_prompt = "You are good at giving a holistic and complete description of multiple observations. You are given multiple descriptions of a region of an indoor house scene, and you should aggregate them into a single description. Such holistic and complete description should provide useful information for object navigation. For example: I need to find a bed, so your description should reflect the information that can help me find the bed. The list of things I need to find is chosen from: chair, couch, potted plant, bed, toilet, and TV."

    if 'claude' in model:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
            futures = []
            for response_list in response_lists:
                futures.append(executor.submit(process_response_list, response_list, model, system_prompt))
                time.sleep(delay)  # Add a delay between submitting requests
            clusters = [future.result() for future in concurrent.futures.as_completed(futures)]
    else:
        clusters = [process_response_list(response_list, model, system_prompt)
                    for response_list in response_lists]

    return clusters


    
# if __name__ == "__main__":
#     ask_llava("/mnt/L3MVN/tmp/dump/llava_nav/episodes/thread_0/eps_20/0-20-Obs-61.png", model="claude-3-haiku-20240307")