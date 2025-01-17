import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel
generated_concept_path = "./data_122/m2conceptbase/generated_descriptions/"
grounded_concepts_path = "./data_122/vscode_home/projects/grounded_concepts.json"
wukong_path = "./data_122/wukong_release/"

# load chatglm2
save_path = "./cache/"
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, cache_dir=save_path)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device="cuda", cache_dir=save_path)
model = model.eval()


def get_generated_concept_dict(load_from_file=False, stage=1, iter=2):
    if load_from_file:
        if stage == 0:
            file_name = "./data_122/vscode_home/projects/concept_description_dict_filtered.json"
        elif stage == 1:
            file_name = f"./data_122/vscode_home/projects/concept_description_dict_filtered_stage_1_iter{iter}.json"
        with open(file_name, 'r', encoding='utf-8') as f:
            concept_descrption_dict = json.load(f)
    else:
        concept_descrption_dict = {}
        concept_description_paths = os.listdir(generated_concept_path)
        for concept_description_path in tqdm(concept_description_paths, total=len(concept_description_paths)):
            filename = os.path.join(generated_concept_path, concept_description_path)
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                concept_descrption_dict.update(data)
    return concept_descrption_dict

def get_caption_and_concepts(file_index, filename_prefix="wukong_100m_"):
    caption_filepath = os.path.join(wukong_path, filename_prefix + str(file_index) + "_output.csv")
    concept_filepath = os.path.join(wukong_path, filename_prefix + str(file_index) + "_id_img_concepts.csv")
    print(caption_filepath)
    
    captions, concepts = [], []
    

    with open(caption_filepath, 'r', encoding='utf-8') as f:
        caption_lines = f.readlines()
        for line in caption_lines[1:]:
            try:
                items = line.strip().split('\t')
                if len(items) == 3:
                    idx, caption, image_name = items
                else:
                    idx, caption, image_name = items[0], ''.join(items[1:-1]), items[-1]
                    # print("line {} exception processed.".format(cnt))
                    print(caption)
                captions.append(caption)
            except Exception as e:
                continue
            
    with open(concept_filepath, 'r', encoding='utf-8') as f:
        concept_lines = f.readlines()
        for line in concept_lines[1:]:
            try:
                items = line.strip().split('\t')
                if len(items) == 3:
                    idx, image_name, concept = items
                else:
                    idx, image_name, concept = items[0], ''.join(items[1:-1]), items[-1]
                    # print("line {} exception processed.".format(cnt))
                    print(concept)
                concepts.append(concept)
            except Exception as e:
                continue
        
    return captions, concepts

def get_result(text, concept, description):
    # text = "一是持续加大\"双一流\"建设支持力度."
    # concept = "力度"
    # descrption = "力度是指在某个领域、行业、竞技或其他形式的竞争中,一个人或组织所表现出的强度、速度、灵活性、持久性、爆发力、掌控力等综合能力。"
    if len(description) > 500:
        return 0
    prompt = f"""
    上下文: “{text}”; 概念: “{concept}”;
    概念描述: “{description}”;
    你的任务是: 判断上下文中的概念的含义是否和概念描述矛盾，矛盾则输出0，不矛盾则输出1。
    """.format(text, concept, description)
    print(prompt)
    response, history = model.chat(tokenizer, prompt, history=[])
    print(response)
    if '0' in response:
        return 0 # 矛盾
    else:
        return 1 # 不矛盾

# stage 1: image context based hallucination eliminating
def check_grounded_and_generated_intersection(concept_descrption_dict):
    with open(grounded_concepts_path, 'r', encoding='utf-8') as f:
        grounded_concepts = json.load(f)
    generated_concepts = set(concept_descrption_dict.keys())
    intersection = set(grounded_concepts) & generated_concepts
    print(len(intersection))
    concept_descrption_dict_s1 = {c: concept_descrption_dict[c] for c in intersection}
    with open("./data_122/vscode_home/projects/concept_description_dict_filtered_stage_1.json", 'w', encoding='utf-8') as f:
        json.dump(concept_descrption_dict_s1, f, ensure_ascii=False, indent=4)
    print("saved.")
    return intersection

if __name__ == "__main__":
    
    print("loading concept_descrption_dict...")
    concept_descrption_dict = get_generated_concept_dict(load_from_file=True)
    print(len(concept_descrption_dict))
    print('done.')
    concept_description_score_dict = {}

    # stage 1: image context based
    # print("check grounded and generated intersection...")
    # intersection = check_grounded_and_generated_intersection(concept_descrption_dict)
    # print("checked.") # 87k
    # exit(1)
    # 
    # stage 2: text context based 
    # 8:2 9-16:3 /32
    # for file_idx in range(16, 32): # 3
    for file_idx in [3, 6, 7]: # 2
    # file_idx = 1
        print("processing file {}".format(file_idx))
        
        captions, concepts = get_caption_and_concepts(file_idx)
        # captions, concepts = captions[:2000], concepts[:2000]
        # half_len = len(captions) // 2
        # captions, concepts = captions[:half_len], concepts[:half_len]
        # captions, concepts = captions[half_len:], concepts[half_len:]
        for caption, concept in tqdm(zip(captions, concepts), total=len(captions)):
            # print(caption)
            # print(concept) # 这个是字符串
            # for c in eval(concept):
            #     print(c)
            for c in eval(concept):
                try:
                    description = concept_descrption_dict.get(c, None)
                    if description == None:
                        continue
                    
                    if c not in concept_description_score_dict:
                        concept_description_score_dict[c] = {"pos": 0, "neg": 0, "res": True}
                    
                    if len(description) > 500:
                        concept_description_score_dict[c]["res"] = False
                        continue
                    
                    result = get_result(caption, c, description)
                    if result == 0: # 矛盾
                        concept_description_score_dict[c]["neg"] += 1
                        # print(concept_description_score_dict[c])
                    else: # 不矛盾
                        concept_description_score_dict[c]["pos"] += 1
                        
                    print(concept_description_score_dict[c])
                except Exception as e:
                    print(e)
                    continue
        
        for k, v in concept_description_score_dict.items():
            if concept_description_score_dict[k]["neg"] > concept_description_score_dict[k]["pos"]:
                concept_description_score_dict[k]["res"] = False
        with open("concept_description_score_dict_f{}_0.json".format(file_idx), 'w', encoding='utf-8') as f:
            json.dump(concept_description_score_dict, f, ensure_ascii=False, indent=4)
        print("saved.")
