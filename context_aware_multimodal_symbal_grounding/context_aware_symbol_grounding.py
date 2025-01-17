import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from cn_clip.clip import load_from_name, available_models
from cn_clip.clip_raw import load_from_name as load_from_name_raw
import cn_clip.clip as clip
import cn_clip.clip_raw as raw_clip
import json
from pypinyin import pinyin, Style
from itertools import chain
import torch
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm

weighted_images_path = "./data_122/m2conceptbase/weighted_images/"
concept_descriptions_path = "./data_122/m2conceptbase/concept_descriptions/"
grounded_concepts_save_path = "./data_122/m2conceptbase/grounded_concepts/"
generated_concept_path = "./projects/m2conceptbase/concept_description_dict_filtered.json"
with open(generated_concept_path, 'r', encoding='utf-8') as f:
    generated_concept_description_dict = json.load(f)

def load_model_preprocess(load_from_name_func = "load_from_name_raw"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "./projects/cache/pretrained_models/" # 模型保存位置
    # ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
    if load_from_name_func == "load_from_name":
        print("here")
        model, preprocess = load_from_name("ViT-H-14", device=device, download_root=save_path) # 加载大模型和预处理函数
    else:
        model, preprocess = load_from_name_raw("ViT-H-14", device=device, download_root=save_path) 
    model.eval() # 将模型设置为测试模式
    return model, preprocess, device

# utils: 汉字转拼音
def to_pinyin(s):
    return ''.join(chain.from_iterable(pinyin(s, style=Style.TONE3)))

def is_basic_meaning(description):
    if "《" not in description and "》" not in description and "产品" not in description:
        return True
    else:
        return False

# 加载一个概念的候选描述
def load_concept_candidate_descriptions(concept_name):
    
    first_pinyin = to_pinyin(concept_name)[0].upper()
    concept_description_path = os.path.join(concept_descriptions_path, first_pinyin + "/" + concept_name + ".json")
    if not os.path.exists(concept_description_path):
        if concept_name in generated_concept_description_dict:
            return [generated_concept_description_dict[concept_name]]
        else:
            print("concept {} not exist.".format(concept_name))
            return None
    # print(concept_description_path)
    try:
        with open(concept_description_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
    except Exception as e:
        print(e)
        return None
    # 过滤非本义描述（启发式规则）
    candidate_descriptions = [description for description in data_dict['candidate_descriptions'] if is_basic_meaning(description)]
    if candidate_descriptions == [] and concept_name in generated_concept_description_dict:
        candidate_descriptions = [generated_concept_description_dict[concept_name]]
    return candidate_descriptions

# 加载一个概念的图片
def load_concept_image(concept_name):
    first_pinyin = to_pinyin(concept_name)[0].upper()
    concept_image_path = os.path.join(weighted_images_path, first_pinyin + "/" + concept_name + ".json")
    pass

# 基于图文对上下文的符号接地: 对于某一个概念
def context_aware_symbol_grounding(concept_name, concept_dir, preprocess, device):
    grounded_concept_dict = {}
    # 已经接地的概念直接跳过
    first_pinyin = to_pinyin(concept_name)[0].upper()    
    if os.path.exists(os.path.join(grounded_concepts_save_path, "{}/{}.json".format(first_pinyin, concept_name))):
        print("{} already grounded.".format(concept_name))
        return None
    
    for concept_image_name in os.listdir(concept_dir): # image_name
        image_path = os.path.join(concept_dir, concept_image_name)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        description_list = load_concept_candidate_descriptions(concept_name)
        if description_list == None:
            continue
        description_list.append("图文不符") # 提供模型否定的选项
        text = raw_clip.tokenize(description_list).to(device)
        with torch.no_grad(): # 
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            # print(text_features.size()) # [4, 512]
            # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
            image_features /= image_features.norm(dim=-1, keepdim=True) 
            text_features /= text_features.norm(dim=-1, keepdim=True)   
            # print(image_features.size()) # [1, 512] 
            # print(text_features.size()) # [4, 512] 
            # logits_per_image, logits_per_text = model(image, text) #clip
            logits_per_image, logits_per_text = model.get_similarity(image, text) # cnclip
            # print(logits_per_image.size()) # [1, 4]
            # print(logits_per_text.size()) # [4, 1]
            probs = logits_per_image.softmax(dim=-1)
            description_id = probs[0].topk(1)[1].item() # index
            concept_description = description_list[description_id] # concept_description
            matching_score = logits_per_image.cpu().detach().numpy()[0][description_id] # concept_description score

            if concept_description == "图文不符": # 匹配失败
                print("{}-{}匹配失败".format(concept_name, concept_image_name))
                pass
            else: # 匹配成功
                concept_id = concept_name + "_" + str(description_id)
                print("concept {} succeed.".format(concept_id))
                if concept_id not in grounded_concept_dict:
                    grounded_concept_dict[concept_id] = {}
                    grounded_concept_dict[concept_id]['concept_description'] = concept_description
                    grounded_concept_dict[concept_id]['concept_images'] = []
                    grounded_concept_dict[concept_id]['matching_scores'] = []
                    grounded_concept_dict[concept_id]['concept_images'].append(concept_image_name)
                    grounded_concept_dict[concept_id]['matching_scores'].append(str(matching_score))
                else:
                    grounded_concept_dict[concept_id]['concept_images'].append(concept_image_name)
                    grounded_concept_dict[concept_id]['matching_scores'].append(str(matching_score))
                
    # save
    first_path = os.path.join(grounded_concepts_save_path, first_pinyin) # 首字母文件夹路径
    # (1) 创建首字母文件夹
    if not os.path.exists(first_path): 
        os.makedirs(first_path)
    grounded_concept_filename = os.path.join(first_path, concept_name + ".json")
    
    if grounded_concept_dict != {}:
        with open(grounded_concept_filename, 'w', encoding='utf-8') as f:
            json.dump(grounded_concept_dict, f, ensure_ascii=False, indent=4)
    return grounded_concept_dict


if __name__ == "__main__":
    # res = load_concept_candidate_descriptions("暗魔")
    # print(res)
    print("model loading...")
    model, preprocess, device = load_model_preprocess()
    
    with open("./projects/m2conceptbase/utils/word_freq_train_data_230531.json", 'r', encoding='utf-8') as f:
        word_dict = json.load(f)

    # p = mp.Pool(10)

    letter_dir_list = sorted(os.listdir(weighted_images_path)) # A/B/C/...

    for letter_dir in letter_dir_list[16:]: # concept
        if letter_dir in ['A', 'B', 'C', 'D', 'E', 'F']:
            print('letter {} processed.'.format(letter_dir))
            continue
        else:
            print('processing letter {}...'.format(letter_dir))
        letter_dir_path = os.path.join(weighted_images_path, letter_dir)
        concept_name_list = os.listdir(letter_dir_path)
        for concept_name in tqdm(concept_name_list, total=len(concept_name_list)):
            # print(concept_name)
            if concept_name not in word_dict:
                continue
            concept_dir = os.path.join(letter_dir_path, concept_name)
            context_aware_symbol_grounding(concept_name, concept_dir, preprocess, device)
            # results =  p.map(context_aware_symbol_grounding, (concept_name, concept_dir, preprocess, device))
    
    print("Done!")