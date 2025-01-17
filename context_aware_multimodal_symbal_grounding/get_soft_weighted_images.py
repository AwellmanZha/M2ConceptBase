import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from cn_clip.clip import load_from_name, available_models
from cn_clip.clip_raw import load_from_name as load_from_name_raw
import cv2
import numpy as np
from PIL import Image
# import clip as en_clip
import cn_clip.clip as clip
import cn_clip.clip_raw as raw_clip
import torch
from itertools import chain
from pypinyin import pinyin, Style
from tqdm import tqdm
import multiprocessing
import json
import pandas as pd
# # import matplotlib.pyplot as plt
# from captum.attr import visualization

# 全局变量/设置为加载函数，以实现防止内存溢出
def load_model_preprocess(load_from_name_func = "load_from_name"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "./projects/cache/pretrained_models/" # 模型保存位置
    # ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
    if load_from_name_func == "load_from_name":
        model, preprocess = load_from_name("ViT-H-14", device=device, download_root=save_path) # 加载大模型和预处理函数
    else:
        model, preprocess = load_from_name_raw("ViT-H-14", device=device, download_root=save_path) 
    model.eval() # 将模型设置为测试模式
    return model, preprocess, device

# utils: 汉字转拼音
def to_pinyin(s):
    return ''.join(chain.from_iterable(pinyin(s, style=Style.TONE3)))

# 获得图像相关性得分
def interpret(image, texts, model, device, start_layer=-1, start_layer_text=-1):
    batch_size = texts.shape[0]
    # print("texts:", texts.shape)
    # images = image.repeat(batch_size, 1, 1, 1)
    # logits_per_image, logits_per_text = model(images, texts)
    logits_per_image, logits_per_text = model.get_similarity(image, texts)
    # probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    # print("probs:", probs)
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
        # calculate index of last layer
        start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens,
                  dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(
            one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    # text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
    # text_attn_blocks = list(dict(model.bert.encoder.named_children()).values())

    # if start_layer_text == -1:
    #   # calculate index of last layer
    #   start_layer_text = len(text_attn_blocks) - 1

    # num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    # R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    # R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    # for i, blk in enumerate(text_attn_blocks):
    #     if i < start_layer_text:
    #       continue
    #     grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
    #     cam = blk.attn_probs.detach()
    #     cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
    #     grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
    #     cam = grad * cam
    #     cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
    #     cam = cam.clamp(min=0).mean(dim=1)
    #     R_text = R_text + torch.bmm(cam, R_text)
    # text_relevance = R_text

    # return text_relevance, image_relevance
    return image_relevance

# 获得加权图像
def get_weighted_image(image_relevance, image):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        # heatmap = mask
        heatmap = np.uint8(255 * mask)
        # print("mask:", mask)
        heatmap = np.float32(heatmap) / 255
        # print("heatmap:", heatmap)
        heatmap = heatmap[:, :, np.newaxis]
        cam = heatmap + np.float32(img)
        # cam = heatmap * np.float32(img)
        cam = cam / np.max(cam)
        return cam

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=224, mode='bilinear')
    # print("R:", image_relevance.size())
    # print(image_relevance)
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / \
        (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    # print("vis:", vis)
    vis = np.uint8(255 * vis)
    # print(np.shape(vis))
    # print("vis:", np.shape(vis))
    weighted_img = Image.fromarray(vis) # Image对象
    # weighted_img = torch.Tensor(vis)
    # img_save_path = "./projects/m2conceptbase/"
    # weighted_img.save(os.path.join(img_save_path, '1' + '.jpg'))
    return weighted_img  # 返回的是Image对象



# from cn_clip.clip.bert_tokenizer import FullTokenizer as _Tokenizer
# _tokenizer = _Tokenizer()

# def show_heatmap_on_text(text, text_encoding, R_text):
#     CLS_idx = text_encoding.argmax(dim=-1)
#     R_text = R_text[CLS_idx, 1:CLS_idx]
#     text_scores = R_text / R_text.sum()
#     text_scores = text_scores.flatten()
#     print(text_scores)
#     text_tokens=_tokenizer.encode(text)
#     text_tokens_decoded=[_tokenizer.decode([a]) for a in text_tokens]
#     vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,text_tokens_decoded,1)]
#     visualization.visualize_text(vis_data_records)

# 测试
# save_path = "./projects/cache/pretrained_models/"
# example_path = "./projects/cache/datasets/wukong_test/wukong_test/00962474-0960.jpg"
# example_path = "./projects/cache/examples/00999456-0285.jpg" # 水果茶
# example_path = "./projects/cache/datasets/wukong_test/wukong_test/00001469-0017.jpg"
# example_path = "./projects/cache/examples/00000307-0442.jpg"  # 手表
# example_path = "./projects/cache/datasets/wukong_test/wukong_test/00998129-0343.jpg"
# example_path = "./projects/cache/datasets/wukong_test/wukong_test/00987549-0941.jpg"
# example_path = "./projects/Chinese-CLIP/" + "examples/pokemon.jpeg"
# model, preprocess = load_from_name("ViT-B-16", device=device, download_root=save_path)
# model, preprocess = load_from_name("ViT-L-14", device=device, download_root=save_path)
# model, preprocess = load_from_name("ViT-H-14", device=device, download_root=save_path)
# model.eval()
# image = preprocess(Image.open(example_path)).unsqueeze(0).to(device)
# texts = ["足球", "西瓜", "水果茶"]
# texts = "缤纷水果茶的美味好喝做法"
# texts = "一杯缤纷水果茶"
# texts = "花盆"
# texts = "一盘食物"
# texts = ["一款精致的手表"]
# texts = ["一款精致的手表", "一款手表"] # 这种方式输入有问题
# texts = "两个大转子"

# texts = "一张粉底液的照片"
# text = clip.tokenize(texts).to(device)
# text = clip.tokenize(["大理石", "纹理", "山水", "风景", "电视", "背景墙"]).to(device)
# text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

# R_text, R_image = interpret(model=model, image=image, texts=text, device=device)
# R_image = interpret(model=model, image=image, texts=text, device=device)
# print(R_image.shape)
# batch_size = text.shape[0]
# for i in range(batch_size):
#     # show_heatmap_on_text(texts[i], text[i], R_text[i])
#     get_image_relevance(R_image[i], image, orig_image=Image.open(example_path))
#     # plt.show()
# print("saved.")


# 函数功能：获取加权图像的类别和匹配得分
# 输入：加权图像，候选概念列表
# 输出：加权图像的类别和图像概念匹配得分
def get_weighted_image_category_and_matching_score(model, preprocess, device, weighted_image, concept_list):
    image = preprocess(weighted_image).unsqueeze(0).to(device)
    text_list = ["一张{}的图片".format(concept) for concept in concept_list]
    text = raw_clip.tokenize(text_list).to(device)
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
        concept_index = probs[0].topk(1)[1].item() # index
        concept_name = concept_list[concept_index] # concept_name
        score = logits_per_image.cpu().detach().numpy()[0][concept_index] # concept_name score
    return concept_name, score


# 获取加权图像，判断图像概念类别，获得匹配得分
# 保存过滤后的加权图像
# 按照首字母-词频-字典序的层次组织文件结构，保存加权图
def save_weighted_image(param_config_dict, image, concept, concept_list, img_save_path="./projects/m2conceptbase/test/", img_save_name="3.jpg"):
    template = "一张{}的图片"
    text = template.format(concept)
    text = clip.tokenize(text).to(device)
    R_image = interpret(model=param_config_dict["model"], image=image, texts=text, device=device)
    # print("R_image", R_image.shape)
    # batch_size = text.shape[0] # batch_size = 1
    # for i in range(batch_size):
        # show_heatmap_on_text(texts[i], text[i], R_text[i])
        # weighted_img = get_weighted_image(R_image[i], image)
    # 获得加权图像：Image对象
    weighted_img = get_weighted_image(R_image[0], image) # PIL Image
    # 判断加权图像的概念类别并获得得分
    concept_name, score = get_weighted_image_category_and_matching_score(
        param_config_dict["judge_model"], 
        param_config_dict["judge_preprocess"], 
        param_config_dict["device"], 
        weighted_img, 
        concept_list)
    # print("concept:", concept)
    # print("concept_name:", concept_name)
    # print("score:", score)
    
    if concept_name == concept: # 如果模型判断该加权图的概念类别 符合 期望的加权图的概念类别，则保存该加权图
        first_pinyin = to_pinyin(concept)[0].upper()
        first_path = os.path.join(img_save_path, first_pinyin) # 首字母文件夹路径
        # (1) 创建首字母文件夹
        if not os.path.exists(first_path): 
            os.makedirs(first_path)
        # (2) 创建概念子文件夹
        concept_path = os.path.join(first_path, concept)
        if not os.path.exists(concept_path):
            os.makedirs(concept_path)
        # (3) 查看目标文件夹下图片数量,图像命名方式：num_score.jpg, 1_48.6.jpg
        image_file_list = os.listdir(concept_path)
        if len(image_file_list) < 20: # 如果概念文件夹下图像文件数量少于20，则直接保存
            img_save_name = str(len(image_file_list) + 1) + '_'  + str(score) + ".jpg"
            is_repeat = False
            for image_file_name in image_file_list:
                temp_num, temp_score = image_file_name.replace(".jpg", "").split('_')
                if abs(score - float(temp_score)) < 0.000001:
                    is_repeat = True
            if is_repeat == False:
                weighted_img.save(os.path.join(concept_path, img_save_name))
            weighted_img.close() # 释放加权图像
        else: # 多于20个文件，则按照匹配得分从大到小，保留20张图片
            min_score = 1000.0
            min_num = 0
            for image_file_name in image_file_list:
                temp_num, temp_score = image_file_name.replace(".jpg", "").split('_')
                if float(temp_score) < min_score:
                    min_num, min_score = int(temp_num), float(temp_score)
            if score > min_score: # 如果 当前图像得分 比 最小得分 大
                # 则 删除 最小得分图像
                os.remove(os.path.join(concept_path, str(min_num) + '_' + str(min_score) + ".jpg"))
                # 当前图像 取代 最小得分图像
                weighted_img.save(os.path.join(concept_path, str(min_num) + '_' + str(score) + ".jpg"))
                weighted_img.close() # 释放加权图像
    
    del weighted_img
        
    # print("saved.")
# save_weighted_image(image, text)


# 获得词频统计字典
def get_word_freq_dict(filename="./projects/m2conceptbase/word_freq_test_data_230102.txt"):
    word_freq_dict = {}
    if filename.split('.')[-1] == 'txt':
        with open(filename, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split()
                if len(items) == 2:
                    word_freq_dict[items[0]] = int(items[1])
    if filename.split('.')[-1] == 'json':
        with open(filename, 'r', encoding='utf-8') as f:
            word_freq_dict = json.load(f)

    print("word_freq_dict file {} load.".format(filename))
    return word_freq_dict

# def get_image_names(filename, preprocess, device):
#     # 读取文件，获得图片列表
#     # image_list = []
#     # candidate_concepts_list = []
#     image_path = "./projects/cache/datasets/wukong_test/wukong_test/{}"
    
#     with open(filename, mode='r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in tqdm(lines, total=len(lines)):
#             items = line.split()
#             image_name, candidate_concepts = items[0], items[1:]
#             img_obj = Image.open(image_path.format(image_name))
#             image = preprocess(img_obj).unsqueeze(0).to(device)
#             # image_list.append(image)
#             # candidate_concepts_list.append(candidate_concepts)
#             img_obj.close()
            # yield image, candidate_concepts
    # return image_list, candidate_concepts_list

def image_batch_generator(file_path, batch_size, image_path="./projects/cache/datasets/wukong_test/wukong_test/{}"):
    # lines = pd.read_csv(file_path, delimiter='\t') # 不要使用pandas，容易出错，直接txt读取
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        image_list = []
        candidate_concepts_list = []
        for line in batch:
            items = line.strip().split('\t')
            image_name, candidate_concepts = items[1], items[2]
            # image_name, candidate_concepts = items[0], items[1:]
            try:
                img_obj = Image.open(image_name)
            except Exception as e:
                print(e)
                continue
            # img_obj = Image.open(image_path.form at(image_name))
            image = preprocess(img_obj).unsqueeze(0).to(device)
            # preprocess the image
            image_list.append(image)
            candidate_concepts_list.append(eval(candidate_concepts)) # 字符串形式的元组转为元组
        yield image_list, candidate_concepts_list


# def process_func(model, preprocess, device, image_list, candidate_concepts_list, test_word_freq):
#     excluded_concepts = ["图片", "图"]
#     img_save_path = "./projects/cache/datasets/weighted_image_test/"
#     for image, candidate_concepts in zip(image_list, candidate_concepts_list):
#         for concept in candidate_concepts:
#             if test_word_freq.get(concept, 0) < 10: # （1）词频过滤：词频<5的舍去（test:5; train: 10）
#                 continue
#             if concept in excluded_concepts: # (2) 人工过滤：过滤掉不适合作为概念的词语
#                 continue
#             save_weighted_image(model, preprocess, device, image, concept, candidate_concepts, img_save_path=img_save_path)


if __name__ == '__main__':
    # filename = "./projects/m2conceptbase/image_candidate_concepts_test.txt"
    img_concept_filepath = "./data_122/wukong_release/wukong_100m_{}_id_img_concepts.csv"
    # image_path = "./projects/cache/datasets/wukong_test/wukong_test/{}"
    # img_wukong_path = "./data_122/wukong_train/wukong_100m_{}/all/{}" # train no need image_path
    # img_save_path = "./projects/cache/datasets/weighted_image_test/"
    img_save_path = "./data_122/m2conceptbase/weighted_images/"
    
    word_freq = get_word_freq_dict(filename="./projects/m2conceptbase/utils/word_freq_train_data_230331.json")
    excluded_concepts = ["图片", "图", "大图", "照片", "版", "们", "子", "时", "人名"]
    
    # 加载模型
    print("model loading...")
    model, preprocess, device = load_model_preprocess(load_from_name_func="load_from_name")
    
    # 加载判别模型：en_clip
    judge_model, judge_preprocess, _ = load_model_preprocess(load_from_name_func="load_from_name_raw")
    # judge_model, judge_preprocess = en_clip.load("ViT-L/14", device=device)
    print("model loaded.")
    param_config_dict = {
        "model": model,
        "preprocess": preprocess,
        "device": device,
        "judge_model": judge_model,
        "judge_preprocess": judge_preprocess,
    }
    
    for file_num in range(0, 4):
        img_concept_filename = img_concept_filepath.format(file_num)
        print("processing file {}...".format(file_num))
        # 使用生成器分批次处理
        batch_size = 128
        for index, (image_list, candidate_concepts_list) in enumerate(image_batch_generator(img_concept_filename, batch_size)):
            for image, candidate_concepts in tqdm(zip(image_list, candidate_concepts_list), total=batch_size, desc="index %i"%index):
                # print(candidate_concepts)
                for concept in candidate_concepts:
                    if word_freq.get(concept, 0) < 9: # （1）词频过滤：词频<5的舍去（test:5; train: 10）
                        continue
                    if concept in excluded_concepts: # (2) 人工过滤：过滤掉不适合作为概念的词语
                        continue
                    # （3）保存加权图
                    save_weighted_image(param_config_dict, image, concept, candidate_concepts, img_save_path=img_save_path)
            del image_list, candidate_concepts_list
            torch.cuda.empty_cache()
    # torch.multiprocessing.set_start_method('spawn')
    # image_list, candidate_concepts_list = get_image_names(filename, preprocess, device)
    # for image, candidate_concepts in zip(image_list, candidate_concepts_list):
    # p = multiprocessing.Pool(processes=10)
    # for image, candidate_concepts in get_image_names(filename, preprocess, device):
        # img_P = multiprocessing.Process(target=process_func, args=(model, preprocess, device, image, candidate_concepts, test_word_freq,))
        # img_P = p.apply(process_func, args=(model, preprocess, device, image, candidate_concepts, test_word_freq,))
        # img_P.start()
        # for concept in candidate_concepts:
        #     if test_word_freq.get(concept, 0) < 5: # （1）词频过滤：词频<5的舍去
        #         continue
        #     if concept in excluded_concepts: # (2) 人工过滤：过滤掉不适合作为概念的词语
        #         continue
        #     save_weighted_image(model, preprocess, device, image, concept, candidate_concepts, img_save_path=img_save_path)
    #     # torch.cuda.empty_cache()
    #     from  numba import cuda
    #     device = cuda.get_current_device()
    #     device.reset()     
        
    # with open(filename, mode='r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in tqdm(lines, total=len(lines)):
    #         items = line.split()
    #         image_name, candidate_concepts = items[0], items[1:]
            
    #         # 对于每张图片每次重新加载模型
    #         model, preprocess, device = load_model_preprocess()
            
    #         img_obj = Image.open(image_path.format(image_name))
    #         image = preprocess(img_obj).unsqueeze(0).to(device)
    #         img_obj.close()
            
    #         for concept in candidate_concepts:
    #             if test_word_freq.get(concept, 0) < 5: # （1）词频过滤：词频<5的舍去
    #                 continue
    #             if concept in excluded_concepts: # (2) 人工过滤：过滤掉不适合作为概念的词语
    #                 continue
    #             save_weighted_image(model, preprocess, device, image, concept, candidate_concepts, img_save_path=img_save_path)
                
    #         # from  numba import cuda
    #         torch.cuda.empty_cache()
            # device.reset()     
        

    print("Done!")
                