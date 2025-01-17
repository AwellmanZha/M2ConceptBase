#coding=utf-8
import jieba
import re
from collections import Counter
import json
import pandas as pd
import jieba.posseg as pseg
from LAC import LAC
from tqdm import tqdm

"""
获得图文对上下文中的候选概念。
"""

train_flag = True
if train_flag:
    print("process train file...")
else:
    print("process test file...")

#导入打开要处理的文本
# file_fmt = './data_122/wukong_release/wukong_100m_{}.csv'
file_fmt = './data_122/wukong_release/wukong_100m_{}_output.csv' # id\tcaption\timage
test_file_path = "./datasets/wukong_test/wukong_test.csv"


#加入停用词
stopwords = {}.fromkeys([line.rstrip() for line in open('./data_122/kg_data/stopwords.txt', 'r', encoding='utf-8')])

# 定义lac分词器
lac = LAC(mode='lac')

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
word_freq = get_word_freq_dict(filename="./data_122/zhazhiwei_133_backup/projects/m2conceptbase/utils/word_freq_train_data_230531.json")

# 获取所有文件中的caption: [0,255], range(0,256)
for file_num in range(5, 33): # 4-8
    
    # caption_list = []
    # image_name_list = []
    
    if train_flag == True:
        filename = file_fmt.format(file_num)
    else:
        filename = test_file_path
    print("processing", filename)
    # 读取csv文件的第二列：caption
    # with open(filename, "r", encoding='utf-8') as f:
        # f_csv = csv.reader(f)
    # print("num of captions:", len(f_csv)) # 24: 390541 25: 390454  test: 33365

    img_concepts_fmt = "./data_122/wukong_release/wukong_100m_{}_id_img_concepts.csv"
    file = open(img_concepts_fmt.format(file_num), "w", encoding='utf-8')
    file.write("id\timage\tconcepts\n")
    
    # 统计一个文件的caption
    #设置初始计数数组
    # candidate_concept_list = []
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.readlines()[1:] # 去除表头
    # 对每个文件的caption进行分词和统计
    for line in tqdm(lines, total=len(lines)):
        idx, caption, image_name = line.strip().split('\t')
        #利用jieba分词
        # words = jieba.lcut(caption)
        words_jieba = []
        res_jieba = pseg.cut(caption)
        for w in res_jieba:
            if w.flag.startswith('n'):
                # print(w.word, w.flag)
                words_jieba.append(w.word)
                
        # print("--------------------")
        words_lac = []
        res_lac = lac.run(caption)
        res_lac = list(zip(*res_lac))
        for item in res_lac:
            if item[1].startswith('n') and len(item[0]) < 6: # ['ORG', 'LOC', 'PER'] # best practice : 6
                # print(item)
                words_lac.append(item[0])
        
        # print(words_jieba)
        # print(words_lac)
        
        # 合并两个分词器的分词结果，重复的删除，用合并后的分词结果进行词频统计
        combined_words_list = []
        combined_words = set(words_jieba + words_lac)
        # combined_words.update(words_lac)
        # 将 set 转换回列表
        combined_words_list = list(combined_words)
        combined_words_list = [word for word in combined_words_list if word not in stopwords] # 去除停用词
        filtered_word_list = [word for word in combined_words_list if word in word_freq] # 仅保留在候选中的概念
        # if len(combined_words_list) > 40:
            # print(i, combined_words_list)
            # print(caption)
            # print(img_name)
        # print("all:", combined_words_list)
        # 将每个caption的最终分词结果保存下来，和每个caption对应
        # candidate_concepts = "null"
        # if combined_words_list:
        #     candidate_concepts = " ".join(combined_words_list)
        # candidate_concept_list.append(candidate_concepts)
        # print(i, str(candidate_concepts))
        file.write("%s\t%s\t%s\n"%(str(idx), image_name, tuple(filtered_word_list)))
    file.close()
    print("File {} saved.".format(file_num))
        # print("====================")
        # 开始遍历计数
        # for w in combined_words_list:
        #     counts[w] = counts.get(w, 0) + 1
        
    # print("num of words:", len(counts))
    # exit(1)
    # 将一个文件的image_name, candidate_concepts保存到txt文件
    # test & train
    # img_concepts_fmt = "/home/zhazhiwei/projects/m2conceptbase/image_candidate_concepts_{}.txt"
    # if train_flag == True:
    #     img_concepts_filename = img_concepts_fmt.format("train")
    # else:
    #     img_concepts_filename = img_concepts_fmt.format("test")
    # img_concepts_fmt = "/home/zhazhiwei/projects/m2conceptbase/image_concepts_train/image_candidate_concepts_train_{}.txt"
    
    # with open(img_concepts_fmt.format(file_num), mode='w', encoding='utf-8') as f:
    # for idx, (img_name, caption, candi_concepts) in enumerate(zip(image_name_list, caption_list, candidate_concept_list)):
    #     file.write("%s\t%s\t%s\t%s\n"%(str(idx), img_name, caption, candi_concepts))
            # new_context = img_name + "  " + candi_concepts + '\n'
            # f.write(new_context)
            # f.write("")
    # file.close()

# 得到所有文件caption的词频统计结果
# 返回遍历得分所有键与值
# items = list(counts.items())
# print("num of words:", len(items))

# # 根据词频进行排序
# items.sort(key=lambda x: x[1], reverse=True)
# # print(items)

# # 将词频数据写入txt文本
# save_file_path_fmt = "/home/zhazhiwei/projects/m2conceptbase/word_freq_{}_data_230103.txt"
# if train_flag == True:
#     save_file_path = save_file_path_fmt.format("train")
# else:
#     save_file_path = save_file_path_fmt.format("test")
# with open(save_file_path, mode='w', encoding='utf-8') as file:
#     #输出词语与词频
#     for i in range(len(items)):
#         word, count = items[i]
#         if count < 5: #词频<5的不保存
#             continue
#         # print("{0:<10}{1:>5}".format(word,count))
#         #写入txt文件
#         new_context = word + "  " + str(count) + '\n'
#         file.write(new_context)

print("done!")
