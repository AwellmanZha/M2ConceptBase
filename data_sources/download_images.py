import os
import requests
from tqdm import tqdm
import json
import time
        
# # 图像链接列表
# image_links = [
#     'https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fdimg05.c-ctrip.com%2Fimages%2Ftg%2F663%2F026%2F799%2F0beb1c361d564e999bd395c459998f80.jpg&refer=http%3A%2F%2Fdimg05.c-ctrip.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1632920511&t=14db9cd7a7aa0933b04faa91159e1efd',
#     # 'https://example.com/image2.png',
#     # 'https://example.com/image3.jpg',
#     # 添加更多的链接
# ]

def download_img_for_file(filename, folder_path, file_index):
    image_links = []

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for link in data:
            image_links.append(link)
    
    link2name = {} # 保存图像链接到图像名词的字典，用作id对齐和image name和concepts(for every file)
    # 循环遍历所有链接并下载图像
    for i, link in tqdm(enumerate(image_links), total=len(image_links)):
        response = requests.get(link)
        time
        image_name = f'image_{i}.jpg'  # 图像文件名
        link2name[link] = image_name
        image_path = os.path.join(folder_path, image_name)  # 图像保存路径
        with open(image_path, 'wb') as f:
            f.write(response.content)
        # print(f'{image_name} saved successfully.')
    
    save_file = "link2name_{}.json".format(file_index)
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(link2name, f, ensure_ascii=False, indent=4)
    
    print(save_file, "saved.")


# 文件格式处理，将有空格开头的txt文件，处理成json格式
def file_processing(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    result = {}
    url = None

    for line in tqdm(lines, total=len(lines)):
        stripped_line = line.strip()
        if stripped_line == '':
            continue
        if stripped_line.startswith('http'):
            url_concepts = stripped_line.split(' ', 1)
            url = url_concepts[0].strip()
            if len(url_concepts) == 1:
                result[url] = ""
            else:
                result[url] = url_concepts[1].strip()
        else:
            result[url] += " " + stripped_line

    # 处理最后一行
    if url not in result:
        result[url] = []

    # print(result)

    with open(filename.split('.')[0] + ".json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        
    print("saved.")


if __name__ == "__main__":
    # file_processing("./projects/m2conceptbase/utils/test.txt", "./")
    # 文件格式处理
    # image_url_file = "./datasets/m2conceptbase/image_concepts_train/image_candidate_concepts_train_{}.txt"
    # # 遍历所有文件（0, 256）
    # for index in range(0, 256):
    #     print("processing file {}...".format(index))
    #     file_processing(image_url_file.format(index))
    # print("Done.")
    
    image_url_file = "./datasets/m2conceptbase/image_concepts_train/image_candidate_concepts_train_{}.json"
    # 遍历所有文件（0, 256）
    for index in range(0, 256):
        print("downloading wukong_train_", index)
        # 用于保存图像的文件夹路径
        folder_path = './datasets/wukong_train/wukong_train_{}'.format(index)
        # 如果文件夹不存在，则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        download_img_for_file(image_url_file.format(index), folder_path, index)
    print("Done.")
