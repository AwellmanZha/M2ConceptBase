"""
concept utils with chatgpt: 
filter_non_concepts, generate_concept_descriptions, classify_concrete_abstract_concepts.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import openai
import json
import random
from tqdm import tqdm
import time
import multiprocessing as mp
# mp.set_start_method('spawn', True)
openai.api_key = "your_openai_api_key"
from transformers import AutoTokenizer, AutoModel

# load chatglm2
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')
# model = model.eval()


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_completion_v2(prompt, model="text-davinci-003"):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0, # this is the degree of randomness of the model's output
        max_tokens=1000
    )
    return response.choices[0].text


# concepts = [
#     "多巴胺受体", "帮川", "船公司", "上报名", "大站点", "黑底白字", "九品莲花", "通知类", "外圆刀", "包子状", 
#     "科学", "生日", "保鲜盒", "梦", "价格", "数学", "房地产税", "文化", "苹果", "杂款", 
#     "小而美的", "能奶", "学生校", "女球", "香蕉姐", "移植仓", "零胆固醇", "南将军", "胃王", "留守男孩",
#     "初霜", "防晒伞", "薄膜开关", "丑陋的", "空床", "墙饰", "收费单", "腰款", "外研社", "普适性", 
#     "烤机", "文化用品", "内啡肽", "文昌帝君", "千里马", "平舌音", "螺蛳粉", "基因型", "玉制", "班人马",
#     "交通生活", "英伦味", "电热棒", "五星城", "小鱼苗", "巨鳖", "斗象台", "渠底", "汗卫", "鸡体",
# ]
def filter_non_concepts():
    save_path = "./data_122/m2conceptbase/concept_labels/"
    
    with open("./projects/m2conceptbase/concepts_to_get_230524.json", 'r', encoding='utf-8') as f:
        candidate_concepts = json.load(f)
    fail_list = []
    batch_size = 60
    for i in tqdm(range(0, len(candidate_concepts), batch_size), total=len(candidate_concepts)//batch_size):
        concepts = candidate_concepts[i: i + batch_size]
        if i == 228420: 
            print(concepts)
            # break
        prompt = f"""
            你是一名专业的概念标注专家，你的任务是标注出概念和非概念。概念是一个有意义的名词，非概念往往是分词错误导致的。
            例如“能奶”、“学生校”似乎包含部分名词，但是分词错误，应标注为非概念；“包子状”是形容词，“运送货物”是动词搭配，因此也应标注为非概念。
            接下来在一个可能包含很多非概念的词语列表中完成概念标注任务：{concepts}。
            以Python字典格式输出，其中键为词语(用双引号)，值为0表示非概念，值为1表示概念。
        """
        try:
            if os.path.exists(os.path.join(save_path, "concept_labels_{}_{}.json".format(i, i + batch_size))):
                # print("File concept_labels_{}_{}.json already done.".format(i, i + batch_size))
                continue
            else:
                print("getting concept_labels_{}_{}.json ......".format(i, i + batch_size))
            response = get_completion(prompt)
            print(response)
            with open(os.path.join(save_path, "concept_labels_{}_{}.json".format(i, i + batch_size)), 'w', encoding='utf-8') as f:
                f.write(response)
            print("File concept_labels_{}_{}.json saved.".format(i, i + batch_size))
        except Exception as e:
            print(e)
            fail_list.append(i)
            time.sleep(10)
            continue

    with open("fail_list.json", 'w', encoding='utf-8') as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=4)

    print("Done.")


with open("./projects/m2conceptbase/concepts_to_generate_230608.json", 'r', encoding='utf-8') as f:
    concepts_to_generate = json.load(f)
fail_list = []

def generate_descriptions(index):
    try:
        save_path = "./data_122/m2conceptbase/generated_descriptions/"
        batch_size = 10
        if os.path.exists(os.path.join(save_path, "concept_descriptions_{}.json".format(index))):
            print("File concept_descriptions_{}.json already done.".format(index))
            return
        else:
            print("getting concept_descriptions_{}.json ......".format(index))
            
            result_dict = {}
            for concept in concepts_to_generate[index:index + batch_size]:
                prompt = f"请为概念\"{concept}\"生成一句基本概念描述, 科学严谨地解释这个概念的基本含义，注意不要自由发挥，但是要尽可能详细。".format(concept)
                
                response = get_completion_v2(prompt)
                print(response)
                
                result_dict[concept] = response.strip()
                
            with open(os.path.join(save_path, "concept_descriptions_{}.json".format(index)), 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=4)
            print("File concept_descriptions_{}.json saved.".format(index))
    except Exception as e:
        print(e)
        fail_list.append(index)
        time.sleep(10)
        return

def generate_descriptions_from_chatglm2(index):
    
    save_path = "./data_122/m2conceptbase/generated_descriptions/"
    batch_size = 10
    
    if os.path.exists(os.path.join(save_path, "concept_descriptions_{}.json".format(index))):
        print("File concept_descriptions_{}.json already done.".format(index))
        return
    else:
        print("getting concept_descriptions_{}.json ......".format(index))
    
    result_dict = {}
    # concepts = ["科学", "危机", "梦想", "自由"]
    for concept in concepts_to_generate[index:index + batch_size]:
        response, history = model.chat(tokenizer, f"请为概念“{concept}”生成一句精确详实的描述来解释其含义。".format(concept), history=[])
        print(response)
        result_dict[concept] = response.strip()
        
    with open(os.path.join(save_path, "concept_descriptions_{}.json".format(index)), 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    print("File concept_descriptions_{}.json saved.".format(index))



def generate_concept_descriptions(): 
    p = mp.Pool(1)
    print("process pool produced.")
    
    # p.map(generate_descriptions, [i for i in range(0, len(concepts_to_generate), 10)])
    p.map(generate_descriptions_from_chatglm2, [i for i in range(0, len(concepts_to_generate), 10)])
        
    with open("fail_generate.json", 'w', encoding='utf-8') as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=4)

    print("Done.")
    

def concret_abstract_concept_classification_by_chatGPT(concepts):
    
    prompt = f"""
        你是一个聪明的概念标注人员，你的任务是标注出具象概念和抽象概念。具象概念是指可以用视觉图像表达的概念，抽象概念是指没有统一视觉表达的概念。
        概念列表如下：{concepts}。
        以Python字典格式输出结果，其中键为概念(双引号)，值为0表示具象概念，值为1表示抽象概念，值为2表示不确定是不是概念。
    """
    response = get_completion(prompt)
    print(response)
    return eval(response.strip())

def concret_abstract_concept_classification_by_chatGPT_batch_processing(concret=True):
    # stage 1: concret
    # if concret==True: 
    #     print("processing concret concepts...")
    #     with open("./projects/m2conceptbase/utils/m2conceptbase_eval_part0_concret_concept_understanding.json", 'r', encoding='utf-8') as f:
    #         concret_concepts = json.load(f)
    #     batch_size = 20
    #     result_dict = {}
    #     for i in tqdm(range(0, len(concret_concepts), batch_size), total=len(concret_concepts) // batch_size):
    #         if i // batch_size <= 37:
    #             continue
    #         concept_list = concret_concepts[i: i + batch_size]
    #         result = concret_abstract_concept_classification_by_chatGPT(concept_list)
    #         result_dict.update(result)
            
    #     with open("concret_concepts_label.json", 'w', encoding='utf-8') as f:
    #         json.dump(result_dict, f, ensure_ascii=False, indent=4)
    # stage 2: abstract
    # else:
    # print("processing abstract concepts...")
    # with open("./projects/m2conceptbase/utils/m2conceptbase_eval_part1_abstract_concept_understanding.json", 'r', encoding='utf-8') as f:
    #     abstract_concepts = json.load(f)
    # batch_size = 20
    # result_dict = {}
    # for i in tqdm(range(0, len(abstract_concepts), batch_size), total=len(abstract_concepts) // batch_size):
    #     if i // batch_size <= 32:
    #         continue
    #     concept_list = abstract_concepts[i: i + batch_size]
    #     result = concret_abstract_concept_classification_by_chatGPT(concept_list)
    #     result_dict.update(result)
        
    # with open("abstract_concepts_label.json", 'w', encoding='utf-8') as f:
    #     json.dump(result_dict, f, ensure_ascii=False, indent=4)
    # # stage 3: more abstract(up to 2000)
    # with open("./projects/m2conceptbase/m2conceptbase_concepts.json", 'r', encoding='utf-8') as f:
    #     m2conceptbase_concepts = json.load(f)
    # with open("./projects/m2conceptbase/utils/m2conceptbase_eval_part0_concret_concept_understanding.json", 'r', encoding='utf-8') as f:
    #     concret_concepts = json.load(f)
    # with open("./projects/m2conceptbase/utils/m2conceptbase_eval_part1_abstract_concept_understanding_v2.json", 'r', encoding='utf-8') as f:
    #     abstract_concepts = json.load(f)
    # cand_abstract_concepts = list(set(m2conceptbase_concepts) - set(concret_concepts) - set(abstract_concepts))
    # random.shuffle(cand_abstract_concepts)
    # left_num = 2000 - len(abstract_concepts)
    # print(left_num)
    # left_abstract_concepts = []
    # batch_size = 20
    # for i in tqdm(range(0, len(cand_abstract_concepts), batch_size), total=len(cand_abstract_concepts) // batch_size):
    #     try:
    #         # if i // batch_size <= 32:
    #         #     continue
    #         concept_list = cand_abstract_concepts[i: i + batch_size]
            
    #         result = concret_abstract_concept_classification_by_chatGPT(concept_list)
            
    #         temp_list = [k for k, v in result.items() if v == 0] # 0 for abstract
    #         print(temp_list)
    #         left_abstract_concepts.extend(temp_list)
    #         print(len(left_abstract_concepts))
    #         if len(left_abstract_concepts) >= left_num:
    #             break
    #     except Exception as e:
    #         print(e)
    #         continue
    # stage 4: m2conceptbase concept classification
    with open("./projects/m2conceptbase/m2conceptbase_concepts.json", 'r', encoding='utf-8') as f:
        m2conceptbase_concepts = json.load(f)


    save_path = "./projects/m2conceptbase/utils/results/"
    result_dict = {}
    batch_size = 20
    i = 0
    while i < len(m2conceptbase_concepts):
    # for i in tqdm(range(0, len(m2conceptbase_concepts), batch_size), total=len(m2conceptbase_concepts) // batch_size):
        try:
            # if i // batch_size <= 32:
            #     continue 
            concept_list = m2conceptbase_concepts[i: i + batch_size]
            
            result = concret_abstract_concept_classification_by_chatGPT(concept_list)
            result_dict.update(result)
            if len(result_dict) >= 10000:
                with open(save_path + f"m2conceptbase_concept_label_{i}.json", 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, ensure_ascii=False, indent=4)
                result_dict = {}

        except Exception as e:
            print(e)
            print("wait 10s and retry...")
            time.sleep(10)
            continue
        i += batch_size
    
    with open(save_path + f"m2conceptbase_concept_label_final_{i}.json", 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    # final_abstract_concepts = abstract_concepts + left_abstract_concepts
    # with open("./projects/m2conceptbase/utils/m2conceptbase_eval_part1_abstract_concept_understanding.json", 'w', encoding='utf-8') as f:
    #     json.dump(final_abstract_concepts, f, ensure_ascii=False, indent=4)
    print("done.")




def generate_concept_knowledge_vqa_question():
    # with open("m2conceptbase_eval_part3_concept_knowledge_vqa.json", 'r', encoding='utf-8') as f:
    #     sampled_knowledge_vqa_concepts = json.load(f)
    # sampled_knowledge_vqa_concepts = sampled_knowledge_vqa_concepts[:10]
    with open("./projects/m2conceptbase/utils/concept_knowledge_vqa_description_dict.json", 'r', encoding='utf-8') as f:
        concept_knowledge_vqa_description_dict = json.load(f)
    # fail_list = []
    with open("./projects/m2conceptbase/utils/question_fail_list.json", 'r', encoding='utf-8') as f:
        sampled_knowledge_vqa_concepts = json.load(f)
    concept_questions = {}
    for concept in tqdm(sampled_knowledge_vqa_concepts, total=len(sampled_knowledge_vqa_concepts)):
        try:
            description = concept_knowledge_vqa_description_dict[concept]
            prompt = f"""
                请根据概念的描述生成一个和概念相关的知识性问题(注意只需生成一个问题且问题中最好出现概念的名称)。
                概念：{concept}
                概念描述如下：{description}
            """
            response = get_completion(prompt)
            print("description: ", description)
            print("question: ", response)
            concept_questions[concept] = str(response.strip())
        except Exception as e:
            fail_list.append(concept)
            print(e)
            continue
    with open("m2conceptbase_eval_part3_concept_knowledge_vqa_questions_p3.json", 'w', encoding='utf-8') as f:
        json.dump(concept_questions, f, ensure_ascii=False, indent=4)
    with open("question_fail_list.json", 'w', encoding='utf-8') as f:
        json.dump(fail_list, f, ensure_ascii=False, indent=4)
    print("done!")
    return concept_questions, fail_list




def m2conceptbase_eval_with_chatGPT(prediction_result_file=None):
    with open(prediction_result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open("./projects/m2conceptbase/utils/m2conceptbase_eval_part3_concept_knowledge_vqa_answers.json", 'r', encoding='utf-8') as f:
        kvqa_answers = json.load(f)

    qa_pairs_task0 = []
    qa_pairs_task1 = []
    qa_pairs_task2 = []
    qa_pairs_task3 = []
    
    for k, v in data.items():
        # qa_pair = v[1]
        qa_pair = v[0]
        if "回答相关知识性问题" in qa_pair[0]:
            qa_pair.append(kvqa_answers[k])
            qa_pairs_task3.append(qa_pair)
            # print(qa_pairs_task3)
        elif "相关吗" in qa_pair[0]:
            qa_pairs_task2.append(qa_pair)
        elif "描绘的是" in qa_pair[0]:
            qa_pairs_task1.append(qa_pair)
        else:
            qa_pairs_task0.append(qa_pair)
            
    print(len(qa_pairs_task0), len(qa_pairs_task1), len(qa_pairs_task2),len(qa_pairs_task3))
    # qa_pairs = [
    #     [
    #         "这张图片中有芦荟吗？",
    #         "这张图片中的植物是芦荟。"
    #     ],
    #     [
    #         "这张图片中有眼科吗？",
    #         "这张图片中没有眼科内容。"
    #     ],
    #     [
    #         "这张图片中有芸豆吗？",
    #         "这张图片中确实包含了一些芸豆。"
    #     ]
    # ]
    for task_idx, qa_pairs in enumerate([qa_pairs_task0, qa_pairs_task1, qa_pairs_task2, qa_pairs_task3]):
        if task_idx == 0:
            # continue
            task_question_template = "这张图像中有这个·概念·吗？"
        elif task_idx == 1:
            # continue
            task_question_template = "这张图像描绘的是这个·概念·吗？"
        elif task_idx == 2:
            # continue
            task_question_template = "这张图像和这个·概念·相关吗？"
        else:
            task_question_template = "请根据这个·概念·的图片内容，回答相关知识性问题。"
            
        print(f"eval task {task_idx}")
        qa_result_list = []
        i = 0
        while i < len(qa_pairs):
            print(f"task: {task_idx}, index:{i}/{len(qa_pairs)}")
            if task_idx == 3:
                qa_pairs_batch = qa_pairs[i: i + 2]
            else:
                qa_pairs_batch = qa_pairs[i: i + 2]
            formatted_qa_pairs = []
            for qa_pair in qa_pairs_batch:
                if task_idx == 3:
                    qa_pair_template = f"""问题：{qa_pair[0]} \n 回答：{qa_pair[1]} \n 参考答案：{qa_pair[2]}"""
                else:
                    qa_pair_template = f"""问题：{qa_pair[0]} \n 回答：{qa_pair[1]}"""
                formatted_qa_pairs.append(qa_pair_template)
                
            
            if task_idx == 3:
                prompt = f"""
                    现在我有一个概念图像数据集，对于每张概念图像，我根据概念的知识性描述，设计了一个问题，问题类似于“{task_question_template}”。
                    由于多模态大模型的回复是自由文本形式，回答可能是正确的，但是没有统一的自动判断对错的方法。因此我想请你充当评估器。
                    对于输入的(问题，回答）对，请根据参考答案的内容判断回答是否正确，注意聚焦于问题是否回答正确，忽略其他内容，只要意思上正确则判断为正确，意思上错误则判断为错误。
                    正确则输出1，错误则输出0，仅输出0或1。
                    这里有一个例子：
                    例子1：
                    问题：请根据“握手”的图片内容，回答相关知识性问题：问题：握手在人际交往中有什么作用？
                    回答：握手是一种社交礼仪动作，通常用于表示尊重和礼貌。通过握手，人们可以表达彼此之间的情感联系、认可和支持。这种互动也可以促进建立良好的关系，增强沟通和理解能力。此外，握手还可以作为礼物交换的象征，例如以握手为礼，表达对对方友谊和感激之情。
                    参考答案：握手是一种礼仪，人与人之间、团体之间、国家之间的交往都赋予这个动作丰富的内涵。一般说来，握手表示友好，是一种交流方式，可以沟通原本隔阂的情感，可以加深双方的理解、信任，可以表示一方对另一方的尊敬、景仰、祝贺、鼓励，也能传达出一些人的淡漠、敷衍、逢迎、虚假、傲慢。团体领袖、国家元首之间的握手则往往象征着合作、和解、和平。握手的次数也许数也数不清，印象深刻的可能只有几次：第一次见面的激动，离别之际的不舍，久别重逢的欣喜，误会消除、恩怨化解的释然等等。
                    判定结果：1
                    接下来输入按照上述格式的(问题，回答）对的列表，请批量回答（结果是一个元素为0或1的Python列表，按顺序表示每个样本的判定结果，不要有其他输出）：
                    {formatted_qa_pairs}
                """
            else:
                prompt = f"""
                    现在我有一个概念图像数据集，对于每张概念图像，我设计了一个问题，问题类似于{task_question_template}。
                    由于我是按照图像中的概念来提问的，因此答案都是肯定的。现在我用了多模态大模型回答了这个问题，得到了（问题，回复）对。
                    由于多模态大模型的回复是自由文本形式，回答可能是正确的，但是没有统一的自动判断对错的方法。因此我想请你充当评估器。
                    对于输入的(问题，回答）对，请判断回答是否正确，只要意思上正确则判断为正确，意思上相反则判断为错误。
                    正确则输出1，错误则输出0，仅输出0或1。
                    这里有两个例子：
                    例子1：
                    问题：这张图片中有天津吗？
                    回答：这张照片中没有天津。
                    判定结果：0
                    例子2：
                    问题：这张图片中有古画吗？
                    回答：这幅图片描绘了一组古代绘画作品的场景，其中包含有几位身着传统服饰的女子正在林中寻找出路或寻找某个特定对象。画面的背景是一棵高大的树和一个石头墙壁，整体呈现出一种古典的氛围。整个画作使用明亮的色彩来突出主题，如黄色和橙色，并点缀着黑色和灰色调。
                    判定结果：1
                    接下来输入按照上述格式的(问题，回答）对的列表，请批量回答（结果是一个元素为0或1的Python列表，按顺序表示每个样本的判定结果，不要有其他输出）：
                    {formatted_qa_pairs}
                """
            # print(prompt)
            try:
                response = get_completion(prompt)
                print(response)
            except Exception as e:
                print(e)
                print(f"error index: {i}. Wait 10s and retry.")
                time.sleep(10)
                continue
            res = eval(response)
            if type(res) == int:
                res = [res]
            else:
                res = [int(r) for r in res]
            qa_result_list.extend(res)
            if task_idx == 3:
                i += 2
            else:
                i += 2
        Acc = sum(qa_result_list) / len(qa_result_list)
        print(f"Task {task_idx}, T={sum(qa_result_list)}, total={len(qa_result_list)}, Acc={Acc}.")
        with open(prediction_result_file.replace(".json", "") + f"_task{task_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(qa_result_list, f, ensure_ascii=False, indent=4)
        print("Done.")

if __name__ == "__main__":
    # generate_concept_descriptions()
    # for i in tqdm(range(20, len(concepts_to_generate), 30), total=len(concepts_to_generate)//30):
    #     generate_descriptions_from_chatglm2(i)
    # concret_abstract_concept_classification_by_chatGPT_batch_processing(False)
    # generate_concept_knowledge_vqa_question()
    
    # prediction_result_file = "./projects/m2conceptbase/utils/m2conceptbase_eval_VisualGLM_prediction_results.json"
    # prediction_result_file = "./projects/m2conceptbase/utils/m2conceptbase_eval_RAMICL_VisualGLM_prediction_results.json"
    prediction_result_file = "./data_122/vscode_home/evaluation/Chinese-CLIP/m2conceptbase_eval_RAMICL_VisualGLM_prediction_results_v3.json"
    m2conceptbase_eval_with_chatGPT(prediction_result_file)
    
    
    
    # concret_abstract_concept_classification_by_chatGPT_batch_processing()
