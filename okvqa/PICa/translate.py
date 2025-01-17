from googletrans import Translator
import json
import time
from tqdm import tqdm

def translate_tags(tag_list, target_lang='zh-cn'):
    translator = Translator()

    translations = {}
    for tag in tag_list:
        try:
            translation = translator.translate(tag, dest=target_lang)
            translations[tag] = translation.text
        except Exception as e:
            print(f"Error translating '{tag}': {e}")

    return translations

# 示例tag列表
# import json
# # tag_list = ["beach", "mountain", "restaurant", "hotel", "shopping", "park"]
# with open("tag_list.json", 'r', encoding='utf-8') as f:
#     tag_list = json.load(f)
# print(len(tag_list))

# # 打印结果
# with open("tag_en2zh.json", 'r', encoding='utf-8') as f:
#     tag_en2zh = json.load(f)

# tag_list = [tag for tag in tag_list if tag not in tag_en2zh.keys()]
# print(len(tag_list))

# translations = translate_tags(tag_list)


# for tag, translation in translations.items():
#     print(f"{tag} -> {translation}")
#     tag_en2zh[tag] = translation
# print(len(tag_en2zh))

# with open("tag_en2zh_v2.json", 'w', encoding='utf-8') as f:
#     json.dump(tag_en2zh, f, ensure_ascii=False, indent=4)

# print("saved.")



with open("tag2description.json", 'r', encoding='utf-8') as f:
    tag_description = json.load(f)
    

def translate_descriptions(tag_description, target_lang='en'):
    translator = Translator()

    new_tag_description = {}
    for tag, desc_list in tqdm(tag_description.items(), total=len(tag_description.items()), desc=f"translating"):
        # for desc in desc_list:
        i = 0
        translations = []
        while i < len(desc_list):
            try: 
                desc = desc_list[i]
                translation = translator.translate(desc, dest=target_lang)
                translations.append(translation.text)
            except Exception as e:
                print(f"Error translating '{tag}': {e}")
                print(f"wait 10s and retry")
                time.sleep(10)
                continue
            i += 1
        
        new_tag_description[tag] = translations
    
    return new_tag_description


new_tag_description = translate_descriptions(tag_description)

with open("new_tag_description.json", 'w', encoding='utf-8') as f:
    json.dump(new_tag_description, f, ensure_ascii=False, indent=4)

print("done.")