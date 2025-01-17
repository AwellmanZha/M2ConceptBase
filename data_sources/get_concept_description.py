import json
import requests
import urllib
from bs4 import BeautifulSoup
import re
import os
import multiprocessing as mp
from tqdm import tqdm
import time
import random
from itertools import chain
from pypinyin import pinyin, Style

ROOT = "./data_122/m2conceptbase/concept_descriptions/"
excluded_concepts = ["图片", "图", "大图", "照片", "版", "们", "子", "时", "人名"]

# 伪装头
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    'Cookie': 'BIDUPSID=6E5DAC66718DC8B6D51890A708CC781C; PSTM=1622962867; __yjs_duid=1_b086b0711dd02869fee4f2cc243ab70e1622988218615; H_WISE_SIDS=107319_110085_180636_194520_194530_196425_197471_204905_208721_209568_209630_210304_210321_210470_211433_212296_212869_213037_213357_214806_215730_216047_216212_216368_216714_216851_216943_217167_218445_218452_218549_218599_219064_219450_219565_219943_219946_220338_220345_220379_220386_220603_220663_221007_221016_221119_221121_221139_221370_221501_221641_221698_221716_221734_221795_221824_221871_221881_221919_222140_222272_222392_222397_222463_222616_222617_222619_222625_222698_222779_222794_222861_222870_222880_223069_223190_223296_223344_223395_223729_223766_223781_223833_223851_224055_224085; H_WISE_SIDS_BFESS=107319_110085_180636_194520_194530_196425_197471_204905_208721_209568_209630_210304_210321_210470_211433_212296_212869_213037_213357_214806_215730_216047_216212_216368_216714_216851_216943_217167_218445_218452_218549_218599_219064_219450_219565_219943_219946_220338_220345_220379_220386_220603_220663_221007_221016_221119_221121_221139_221370_221501_221641_221698_221716_221734_221795_221824_221871_221881_221919_222140_222272_222392_222397_222463_222616_222617_222619_222625_222698_222779_222794_222861_222870_222880_223069_223190_223296_223344_223395_223729_223766_223781_223833_223851_224055_224085; FPTOKEN=30$ndhUpR/c2sqNAvvDjZL1Ju1TYJCkE3tHK7T1i1n9hNkhTrlRbbCqkpFC9rvfHK5zTBySCCPkMKOU2beyc4qpZSu4H4Pm0VE41tv8Fq43tUccOMlj+TUAfDpmDNPgviFaYvQhpBbtWOcxWY26Nvb+uOaQjIXUdkef1Mf008+1rNwjW0jt3bt+koYLLmdUErNb3pmB3cWF6SgagtHHA0IiPjwQYU9IkZmYwL4glJ3lbNBSB13CDsKiMPkJ91eZRCmTWzLBf2v/I+ED/bpSBPgqi8LMFdUnd7ORIJ6GiZLNMHAlKBCnpcdalyY82uDwKAE7MTfy1vcKvEEDlTblxKrOHOtS2/ORdQzG3faAR2syC5OuYiU5lW8PT3EeKHedUhF2auKeOSvlFg/ab/yiFTcD9f6C9bYrrhAzj5EmwZM4EY4=|lNm1v6HNEgWIFlQlpcZvgorGOZF7lTqFqp85PZHh58o=|10|4d585c4e1b7f3fe478763bd8787a3517; BAIDUID=4915043CD25AB00C5BA1C9F27675FBED:FG=1; MCITY=-289%3A; BK_SEARCHLOG=%7B%22key%22%3A%5B%22%E5%A5%B3%E5%A4%A7%E5%AD%A6%E7%94%9F%22%2C%22%E5%B0%8F%E7%BA%A2%E4%B9%A6%22%2C%22%E6%89%8B%E8%89%BA%22%2C%22%E8%8A%B1%E9%A6%99%22%2C%22%E9%AA%A8%E6%9E%B6%22%2C%22%E8%8B%B9%E6%9E%9C%22%2C%22%E5%B8%82%E5%9C%BA%22%2C%22%E7%A9%BA%E9%97%B4%22%2C%22%E5%88%A9%E7%9B%8A%E5%88%86%E6%9E%90%E5%92%8C%E9%98%B6%E7%BA%A7%E9%98%B6%E5%B1%82%E5%88%86%E6%9E%90%22%2C%22%E4%B8%AA%E4%BD%93%E5%BF%83%E7%90%86%E5%AD%A6%22%5D%7D; BDORZ=FFFB88E999055A3F8A630C64834BD6D0; BA_HECTOR=8k81a0052g0k2581a08004561i41oan1n; BAIDUID_BFESS=4915043CD25AB00C5BA1C9F27675FBED:FG=1; ZFY=hgUWIsAhuRpQuUQ9jpyS29xiXxuCPDeJxQVZmZ0lIAM:C; BDRCVFR[1kRcOFa5hin]=mk3SLVN4HKm; H_PS_PSSID=26350; PSINO=5; delPer=0; zhishiTopicRequestTime=1682042422067; Hm_lvt_55b574651fcae74b0a9f1cf9c8d7c93a=1681288913,1681536767,1681899792,1682042432; ZD_ENTRY=baidu; FPTOKEN=X8QQ3eOYAN3epRQePEAH6CoH4oWh+LCkT955CUjWzH9X0943dtDp9AwAzy4jVXTfAedkKtoVt7ESKtQqSzwCoR//skx/jdjYYMZA/Uo/nkZ6I0HnCiC5AVukb7JfUz3ucQel279PZt/Z1UQW/OaVMpb6QeBKKL3OQAXRDp/lzivbDvl/wnsrE8QLW1MlhROzsh+46ZNAtdKnDABDgUidxRLYzTSM0w+qlZJN6Q6cg+QeqJ5sM3jUoNlSXCwtuyjf0fvftUE0RxHBW8Ty6h2DunVj23uEhYHg2wUsLTSHL9gpSPwAikSrsX5fUGy135Cup8cdAogGwNhRMcIgHgz0w6F6edfLNL+Mn/60BLcAjN2yFrLNmh/JaxmhU0OhwBlczKEnVM7pS5R4FCVnbeRRg7HQWlwLHE2l6z79PtoG+o6/44JNQVVNcMbbzA688Z0y|l0dgcbG9++e/TfCqSLD5TZyyNN8IbscAWvg/jihbBFM=|10|e0cf06c66bc34b281e6c80578db2cf9c; __bid_n=1878366d898e9fc3674207; baikeVisitId=aeed926b-3677-44d0-98e4-3e43bc4945ff; BDUSS=UktRDRFQ3RsOFJVMkRZdXZONW93fnlMWVc0eXo2S1pKMjE0MXRlaUVNelhnV2xrRVFBQUFBJCQAAAAAAAAAAAEAAACJK-DV0rvX1s7ezOIyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANf0QWTX9EFkO; BDUSS_BFESS=UktRDRFQ3RsOFJVMkRZdXZONW93fnlMWVc0eXo2S1pKMjE0MXRlaUVNelhnV2xrRVFBQUFBJCQAAAAAAAAAAAEAAACJK-DV0rvX1s7ezOIyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANf0QWTX9EFkO; X_ST_FLOW=0; ab_sr=1.0.1_ZDM1NTY5NDhkMzgzZjBiZTZmMDZhODIyNWFmZjI1ZGE4MDJhNDQwNmU0YWY3NWIyZGZjMmU1Y2YwMzA5YmUyODgwYjg3MjJhN2M1MDQ1YTI1NDk3NDBlZDA2NjUxZGE4MTdkYjUxNmU0ZWQxMmJkNmI5ZDRmMGY3OTc5NzIxNGFkZjY5Y2I2NzhjYjU3ZDllNWI2ODA4MjcyZDVkODA3NmZjNjFhZDAzMGZmM2Q3NWFkNGVkNTg4Mjc5MzVmN2E5; RT="z=1&dm=baidu.com&si=2acd8285-5178-407f-9171-93ccd19ce80e&ss=lgpwmkud&sl=1n&tt=ztg&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf"; Hm_lpvt_55b574651fcae74b0a9f1cf9c8d7c93a=1682044816',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Host': 'baike.baidu.com'
}

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
    "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (Linux; U; Android 2.3.7; en-us; Nexus One Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10",
    "Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) AppleWebKit/534.13 (KHTML, like Gecko) Version/4.0 Safari/534.13",
    "Mozilla/5.0 (BlackBerry; U; BlackBerry 9800; en) AppleWebKit/534.1+ (KHTML, like Gecko) Version/6.0.0.337 Mobile Safari/534.1+",
    "Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 (KHTML, like Gecko) wOSBrowser/233.70 Safari/534.6 TouchPad/1.0",
    "Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019; Profile/MIDP-2.1 Configuration/CLDC-1.1) AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; HTC; Titan)",
    "UCWEB7.0.2.37/28/999",
    "NOKIA5700/ UCWEB7.0.2.37/28/999",
    "Openwave/ UCWEB7.0.2.37/28/999",
    "Mozilla/4.0 (compatible; MSIE 6.0; ) Opera/UCWEB7.0.2.37/28/999",
    # iPhone 6：
    "Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25",
]

# utils: 汉字转拼音
def to_pinyin(s):
    return ''.join(chain.from_iterable(pinyin(s, style=Style.TONE3)))

def get_proxy():
    # proxyAPI = "http://18966495250.user.xiecaiyun.com/api/proxies?action=getJSON&key=NP5285DBB6&count=&word=&rand=false&norepeat=false&detail=false&ltime=&idshow=false"
    # proxyAPI = "http://proxy.siyetian.com/apis_get.html?token=gHbi1ST6l0MNR0Yy8EVRRzTR1STqFUeNpXQw0kanBjTqFFNPRUV08ERjVzTUlVM.QNxYTO1IzM4YTM&limit=1&type=0&time=&split=1&split_text=&area=0&repeat=0&isp=0"
    proxyAPI = "http://proxy.siyetian.com/apis_get.html?token=AesJWLORVQ41EVJdXTq10dORVQ45kaFhXTR1STqFUeNpXQw0kanBjTqFFNPRUV08ERjVzTUlVM.ANwEzMxkjM4YTM&limit=15&type=0&time=&split=1&split_text=&area=0&repeat=0&isp=0"
    r = requests.get(proxyAPI)
    print(r.text)
    time.sleep(random.random() * random.randint(1, 2))
    ipport_pool = r.text.split('<br />')
    # ipport = res['http'].split('<br />')
    proxies_pool = [{'http':ipport,'https':ipport} for ipport in ipport_pool]
    return proxies_pool
    # ipport = r.text
    # proxies={
    #     'http':ipport,
    #     'https':ipport
    # }
    # return proxies

    # print(r)
    # proxyusernm = "18966495250"  # 代理帐号
    # proxypasswd = "18966495250"
    # if (r.status_code == 200):
    #     # print(r.text)
    #     j = json.loads(r.text)
    #     if (j["success"] and len(j["result"]) > 0):
    #         p=j["result"][0]
    #             # name = input();
    #         proxyurl = "http://" + proxyusernm + ":" + proxypasswd + "@" + p["ip"] + ":" + "%d" % p["port"]
    # return {'http':proxyurl,'https':proxyurl}

ip_queue = mp.Queue()
proxy_pool = get_proxy()
for ip in proxy_pool:
    ip_queue.put(ip)
lock = mp.Lock()

def get_description_and_images(response, proxy, headers):
    html = response.content.decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    
    # headers['user-agent'] = random.choice(USER_AGENTS)
    # proxy = get_proxy()
    res_list = soup.find_all('div', {'class': "lemma-summary J-summary"})
    filtered_text_list = []
    for res in res_list:
        text = res.text.strip()
        filtered_text = re.sub(r'\[\d+\]\s*', '', text)
        filtered_text_list.append(filtered_text)
        # print(filtered_text)
    description = "".join(filtered_text_list).replace("\n", "")
    print(description)
    print(len(description))
    # if len(description) == 0:
    #     print(response.url, response.status_code)
    #     description = response.url
    # 获取概述图url
    concept_img_link = soup.find('div', class_="summary-pic")
    url_list = []
    if concept_img_link:
        href = concept_img_link.a.get("href")
        url = "https://baike.baidu.com" + href
        # print(url)
        # 图像url，多张，去提取这个页面的图像link列表
        response = requests.get(url, headers=headers, proxies=proxy,timeout=5)
        # time.sleep(random.random())
        
        if "anticrawl" in response.url:
            url_list = [url, "failed"]
        else:
            img_html = response.content.decode('utf-8')
            soup3 = BeautifulSoup(img_html, 'html.parser')
            item_list = soup3.find_all('a', class_='pic-item')
            for item in item_list:
                url = item.img.get("src")
                url_list.append(url)
            # print(url_list)
    
    return description, url_list


def grab_description(keyword):
    # 已经下载过的直接返回
    first_pinyin = to_pinyin(keyword)[0].upper()    
    if os.path.exists(os.path.join(ROOT, "{}/{}.json".format(first_pinyin, keyword) )):
        print("{} already finished.".format(keyword))
        return keyword, ["done"], "done", False
        
    try:
        if keyword not in excluded_concepts:
            url = "https://baike.baidu.com/item/" + urllib.parse.quote(keyword) + "?force=1"
        else: # 排除的概念不用爬
            return
        
        print("try to crawl concept {}".format(keyword))
    
        # max_retries = 5
        # retry_count = 1
        # # while True:
        try:
            with lock:
                if ip_queue.empty():
                    print("no ip.")
                    proxy_pool = get_proxy()
                    for ip in proxy_pool:
                        ip_queue.put(ip)
                    print("===========================add new ip===========================")
                    return keyword, ["no ip"], "done", "done" 
                proxy = ip_queue.get()
            headers['user-agent'] = random.choice(USER_AGENTS)        
            # proxy = get_proxy() # 成功获取一次代理后，访问一个概念爬取的全流程
            response = requests.get(url, headers=headers, proxies=proxy, timeout=5)
            # time.sleep(random.random() * random.randint(1, 3))
            time.sleep(random.random())
            if response.status_code == 200 and "anticrawl" not in response.url:
                print("请求成功")
                print(response.url)
                ip_queue.put(proxy)
                # break
            else:
                with lock:
                    proxy_pool.remove(proxy)
                print("请求失败")
        except:
            with lock:
                proxy_pool.remove(proxy)
        
                # if retry_count < max_retries:
                #     print("第{}次请求失败，正在进行第{}次尝试".format(retry_count, retry_count + 1))
                #     time.sleep(random.random() * random.randint(1, 2))
                #     retry_count += 1
                # # elif retry_count == max_retries:
                # #     proxy_pool = get_proxy()
                # else:
                #     # print("请求失败，达到最大重试次数")
                #     print("请求失败，ip池用完")
                #     break
                    # raise
                    # return keyword, "no proxy", "done", "done"
        
        candidate_descriptions, image_urls = [], []
        if "error.html" in response.url: # 概念不存在
            print("no concept {}(status_code:{})".format(keyword, response.status_code))
            description, url_list = "no concept", [url]
            # retry_list.append(keyword)
            candidate_descriptions.append(description)
            image_urls.append(url)
            # return keyword, description, url_list
        elif "anticrawl" in response.url: # 反爬
            print("anticrawl {}(status_code:{})".format(keyword, response.status_code))
            description, url_list = "anticrawl", [url]
            # retry_list.append(keyword)
            candidate_descriptions.append(description)
            image_urls.append(url_list)
            # return keyword, description, url_list
        else: # 成功
            # 解析        
            html = response.content.decode('utf-8')
            soup = BeautifulSoup(html, 'html.parser')
            # print(soup)
            # 抓取多义词项链接
            link_list = []
            sub_lemma_list = soup.find_all('li', class_="list-dot list-dot-paddingleft")
            # print(sub_lemma_list)
            if sub_lemma_list: # 存在多义词项
                for lemma in sub_lemma_list:
                    link_item = lemma.div.a.get("href")
                    link_list.append(link_item)
                # 处理多义词项
                for link in link_list[:3]:
                    url = "https://baike.baidu.com"  + link
                    # print(url)
                    response = requests.get(url, headers=headers, proxies=proxy, timeout=5)
                    # time.sleep(random.random() * random.randint(1, 2))
                    time.sleep(random.random())
                    
                    if "anticrawl" in response.url:
                        print("anticrawl {}(status_code:{})".format(keyword, response.status_code))
                        description, url_list = "anticrawl", []
                        # retry_list.append(keyword)
                    else:
                        description, url_list = get_description_and_images(response, proxy, headers)
                        if len(description) == 0:
                            # retry_list.append(keyword)
                            print(response.url, response.status_code)
                    candidate_descriptions.append(description)
                    image_urls.append(url_list)
            else: # 没有多义词的情况(存在一些情况，有多义词，但是还是进入这个else分支)
                print("no multiple lemma:", keyword)
                # retry_list.append(keyword)
                description, url_list = get_description_and_images(response, proxy, headers)
                if len(description) == 0:
                    # retry_list.append(keyword)
                    print(response.url, response.status_code)
                candidate_descriptions.append(description)
                image_urls.append(url_list)
            
        # 如果有返回，直接保存
        failed = False
        for desc in candidate_descriptions:
            if len(desc) == 0 or desc == "anticrawl" or desc == "no concept":
                print("desc:", desc)
                failed = True
                break

        if failed == True:
            print("concept {} failed.".format(keyword))
        else:
            save_dict = {}
            save_dict["concept"] = keyword
            save_dict["candidate_descriptions"] = candidate_descriptions
            save_dict["image_urls"] = image_urls

            # 保存
            first_pinyin = to_pinyin(keyword)[0].upper()
            first_path = os.path.join(ROOT, first_pinyin) # 首字母文件夹路径
            # 创建首字母文件夹
            if not os.path.exists(first_path): 
                os.makedirs(first_path)
            with open(os.path.join(ROOT, "{}/{}.json".format(first_pinyin, keyword)), 'w', encoding='utf-8') as f:
                json.dump(save_dict, f, ensure_ascii=False, indent=4)
            
            print("concept {} saved.".format(keyword))
                
        return keyword, candidate_descriptions, image_urls, failed   
    except Exception as e:
        print("Unknown error of {}:".format(keyword), e)
        return keyword, ["unknown error"], [], True
    
if __name__ == "__main__":
    
    start_time = time.time()
    # 1 加载概念搜索词
    candidate_concepts_path = "./projects/m2conceptbase/utils/word_freq_train_data_230331.json"
    with open(candidate_concepts_path, 'r', encoding='utf-8') as f:
        cand_concepts_dict = json.load(f)
    # cand_concepts_dict = {
    #     "人名": 7012265,
    #     "中国": 2521113,
    #     "图片": 1271515,
    #     "企业": 851620,
    #     "技术": 785123,
    #     "世界": 775091,
    #     "工作": 754635,
    #     "项目": 725662,
    #     "活动": 721523
    # }
    keyword_list = [keyword for keyword, count in cand_concepts_dict.items() if count >= 9]
    print(len(keyword_list))

    
    
    # filename = "concept_cand_descriptions_img_urls_7.json"
    p = mp.Pool(10)
    print("process pool produced.")
    begin_idx, end_idx = 0, 100000 # round 2
    # begin_idx, end_idx = 862027, 862081 # round 2
    
    # 二次爬取
    # with open(ROOT + "retry_list_{}_{}.json".format(begin_idx, end_idx), 'r', encoding='utf-8') as f:
    #     retry_list = json.load(f)
    #     results =  p.map(grab_description, keyword_list[begin_idx: end_idx])
    

    results =  p.map(grab_description, keyword_list[begin_idx: end_idx])
    
    # results =  p.map(grab_description, keyword_list[begin_idx: end_idx])
    p.close()
    retry_list = [line[0] for line in results if line != None if line[-1] == True]
    error_list = [line[0] for line in results if line != None if line[1][0] == "no concept"]
    print("retry_list", retry_list)
    print("error_list:", error_list)
    with open(ROOT + "retry_list_{}_{}.json".format(begin_idx, end_idx), 'w', encoding='utf-8') as f:
        json.dump(retry_list, f, ensure_ascii=False, indent=4)
    with open(ROOT + "error_list{}_{}.json".format(begin_idx, end_idx), 'w', encoding='utf-8') as f:
        json.dump(error_list, f, ensure_ascii=False, indent=4)
    print("{} concepts failed. Try again for a moment.".format(len(retry_list)))
    print("{} concepts may not exist. Try again for a moment.".format(len(error_list)))
    
    print("All {} concepts done, {} succeed.(total {})".format(len([_ for _ in results if _ != None]), len([line for line in results if line != None if line[-1] == False]), len(results)))
    print("Used time: {}min".format((time.time() - start_time) / 60))
    # for result in tqdm(results, total=len(results)):
    #     # print(row)
    #     if result is None: 
    #         continue
        
    #     keyword, candidate_descriptions, image_urls = result
        # if candidate_descriptions != "done":
        #     save_dict = {}
        #     save_dict["concept"] = keyword
        #     save_dict["candidate_descriptions"] = candidate_descriptions
        #     save_dict["image_urls"] = image_urls

        #     # 保存
        #     first_pinyin = to_pinyin(keyword)[0].upper()
        #     first_path = os.path.join(ROOT, first_pinyin) # 首字母文件夹路径
        #     # 创建首字母文件夹
        #     if not os.path.exists(first_path): 
        #         os.makedirs(first_path)
        #     with open(os.path.join(ROOT, "{}/{}.json".format(first_pinyin, keyword)), 'w', encoding='utf-8') as f:
        #         json.dump(save_dict, f, ensure_ascii=False, indent=4)
        # else:
        #     print("Finished", keyword)
    # print("All done.")