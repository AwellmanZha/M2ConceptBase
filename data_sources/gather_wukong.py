import requests
import os
import multiprocessing as mp
from io import BytesIO
import numpy as np
import PIL
from PIL import Image
import pickle
import sys
import pandas as pd
from tqdm import tqdm
import json
import time
import random

def grab(line):
    """
    Download a single image from the CSV.
    """
    uid, split, line = line
    try:
        url = line['url']
        caption = line['caption']
        caption = caption.replace('\n', ' ')
    except:
        print("Parse error")
        return

    if os.path.exists(ROOT+"/%s/%d/%d.jpg"%(split,uid%1000,uid)):
        print("Finished", uid)
        return uid, caption, url

    # Let's not crash if anythign weird happens
    try:
        # headers = {
        #     'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'
        # }
        dat = requests.get(url, timeout=20)
        # time.sleep(random.random() * 0.2) # 40w * 0.1 = 4w s = 11 h
        if dat.status_code != 200:
            # print("404 file", url)
            return

        # Try to parse this as an Image file, we'll fail out if not
        im = Image.open(BytesIO(dat.content))
        
        # image format covertion
        if im.mode == 'RGBA': 
            im = im.convert('RGB')
        
        im.thumbnail((512, 512), PIL.Image.BICUBIC)
        if min(*im.size) < max(*im.size)/3 or min(*im.size) < 10:
            # print("Too small", url)
            return

        im.save(ROOT+"/%s/%d/%d.jpg"%(split,uid%1000,uid))

        # Another try/catch just because sometimes saving and re-loading
        # the image is different than loading it once.
        try:
            o = Image.open(ROOT+"/%s/%d/%d.jpg"%(split,uid%1000,uid))
            o = np.array(o)

            print("Success", o.shape, uid, url)
            return uid, caption, url
        except:
            print("Failed", uid, url)
            
    except Exception as e:
        print("Unknown error", e)
        pass

if __name__ == "__main__":
    # ROOT = "wukong_100m_0"
    
    csv_file_path = "./data_122/wukong_release/wukong_100m_{}.csv"
    file_idx_list = [57, 58, 59, 60]
    
    # for csv_idx in sys.argv[1:]:    
    for csv_idx in file_idx_list:
        ROOT = "./data_122/wukong_train/wukong_100m_{}".format(csv_idx)
        p = mp.Pool(300)
        print("process pool produced.")
        # url2img = {}
        
        if not os.path.exists(ROOT):
            os.mkdir(ROOT)
            os.mkdir(os.path.join(ROOT, "all"))
        for i in range(1000):
            if not os.path.exists(os.path.join(ROOT, "all", str(i))):
                os.mkdir(os.path.join(ROOT, "all", str(i)))
        
        csv_file = csv_file_path.format(csv_idx)
        print("Processing file", csv_file)
        split = 'all'
        csv_data = pd.read_csv(csv_file)
        results =  p.map(grab, [(i, split, line) for i, line in csv_data.iterrows()])
        
        out = open(csv_file.replace(".csv","_output.csv"), "w")
        url2img_out = open(csv_file.replace(".csv","_url2img.csv"), "w")
        out.write("id\tcaption\timage\n")
        url2img_out.write("url\timage\n")
        
        print("saving id, caption, image name ...")
        item_index = 0
        for row in tqdm(results, total=len(results)):
            # print(row)
            if row is None: 
                continue
            id, caption, url = row
            filepath = os.path.join(ROOT, split, str(id % 1000), str(id) + ".jpg")
            if os.path.exists(filepath):
                out.write("%s\t%s\t%s\n"%(str(item_index), caption, filepath))
                url2img_out.write("%s\t%s\n"%(str(url), filepath))
                # url2img[str(url)] = filepath
                item_index += 1
            else:
                print("Drop", id)
        out.close()
        url2img_out.close()
        # with open(csv_file.replace(".csv","_url2img.json"), 'w', encoding='utf-8') as f:
        #     json.dump(url2img, f, ensure_ascii=False, indent=4)
        p.close()
        print("File {} saved.".format(csv_idx))
