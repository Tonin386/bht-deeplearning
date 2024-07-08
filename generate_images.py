from PIL import Image
from tqdm import tqdm
import random
import os

batches = os.listdir("Data/Raw")

size = 64
x = 0
y = 0
im_num = 0

def generating_waldos(use_background=True):
    background_use_num = 28
    head_use_num = 500
    im_num = 0
    for i in tqdm(range(1, background_use_num+1)):
        for head_name in os.listdir("Data/Clean/OnlyWaldoHeads"):
            for _ in range(head_use_num):
                if random.randint(0, 9) < 5:
                    num = random.randint(-15, 15)
                    foreground = Image.open(
                        "Data/Clean/OnlyWaldoHeads/" + head_name).rotate(
                        num)
                else:
                    foreground = Image.open("Data/Clean/OnlyWaldoHeads/" + head_name)

                if random.randint(0, 9) < 7:
                    scale = random.uniform(1.2, 1.8)
                    w, h = foreground.size
                    foreground = foreground.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                if use_background:
                    background = Image.open("Data/Clean/ClearedWaldos/" + str(i) + ".jpg")
                else:
                    background = Image.open("Data/Clean/black_background.jpg")

                bck_w, bck_h = background.size
                frg_w, frg_h = foreground.size

                bck_x = random.randint(0, bck_w - size)
                bck_y = random.randint(0, bck_h - size)
                
                frg_x = random.randint(0, 64 - frg_w)
                frg_y = random.randint(0, 64 - frg_h)

                cropped = background.crop((bck_x, bck_y, bck_x + size, bck_y + size))
                cropped.save("Data/NotWaldo/n" + str(im_num) + ".png")
                cropped.paste(foreground, (frg_x, frg_y), foreground)
                cropped.save("Data/Waldo/" + str(im_num) + ".png")
                im_num += 1

generating_waldos(use_background=True)