# 参考 Bankruptcy prediction using imaged financial ratios and convolutional neural networks 论文，建立公司特征图片
# https://www.rs.tus.ac.jp/hosaka-t/img/file1.pdf
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
seed = 12345
random.seed(seed)
wide = 10

# 计算对调位置后的损失差距
def energy(pic, cor, num1, num2):
    e_diff = 0
    wide = 10
    for i, x in enumerate(pic):
        dis1 = (num1//wide - i//wide)**2 + (num1%wide - i%wide)**2
        dis2 = (num2 // wide - i // wide) ** 2 + (num2 % wide - i % wide) ** 2

        e_diff += abs(min(cor.get(x + 'COR' + pic[num1], 1), cor.get(pic[num1] + 'COR' + x, 1))) * (dis1 - dis2)
        e_diff += abs(min(cor.get(x + 'COR' + pic[num2], 1), cor.get(pic[num2] + 'COR' + x, 1))) * (dis2 - dis1)
    if e_diff >= 0:
        return True
    else:
        return False


def get_ratio_data():
    # 使用2009的财报
    date = ['20090331','20090630','20090930','20091231']
    fsa_Z = {}
    df = pd.read_csv('fsa_chinese_Z/600006.SH.csv')
    columns = list(df.columns)
    col = [x for x in columns if '_Z' in x or '/' in x]
    for x in col:
        fsa_Z[x] = []

    # 取出所有股票的财务比率Z值
    for root, dirs, files in os.walk('fsa_chinese_Z'):
        for f in files:
            df = pd.read_csv('fsa_chinese_Z/' + f)
            df = df.loc[df['end_date'] <= 20091231]
            df = df.loc[:,['end_date'] + col]
            df = df.replace(np.nan, 0)
            fsa = sorted(df.values.tolist())
            fsa_dict = {}
            for x in fsa:
                fsa_dict[str(int(x[0]))] = x[1:]
            for d in date:
                if d in fsa_dict:
                    for i,x in enumerate(col):
                        fsa_Z[x].append(fsa_dict[d][i])
                else:
                    for i,x in enumerate(col):
                        fsa_Z[x].append(0)

    # 计算cor
    cor = {}  # factor*factor: cor
    cor_pair = []
    for i1,x1 in enumerate(col):
        for x2 in col[i1+1:]:
            cor[x1+'COR'+x2] = stats.pearsonr(fsa_Z[x1], fsa_Z[x2])[0]
            cor_pair.append(x1+'COR'+x2)

    # 初始化每个比率在图中位置
    pic = []
    for x in col:
        pic.append(x)
    random.shuffle(pic)

    # 使energy尽可能小(让高相关性的距离近)
    count = 0
    iter = 0
    while count < 3*len(col):
        iter += 1
        num1 = random.randint(0, len(col)-1)
        num2 = num1 + random.randint(1, len(col)-1)
        if num2 >= len(col): num2 -= len(col)
        if energy(pic, cor, num1, num2):
            count = 0
            pic[num1], pic[num2] = pic[num2], pic[num1]
        else:
            count += 1
        if iter%1000 == 0:
            print(iter)

    with open('company_pic_10.txt', 'w') as f:
        f.write(','.join(pic))


def show_pic(wide):
    with open('company_pic_'+str(wide)+'.txt', 'r') as f:
        text = f.read().split(',')
        for x in range(wide):
            print(text[wide*x:wide*(x+1)])


def generate_pic(pic):
    # 找出对应位置的ratio
    with open(pic, 'r') as f:
        pic_ratio = f.read().split(',')


    # 取得日期
    quarter = []
    for root, dirs, files in os.walk('company_pic'):
        if '\\' in root:
            quarter.append(root.split('\\')[1])

    # 每一家公司生成图片
    for root, dirs, files in os.walk('fsa_chinese_Z'):
        # company
        for f in files:
            df = pd.read_csv('fsa_chinese_Z/' + f)
            df = df.replace(np.nan, 0)
            df = df.set_index('end_date')
            company_pic = [127]*wide**2
            # quarter
            for q in quarter:
                for i, x in enumerate(pic_ratio):
                    try:
                        company_pic[i] = int((float(df.at[int(q), x])+3) * 255 / 6)
                    except:
                        # 避免日期缺失
                        print(f, q)
                        company_pic[i] = 127
                np_pic = np.array(company_pic)
                np_pic = np_pic.reshape(wide, wide)
                image = Image.fromarray(np_pic)
                image = image.convert('L')
                image.save('company_pic/' + q + '/' + f.split('.')[0] + '.jpg')


if __name__ == '__main__':
    # get_ratio_data()
    # show_pic(10)
    generate_pic('company_pic_10.txt')



