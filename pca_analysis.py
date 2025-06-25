import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
df1 = pd.read_excel("lec11_data_nagoya.xlsx")

# 分析に使用する変数を定義
variables = ["人口総数", "15歳未満人口", "15～64歳人口", "65歳以上人口","外国人人口","昼間人口"]

# データの抽出と標準化
D = df1[variables].values
mean = D.mean(axis=0)
std = D.std(axis = 0,ddof = 1)
SD = (D - mean) / std

print("データの標準化が完了しました。")