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

# --- ここから追記 ---
# 相関行列の計算 (標準化データからなので共分散行列が相関行列になる)
R = np.cov(SD, rowvar=False)
print("R = ", R)

# 固有値と固有ベクトルを計算
w, v = np.linalg.eig(R)

print("相関行列の計算と固有値分解が完了しました。")

# --- ここから追記 ---
# 相関行列の計算 (標準化データからなので共分散行列が相関行列になる)
R = np.cov(SD, rowvar=False)
print("R = ", R)

# 固有値と固有ベクトルを計算
w, v = np.linalg.eig(R)

print("相関行列の計算と固有値分解が完了しました。")

# --- ここから追記 ---
# 固有値と固有ベクトルを降順にソート
sort_index = np.argsort(w)[::-1]
sort_w = w[sort_index]
sort_v = v[:, sort_index]
print("sort v = ", sort_v)

# 寄与率と累積寄与率の計算
contribution_rate = sort_w / np.sum(sort_w)
cumulative_contribution_rate = np.cumsum(contribution_rate)

# 結果をDataFrameで分かりやすく表示
pca_results = pd.DataFrame({
    '固有値': sort_w,
    '寄与率': contribution_rate,
    '累積寄与率': cumulative_contribution_rate
})
pca_results.index = [f'第{i+1}主成分' for i in range(len(sort_w))]

# 因子負荷量の計算
factor_loadings = sort_v * np.sqrt(sort_w)
factor_loadings_df = pd.DataFrame(factor_loadings,
                                  index=variables,
                                  columns=pca_results.index)

# 計算結果の表示
print("\n--- 固有値と寄与率 ---")
print(pca_results)
print("\n--- 因子負荷量 ---")
print(factor_loadings_df)