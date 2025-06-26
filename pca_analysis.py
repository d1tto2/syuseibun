import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_pca_analysis(filepath, columns):
    """主成分分析を実行し、結果を表示・保存する関数"""

    # データの読み込み
    df1 = pd.read_excel(filepath)

    # データの抽出と標準化
    D = df1[columns].values
    mean = D.mean(axis=0)
    std = D.std(axis=0, ddof=1)
    SD = (D - mean) / std

    # 相関行列の計算と固有値分解
    R = np.cov(SD, rowvar=False)
    w, v = np.linalg.eig(R)

    # 固有値と固有ベクトルを降順にソート
    sort_index = np.argsort(w)[::-1]
    sort_w = w[sort_index]
    sort_v = v[:, sort_index]

    # (中略... 因子負荷量、主成分得点の計算など、これまでのロジックはすべてこのインデント内に収める)
    # ...
    # ...

    # 主成分スコアをグラフで可視化
    plt.figure(figsize=(10, 8))
    # (中略... グラフ描画のコード)
    plt.savefig("pca_scores_scatter_plot.png")

    print("\n主成分分析が完了し、散布図を 'pca_scores_scatter_plot.png' として保存しました。")


# --- メインの実行部分 ---
if __name__ == '__main__':
    # 設定
    excel_file = "lec11_data_nagoya.xlsx"
    variables_to_analyze = ["人口総数", "15歳未満人口", "15～64歳人口", "65歳以上人口", "外国人人口", "昼間人口"]

    # 関数の実行
    run_pca_analysis(excel_file, variables_to_analyze)