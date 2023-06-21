import numpy as np
from sklearn.preprocessing import MinMaxScaler

# サンプルバッチデータ
batch_data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# MinMaxScalerのインスタンスを作成
scaler = MinMaxScaler()

# バッチデータに対してMinMaxScalerを適用
print(batch_data.T)
normalized_batch_data = scaler.fit_transform(batch_data.T).T

print(normalized_batch_data)