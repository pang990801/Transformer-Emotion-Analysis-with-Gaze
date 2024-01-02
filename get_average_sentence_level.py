import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 文件夹路径
folder_path = 'et_csv_data'

# 读取所有 CSV 文件
dataframes = []
for i in range(0, 12):
    file_path = os.path.join(folder_path, f"{i + 1}_SR.csv")
    df = pd.read_csv(file_path)
    # 选择除第 0 列以外的所有列
    columns_except_first = df.columns[1:]
    # 只将这些列中的 0 值替换为 NaN
    df[columns_except_first] = df[columns_except_first].replace(0, np.nan)
    dataframes.append(df)

# 计算平均值
average_df = pd.concat(dataframes).groupby(level=0).mean()

# 将索引转换为 'id' 列，并转换为整数
average_df.reset_index(inplace=True)
average_df['id'] = average_df['id'].astype(int)

# 分离 'id' 列
id_column = average_df['id']
average_df_no_id = average_df.drop(columns=['id'])

# 确保 id_column 是一维的
id_column = id_column.values

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
scaled_data_min_max = min_max_scaler.fit_transform(average_df_no_id)
scaled_average_df_min_max = pd.DataFrame(scaled_data_min_max, columns=average_df_no_id.columns)

# 将 'id' 列设置为索引
scaled_average_df_min_max['id'] = id_column
scaled_average_df_min_max.set_index('id', inplace=True)

# 保存 Min-Max Scaling 结果
scaled_average_df_min_max.to_csv(os.path.join(folder_path, 'min_max_scaled_average_data.csv'))

# Standard Scaling
standard_scaler = StandardScaler()
scaled_data_standard = standard_scaler.fit_transform(average_df_no_id)
scaled_average_df_standard = pd.DataFrame(scaled_data_standard, columns=average_df_no_id.columns)

# 将 'id' 列设置为索引
scaled_average_df_standard['id'] = id_column
scaled_average_df_standard.set_index('id', inplace=True)

# 保存 Standard Scaling 结果
scaled_average_df_standard.to_csv(os.path.join(folder_path, 'standard_scaled_average_data.csv'))