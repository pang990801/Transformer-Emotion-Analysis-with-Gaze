from utils_ZuCo import *
import os

datatransform_t1 = DataTransformer('task1', level='sentence', scaling='raw', fillna='zeros')

# 处理并保存每个受试者的数据
sbjs_t1 = []
for i in range(12):
    # 转换受试者数据
    sbj_data = datatransform_t1(i)

    # 将索引转换为列，并命名为 'id'
    sbj_data.reset_index(inplace=True)
    sbj_data.rename(columns={'index': 'id'}, inplace=True)

    sbjs_t1.append(sbj_data)

    # 生成文件名
    filename = f"{i+1}_SR.csv"
    # 定义保存路径
    path = os.path.join("et_csv_data", filename)

    # 显示受试者 1 的前几行数据（仅对第一个受试者执行）
    if i == 0:
        print(sbj_data.head())

    # 保存到 CSV 文件
    sbj_data.to_csv(path, index=False)  # 确保不将索引（现在是 'id' 列）再次作为索引保存

# 此处可以访问 sbjs_t1 列表中的 DataFrame