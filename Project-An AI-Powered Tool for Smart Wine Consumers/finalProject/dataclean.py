import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

print("--- 开始执行数据清洗与特征工程脚本 ---")

# --- 1. 数据加载 ---
try:
    # 加载数据集
    df = pd.read_csv('winemag-data-130k-v2.csv')
    print(f"成功加载数据集，原始形状: {df.shape}")
except FileNotFoundError:
    print("错误: 'winemag-data-130k-v2.csv' 文件未找到。请确保文件在正确的目录下。")
    exit()

# --- 2. 初始数据清洗 ---
print("\n--- 阶段一：初始数据清洗 ---")

# 删除不必要的列
# 'Unnamed: 0' 是旧的索引
# 'designation', 'region_2', 'taster_twitter_handle' 缺失值过多或信息冗余
cols_to_drop = ['Unnamed: 0', 'designation', 'region_2', 'taster_twitter_handle']
df.drop(columns=cols_to_drop, inplace=True)
print(f"删除了列: {cols_to_drop}")

# 处理目标变量 'price'，删除价格缺失的行
initial_rows = len(df)
df.dropna(subset=['price'], inplace=True)
print(f"处理了目标变量 'price'，删除了 {initial_rows - len(df)} 行价格缺失的数据。")

# 为关键分类特征填充缺失值
for col in ['country', 'province', 'region_1', 'variety', 'taster_name', 'winery']:
    df[col].fillna('Unknown', inplace=True)
print("为关键分类特征填充了缺失值。")


# --- 3. 特征工程 ---
print("\n--- 阶段二：特征工程 ---")

# (A) 从 'title' 中提取年份
def extract_year(text):
    """使用正则表达式从文本中提取四位数年份"""
    match = re.search(r'\b(19|20)\d{2}\b', str(text))
    if match:
        return int(match.group(0))
    return np.nan

df['year'] = df['title'].apply(extract_year)

# 用所有年份的中位数填充提取失败的年份
median_year = df['year'].median()
df['year'].fillna(median_year, inplace=True)
df['year'] = df['year'].astype(int)
print(f"从 'title' 提取了年份，并用中位数 {int(median_year)} 填充了缺失值。")
print(df[['title', 'year']].head())


# (B) 简化高基数分类特征
def simplify_categorical_feature(df, column_name, top_n=30):
    """保留前N个最常见的类别，其余归为'Other'"""
    top_categories = df[column_name].value_counts().nlargest(top_n).index
    new_column_name = f"{column_name}_simplified"
    df[new_column_name] = df[column_name].where(df[column_name].isin(top_categories), 'Other')
    print(f"简化了 '{column_name}' 特征，保留了Top {top_n}，其余归为 'Other'。")
    return df

df = simplify_categorical_feature(df, 'variety', top_n=30)
df = simplify_categorical_feature(df, 'winery', top_n=50) # 酒庄数量更多，保留top 50


# --- 4. 文本量化 (TF-IDF) ---
print("\n--- 阶段三：使用TF-IDF进行文本量化 ---")

# 确保 'description' 列没有缺失值
df['description'].fillna('', inplace=True)

# 初始化TfidfVectorizer
# - stop_words='english': 移除常见的英文停用词 (如 'the', 'a', 'is')
# - ngram_range=(1, 2): 同时考虑单个词和两个词构成的词组
# - max_features=5000: 只保留最重要的5000个词/词组作为特征
# - min_df=5: 忽略在少于5个文档中出现的词
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000,
    min_df=5
)

# 拟合并转换 'description' 列
tfidf_matrix = tfidf.fit_transform(df['description'])
print(f"TF-IDF矩阵已生成，形状: {tfidf_matrix.shape}")

# 将稀疏矩阵转换为DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
# 为避免列名冲突，给TF-IDF的列名加上前缀
tfidf_df = tfidf_df.add_prefix('tfidf_')

# 重置原始DataFrame的索引，以便安全合并
df.reset_index(drop=True, inplace=True)

# 将TF-IDF特征与原始数据合并
df_final = pd.concat([df, tfidf_df], axis=1)
print(f"合并TF-IDF特征后的最终数据集形状: {df_final.shape}")


# --- 5. 最后清理与保存 ---
print("\n--- 阶段四：最后清理与保存 ---")

# 删除原始的、已被处理的列
df_final.drop(columns=['description', 'title', 'variety', 'winery', 'region_1', 'province', 'taster_name'], inplace=True)
print("删除了原始的文本列和高基数列。")

# 保存处理后的数据
output_filename = 'wine_data_processed_with_tfidf.csv'
df_final.to_csv(output_filename, index=False)

print(f"\n处理完成！最终数据集形状: {df_final.shape}")
print(f"已将清洗和特征工程后的数据保存至 '{output_filename}'")
print("\n最终数据集的前5行预览:")
print(df_final.head())
