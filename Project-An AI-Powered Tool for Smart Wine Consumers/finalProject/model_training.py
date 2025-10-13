import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("--- 开始执行模型训练脚本 (分类任务) ---")

# --- 1. 加载预处理好的数据 ---
try:
    df = pd.read_csv('wine_data_processed_with_tfidf.csv')
    print(f"成功加载预处理数据集，形状: {df.shape}")
except FileNotFoundError:
    print("错误: 'wine_data_processed_with_tfidf.csv' 文件未找到。请先运行data_preprocessing.py脚本。")
    exit()

# --- 2. 创建价格区间 (将回归问题转为分类问题) ---
price_bins = [0, 20, 40, 80, 200, float('inf')]
price_labels = ['1. Entry-level (<= $20)', '2. Quality ($21-$40)', '3. Premium ($41-$80)', '4. Luxury ($81-$200)', '5. Collectible (> $200)']

df['price_range'] = pd.cut(df['price'], bins=price_bins, labels=price_labels, right=True)

print("\n已将'price'转换为分类的'price_range':")
print(df['price_range'].value_counts().sort_index())

# --- 3. 定义特征(X)和新的目标(y) ---
# 'price' 和 'price_range' 都不应作为特征
X = df.drop(['price', 'price_range'], axis=1)
y = df['price_range']

# 识别出需要进行独热编码的分类特征列
categorical_features = ['country', 'variety_simplified', 'winery_simplified']

# --- 4. 分割数据为训练集和测试集 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # stratify=y 确保训练集和测试集中各价格区间的比例相同
)
print(f"\n数据已分割: 训练集 {X_train.shape[0]} 行, 测试集 {X_test.shape[0]} 行。")

# --- 5. 创建并应用独热编码器 ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

print("\n正在对训练数据拟合预处理器 (OneHotEncoder)...")
X_train_processed = preprocessor.fit_transform(X_train)
print("正在转换训练集和测试集...")
X_test_processed = preprocessor.transform(X_test)

print(f"独热编码和处理完成！")
print(f"处理后的训练集特征形状: {X_train_processed.shape}")

# --- 6. 训练随机森林分类模型 ---
print("\n--- 正在训练一个随机森林分类模型 ---")
# n_jobs=-1 使用所有可用的CPU核心来加速训练
# verbose=1 会在训练时打印详细进度
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
model.fit(X_train_processed, y_train)
print("模型训练完成。")

# --- 7. 在测试集上进行预测和评估 ---
print("\n--- 正在测试集上进行评估 ---")
predictions = model.predict(X_test_processed)

# 计算并打印准确率
accuracy = accuracy_score(y_test, predictions)
print(f"模型在测试集上的整体准确率: {accuracy:.2%}")
print("\n这意味着模型能够正确预测约 {:.0f}% 的葡萄酒所属的价格区间。".format(accuracy * 100))

# 打印详细的分类报告
print("\n--- 各价格区间的详细评估报告 ---")
# precision: 预测为该类别的样本中，实际为该类别的比例 (查准率)
# recall: 该类别的所有样本中，被模型成功预测对的比例 (查全率)
# f1-score: precision和recall的调和平均数
print(classification_report(y_test, predictions, target_names=price_labels))