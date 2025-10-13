import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

print("--- 开始执行模型训练脚本 (V3: XGBoost + 手动调整权重) ---")

# --- 1. 加载预处理好的数据 ---
try:
    df = pd.read_csv('wine_data_processed_with_tfidf.csv')
    print(f"成功加载预处理数据集，形状: {df.shape}")
except FileNotFoundError:
    print("错误: 'wine_data_processed_with_tfidf.csv' 文件未找到。请先运行data_preprocessing.py脚本。")
    exit()

# --- 2. 创建价格区间 ---
price_bins = [0, 20, 40, 80, 200, float('inf')]
price_labels = ['1. Entry-level (<= $20)', '2. Quality ($21-$40)', '3. Premium ($41-$80)', '4. Luxury ($81-$200)', '5. Collectible (> $200)']
df['price_range'] = pd.cut(df['price'], bins=price_bins, labels=price_labels, right=True)

print("\n已将'price'转换为分类的'price_range':")
print(df['price_range'].value_counts().sort_index())

# --- 3. 定义特征(X)和目标(y) ---
X = df.drop(['price', 'price_range'], axis=1)
y = df['price_range']
categorical_features = ['country', 'variety_simplified', 'winery_simplified']

# --- 4. 分割数据 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n数据已分割: 训练集 {X_train.shape[0]} 行, 测试集 {X_test.shape[0]} 行。")

# --- 5. 独热编码 ---
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
    remainder='passthrough'
)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"独热编码完成！处理后的训练集特征形状: {X_train_processed.shape}")

# --- 6. 解决数据不均衡问题 (手动调整权重) ---
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# **V3 改进: 手动设置权重，而不是完全自动计算**
# 目标：保持对稀有类别的关注，但减轻惩罚力度，寻求更好的平衡。
manual_weights = {
    '1. Entry-level (<= $20)': 0.6,
    '2. Quality ($21-$40)': 0.7,
    '3. Premium ($41-$80)': 1.0,
    '4. Luxury ($81-$200)': 5.0,     # 之前是4.26，略微提高
    '5. Collectible (> $200)': 20.0   # 之前是35.45，大幅降低
}
# 将文本标签的权重字典，转换为XGBoost需要的数字标签格式
d_class_weights = {i: manual_weights[le.classes_[i]] for i in range(len(le.classes_))}

print("\n手动设置的类别权重，寻求更好的平衡:")
print({le.classes_[i]: weight for i, weight in d_class_weights.items()})

# --- 7. 训练XGBoost模型 ---
print("\n--- 正在训练一个XGBoost分类模型 ---")
sample_weights = np.array([d_class_weights[label] for label in y_train_encoded])

model = xgb.XGBClassifier(
    objective='multi:softmax',
    n_estimators=150,
    learning_rate=0.1,
    max_depth=7,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_processed, y_train_encoded, sample_weight=sample_weights)
print("模型训练完成。")

# --- 8. 在测试集上进行预测和评估 ---
print("\n--- 正在测试集上进行评估 ---")
predictions_encoded = model.predict(X_test_processed)
predictions = le.inverse_transform(predictions_encoded)

accuracy = accuracy_score(y_test, predictions)
print(f"模型在测试集上的整体准确率: {accuracy:.2%}")
print("\n--- 各价格区间的详细评估报告 (XGBoost with Manual Weights) ---")
print(classification_report(y_test, predictions, target_names=price_labels))

# --- 9. 保存模型和预处理器 ---
print("\n--- 正在保存模型和预处理器 ---")
joblib.dump(model, 'wine_price_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(le, 'label_encoder.joblib')
print("模型、数据预处理器和标签编码器已成功保存！")

