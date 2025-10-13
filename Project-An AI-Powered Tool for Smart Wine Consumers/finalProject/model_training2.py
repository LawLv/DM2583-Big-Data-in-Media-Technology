import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("--- 开始执行模型训练脚本 (V6: 修复内存问题 - 专注GPU) ---")

# --- 1. 加载预处理好的数据 ---
try:
    df = pd.read_csv('wine_data_processed_with_tfidf.csv')
    print(f"成功加载预处理数据集，总行数: {df.shape[0]}")
except FileNotFoundError:
    print("错误: 'wine_data_processed_with_tfidf.csv' 文件未找到。请先运行data_preprocessing.py脚本。")
    exit()

# --- 2. 筛选数据，专注主流消费区间 ---
df_focused = df[df['price'] <= 80].copy()
print(f"筛选后，专注于价格 <= $80 的数据，剩余行数: {df_focused.shape[0]}")


# --- 3. 创建新的、简化的价格区间 ---
price_bins = [0, 20, 40, 80]
price_labels = ['1. 入门级 (<= $20)', '2. 品质之选 ($21-$40)', '3. 优质佳酿 ($41-$80)']
df_focused['price_range'] = pd.cut(df_focused['price'], bins=price_bins, labels=price_labels, right=True)

print("\n已将'price'转换为新的3个分类区间:")
print(df_focused['price_range'].value_counts().sort_index())

# --- 4. 定义特征(X)和目标(y) ---
X = df_focused.drop(['price', 'price_range'], axis=1)
y = df_focused['price_range']
categorical_features = ['country', 'variety_simplified', 'winery_simplified']

# --- 5. 分割数据 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n数据已分割: 训练集 {X_train.shape[0]} 行, 测试集 {X_test.shape[0]} 行。")

# --- 6. 独热编码 ---
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
    remainder='passthrough'
)
# **内存优化**: 转换为 float32 将内存使用量减半
X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
X_test_processed = preprocessor.transform(X_test).astype(np.float32)
print(f"独热编码完成！处理后的训练集特征形状: {X_train_processed.shape}")

# --- 7. 编码目标变量 ---
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# --- 8. V6 核心改进: 超参数调优 ---
print("\n--- 正在进行超参数调优 ---")

# 定义要搜索的参数范围
param_dist = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [5, 7, 9],
    'colsample_bytree': [0.7, 0.8, 0.9], # 每棵树用的特征比例
    'subsample': [0.7, 0.8, 0.9]       # 每棵树用的数据比例
}

# 初始化XGBoost分类器
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax',
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    # **GPU 选项**: 启用CUDA加速
    device='cuda',
)

# 使用随机搜索进行调优
random_search = RandomizedSearchCV(
    xgb_classifier,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    verbose=2,
    random_state=42,
    # **内存修复**: 将 n_jobs 设置为 1，关闭CPU并行，避免内存错误
    n_jobs=1
)

# 在训练数据上执行搜索
random_search.fit(X_train_processed, y_train_encoded)

print("\n超参数调优完成！")
print(f"找到的最佳参数是: {random_search.best_params_}")

# 获取并使用最佳模型
best_model = random_search.best_estimator_

# --- 9. 在测试集上进行预测和评估 ---
print("\n--- 正在使用最佳模型进行评估 ---")
predictions_encoded = best_model.predict(X_test_processed)
predictions = le.inverse_transform(predictions_encoded)

accuracy = accuracy_score(y_test, predictions)
print(f"调优后模型在测试集上的整体准确率: {accuracy:.2%}")
print("\n--- 各价格区间的详细评估报告 (调优后模型) ---")
print(classification_report(y_test, predictions, target_names=price_labels))

# --- 10. 保存最佳模型和预处理器 ---
print("\n--- 正在保存最佳模型和预处理器 ---")
joblib.dump(best_model, 'wine_price_model_tuned.joblib')
joblib.dump(preprocessor, 'preprocessor_focused.joblib') # 预处理器不变
joblib.dump(le, 'label_encoder_focused.joblib')       # 编码器不变
print("调优后的最佳模型、预处理器和编码器已成功保存！")

