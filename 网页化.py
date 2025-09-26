'''
streamlit==1.30.0
joblib==1.4.2
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.0
scikit-learn==1.5.1
shap==0.45.1
'''
import streamlit as st
import joblib
import numpy as np
print(np.__version__)
import pandas as pd
print(pd.__version__)
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('GNB_RF【final_model】.pkl')
# 加载保存的StandardScaler
scaler = joblib.load('feature_scaler.pkl')
# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "CHI3L1": {"type": "numerical", "min": 0.000, "max": 2000.000, "default": 79.000},
    "ALP": {"type": "numerical", "min": 0.000, "max": 1000.000, "default": 24.555},
    "Fibrinogen": {"type": "numerical", "min": 0.000, "max": 10.000, "default": 4.000},
    "Chlid": {"type": "numerical", "min": 0, "max": 10, "default": 6},
    "PIVAK_2": {"type": "numerical", "min": 0.000, "max": 100000.000, "default": 40.000},
    "Fer": {"type": "numerical", "min": 0.000, "max": 10000.000, "default": 400.000},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features_raw = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # # 显示输入的原始数据
    # st.subheader("输入数据处理过程:")
    # st.write("**原始输入数据:**")
    # raw_data_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    # st.dataframe(raw_data_df)
    
    # 对输入数据进行标准化处理
    features_scaled = scaler.transform(features_raw)
    
    # # 显示标准化后的数据
    # st.write("**标准化后的数据:**")
    # scaled_data_df = pd.DataFrame(features_scaled, columns=feature_ranges.keys())
    # st.dataframe(scaled_data_df)
    
    # 模型预测（使用标准化后的数据）
    predicted_class = model.predict(features_scaled)[0]
    predicted_proba = model.predict_proba(features_scaled)[0]

    # 提取失代偿发生的概率（通常是类别1的概率）
    # 对于二分类问题：类别0=无失代偿，类别1=有失代偿
    decompensation_probability = predicted_proba[1] * 100  # 失代偿发生的概率
    
    # 使用失代偿发生的概率作为主要显示结果
    probability = decompensation_probability

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Predicted possibility of Postoperative Decompensation after TACE is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值（使用标准化后的数据）
    features_df_scaled = pd.DataFrame(features_scaled, columns=feature_ranges.keys())
    
    # 为堆叠模型创建一个可调用的预测函数
    def model_predict(X):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            return model.predict(X)
    
    # 使用KernelExplainer处理堆叠模型
    # 创建背景数据集（使用训练数据的样本）
    background = shap.sample(features_df_scaled, min(100, len(features_df_scaled)))
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(features_df_scaled)

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    
    # 处理KernelExplainer返回的SHAP值
    if isinstance(shap_values, list):
        # 多分类情况，选择对应类别的SHAP值
        if len(shap_values) > class_index:
            shap_values_for_class = shap_values[class_index][0]  # 取第一个样本
        else:
            shap_values_for_class = shap_values[0][0]  # 如果索引超出范围，使用第一个类别
        expected_value = explainer.expected_value[class_index] if len(explainer.expected_value) > class_index else explainer.expected_value[0]
    else:
        # 二分类或单一数组情况
        if len(shap_values.shape) > 1:
            shap_values_for_class = shap_values[0]  # 取第一个样本
        else:
            shap_values_for_class = shap_values
        expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
    
    # 为了可读性，在SHAP图中显示原始特征值
    features_df_original = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    
    # 生成 SHAP 力图
    plt.figure(figsize=(12, 3))
    shap.force_plot(
        expected_value,
        shap_values_for_class,
        features_df_original.iloc[0],  # 传递Series而不是DataFrame
        matplotlib=True,
        show=False
    )
    
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    # plt.close()  # 关闭当前图形以释放内存
    st.image("shap_force_plot.png")
    
    # # 清理临时文件
    # import os
    # import time
    # time.sleep(1)  # 等待图片显示完成
    # try:
    #     os.remove("shap_force_plot.png")
    # except FileNotFoundError:
    #     pass



