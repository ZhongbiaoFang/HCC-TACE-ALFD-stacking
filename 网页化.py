'''
streamlit==1.30.0
joblib==1.4.2
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.0
scikit-learn==1.5.1
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
    # 返回失代偿类别（类别1）的概率
    def model_predict(X):
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            return proba[:, 1]  # 只返回失代偿类别的概率
        else:
            return model.predict(X)
    
    # 使用KernelExplainer处理堆叠模型
    # 创建背景数据集（使用训练数据的样本）
    background = shap.sample(features_df_scaled, min(100, len(features_df_scaled)))
    explainer = shap.KernelExplainer(model_predict, background)
    
    # 只计算当前输入样本的SHAP值
    current_sample = features_df_scaled.iloc[[0]]  # 保持DataFrame格式
    shap_values = explainer.shap_values(current_sample)
    

    # 生成 SHAP 力图
    # 现在预测函数直接返回失代偿概率，SHAP值处理更简单
    
    # 处理SHAP值 - 现在应该是单一输出的回归格式
    if isinstance(shap_values, list):
        # 如果是列表，取第一个元素
        shap_values_for_class = shap_values[0][0] if isinstance(shap_values[0], np.ndarray) else shap_values[0]
        expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        # 直接使用SHAP值
        shap_values_for_class = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
    
    # 为了可读性，在SHAP图中显示原始特征值
    features_df_original = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    
    # 生成标准的 SHAP 力图
    try:
        # 确保数据类型正确
        expected_value = float(expected_value) if not isinstance(expected_value, (list, np.ndarray)) else expected_value[0] if len(expected_value) > 0 else 0.0
        
        # 确保SHAP值是numpy数组
        shap_values_array = np.array(shap_values_for_class).flatten()
        
        # 获取特征名称和值
        feature_names = list(feature_ranges.keys())
        feature_vals = features_df_original.iloc[0].values
        shap_vals = shap_values_array
        
        # 创建标准SHAP力图样式
        fig, ax = plt.subplots(figsize=(16, 3))
        
        # 计算累积SHAP值
        cumulative = [expected_value]
        for shap_val in shap_vals:
            cumulative.append(cumulative[-1] + shap_val)
        
        # 绘制标准的水平瀑布图样式
        y_pos = 0.5
        bar_height = 0.8
        
        # 从基准值开始累积绘制
        current_x = expected_value
        
        # 先打印调试信息，确保SHAP值不为空
        st.write("调试信息:")
        st.write(f"特征数量: {len(feature_names)}")
        st.write(f"SHAP值数量: {len(shap_vals)}")
        st.write("前5个SHAP值:", shap_vals[:5] if len(shap_vals) > 0 else "无SHAP值")
        
        # 按照SHAP值绝对值排序，确保重要特征优先显示
        feature_importance = list(zip(feature_names, feature_vals, shap_vals))
        feature_importance.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # 绘制所有特征的贡献（不设置最小阈值）
        for i, (name, val, shap_val) in enumerate(feature_importance):
            # 选择颜色：正值红色，负值蓝色
            color = '#ff4757' if shap_val > 0 else '#3742fa'
            
            # 计算条形的起始位置
            start_x = current_x
            
            # 绘制条形（使用带符号的shap_val作为宽度）
            rect = plt.Rectangle((start_x, y_pos - bar_height/2), 
                               shap_val, bar_height,
                               facecolor=color, alpha=0.7, 
                               edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            
            # 只为前10个最重要的特征添加标签，避免拥挤
            if i < 10:
                # 添加特征标签（在条形上方）
                mid_x = start_x + shap_val/2
                ax.text(mid_x, y_pos + bar_height/2 + 0.05, 
                       f'{name}\n{val:.2f}', 
                       ha='center', va='bottom', fontsize=7, 
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                # 添加SHAP值标签（在条形下方）
                ax.text(mid_x, y_pos - bar_height/2 - 0.05, 
                       f'{shap_val:.3f}', 
                       ha='center', va='top', fontsize=7, 
                       color=color, fontweight='bold')
            
            # 更新累积位置
            current_x += shap_val
        
        # 添加顶部的higher/lower标识
        ax.text(0.02, 0.95, 'higher', transform=ax.transAxes, 
               fontsize=12, color='#ff4757', fontweight='bold', ha='left')
        ax.text(0.98, 0.95, 'lower', transform=ax.transAxes, 
               fontsize=12, color='#3742fa', fontweight='bold', ha='right')
        
        # 添加垂直线标记基准值和最终值
        ax.axvline(x=expected_value, color='gray', linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(x=current_x, color='black', linestyle='-', alpha=0.8, linewidth=2)
        
        # 添加base value标识
        ax.text(expected_value, y_pos - bar_height/2 - 0.2, 'base value', 
               ha='center', va='top', fontsize=10, color='gray', fontweight='bold')
        
        # 添加f(x)预测值标识，应该与实际预测概率一致
        # 确保显示的是失代偿的概率
        actual_decompensation_prob = predicted_proba[1] * 100
        ax.text(current_x, y_pos + bar_height/2 + 0.15, f'f(x)\n{actual_decompensation_prob:.2f}%', 
               ha='center', va='bottom', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        # 设置图表属性，确保所有内容都能显示
        x_range = abs(current_x - expected_value)
        margin = max(0.02, x_range * 0.1)  # 动态边距
        x_min = min(expected_value, current_x) - margin
        x_max = max(expected_value, current_x) + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.1, 1.2)
        
        # 移除y轴刻度和标签
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # 设置x轴
        ax.set_xlabel('Model Output Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 设置标题
        ax.set_title(f'SHAP Force Plot for Decompensation Risk (Probability: {probability:.2f}%)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        st.image("shap_force_plot.png")
        
    except Exception as e:
        st.error(f"SHAP force plot generation failed: {str(e)}")
        st.info("Using backup SHAP feature importance chart")
        
        # 备用方案：生成SHAP特征重要性条形图
        try:
            plt.figure(figsize=(10, 6))
            
            # 获取特征名称和SHAP值
            shap_vals = np.array(shap_values_for_class).flatten()
            
            # 使用特征名称
            feature_names = list(feature_ranges.keys())
            
            # 创建特征重要性排序
            importance_data = list(zip(feature_names, shap_vals))
            importance_data.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # 绘制条形图
            features, values = zip(*importance_data)
            colors = ['red' if v < 0 else 'blue' for v in values]
            
            plt.barh(range(len(features)), values, color=colors, alpha=0.7)
            plt.yticks(range(len(features)), features)
            plt.xlabel('SHAP Value (Feature Impact on Prediction)')
            plt.title('Feature Impact on Prediction Results (SHAP Analysis)')
            plt.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(values):
                plt.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}', 
                        va='center', ha='left' if v >= 0 else 'right')
            
            plt.tight_layout()
            plt.savefig("shap_bar_plot.png", bbox_inches='tight', dpi=300)
            plt.close()
            st.image("shap_bar_plot.png")
            
        except Exception as e2:
            st.error(f"Backup SHAP chart generation also failed: {str(e2)}")
            st.write("SHAP值:", shap_values_for_class)
    
    # # 清理临时文件
    # import os
    # import time
    # time.sleep(1)  # 等待图片显示完成
    # try:
    #     os.remove("shap_force_plot.png")
    # except FileNotFoundError:
    #     pass



