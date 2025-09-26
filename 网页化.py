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
    def model_predict(X):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
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
        # 处理形状为 (1, n_features, n_classes) 的情况
        if len(shap_values.shape) == 3:  # (1, n_features, n_classes)
            shap_values_for_class = shap_values[0, :, class_index] if class_index < shap_values.shape[2] else shap_values[0, :, 0]
            # 处理expected_value
            if hasattr(explainer, 'expected_value'):
                if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > class_index:
                    expected_value = explainer.expected_value[class_index]
                elif isinstance(explainer.expected_value, (list, np.ndarray)):
                    expected_value = explainer.expected_value[0]
                else:
                    expected_value = explainer.expected_value
            else:
                expected_value = 0
        else:
            # 其他情况
            shap_values_for_class = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
    
    # 为了可读性，在SHAP图中显示原始特征值
    features_df_original = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    
    # 生成 SHAP 力图
    try:
        # 确保数据类型正确
        expected_value = float(expected_value) if not isinstance(expected_value, (list, np.ndarray)) else expected_value[0] if len(expected_value) > 0 else 0.0
        
        # 确保SHAP值是numpy数组（不flatten，保持原始特征维度）
        shap_values_array = np.array(shap_values_for_class)
        
        # 使用自定义方法创建带颜色的SHAP力图
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # 获取特征名称和值
        feature_names = list(feature_ranges.keys())
        feature_vals = features_df_original.iloc[0].values
        shap_vals = shap_values_array.flatten()
        
        # 计算累积位置
        cumulative = [expected_value]
        for i in range(len(shap_vals)):
            cumulative.append(cumulative[-1] + shap_vals[i])
        
        # 绘制基线
        base_y = 0.5
        ax.plot([expected_value, cumulative[-1]], [base_y, base_y], 'k-', linewidth=2, alpha=0.7)
        
        # 绘制每个特征的贡献
        current_pos = expected_value
        for i, (name, val, shap_val) in enumerate(zip(feature_names, feature_vals, shap_vals)):
            # 确定颜色：正值用红色，负值用蓝色
            color = '#ff6b6b' if shap_val > 0 else '#4ecdc4'  # 红色为正面影响，青色为负面影响
            
            # 绘制箭头
            if abs(shap_val) > 0.001:  # 只显示有意义的贡献
                # 绘制矩形表示贡献大小
                rect_height = 0.15
                rect = plt.Rectangle((current_pos, base_y - rect_height/2), 
                                   shap_val, rect_height, 
                                   facecolor=color, alpha=0.8, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                
                # 添加特征标签
                mid_pos = current_pos + shap_val/2
                ax.text(mid_pos, base_y + 0.25, f'{name}\n{val:.2f}', 
                       ha='center', va='bottom', fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # 添加SHAP值标签
                ax.text(mid_pos, base_y - 0.25, f'{shap_val:.3f}', 
                       ha='center', va='top', fontsize=8, 
                       color='darkred' if shap_val > 0 else 'darkblue', fontweight='bold')
            
            current_pos += shap_val
        
        # 添加起始和结束标记
        ax.axvline(x=expected_value, color='gray', linestyle='--', alpha=0.7)
        ax.text(expected_value, base_y - 0.4, f'Base Value\n{expected_value:.3f}', 
               ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        ax.axvline(x=cumulative[-1], color='black', linestyle='-', linewidth=2)
        ax.text(cumulative[-1], base_y - 0.4, f'Prediction\n{cumulative[-1]:.3f}', 
               ha='center', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        # 设置图表属性
        ax.set_ylim(-0.6, 1.1)
        ax.set_xlabel('Model Output Value', fontsize=12)
        ax.set_title(f'SHAP Force Plot - Feature Contributions to Prediction (Predicted Class: {predicted_class})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_yticks([])
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff6b6b', alpha=0.8, label='Positive Impact (Increases Prediction)'),
            Patch(facecolor='#4ecdc4', alpha=0.8, label='Negative Impact (Decreases Prediction)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
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



