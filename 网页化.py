'''
streamlit==1.30.0
joblib==1.4.2
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.0
scikit-learn==1.5.1
'''
import streamlit as st
import streamlit.components.v1
import joblib
import numpy as np
print(np.__version__)
import pandas as pd
print(pd.__version__)
import shap
import matplotlib.pyplot as plt
import tempfile
import os

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
    # 创建背景数据集 - 使用标准化数据的均值作为基准
    # 注意：features_scaled是1D数组，需要reshape为2D
    current_sample_2d = features_scaled.reshape(1, -1)
    background_data = np.zeros_like(current_sample_2d)  # 创建同样形状的零向量
    
    # 调试信息
    st.write(f"背景数据形状: {background_data.shape}")
    st.write(f"当前样本2D形状: {current_sample_2d.shape}")
    st.write(f"特征数量: {current_sample_2d.shape[1]}")
    
    # 测试模型预测函数
    try:
        test_pred = model_predict(background_data)
        st.write(f"背景数据预测成功: {test_pred}")
        test_pred2 = model_predict(current_sample_2d)
        st.write(f"当前样本预测成功: {test_pred2}")
    except Exception as e:
        st.error(f"模型预测测试失败: {e}")
        st.stop()
    
    explainer = shap.KernelExplainer(model_predict, background_data)
    
    # 计算当前输入样本的SHAP值
    shap_values = explainer.shap_values(current_sample_2d, nsamples=100)  # 限制采样数量加快计算
    

    # 生成 SHAP 力图
    # 现在预测函数直接返回失代偿概率，SHAP值处理更简单
    
    # 处理SHAP值 - 现在应该是单一输出的回归格式
    st.write(f"SHAP值类型: {type(shap_values)}")
    st.write(f"SHAP值形状: {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")
    st.write(f"Expected value: {explainer.expected_value}")
    
    # 正确提取SHAP值
    if isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 2:
            shap_values_for_class = shap_values[0]  # 取第一行（第一个样本）
        else:
            shap_values_for_class = shap_values
    else:
        shap_values_for_class = shap_values
    
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0] if len(expected_value) > 0 else 0
    
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
        
        # 主要调试信息
        st.write(f"主力图 - 提取的SHAP值: {shap_vals}")
        st.write(f"主力图 - SHAP值总和: {np.sum(shap_vals)}")
        st.write(f"主力图 - 基准值: {expected_value}")
        st.write(f"主力图 - 预测值: {expected_value + np.sum(shap_vals)}")
        
        # 方法1：使用SHAP原生力图
        st.subheader("📊 SHAP Force Plot (Original Style)")
        
        # 准备SHAP原生力图数据
        # 需要创建shap.Explanation对象
        explanation = shap.Explanation(
            values=shap_values_array,
            base_values=expected_value,
            data=feature_vals,
            feature_names=feature_names
        )
        
        # 生成SHAP力图并保存为HTML
        force_plot = shap.force_plot(
            base_value=expected_value,
            shap_values=shap_values_array,
            features=feature_vals,
            feature_names=feature_names,
            out_names="Decompensation Risk",
            matplotlib=False  # 使用HTML版本
        )
        
        # 将SHAP力图保存为HTML文件并显示
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            shap.save_html(f.name, force_plot)
            html_file = f.name
        
        # 读取并显示HTML
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=300, scrolling=True)
        
        # 方法2：创建自定义的经典风格力图
        st.subheader("📈 Custom SHAP Force Plot")
        
        # 创建标准SHAP力图样式
        fig, ax = plt.subplots(figsize=(16, 4))
        
        # 绘制标准的水平瀑布图样式
        y_pos = 0.5
        bar_height = 0.6
        
        # 从基准值开始累积绘制
        current_x = expected_value
        
        # 按照SHAP值绝对值排序，确保重要特征优先显示
        feature_importance = list(zip(feature_names, feature_vals, shap_vals))
        feature_importance.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # 只显示前8个最重要的特征，避免过于拥挤
        top_features = feature_importance[:8]
        
        # 绘制特征贡献
        for i, (name, val, shap_val) in enumerate(top_features):
            # 选择颜色：正值红色，负值蓝色
            color = '#ff0051' if shap_val > 0 else '#008bfb'
            
            # 计算条形的起始位置
            start_x = current_x
            
            # 绘制条形（使用带符号的shap_val作为宽度）
            rect = plt.Rectangle((start_x, y_pos - bar_height/2), 
                               shap_val, bar_height,
                               facecolor=color, alpha=0.8, 
                               edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            
            # 添加特征标签（在条形上方）
            mid_x = start_x + shap_val/2
            ax.text(mid_x, y_pos + bar_height/2 + 0.05, 
                   f'{name} = {val:.2f}', 
                   ha='center', va='bottom', fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=color))
            
            # 添加SHAP值标签（在条形中央）
            ax.text(mid_x, y_pos, 
                   f'{shap_val:.3f}', 
                   ha='center', va='center', fontsize=10, 
                   color='white', fontweight='bold')
            
            # 更新累积位置
            current_x += shap_val
        
        # 添加顶部的higher/lower标识箭头
        ax.annotate('higher', xy=(0.02, 0.85), xycoords='axes fraction',
                   fontsize=14, color='#ff0051', fontweight='bold', ha='left',
                   arrowprops=dict(arrowstyle='->', color='#ff0051', lw=2))
        ax.annotate('lower', xy=(0.98, 0.85), xycoords='axes fraction',
                   fontsize=14, color='#008bfb', fontweight='bold', ha='right',
                   arrowprops=dict(arrowstyle='<-', color='#008bfb', lw=2))
        
        # 添加垂直线标记基准值和最终值
        ax.axvline(x=expected_value, color='gray', linestyle='--', alpha=0.8, linewidth=3)
        ax.axvline(x=current_x, color='black', linestyle='-', alpha=0.9, linewidth=3)
        
        # 添加base value标识
        ax.text(expected_value, y_pos - bar_height/2 - 0.15, 'base value', 
               ha='center', va='top', fontsize=12, color='gray', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.8))
        
        # 添加f(x)预测值标识
        actual_decompensation_prob = predicted_proba[1] * 100
        ax.text(current_x, y_pos + bar_height/2 + 0.2, f'f(x)\n{actual_decompensation_prob:.2f}%', 
               ha='center', va='bottom', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='yellow', alpha=0.9, edgecolor='orange'))
        
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
            
            # 最终调试信息
            st.write(f"提取的SHAP值: {shap_vals}")
            st.write(f"SHAP值总和: {np.sum(shap_vals)}")
            st.write(f"基准值: {expected_value}")
            st.write(f"预测值: {expected_value + np.sum(shap_vals)}")
            
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



