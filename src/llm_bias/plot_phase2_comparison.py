from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os

def load_phase2_results(file_path: str) -> List[Dict]:
    """加载阶段二结果文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"结果文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是字典格式，提取combined_rows
    if isinstance(data, dict) and 'combined_rows' in data:
        return data['combined_rows']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"无法识别的结果文件格式: {file_path}")

def get_chinese_label(english_id: str) -> str:
    """将英文ID转换为中文标签"""
    label_mapping = {
        "Residence": "居住地",
        "Ethnicity": "民族", 
        "Gender": "性别",
        "Age": "年龄",
        "Disability": "健康状况",
        "Religion": "宗教",
        "Nationality": "国籍",
        "SocioeconomicStatus": "社会经济地位"
    }
    return label_mapping.get(english_id, english_id)

def get_chinese_context(english_context: str) -> str:
    """将英文情境转换为中文标签"""
    context_mapping = {
        "平时情境": "平时情境",
        "危机情境": "危机情境", 
        "灾后恢复情境": "灾后恢复情境"
    }
    return context_mapping.get(english_context, english_context)

def plot_phase2_comparison(deepseek_file: str, qianwen_file: str, output_path: str = None):
    """
    将DeepSeek和千问的阶段二结果绘制在同一张图上
    
    Args:
        deepseek_file: DeepSeek结果文件路径
        qianwen_file: 千问结果文件路径  
        output_path: 输出图片路径
    """
    
    # 加载两个模型的结果
    deepseek_results = load_phase2_results(deepseek_file)
    qianwen_results = load_phase2_results(qianwen_file)
    
    # 定义固定的维度顺序
    dimension_order = ["Residence", "Ethnicity", "Gender", "Age", "Disability", 
                      "Religion", "Nationality", "SocioeconomicStatus"]
    
    # 定义情境顺序
    context_order = ["平时情境", "危机情境", "灾后恢复情境"]
    
    # 创建映射字典以便快速查找
    deepseek_dict = {}
    qianwen_dict = {}
    
    # 构建 (维度, 情境) -> 结果 的映射
    for result in deepseek_results:
        key = (result['id'], result['context'])
        deepseek_dict[key] = result
        
    for result in qianwen_results:
        key = (result['id'], result['context'])
        qianwen_dict[key] = result
    
    # 设置字体 - 使用通用字体避免中文字体警告
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10.5  # 五号字体
    
    # 创建图形 - 3个子图，每个情境一个
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for context_idx, context in enumerate(context_order):
        ax = axes[context_idx]
        
        # 准备当前情境的数据
        chinese_labels = []
        deepseek_means = []
        deepseek_err_lo = []
        deepseek_err_hi = []
        qianwen_means = []
        qianwen_err_lo = []
        qianwen_err_hi = []
        
        for dim in dimension_order:
            chinese_labels.append(get_chinese_label(dim))
            
            # 获取DeepSeek数据
            key = (dim, context)
            if key in deepseek_dict:
                deepseek_data = deepseek_dict[key]
                deepseek_means.append(deepseek_data['bias_mean'])
                deepseek_err_lo.append(deepseek_data['bias_mean'] - deepseek_data['bias_ci_lo'])
                deepseek_err_hi.append(deepseek_data['bias_ci_hi'] - deepseek_data['bias_mean'])
            else:
                deepseek_means.append(0.5)  # 默认中性值
                deepseek_err_lo.append(0)
                deepseek_err_hi.append(0)
                
            # 获取千问数据
            if key in qianwen_dict:
                qianwen_data = qianwen_dict[key]
                qianwen_means.append(qianwen_data['bias_mean'])
                qianwen_err_lo.append(qianwen_data['bias_mean'] - qianwen_data['bias_ci_lo'])
                qianwen_err_hi.append(qianwen_data['bias_ci_hi'] - qianwen_data['bias_mean'])
            else:
                qianwen_means.append(0.5)  # 默认中性值
                qianwen_err_lo.append(0)
                qianwen_err_hi.append(0)
        
        x_positions = range(len(chinese_labels))
        
        # 绘制DeepSeek数据点和误差线（圆形点，更小）
        ax.errorbar(x_positions, deepseek_means, 
                    yerr=[deepseek_err_lo, deepseek_err_hi], 
                    fmt='o', capsize=4, label='DeepSeek', 
                    color='#1f77b4', markersize=4, linewidth=1.5)
        
        # 绘制千问数据点和误差线（方形点，更小）
        ax.errorbar(x_positions, qianwen_means, 
                    yerr=[qianwen_err_lo, qianwen_err_hi], 
                    fmt='s', capsize=4, label='千问', 
                    color='#ff7f0e', markersize=4, linewidth=1.5)
        
        # 添加辅助线
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(0.5, color="red", linestyle=":", linewidth=1, label='中性线 (0.5)')
        
        # 设置坐标轴
        ax.set_xticks(x_positions)
        ax.set_xticklabels([''] * len(chinese_labels))  # 预留位置，不显示标签
        ax.set_ylabel("偏见值 [0, 1]", fontsize=10.5)
        ax.set_title(get_chinese_context(context), fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        
        # 只在第一个子图添加图例到左上角
        if context_idx == 0:
            ax.legend(loc='upper left', fontsize=9)
    
    # 添加整体标题
    fig.suptitle("Phase 2: 相对决策偏见对比", fontsize=12, y=0.98)
    
    plt.tight_layout()
    
    # 保存图片到outputs文件夹
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    
    return fig, axes

if __name__ == "__main__":
    # 文件路径
    deepseek_file = "/Users/lun/Downloads/work/研究生课程/危机决策与沟通/Data-for-Emergency 阶段一中文未改/outputs/phase2_deepseek.json"
    qianwen_file = "/Users/lun/Downloads/work/研究生课程/危机决策与沟通/Data-for-Emergency 阶段一中文未改/outputs/phase2_千问.json"
    output_file = "/Users/lun/Downloads/work/研究生课程/危机决策与沟通/Data-for-Emergency 阶段一中文未改/outputs/phase2_model_comparison.png"
    
    # 生成对比图
    try:
        fig, axes = plot_phase2_comparison(deepseek_file, qianwen_file, output_file)
        print("Phase2对比图生成完成！")
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"生成图表时出错: {e}")
        import traceback
        traceback.print_exc()