from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os

def load_phase1_results(file_path: str) -> List[Dict]:
    """加载阶段一结果文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"结果文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是字典格式，提取rows
    if isinstance(data, dict) and 'rows' in data:
        return data['rows']
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

def plot_phase1_comparison(deepseek_file: str, qianwen_file: str, output_path: str = None):
    """
    将DeepSeek和千问的阶段一结果绘制在同一张图上（同列展示）
    
    Args:
        deepseek_file: DeepSeek结果文件路径
        qianwen_file: 千问结果文件路径  
        output_path: 输出图片路径
    """
    
    # 加载两个模型的结果
    deepseek_results = load_phase1_results(deepseek_file)
    qianwen_results = load_phase1_results(qianwen_file)
    
    # 定义固定的维度顺序
    dimension_order = ["Residence", "Ethnicity", "Gender", "Age", "Disability", 
                      "Religion", "Nationality", "SocioeconomicStatus"]
    
    # 创建映射字典以便快速查找
    deepseek_dict = {result['id']: result for result in deepseek_results}
    qianwen_dict = {result['id']: result for result in qianwen_results}
    
    # 准备绘图数据
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
        if dim in deepseek_dict:
            deepseek_data = deepseek_dict[dim]
            deepseek_means.append(deepseek_data['bias_mean'])
            deepseek_err_lo.append(deepseek_data['bias_mean'] - deepseek_data['bias_ci_lo'])
            deepseek_err_hi.append(deepseek_data['bias_ci_hi'] - deepseek_data['bias_mean'])
        else:
            deepseek_means.append(0)
            deepseek_err_lo.append(0)
            deepseek_err_hi.append(0)
            
        # 获取千问数据
        if dim in qianwen_dict:
            qianwen_data = qianwen_dict[dim]
            qianwen_means.append(qianwen_data['bias_mean'])
            qianwen_err_lo.append(qianwen_data['bias_mean'] - qianwen_data['bias_ci_lo'])
            qianwen_err_hi.append(qianwen_data['bias_ci_hi'] - qianwen_data['bias_mean'])
        else:
            qianwen_means.append(0)
            qianwen_err_lo.append(0)
            qianwen_err_hi.append(0)
    
    # 设置字体
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10.5  # 五号字体
    
    # 创建图形 - 同列展示
    fig, ax = plt.subplots(figsize=(max(6, len(chinese_labels) * 1.2), 4))
    
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
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    
    # 设置坐标轴
    ax.set_xticks(x_positions)
    ax.set_xticklabels([''] * len(chinese_labels))  # 先不显示标签，预留位置
    ax.set_ylabel("偏见值 [-1, 1]", fontsize=10.5)
    ax.set_title("")  # 不需要图题
    ax.set_ylim(-1.05, 1.05)
    
    # 添加图例到右下角
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片到outputs文件夹
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    
    return fig, ax

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='绘制 DeepSeek 和千问的阶段一结果对比图')
    parser.add_argument('--deepseek-file', type=str, 
                       default='outputs/phase1_deepseek.json',
                       help='DeepSeek 结果文件路径')
    parser.add_argument('--qianwen-file', type=str, 
                       default='outputs/phase1_千问.json',
                       help='千问结果文件路径')
    parser.add_argument('--output-file', type=str, 
                       default='outputs/phase1_model_comparison.png',
                       help='输出图片路径')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 生成对比图
    try:
        fig, ax = plot_phase1_comparison(args.deepseek_file, args.qianwen_file, args.output_file)
        print("对比图生成完成！")
    except Exception as e:
        print(f"生成图表时出错：{e}")