from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os

def load_phase_results(file_path: str) -> List[Dict]:
    """加载阶段结果文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"结果文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是字典格式，提取combined_rows或rows
    if isinstance(data, dict):
        if 'combined_rows' in data:
            return data['combined_rows']
        elif 'rows' in data:
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

def get_chinese_context(english_context: str) -> str:
    """将英文情境转换为中文标签"""
    context_mapping = {
        "平时情境": "平时情境",
        "危机情境": "危机情境", 
        "灾后恢复情境": "灾后恢复情境"
    }
    return context_mapping.get(english_context, english_context)

def plot_phases_qianwen_comparison(phase2_file: str, phase3_file: str, phase4_file: str, output_path: str = None):
    """
    将千问在三个阶段的结果绘制在同一张图上
    
    Args:
        phase2_file: Phase2结果文件路径
        phase3_file: Phase3结果文件路径
        phase4_file: Phase4结果文件路径
        output_path: 输出图片路径
    """
    
    # 加载三个阶段的结果
    phase2_results = load_phase_results(phase2_file)
    phase3_results = load_phase_results(phase3_file)
    phase4_results = load_phase_results(phase4_file)
    
    # 定义固定的维度顺序
    dimension_order = ["Residence", "Ethnicity", "Gender", "Age", "Disability", 
                      "Religion", "Nationality", "SocioeconomicStatus"]
    
    # 定义情境顺序（主要用于phase2）
    context_order = ["平时情境", "危机情境", "灾后恢复情境"]
    
    # 创建映射字典以便快速查找
    phase2_dict = {}
    phase3_dict = {}
    phase4_dict = {}
    
    # 构建数据映射
    for result in phase2_results:
        if 'context' in result:  # Phase2有情境
            key = (result['id'], result['context'])
            phase2_dict[key] = result
        else:  # Phase1格式
            key = result['id']
            phase2_dict[key] = result
            
    for result in phase3_results:
        if 'context' in result:  # Phase3有情境
            key = (result['id'], result['context'])
            phase3_dict[key] = result
        else:
            key = result['id']
            phase3_dict[key] = result
            
    for result in phase4_results:
        if 'context' in result:  # Phase4有情境
            key = (result['id'], result['context'])
            phase4_dict[key] = result
        else:
            key = result['id']
            phase4_dict[key] = result
    
    # 设置字体 - 使用通用字体避免中文字体警告
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10.5  # 五号字体
    
    # 创建图形 - 3个子图，每个情境一个（主要针对phase2的三个情境）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for context_idx, context in enumerate(context_order):
        ax = axes[context_idx]
        
        # 准备当前情境的数据
        chinese_labels = []
        phase2_means = []
        phase2_err_lo = []
        phase2_err_hi = []
        phase3_means = []
        phase3_err_lo = []
        phase3_err_hi = []
        phase4_means = []
        phase4_err_lo = []
        phase4_err_hi = []
        
        for dim in dimension_order:
            chinese_labels.append(get_chinese_label(dim))
            
            # 获取Phase2数据（按情境）
            phase2_key = (dim, context)
            if phase2_key in phase2_dict:
                phase2_data = phase2_dict[phase2_key]
                phase2_means.append(phase2_data['bias_mean'])
                phase2_err_lo.append(phase2_data['bias_mean'] - phase2_data['bias_ci_lo'])
                phase2_err_hi.append(phase2_data['bias_ci_hi'] - phase2_data['bias_mean'])
            else:
                phase2_means.append(0.5)  # 默认中性值
                phase2_err_lo.append(0)
                phase2_err_hi.append(0)
                
            # 获取Phase3数据（如果没有情境，则使用维度ID）
            phase3_key = (dim, context) if (dim, context) in phase3_dict else dim
            if phase3_key in phase3_dict:
                phase3_data = phase3_dict[phase3_key]
                phase3_means.append(phase3_data['bias_mean'])
                phase3_err_lo.append(phase3_data['bias_mean'] - phase3_data['bias_ci_lo'])
                phase3_err_hi.append(phase3_data['bias_ci_hi'] - phase3_data['bias_mean'])
            else:
                phase3_means.append(0.5)  # 默认中性值
                phase3_err_lo.append(0)
                phase3_err_hi.append(0)
                
            # 获取Phase4数据（如果没有情境，则使用维度ID）
            phase4_key = (dim, context) if (dim, context) in phase4_dict else dim
            if phase4_key in phase4_dict:
                phase4_data = phase4_dict[phase4_key]
                phase4_means.append(phase4_data['bias_mean'])
                phase4_err_lo.append(phase4_data['bias_mean'] - phase4_data['bias_ci_lo'])
                phase4_err_hi.append(phase4_data['bias_ci_hi'] - phase4_data['bias_mean'])
            else:
                phase4_means.append(0.5)  # 默认中性值
                phase4_err_lo.append(0)
                phase4_err_hi.append(0)
        
        x_positions = range(len(chinese_labels))
        
        # 绘制Phase2数据点和误差线（圆形点，浅黄色）
        ax.errorbar(x_positions, phase2_means, 
                    yerr=[phase2_err_lo, phase2_err_hi], 
                    fmt='o', capsize=4, label='Phase 2', 
                    color='#fee08b', markersize=4, linewidth=1.5)
        
        # 绘制Phase3数据点和误差线（三角形点，中黄色）
        ax.errorbar(x_positions, phase3_means, 
                    yerr=[phase3_err_lo, phase3_err_hi], 
                    fmt='^', capsize=4, label='Phase 3', 
                    color='#fdae61', markersize=4, linewidth=1.5)
        
        # 绘制Phase4数据点和误差线（方形点，深黄色）
        ax.errorbar(x_positions, phase4_means, 
                    yerr=[phase4_err_lo, phase4_err_hi], 
                    fmt='s', capsize=4, label='Phase 4', 
                    color='#e6550d', markersize=4, linewidth=1.5)
        
        # 添加辅助线
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(0.5, color="red", linestyle=":", linewidth=1, label='中性线 (0.5)')
        
        # 设置坐标轴
        ax.set_xticks(x_positions)
        ax.set_xticklabels([''] * len(chinese_labels))  # 预留位置，不显示标签
        ax.set_ylabel("偏见值 [0, 1]", fontsize=10.5)
        ax.set_title(get_chinese_context(context), fontsize=11)
        
        # 统一Y轴范围为[0,1]
        ax.set_ylim(-0.05, 1.05)
        
        # 只在第一个子图添加图例占位符（不实际显示图例）
        if context_idx == 0:
            # 添加透明图例占位符，不实际显示
            ax.plot([], [], 'o', color='#fee08b', label='Phase 2', markersize=4)
            ax.plot([], [], '^', color='#fdae61', label='Phase 3', markersize=4)
            ax.plot([], [], 's', color='#e6550d', label='Phase 4', markersize=4)
    
    # 添加整体标题
    fig.suptitle("千问在各阶段的偏见表现对比", fontsize=12, y=0.98)
    
    plt.tight_layout()
    
    # 保存主图表到outputs文件夹
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"主图表已保存到: {output_path}")
    
    # 创建单独的图例图表
    if output_path:
        legend_path = output_path.replace('.png', '_legend.png')
        create_legend_chart(legend_path)
        print(f"图例图表已保存到: {legend_path}")
    
    return fig, axes

def create_legend_chart(output_path: str):
    """创建单独的图例图表"""
    fig, ax = plt.subplots(figsize=(6, 2))
    
    # 创建图例项
    ax.plot([], [], 'o', color='#fee08b', label='Phase 2', markersize=6)
    ax.plot([], [], '^', color='#fdae61', label='Phase 3', markersize=6)
    ax.plot([], [], 's', color='#e6550d', label='Phase 4', markersize=6)
    
    # 添加中性线图例
    ax.plot([], [], '-', color='red', linestyle=':', label='中性线 (0.5)', linewidth=1)
    
    # 显示图例
    legend = ax.legend(loc='center', fontsize=10, frameon=True, 
                      fancybox=True, shadow=True, ncol=4)
    
    # 隐藏坐标轴
    ax.axis('off')
    
    # 保存图例图表
    fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='绘制千问多阶段对比图')
    parser.add_argument('--phase2-file', type=str, 
                       default='outputs/phase2_千问.json',
                       help='Phase 2 结果文件路径')
    parser.add_argument('--phase3-file', type=str, 
                       default='outputs/phase3_千问.json',
                       help='Phase 3 结果文件路径')
    parser.add_argument('--phase4-file', type=str, 
                       default='outputs/phase4_千问.json',
                       help='Phase 4 结果文件路径')
    parser.add_argument('--output-file', type=str, 
                       default='outputs/phases_qianwen_comparison.png',
                       help='输出图片路径')
    args = parser.parse_args()
    
    # 生成对比图
    try:
        fig, axes = plot_phases_qianwen_comparison(args.phase2_file, args.phase3_file, args.phase4_file, args.output_file)
        print("千问多阶段对比图生成完成！")
    except Exception as e:
        print(f"生成图表时出错：{e}")
        import traceback
        traceback.print_exc()
