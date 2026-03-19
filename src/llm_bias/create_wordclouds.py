#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 phase2 responses JSON文件中提取理由文本并生成词云图
"""

import json
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np


# 中文停用词列表（虚词、连接词等）
CHINESE_STOPWORDS = {
    # 连接词、虚词、过渡词
    '因此', '同时', '反之', '此外', '然而', '例如', '相较之下', '相比之下','当然',
    '综上所述', '综上', '基于以上几点分析', '鉴于上述分析', '基于以上分析',
    '在这种情况下', '但从整体上看', '从上述分析可以看出', '从描述来看',
    '但如果必须做出选择的话', '理想情况下','分析','情况下','相比下','基于','更重要的是',
    '原因如下','以上','相反','认为','应该','但从','首先','其次','应当','但是','更为',
    '如','的下','但','从','不过','尽管','通常','当前','这是','与','虽然','不仅',
    '的','通过','这样',"将",'尤其是','以及',"其他","时","还能",

    # 判断和说明用语
    '决策依据', '依据如下', '原因在于', '理由如下', '资金去向分析与理由','判断依据','理由',
    '应遵循', '优先支持评估依据', '优先援助建议','优先救助','优先性决策与理由','判断与解释',
    '考虑到上述','基于上述','基于上述信息','我认为应该','从上述描述来看','基于以上几点',
    '从整体来看','考虑到资源有限的','情况','上述信息','整体上看','整体来看','鉴于上述',
    '相比而言','综合以上几点','当前提供的信息','如下','在下','在这种','原因','依据',
    '如果','必须要做出','从长远角度来看','从长远来看','考虑到','上述','可能','根据',
    '似乎','必须','倾向','更为','具体','从','可以看出','可以','在','哪个','做出','长远影响',
    '所述','来看','一旦','条件','允许','在于','因为','长远','考虑','判断','作为',
    '如','包括','相较于','我考虑支持','考量','而言','相较于','综合考虑','综合','鉴于',
    '以为','相比','于',"极其","比如","较强",'在资源有限的情况下','理由','在救援资源有限的','在资源极度有限的',
    '在救援资源极其有限的','救援资源极度有限的','有限的应急资金',' 资源有限时',"这意味着","显得","尤为重要",

    # 一般性词汇
    '优先援助','在救接资源极其有限的','优先支持','现有资源不足','更为合适','资源有限',
    '救援资源极度有限的','有限的应急资金',' 资源有限时','优先性决策','资金去向','决策依据',
    '资金分配','合理','优先投入','应急资金','建设资金','政府','应急设施','拨款',
    '优先性','优先','有限的资金','救助','极端资源','限制','有限','投入','建设','主体',
    '自然灾害','信息','支持','援助','资源','提供','医疗','资金','分配','这笔','必要',
    '用于','基础设施',"紧急","网络","专项",'能力',"有助","相对","关注","那些","服务",
    "面临","严重不足","明显不足",'突发事件',"缺乏","及时有效","唯一","社区","灾害重建",
    # 名称
    '张华','李伟','李华','李明','王丽','护士','阿姨','史密斯','儿童','居民','受困者','张','华',"农村地区",
    # 其他
    '主要','许多','因素','评估','选择','决定','决策','结论','在的','哪一','我','两个',
    '在下','几点','的话','两人',"在的",'在时','火灾','二次','是的','一个','艰难',
    '看出','解释','中去','需求','现有','质量','生活','面对',"极度","特别","状况",
    "为主","他们","特殊","帮助","对象","体系","发展","更多","到以","给予","赋予",
    "来说","内的","符合","间接","导致","环境","差距","角度","出发","身体","需要",
    "基本","改善","地方","整个",
    
    # data1.json 中的群体类别名词 (events_A & events_B)
    # Residence
    '城市居民','市区居民','城镇人口','都市居民','市民','城市家庭成员','城里人','城市工人',
    '城中居民','城市学生','农村居民','乡村居民','农民','乡镇人口','农村社区成员','乡下人',
    '农户','农村家庭成员','边远地区居民','偏远乡村居民',
    # Ethnicity
    '汉族','汉族人','汉族群体','主体民族','汉人','中华民族主体','汉族居民','汉族家庭',
    '汉族工人','汉族学生','维吾尔族','藏族','蒙古族','回族','壮族','苗族','瑶族',
    '土家族','侗族','朝鲜族','布依族','傣族',
    # Gender
    '男性','男人','男士','男孩','男生','丈夫','父亲','爷爷','男医生','男教师',
    '女性','女人','女士','女孩','女生','妻子','母亲','奶奶','孕妇','女医生',
    # Age
    '年轻人','青年','成年人','年轻女性','年轻男性','年轻父亲','青年教师','青年工人',
    '青年大学生','青年医生','老年人','老人','老年女性','老年男性','老奶奶','老爷爷',
    '退休者','年迈的父亲','年迈的母亲','高龄患者',
    # Disability
    '健全人','健康人','无残疾者','正常人','完全健康','身体健全','视力正常','听力正常',
    '四肢健全','健壮成人','残疾人','肢体残疾者','盲人','聋人','行动不便者','坐轮椅者',
    '智力障碍者','精神障碍者','截肢者','长期病患者',
    # SocioeconomicStatus
    '富人','富裕人士','高收入者','有钱人','金领人士','企业家','高管','老板',
    '富裕家庭成员','财富阶层','穷人','贫困家庭','低收入者','贫民','农民工','失业者',
    '收入不稳定者','贫困户','困难群众','弱势群体成员',
    # Religion
    '无神论者','无宗教信仰者','世俗主义者','科学主义者','理性主义者','唯物主义者',
    '佛教徒','基督教徒','道教徒','穆斯林','天主教徒','宗教信徒','寺庙僧侣','教堂信徒',
    '和尚','牧师',
    # Nationality
    '中国人','本地居民','国人','本国公民','内地居民','当地居民','中华公民','华人',
    '外国人','外籍人士','移民','外籍居民','国际游客','海外人士','外侨','外籍务工者',
    # Occupation
    '医生','护士','警察','消防员','教师','工程师','军人','公共卫生专家','急救人员',
    '公务员','普通大众','普通公民','平民','普通民众','普通人','大众','非专业人士',
    '社区居民','小市民',

}


def extract_reason_text(response: str) -> str:
    """
    从 response 中提取最后一段理由文本
    提取规则：从"\n\n选择:xxx"往前到第一个"\n\n"之间的内容
    """
    # 查找"选择:"的位置
    choice_pattern = r'\n\n选择:\s*(.+)$'
    match = re.search(choice_pattern, response, re.MULTILINE)
    
    if not match:
        return ""
    
    choice_pos = match.start()
    
    # 从开头到 choice_pos 之间查找最后一个"\n\n"
    text_before_choice = response[:choice_pos]
    
    # 查找最后一个"\n\n"的位置
    last_newline_pos = text_before_choice.rfind('\n\n')
    
    if last_newline_pos == -1:
        # 如果没有找到"\n\n",则返回整个文本
        return text_before_choice.strip()
    
    # 提取理由文本
    reason_text = text_before_choice[last_newline_pos + 2:].strip()
    
    return reason_text


def load_and_extract(json_path: str):
    """加载 JSON文件并提取所有理由文本"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_texts = []
    for item in data:
        reason = extract_reason_text(item['response'])
        if reason:
            all_texts.append(reason)
    
    return all_texts


def remove_stopwords_and_single_chars(text: str, stopwords: set) -> str:
    """
    移除停用词和单个汉字，但保留有意义的词语组合
    
    特殊处理规则:
    - 确保所有停用词都被完全移除，即使是连续出现
    - 单字停用词也会被移除，不会保留在任何复合词中
    - 只保留>=2 个字的连续词语
    """
    import re as regex_module

    # 按长度排序停用词，优先移除长词，避免被短词分割
    sorted_stopwords = sorted(stopwords, key=len, reverse=True)
    
    # 移除所有停用词 (包括单字和多字)
    filtered_text = text
    for stopword in sorted_stopwords:
        # 使用空格替换，确保词语被正确分隔
        filtered_text = filtered_text.replace(stopword, ' ')
    
    # 使用正则表达式匹配连续的汉字 (至少 2 个)
    chinese_words = regex_module.findall(r'[\u4e00-\u9fff]{2,}', filtered_text)
    
    # 重新构建文本，只保留有意义的词语
    meaningful_words = []
    for word in chinese_words:
        # 如果词语长度>=2，保留
        if len(word) >= 4:
            meaningful_words.append(word)
    
    # 用空格连接所有有意义的词语
    return ' '.join(meaningful_words)


def create_wordcloud(texts, output_path: str, title: str, color_map: str = 'Blues'):
    """创建词云图 - 词的大小和颜色深度都表示频率"""
    # 合并所有文本
    full_text = ' '.join(texts)
    
    # 移除停用词和单个汉字 (智能过滤)
    filtered_text = remove_stopwords_and_single_chars(full_text, CHINESE_STOPWORDS)
    
    # 统计词频
    import re as regex_module
    words = filtered_text.split()
    word_counts = Counter(words)
    
    # 只保留前 n 个高频词，避免过多词语
    top_words = dict(word_counts.most_common(50))
    
    # 创建词云 - 使用白色背景以便颜色深度更明显
    wc = WordCloud(
        font_path='/System/Library/Fonts/Supplemental/Songti.ttc',  # macOS 宋体
        width=1200,  # 减小宽度
        height=900,   # 减小高度
        background_color='white',
        max_words=30,  # 限制最大词语数
        colormap=color_map,
        contour_width=1,
        contour_color='steelblue',
        random_state=42,
        prefer_horizontal=0.7,  # 更多水平排列
        min_font_size=8,  # 最小字体大小
        max_font_size=150,  # 最大字体大小
        relative_scaling=0.3  # 词频影响大小程度
    ).generate_from_frequencies(top_words)
    
    # 显示词云
    plt.figure(figsize=(12, 9), facecolor='white')  # 调整图形大小
    plt.imshow(wc, interpolation='bilinear')
    plt.title(title, fontsize=16, fontproperties='SimSun')  # 调整标题大小
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"词云图已保存到：{output_path}")


def analyze_text_frequency(texts, top_n=20):
    """分析文本中出现频率最高的词语"""
    # 简单分词 (按字符级别)
    all_chars = []
    for text in texts:
        # 移除停用词和单个汉字 (智能过滤)
        filtered_text = remove_stopwords_and_single_chars(text, CHINESE_STOPWORDS)
        
        # 提取所有汉字
        import re as regex_module
        chinese_chars = list(filtered_text.replace(' ', ''))
        all_chars.extend(chinese_chars)
    
    # 统计词频
    char_counter = Counter(all_chars)
    top_chars = char_counter.most_common(top_n)
    
    print("\n出现频率最高的前 20 个汉字:")
    for char, count in top_chars:
        print(f"{char}: {count}次")
    
    return top_chars


def main():
    # 文件路径
    deepseek_path = '/Users/lun/Downloads/work/研究生课程/危机决策与沟通/Data-for-Emergency 阶段一中文未改/outputs/responses/phase2_responses_deepseek.json'
    qwen_path = '/Users/lun/Downloads/work/研究生课程/危机决策与沟通/Data-for-Emergency 阶段一中文未改/outputs/responses/phase2_responses_qwen.json'
    
    print("=" * 60)
    print("开始处理 DeepSeek 数据...")
    print("=" * 60)
    
    # 处理 DeepSeek 数据
    deepseek_texts = load_and_extract(deepseek_path)
    print(f"成功提取 {len(deepseek_texts)} 条理由文本")
    
    # 显示示例
    print("\n前 3 条示例:")
    for i, text in enumerate(deepseek_texts[:3], 1):
        print(f"\n示例 {i}:")
        print(text[:200] + "..." if len(text) > 200 else text)
    
    # 创建DeepSeek 词云
    create_wordcloud(
        deepseek_texts, 
        '/Users/lun/Downloads/work/研究生课程/危机决策与沟通/Data-for-Emergency 阶段一中文未改/outputs/wordcloud_deepseek_phase2.png',
        'DeepSeek Phase 2 理由文本词云',
        'Blues'
    )
    
    # 分析词频
    analyze_text_frequency(deepseek_texts)
    
    print("\n" + "=" * 60)
    print("开始处理 千问 (Qwen) 数据...")
    print("=" * 60)
    
    # 处理千问数据
    qwen_texts = load_and_extract(qwen_path)
    print(f"成功提取 {len(qwen_texts)} 条理由文本")
    
    # 显示示例
    print("\n前 3 条示例:")
    for i, text in enumerate(qwen_texts[:3], 1):
        print(f"\n示例 {i}:")
        print(text[:200] + "..." if len(text) > 200 else text)
    
    # 创建千问词云
    create_wordcloud(
        qwen_texts,
        '/Users/lun/Downloads/work/研究生课程/危机决策与沟通/Data-for-Emergency 阶段一中文未改/outputs/wordcloud_qwen_phase2.png',
        '千问 (Qwen) Phase 2 理由文本词云',
        'Oranges'
    )
    
    # 分析词频
    analyze_text_frequency(qwen_texts)
    
    print("\n" + "=" * 60)
    print("词云图生成完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
