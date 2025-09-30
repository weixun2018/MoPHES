import json
import os
from typing import Dict, List, Any

class MentalHealthDataClassifier:
    def __init__(self):
        # 定义四个分类类别
        self.categories = {
            "normal": lambda d, a: d == 0 and a == 0,
            "depression_only": lambda d, a: d != 0 and a == 0,
            "anxiety_only": lambda d, a: d == 0 and a != 0,
            "mixed": lambda d, a: d != 0 and a != 0
        }
    
    def parse_assistant_content(self, content: str) -> Dict[str, int]:
        """
        解析assistant回复中的JSON内容
        """
        try:
            # 尝试解析JSON字符串
            parsed_data = json.loads(content)
            
            # 提取depression和anxiety值
            depression = parsed_data.get('depression', 0)
            anxiety = parsed_data.get('anxiety', 0)
            
            return {
                'depression': depression,
                'anxiety': anxiety,
                'valid': True
            }
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始内容: {content}")
            return {
                'depression': 0,
                'anxiety': 0,
                'valid': False,
                'error': str(e)
            }
    
    def classify_data(self, depression: int, anxiety: int) -> str:
        """
        根据depression和anxiety值分类
        """
        for category_key, condition in self.categories.items():
            if condition(depression, anxiety):
                return category_key
        
        # 如果没有匹配到任何类别，返回默认类别
        return "unknown"
    
    def get_category_for_record(self, record: Dict[str, Any]) -> str:
        """
        获取单条记录的分类
        """
        messages = record.get('messages', [])
        
        # 找到assistant的回复
        for message in messages:
            if message.get('role') == 'assistant':
                assistant_content = message.get('content', '')
                
                # 解析assistant的回复
                parsed_result = self.parse_assistant_content(assistant_content)
                
                if not parsed_result['valid']:
                    return 'parse_error'
                
                # 分类
                return self.classify_data(parsed_result['depression'], parsed_result['anxiety'])
        
        return 'no_assistant_reply'
    
    def process_data_file(self, input_file: str) -> Dict[str, List[Dict]]:
        """
        处理整个数据文件，保持原始格式
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果输入是单个记录，转换为列表
            if isinstance(data, dict):
                data = [data]
            
            # 初始化结果字典
            results = {
                'normal': [],
                'depression_only': [],
                'anxiety_only': [],
                'mixed': [],
                'parse_error': [],
                'no_assistant_reply': [],
                'unknown': []
            }
            
            # 分类每条记录
            for record in data:
                category = self.get_category_for_record(record)
                
                if category in results:
                    results[category].append(record)  # 保持原始格式
                else:
                    results['unknown'].append(record)
            
            return results
            
        except FileNotFoundError:
            print(f"文件未找到: {input_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"JSON文件格式错误: {e}")
            return {}
        except Exception as e:
            print(f"处理文件时发生错误: {e}")
            return {}
    
    def save_results(self, results: Dict[str, List[Dict]], output_dir: str = "output"):
        """
        保存分类结果到文件，保持原始JSON格式
        """
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存每个类别到单独的JSON文件
        for category, records in results.items():
            if records:  # 只保存非空的类别
                output_file = os.path.join(output_dir, f"{category}.json")
                
                # 直接保存原始记录列表
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                
                print(f"已保存 {len(records)} 条 '{category}' 类别的数据到 {output_file}")
    
    def print_summary(self, results: Dict[str, List[Dict]]):
        """
        打印分类统计摘要
        """
        print("\n" + "="*50)
        print("分类结果统计")
        print("="*50)
        
        total_records = sum(len(records) for records in results.values())
        
        category_names = {
            'normal': '正常状态 (depression=0, anxiety=0)',
            'depression_only': '仅抑郁 (depression≠0, anxiety=0)',
            'anxiety_only': '仅焦虑 (depression=0, anxiety≠0)',
            'mixed': '抑郁焦虑混合 (depression≠0, anxiety≠0)',
            'parse_error': 'JSON解析错误',
            'no_assistant_reply': '无assistant回复',
            'unknown': '未知类别'
        }
        
        for category, records in results.items():
            count = len(records)
            if count > 0:
                percentage = (count / total_records * 100) if total_records > 0 else 0
                category_name = category_names.get(category, category)
                print(f"{category_name:35} : {count:4d} 条 ({percentage:5.1f}%)")
        
        print(f"{'总计':35} : {total_records:4d} 条")
        print("="*50)


def main():
    """
    主函数 - 处理固定路径的JSON文件
    """
    input_file = "auto_eval_judgment_data.json"  # 固定输入路径
    
    classifier = MentalHealthDataClassifier()
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件 '{input_file}' 不存在!")
        return
    
    print(f"开始处理文件: {input_file}")
    
    # 处理数据
    results = classifier.process_data_file(input_file)
    
    if not results:
        print("处理失败或文件为空")
        return
    
    # 打印统计信息
    classifier.print_summary(results)
    
    # 保存结果
    classifier.save_results(results)
    
    print(f"\n处理完成！结果已保存到 'output' 目录中。")
    print("每个分类的原始JSON数据都保存在对应的文件中。")


if __name__ == "__main__":
    main()