import os
import json

def list_files_structure(directory):
    def add_files_to_dict(base_path):
        directory_structure = {}
        for root, dirs, files in os.walk(base_path):
            relative_path = os.path.relpath(root, base_path)
            if relative_path == '.':
                relative_path = ''
            current_dir = directory_structure
            for part in relative_path.split(os.sep):
                if part not in current_dir:
                    current_dir[part] = {}
                current_dir = current_dir[part]
            i=0
            for file in files:
                current_dir[i] = file
                i+=1
        return directory_structure
    
    return add_files_to_dict(directory)

# 使用示例
directory_path = r"C:\Users\Hepisces\Desktop\提交材料\支撑材料"  # 确保路径正确
structure = list_files_structure(directory_path)

# 将结构保存为 JSON 文件
with open('directory_structure.json', 'w', encoding='utf-8') as f:
    json.dump(structure, f, indent=2, ensure_ascii=False)  # 确保 UTF-8 编码支持非 ASCII 字符
