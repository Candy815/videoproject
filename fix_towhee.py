# fix_towhee.py
import os
import re


def fix_map_calls(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复 .map('col1', 'col1', func()) 为 .map('col1', 'col1', func())
    # 匹配 .map('单引号内容', 后面跟着的字符
    pattern1 = r'\.map\(([\'"])([^\'"]+)\1\s*,\s*([^,\)]+)\(\)\)'
    replacement1 = r'.map(\1\2\1, \1\2\1, \3())'

    # 修复 .map("col1", "col1", func()) 为 .map("col1", "col1", func())
    fixed_content = re.sub(pattern1, replacement1, content)

    if content != fixed_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"已修复: {file_path}")
        return True
    return False


# 查找所有 Python 文件
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            fix_map_calls(file_path)