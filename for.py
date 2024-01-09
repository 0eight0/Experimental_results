import os

# 输入文件夹和输出文件夹的路径
input_folder = '/mnt/storage/wubinghong.wbh/deepl/torch_ner_new/datasets/dataset'  # 替换为实际的输入文件夹路径
output_folder = '/mnt/storage/wubinghong.wbh/deepl/torch_ner_new/datasets/dataset/format'  # 替换为实际的输出文件夹路径

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有.txt文件
for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        input_file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, file)

        # 读取文件内容并去除重复行
        with open(input_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            unique_lines = set(lines)

        # 将去除重复行后的内容写入新文件
        with open(output_file_path, 'a', encoding='utf-8') as f:
            f.writelines(unique_lines)
