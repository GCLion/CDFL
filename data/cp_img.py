import os
import shutil
from tqdm import tqdm  # 进度条库，可选

def copy_ori_mask_from_txt(txt_path, target_parent_folder):
    """
    从txt文件读取原图和mask路径，分别复制到target_parent_folder下的ori和mask文件夹
    :param txt_path: 包含路径的txt文件路径
    :param target_parent_folder: 目标父文件夹（ori和mask将作为其子文件夹）
    """
    # 1. 验证输入txt文件
    if not os.path.exists(txt_path):
        print(f"错误：txt文件不存在 → {txt_path}")
        return
    
    # 2. 创建目标文件夹（ori和mask子文件夹）
    ori_folder = os.path.join(target_parent_folder, "ori")
    mask_folder = os.path.join(target_parent_folder, "mask")
    os.makedirs(ori_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    print(f"目标文件夹结构：\n- 原图：{ori_folder}\n- Mask：{mask_folder}")
    
    # 3. 读取txt文件内容并解析路径
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]  # 过滤空行
    
    if not lines:
        print("警告：txt文件中未找到有效内容")
        return
    
    # 4. 统计变量
    total = len(lines)
    success_ori = 0  # 原图复制成功数
    success_mask = 0  # mask复制成功数
    fail_ori_paths = []  # 原图复制失败路径
    fail_mask_paths = []  # mask复制失败路径
    invalid_lines = []  # 无效行（未按逗号分割出两个路径）
    
    # 5. 批量复制
    for idx, line in enumerate(tqdm(lines, desc="处理进度")):
        # 分割原图和mask路径（处理可能的空格）
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 2:
            invalid_lines.append(f"第{idx+1}行：{line}（未找到有效的两个路径）")
            continue
        
        ori_path, mask_path = parts
        
        # 复制原图到ori文件夹
        if os.path.exists(ori_path):
            ori_filename = os.path.basename(ori_path)
            target_ori = os.path.join(ori_folder, ori_filename)
            # 处理文件名冲突
            if os.path.exists(target_ori):
                name, ext = os.path.splitext(ori_filename)
                suffix = 1
                while os.path.exists(os.path.join(ori_folder, f"{name}_{suffix}{ext}")):
                    suffix += 1
                target_ori = os.path.join(ori_folder, f"{name}_{suffix}{ext}")
            # 执行复制
            try:
                shutil.copy2(ori_path, target_ori)
                success_ori += 1
            except Exception as e:
                fail_ori_paths.append(f"原图复制失败：{ori_path}，原因：{str(e)}")
        else:
            fail_ori_paths.append(f"原图不存在：{ori_path}")
        
        # 复制mask到mask文件夹
        if os.path.exists(mask_path):
            mask_filename = os.path.basename(mask_path)
            target_mask = os.path.join(mask_folder, mask_filename)
            # 处理文件名冲突
            if os.path.exists(target_mask):
                name, ext = os.path.splitext(mask_filename)
                suffix = 1
                while os.path.exists(os.path.join(mask_folder, f"{name}_{suffix}{ext}")):
                    suffix += 1
                target_mask = os.path.join(mask_folder, f"{name}_{suffix}{ext}")
            # 执行复制
            try:
                shutil.copy2(mask_path, target_mask)
                success_mask += 1
            except Exception as e:
                fail_mask_paths.append(f"Mask复制失败：{mask_path}，原因：{str(e)}")
        else:
            fail_mask_paths.append(f"Mask不存在：{mask_path}")
    
    # 6. 输出统计结果
    print("\n" + "="*60)
    print(f"总处理行数：{total}")
    print(f"有效行：{total - len(invalid_lines)}，无效行：{len(invalid_lines)}")
    print(f"原图复制：成功{success_ori}/{total}，失败{len(fail_ori_paths)}")
    print(f"Mask复制：成功{success_mask}/{total}，失败{len(fail_mask_paths)}")
    
    if invalid_lines:
        print("\n无效行详情：")
        for line in invalid_lines[:5]:  # 只显示前5条，避免过长
            print(f"- {line}")
        if len(invalid_lines) > 5:
            print(f"- 还有{len(invalid_lines)-5}条无效行未显示...")
    
    if fail_ori_paths or fail_mask_paths:
        print("\n失败详情：")
        for p in fail_ori_paths[:3] + fail_mask_paths[:3]:  # 各显示前3条
            print(f"- {p}")
        if len(fail_ori_paths) + len(fail_mask_paths) > 6:
            print(f"- 还有{len(fail_ori_paths) + len(fail_mask_paths) - 6}条失败记录未显示...")


# -------------------------- 配置参数 --------------------------
txt_file_path = "image_paths.txt"  # 你的txt文件路径（相对或绝对路径）
target_parent = "./"  # 目标父文件夹（ori和mask会创建在此目录下）
# --------------------------------------------------------------

if __name__ == "__main__":
    copy_ori_mask_from_txt(txt_file_path, target_parent)