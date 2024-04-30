import subprocess
import re

def run_detection(image_path):
    command = f"python detect.py --source {image_path}"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    
    # 打印完整的控制台输出以供调试
    print("Command Output:\n", result)
    
    # 使用正则表达式来查找人数
    matches = re.findall(r"(\d+) persons?", result.stdout)
    
    # 计算检测到的总人数
    total_persons_detected = sum(int(match) for match in matches)
    
    # 根据人数提供反馈
    if total_persons_detected > 0:
        print(f"Person detected in the image. Total persons: {total_persons_detected}.")
        return True
    else:
        print("No person detected in the image.")
        return False


if __name__ == "__main__":
    image_path = "data/images/bus.jpg"
    run_detection(image_path)
