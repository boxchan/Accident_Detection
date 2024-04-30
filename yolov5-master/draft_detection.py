import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run_detection(img_path, weights='yolov5s.pt', conf_thres=0.25, iou_thres=0.45, max_det=1000, device=''):
    # 确定设备
    device = select_device(device)

    # 加载模型
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = [640, 640]  # inference size (height, width)
    imgsz = check_img_size(imgsz, s=stride)  # 更新此处以使用utils.general.check_img_size

    # 数据加载器
    dataset = LoadImages(img_path, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch size

    # 运行检测
    person_detected = False
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # 添加batch维度

        # 推理
        pred = model(im)

        # 执行非最大抑制
        # pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic_nms=False, max_det=max_det)

        # 检查是否有人
        for i, det in enumerate(pred):  # 每张图片
            if len(det):
                # 将边界框从img_size缩放到im0大小
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                # 打印结果
                for *xyxy, conf, cls in reversed(det):
                    print("cls: ", cls)
                    if names[int(cls.item())] == 'person':  # 修改此处
                        person_detected = True
                        break
                if person_detected:
                    break

    return person_detected

if __name__ == "__main__":
    img_path = "./data/images/bus.jpg"  # 您需要根据实际路径修改此处
    person_detected = run_detection(img_path)
    print(f"Person Detected: {person_detected}")
