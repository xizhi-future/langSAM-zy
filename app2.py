import os
import warnings
import torch
import gradio as gr
import numpy as np
from PIL import Image
from lang_sam import LangSAM
from lang_sam import SAM_MODELS
from lang_sam.utils import draw_image
from lang_sam.utils import load_image

warnings.filterwarnings("ignore")
SAM_MODELS = {
    "vit_h": "/root/lang-segment-anything-main/weight/sam-weight/sam_vit_h_4b8939.pth",
    # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

class LitGradio:
    def __init__(self, sam_type="vit_h"):
        self.model = LangSAM(sam_type)
        self.sam_type = sam_type

    def predict(self, sam_type, box_threshold, text_threshold, image_path, text_prompt):
        print("Predicting... ", sam_type, box_threshold, text_threshold, image_path, text_prompt)
        if sam_type != self.sam_type:
            self.model.build_sam(sam_type)
        image_pil = load_image(image_path)
        masks, boxes, phrases, logits = self.model.predict(image_pil, text_prompt, box_threshold, text_threshold)
        labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
        image_array = np.asarray(image_pil)
        
        # 将 PyTorch 张量转换为 NumPy 数组，并确保形状正确
        if isinstance(boxes, torch.Tensor):
            # 确保 boxes 是在 CPU 上
            boxes = boxes.detach().cpu().numpy()
            # 调整形状以匹配 (N, 4)，其中 N 是边界框的数量
            if boxes.ndim == 2 and boxes.shape[0] == 4:
                boxes = boxes.reshape(-1, 4)  # 重新调整形状以匹配 (N, 4)
        
        # 将 PyTorch 张量转换为 NumPy 数组，并使用 .bool() 转换 masks 为布尔类型
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
            if masks.dtype == 'uint8':
                masks = masks.astype(bool)
            else:
                masks = masks > 0.5  # 假设 masks 是二值化的，需要阈值处理
        elif isinstance(masks, np.ndarray):
            if masks.dtype == 'uint8':
                masks = masks.astype(bool)
            else:
                masks = masks > 0.5  # 假设 masks 是二值化的，需要阈值处理
        else:
            raise ValueError("Unsupported type for masks")
        
        probs = logits.detach().cpu().numpy()  # 假设 logits 可以直接用作 probs 参数
        
        image = draw_image(image_array, masks, boxes, probs, labels)
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        return image

    def setup_interface(self):
        print(list(SAM_MODELS.keys()))
        inputs = [
            gr.Dropdown(choices=list(SAM_MODELS.keys()), label="SAM model", value="vit_h"),
            gr.Slider(0, 1, value=0.3, label="Box threshold"),
            gr.Slider(0, 1, value=0.25, label="Text threshold"),
            gr.Image(type="filepath", label='Image'),
            gr.Textbox(lines=1, label="Text Prompt"),
        ]
        outputs = [gr.Image(type="pil", label="Output Image")]  # 修正这里
    
        examples = [
            [
                'vit_h',
                0.36,
                0.25,
                os.path.join(os.path.dirname(__file__), "assets", "fruits.jpg"),
                "kiwi",
            ],
            [
                'vit_h',
                0.3,
                0.25,
                os.path.join(os.path.dirname(__file__), "assets", "car.jpeg"),
                "wheel",
            ],
            [
                'vit_h',
                0.3,
                0.25,
                os.path.join(os.path.dirname(__file__), "assets", "food.jpg"),
                "food",
            ],
        ]
    
        demo = gr.Interface(
            fn=self.predict,
            inputs=inputs,
            outputs=outputs,
            examples=examples,
            title="LangSAM Gradio Interface"
        )
        return demo


if __name__ == "__main__":
    lg = LitGradio()
    app = lg.setup_interface()
    app.launch()