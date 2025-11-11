import cv2
import torch
import numpy as np
import torchvision
from PIL import Image

class Segmenter:
    def __init__(self, device: str = "cuda"):
        use_cuda = torch.cuda.is_available() and device == "cuda"
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model, self.preproc = self._init_model()

    def _init_model(self):
        try:
            # Prefer a lighter model on CPU for responsiveness
            if self.device.type == "cpu":
                try:
                    weights = torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
                    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=weights)
                    preproc = weights.transforms()
                except Exception:
                    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)
                    from torchvision import transforms
                    preproc = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                    ])
            else:
                try:
                    weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
                    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
                    preproc = weights.transforms()
                except Exception:
                    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
                    from torchvision import transforms
                    preproc = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                    ])
            model = model.to(self.device).eval()
            return model, preproc
        except Exception as e:
            print(f"[Segmenter] Failed to load segmentation model: {e}. Falling back to color-threshold segmentation.")
            return None, None

    @torch.inference_mode()
    def infer_road_mask(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        if self.model is not None and self.preproc is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inp_img = Image.fromarray(rgb)
            inp = self.preproc(inp_img).unsqueeze(0).to(self.device)
            out = self.model(inp)["out"]  # (1, C, H, W)
            prob = torch.softmax(out, dim=1).squeeze(0)  # (C, H, W)
            prob_np = prob.cpu().numpy()
            # COCO/VOC-like classes: cars(7), bus(6), train(21), bicycle(2), motorcycle(14), person(15)
            # Indices vary by weights; use safe bounds and fallbacks.
            idxs = [2, 6, 7, 14, 15, 21]
            idxs = [i for i in idxs if i < prob_np.shape[0]]
            obstacle = np.zeros((prob_np.shape[1], prob_np.shape[2]), dtype=np.float32)
            for i in idxs:
                obstacle = np.maximum(obstacle, prob_np[i])
            obstacle_mask = (obstacle > 0.30).astype(np.uint8)
            road_like = (1 - obstacle_mask).astype(np.uint8)
            return road_like
        # Fallback: basic color+brightness heuristic for asphalt-like region
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        road_like = ((gray > 50) & (gray < 200)).astype(np.uint8)
        return road_like
