import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, device: str = "cuda"):
        # Select device
        use_cuda = torch.cuda.is_available() and device == "cuda"
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.ok = False
        self._init_model()

    def _init_model(self):
        # On CPU, default to lightweight fallback to keep the demo responsive
        if self.device.type != "cuda":
            print("[DepthEstimator] CPU device detected; using lightweight fallback depth.")
            self.model = None
            self.transform = None
            self.ok = False
            return
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(self.device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            # Use the default DPT transform
            self.transform = transforms.dpt_transform
            self.ok = True
        except Exception as e:
            # Graceful fallback if model cannot be loaded (e.g., no internet)
            print(f"[DepthEstimator] Failed to load MiDaS model: {e}. Falling back to simple gradient depth.")
            self.model = None
            self.transform = None
            self.ok = False

    @torch.inference_mode()
    def infer(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        if self.ok and self.model is not None and self.transform is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inp = self.transform(rgb).to(self.device)
            pred = self.model(inp)
            depth = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze().float().cpu().numpy()
            # Normalize to 0..1 (larger = farther)
            d = (depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth) + 1e-6)
            return d.astype(np.float32)
        # Fallback: synthetic but stable depth heuristic (center closer, horizon far)
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        x = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        d = np.clip(0.3 + 0.7 * (y + 0.15 * (x * x)), 0, 1)
        return d.astype(np.float32)
