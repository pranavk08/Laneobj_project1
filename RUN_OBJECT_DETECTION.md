# 🚀 Quick Start - Object Detection System

## ✅ System Ready!

All dependencies are installed. You can run the system immediately.

---

## 🎯 Run Now

### **Option 1: Default (Recommended)**
```powershell
python main.py
```
This will:
- Use `project_video.mp4` as input
- Show live display window
- Save to `output.mp4`
- Run on GPU if available

---

### **Option 2: Custom Video**
```powershell
python main.py --video your_video.mp4
```

---

### **Option 3: Webcam**
```powershell
python main.py --cam 0
```

---

### **Option 4: Fast Mode (No Display)**
```powershell
python main.py --video project_video.mp4 --no-display --output result.mp4
```

---

## 📊 What You'll See

The output video will show:
- ✅ **Bounding boxes** around detected objects (cars, people, trucks, buses, motorcycles)
- ✅ **Class labels** with confidence scores
- ✅ **Tracking IDs** - each object gets a unique ID
- ✅ **Lane overlay** - detected lanes in green/blue
- ✅ **FPS counter** - real-time performance
- ✅ **Object count** - number of objects detected

---

## 🎮 Controls

While running:
- **Press 'q'** to quit
- Window closes automatically when done

---

## ⚙️ Advanced Options

```powershell
# Use different YOLO model (faster)
python main.py --model yolov8n.pt

# Higher confidence threshold
python main.py --conf 0.7

# Force CPU
python main.py --device cpu

# Force GPU
python main.py --device cuda

# Custom output
python main.py --video input.mp4 --output my_result.mp4
```

---

## 📈 Expected Performance

With your RTX 3060:
- **FPS**: 15-20 with lane detection
- **Accuracy**: High
- **GPU Memory**: ~3GB

---

## 🔍 Example Output

```
🔧 Initializing Driving Assistant...
   Device: cuda
📦 Loading YOLOv8 model: yolov8s.pt
   ✓ Model loaded on GPU
📹 Opening video: project_video.mp4
   Resolution: 1280x720
✓ Lane detector initialized

🚀 Starting detection...
Progress: 15.9% | Frame 200/1260 | FPS: 18.5
Progress: 31.7% | Frame 400/1260 | FPS: 18.7

✅ Processing complete!
   Output saved: output.mp4

📊 Detection Statistics:
   car: 3456
   person: 145
   truck: 89
```

---

## ❓ Need Help?

**Check GPU:**
```powershell
python check_gpu.py
```

**View detailed docs:**
- See `OBJECT_DETECTION_README.md`

---

## 🎉 Ready to Go!

Just run:
```powershell
python main.py
```

That's it! Your object detection + lane integration system is production-ready!
