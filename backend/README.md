# 车牌识别后端说明文档

## 目录

1. [项目简介](#项目简介)
2. [主要依赖](#主要依赖)
3. [系统架构](#系统架构)
4. [核心模块说明](#核心模块说明)
   - [app.py](#apppy)
   - [plate_recognizer.py](#plate_recognizerpy)
5. [接口说明](#接口说明)
   - [HTTP接口](#http接口)
   - [WebSocket接口](#websocket接口)
6. [模型说明](#模型说明)
7. [如何接入自定义OCR模型](#如何接入自定义ocr模型)
8. [日志与调试](#日志与调试)
9. [常见问题](#常见问题)
10. [致谢](#致谢)

---

## 项目简介

本后端服务基于 Flask + Flask-Sock，结合 YOLOv8（自训练模型）和 EasyOCR，实现了车牌检测与识别。支持标准 WebSocket 实时通信和 HTTP 健康检查。后续集成自定义 OCR 模型以提升车牌码识别置信度。

---

## 主要依赖

- Python 3.7+
- Flask
- Flask-Sock
- torch
- ultralytics
- easyocr
- opencv-python
- pillow
- numpy

安装依赖（示例）：

```bash
pip install flask flask-sock torch ultralytics easyocr opencv-python pillow numpy
```

---

## 系统架构

```
┌────────────┐
│前端/客户端 │
└─────┬──────┘
      │ WebSocket/HTTP
┌─────▼──────┐
│  Flask后端 │
│  app.py    │
└─────┬──────┘
      │
┌─────▼────────────┐
│ plate_recognizer │
│ YOLOv11检测+OCR  │
└──────────────────┘
```

---

## 核心模块说明

### app.py

- Flask 应用主入口，负责：
  - 提供 `/health` 健康检查 HTTP API
  - 提供 `/ws` WebSocket 服务，支持实时图片识别请求
  - 统一日志输出
  - 处理消息类型（车牌识别、心跳、异常等）

### plate_recognizer.py

- 负责模型加载、图片预处理、车牌检测（YOLO）、车牌码识别（EasyOCR/自定义OCR）
- 支持模型热加载，首次调用自动加载模型
- 识别结果结构化输出，包括车牌号、置信度、位置等

---

## 接口说明

### HTTP接口

#### 1. 健康检查

- **URL**：`/health`
- **方法**：GET
- **返回示例**：

```json
{
  "status": "ok",
  "timestamp": 1710000000000
}
```

### WebSocket接口

#### 1. 连接

- **URL**：`ws://<host>:<port>/ws`
- **协议**：标准WebSocket

#### 2. 消息格式

- **请求消息**（车牌识别）：

```json
{
  "type": "plate_recognition",
  "data": {
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."  // base64图片
  }
}
```

- **响应消息**（识别结果）：

```json
{
  "type": "plate_result",
  "data": {
    "plates": [
      {
        "plate_no": "粤B12345",
        "plate_color": "未知",
        "rect": [x1, y1, x2, y2],
        "detect_conf": 0.98,
        "ocr_conf": 0.92,
        "roi_height": 40,
        "color_conf": 0.0,
        "plate_type": 0
      }
    ]
  },
  "timestamp": 1710000000000
}
```

- **心跳包**：

  - 请求：`{"type": "heartbeat"}`
  - 响应：`{"type": "heartbeat_ack", "timestamp": ...}`
- **异常/系统消息**：

  - 统一返回 `type: system_message`，`data.message` 字段为具体内容

---

## 模型说明

- **检测模型**：YOLOv11 自主训练模型 `best11n.pt`，用于车牌定位
- **识别模型**：当前集成 EasyOCR（支持中英文），后续替换为自训练OCR模型
- **模型路径**：`xx/YOLO/backend/weights/`
- **置信度阈值**：可在 `plate_recognizer.py` 中调整 `CONF_THRESHOLD`、`IOU_THRESHOLD`

---

## 如何接入自定义OCR模型

1. **训练好OCR模型**，保存为可加载的格式（如 PyTorch、ONNX 等）
2. **在 `plate_recognizer.py` 中加载自定义OCR模型**，替换 EasyOCR 相关部分
   - 推荐将 OCR 相关逻辑封装为独立函数，如 `custom_ocr(roi_img)`，并在 `_perform_recognition_internal` 中调用
3. **输出格式保持一致**，即返回 `plate_no`、`ocr_conf` 等字段
4. **测试集成效果**，确保接口兼容

**示例伪代码：**

```python
# 加载自定义OCR模型
CUSTOM_OCR_MODEL = torch.load('your_ocr_model.pt')

def custom_ocr(roi_img):
    # 预处理
    # 推理
    # 后处理
    return plate_text, ocr_conf

# 替换 EasyOCR 部分
plate_text, ocr_conf = custom_ocr(roi_img)
```

---

## 日志与调试

- 日志默认输出到控制台，包含模型加载、识别流程、异常等信息
- 可根据需要调整日志级别（INFO/DEBUG）
- 识别异常会通过 WebSocket 返回 `system_message`，便于前端提示

---

## 常见问题

1. **模型加载失败**
   - 检查模型路径、文件名是否正确
   - 检查依赖库版本
2. **图片识别无结果**
   - 检查图片格式、分辨率
   - 检查模型置信度阈值设置
3. **WebSocket无法连接**
   - 检查端口、防火墙
   - 检查 Flask-Sock 是否正常安装

---

## 致谢

- YOLO/Ultralytics
- EasyOCR
- Flask/Flask-Sock
