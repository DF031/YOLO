# 车牌识别系统后端

这是智能车牌识别系统的后端服务，基于Flask和SocketIO实现。

## 功能

- 通过WebSocket接收前端发送的图片
- 处理图片并识别车牌
- 将识别结果返回给前端
- 提供健康检查API

## 安装

1. 确保已安装Python 3.8或更高版本
2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 运行

```bash
python app.py
```

默认情况下，服务器将在`http://localhost:5000`上运行。

## API

### WebSocket端点

- 连接：`ws://localhost:5000/socket.io/`
- 事件：
  - `plate_recognition`：接收图片进行车牌识别
  - `plate_result`：发送识别结果
  - `system_message`：发送系统消息
  - `heartbeat`：心跳检测
  - `heartbeat_ack`：心跳响应

### HTTP API

- 健康检查：`GET /health`
  - 返回：`{"status": "ok", "timestamp": 1678886400000}`

## 消息格式

### 接收图片（前端 -> 后端）

```json
{
  "type": "plate_recognition",
  "data": {
    "image": "<base64_encoded_image_string>"
  },
  "timestamp": 1678886400000
}
```

### 识别结果（后端 -> 前端）

```json
{
  "type": "plate_result",
  "data": {
    "plates": [
      { "text": "京A88888", "confidence": 0.95, "box": [100, 50, 200, 80] },
      { "text": "沪B66666", "confidence": 0.89, "box": [300, 150, 400, 180] }
    ]
  },
  "timestamp": 1678886401000
}
```

### 系统消息（后端 -> 前端）

```json
{
  "type": "system_message",
  "data": {
    "message": "未检测到车牌"
  },
  "timestamp": 1678886402000
}
```
