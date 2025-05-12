# 智能车牌识别系统

这是一个基于鸿蒙前端和Python Flask后端的智能车牌识别系统。用户可以通过鸿蒙应用选择图片，系统会将图片发送到后端服务器进行车牌识别，并将识别结果返回给前端展示。

## 项目结构

```
YOLO1/
├── AppScope/                 # 应用级配置
├── backend/                  # Python Flask后端
│   ├── app.py                # Flask应用入口
│   ├── plate_recognizer.py   # 车牌识别模块
│   ├── requirements.txt      # Python依赖
│   └── README.md             # 后端说明文档
└── entry/                    # 鸿蒙前端
    ├── src/main/ets/
    │   ├── common/           # 通用工具和常量
    │   ├── components/       # UI组件
    │   ├── model/            # 数据模型
    │   ├── pages/            # 页面
    │   └── services/         # 服务类
    └── ...                   # 其他配置文件
```

## 功能特点

- 从相册选择图片
- 实时车牌识别
- WebSocket双向通信
- 网络状态监控
- 识别结果可视化展示

## 技术栈

### 前端 (鸿蒙 ArkTS)

- 鸿蒙应用开发框架
- WebSocket通信
- 图片处理和Base64转换
- UI组件和状态管理

### 后端 (Python Flask)

- Flask Web框架
- Flask-SocketIO WebSocket服务
- 图像处理
- 车牌识别算法（模拟版本）

## 开发环境要求

### 前端

- DevEco Studio 4.0或更高版本
- 鸿蒙SDK 5.0.5(17)或更高版本

### 后端

- Python 3.8或更高版本
- 安装requirements.txt中列出的依赖

## 运行说明

### 后端

1. 进入backend目录
2. 安装依赖：`pip install -r requirements.txt`
3. 运行服务器：`python app.py`

### 前端

1. 使用DevEco Studio打开项目
2. 修改`CommonConstants.ets`中的`SERVER_URL`为后端服务器地址
3. 构建并运行应用

## 通信协议

前端和后端通过WebSocket进行通信，消息格式为JSON。详细的消息格式请参考`开发文档.md`。

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交问题和改进建议。
