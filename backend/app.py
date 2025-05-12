#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flask应用入口点
提供标准WebSocket服务和HTTP健康检查API
"""

import os
import json
import logging
import base64
import time
from datetime import datetime
from io import BytesIO
from PIL import Image

from flask import Flask, request, jsonify
# from flask_socketio import SocketIO, emit # 移除
from flask_sock import Sock # 导入 Flask-Sock

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 初始化Sock
sock = Sock(app)

# 导入车牌识别模块
from plate_recognizer import recognize_plate

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查API"""
    return jsonify({"status": "ok", "timestamp": int(time.time() * 1000)})

@sock.route('/ws') # 定义 WebSocket 路由，例如 /ws
def websocket_connection(ws): # ws 是 WebSocket 连接对象
    logger.info(f"WebSocket client connected: {request.remote_addr}")

    try:
        ws.send(json.dumps({
        'type': 'system_message',
            'data': {'message': '已连接到服务器 (standard WebSocket)'},
        'timestamp': int(time.time() * 1000)
        }))
    except Exception as e:
        logger.error(f"Error sending initial system message: {str(e)}")

    while True:
        try:
            message_str = ws.receive(timeout=None)
            if message_str is None:
                logger.info(f"WebSocket client disconnected: {request.remote_addr}")
                break

            logger.info(f"Received WebSocket message: {message_str} from {request.remote_addr}")
            message = json.loads(message_str)
            message_type = message.get('type')
        
            if message_type == 'plate_recognition':
                image_data_b64 = message.get('data', {}).get('image', '')
                if not image_data_b64:
                    raise ValueError("No image data provided in WebSocket message")

                if 'base64,' in image_data_b64:
                    image_data_b64 = image_data_b64.split('base64,')[1]
            
                image_bytes = base64.b64decode(image_data_b64)
                image = Image.open(BytesIO(image_bytes))
            
                plates = recognize_plate(image)
            
                ws.send(json.dumps({
                'type': 'plate_result',
                'data': {'plates': plates},
                'timestamp': int(time.time() * 1000)
                }))
                logger.info(f"Sent recognition results to {request.remote_addr}: {len(plates)} plates found")
            
            elif message_type == 'heartbeat':
                ws.send(json.dumps({
                    'type': 'heartbeat_ack',
                    'timestamp': int(time.time() * 1000)
                }))
                logger.debug(f"Sent heartbeat_ack to {request.remote_addr}")
            
            else:
                logger.warn(f"Unknown WebSocket message type received: {message_type}")
                ws.send(json.dumps({
                    'type': 'system_message',
                    'data': {'message': f'未知消息类型: {message_type}'},
                    'timestamp': int(time.time() * 1000)
                }))

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from WebSocket message: {str(e)}")
            try:
                ws.send(json.dumps({
                    'type': 'system_message',
                    'data': {'message': f'JSON解析错误: {str(e)}'},
                    'timestamp': int(time.time() * 1000)
                }))
            except Exception as send_e:
                logger.error(f"Error sending JSON error message: {str(send_e)}")
        except ValueError as e:
            logger.error(f"ValueError processing WebSocket message: {str(e)}")
            try:
                ws.send(json.dumps({
                    'type': 'system_message',
                    'data': {'message': f'处理请求时发生错误: {str(e)}'},
                    'timestamp': int(time.time() * 1000)
                }))
            except Exception as send_e:
                logger.error(f"Error sending ValueError message: {str(send_e)}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message from {request.remote_addr}: {str(e)}", exc_info=True)
            try:
                ws.send(json.dumps({
                'type': 'system_message',
                    'data': {'message': f'服务器内部错误: {str(e)}'},
                'timestamp': int(time.time() * 1000)
                }))
            except Exception as send_e:
                logger.error(f"Error sending generic error message to {request.remote_addr}: {str(send_e)}")
    
    logger.info(f"WebSocket connection with {request.remote_addr} closed from server side or loop ended.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask server with Flask-Sock on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
