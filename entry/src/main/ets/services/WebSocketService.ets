/**
 * WebSocket服务类
 * 负责与后端服务器建立WebSocket连接，发送和接收消息
 */

import webSocket from '@ohos.net.webSocket';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { CommonConstants } from '../common/constants/CommonConstants';
import { PlateInfo } from '../model/MessageTypes';
import util from '@ohos.util';
import { BusinessError } from '@kit.BasicServicesKit';

const DOMAIN = 0x0000;
const TAG = 'WebSocketService';

// 定义服务器端原始 plate 对象的接口
interface ServerSidePlateObject {
  plate_no: string;
  ocr_conf: number;
  rect: number[];
}

// 定义消息数据类型
interface MessageData {
  image?: string;
  message?: string;
  plates?: Array<ServerSidePlateObject>;
}

// 定义消息类型
interface WebSocketMessage {
  type: string;
  data?: MessageData;
  timestamp?: number;
}

export class WebSocketService {
  private socket: webSocket.WebSocket | null = null;
  private serverUrl: string = CommonConstants.SERVER_URL;
  private isConnected: boolean = false;
  private reconnectTimer: number | null = null;
  private heartbeatTimer: number | null = null;

  // 回调函数
  private onConnectCallback: (() => void) | null = null;
  private onDisconnectCallback: (() => void) | null = null;
  private onErrorCallback: ((error: string) => void) | null = null;
  private onPlateResultCallback: ((plates: PlateInfo[]) => void) | null = null;
  private onSystemMessageCallback: ((message: string) => void) | null = null;

  /**
   * 构造函数
   * @param serverUrl WebSocket服务器URL
   */
  constructor(serverUrl?: string) {
    if (serverUrl) {
      this.serverUrl = serverUrl;
    }
    hilog.info(DOMAIN, TAG, 'WebSocketService created');
  }

  /**
   * 连接到WebSocket服务器
   */
  connect(): void {
    if (this.isConnected) {
      hilog.info(DOMAIN, TAG, 'Already connected');
      return;
    }

    try {
      hilog.info(DOMAIN, TAG, `Connecting to ${this.serverUrl}`);

      // 创建WebSocket对象
      this.socket = webSocket.createWebSocket();

      // 注册WebSocket事件回调
      this.socket.on('open', (err: BusinessError, value: Object) => {
        if (err) {
          hilog.error(DOMAIN, TAG, `WebSocket open error: ${JSON.stringify(err)}`);
          if (this.onErrorCallback) {
            this.onErrorCallback(`连接建立时发生错误: ${err.message} (代码: ${err.code})`);
          }
          return;
        }
        hilog.info(DOMAIN, TAG, `WebSocket connection opened. Status: ${JSON.stringify(value)}`);
        this.isConnected = true;
        this.startHeartbeat();

        if (this.onConnectCallback) {
          this.onConnectCallback();
        }
      });

      this.socket.on('message', (err: BusinessError, data: string | ArrayBuffer) => {
        if (err) {
          hilog.error(DOMAIN, TAG, `WebSocket message event error: ${JSON.stringify(err)}`);
          if (this.onErrorCallback) {
            this.onErrorCallback(`接收消息时发生错误: ${err.message} (代码: ${err.code})`);
          }
          return;
        }
        this.handleMessage(data);
      });

      this.socket.on('close', (err: BusinessError, value: webSocket.CloseResult) => {
        let eventErrorMsg = "";
        if (err) {
          eventErrorMsg = ` (事件错误: ${err.message} [${err.code}])`;
          hilog.error(DOMAIN, TAG, `WebSocket close event error: ${JSON.stringify(err)}`);
        }
        const code = value ? value.code : -1;
        const reason = value ? value.reason : "No reason provided";
        hilog.info(DOMAIN, TAG, `WebSocket closed: Code ${code} - Reason '${reason}'${eventErrorMsg}`);
        this.isConnected = false;
        this.stopHeartbeat();

        if (this.onDisconnectCallback) {
          this.onDisconnectCallback();
        }
        this.scheduleReconnect();
      });

      this.socket.on('error', (errorData: BusinessError) => {
        hilog.error(DOMAIN, TAG, `WebSocket error: ${JSON.stringify(errorData)}`);
        if (this.onErrorCallback) {
          this.onErrorCallback(`WebSocket错误: ${errorData.message} (代码: ${errorData.code})`);
        }
      });

      // 连接到服务器
      this.socket.connect(this.serverUrl, (err: BusinessError, value: boolean) => {
        if (err) {
          hilog.error(DOMAIN, TAG, `Failed to connect: ${JSON.stringify(err)}`);
          if (this.onErrorCallback) {
            this.onErrorCallback(`连接失败: ${err.message} (代码: ${err.code})`);
          }
          this.scheduleReconnect();
        } else {
          hilog.info(DOMAIN, TAG, `Connection process initiated successfully: ${value}`);
          // 'open' 事件将确认实际的连接成功
        }
      });
    } catch (error) {
      hilog.error(DOMAIN, TAG, `Exception during connect: ${error}`);

      if (this.onErrorCallback) {
        this.onErrorCallback(`连接异常: ${error}`);
      }
    }
  }

  /**
   * 断开WebSocket连接
   */
  disconnect(): void {
    if (!this.isConnected || !this.socket) {
      return;
    }

    try {
      this.stopHeartbeat();
      this.cancelReconnect();

      this.socket.close((err) => {
        if (err) {
          hilog.error(DOMAIN, TAG, `Error closing connection: ${JSON.stringify(err)}`);
        } else {
          hilog.info(DOMAIN, TAG, 'Connection closed successfully');
          this.isConnected = false;
        }
      });
    } catch (error) {
      hilog.error(DOMAIN, TAG, `Exception during disconnect: ${error}`);
    }
  }

  /**
   * 发送车牌识别请求
   * @param imageBase64 Base64编码的图片数据
   */
  sendPlateRecognitionRequest(imageBase64: string): void {
    if (!this.isConnected || !this.socket) {
      hilog.error(DOMAIN, TAG, 'Cannot send message: not connected');

      if (this.onErrorCallback) {
        this.onErrorCallback('未连接到服务器');
      }
      return;
    }

    try {
      // 创建图片数据
      const imageData: MessageData = {
        image: imageBase64
      };

      // 创建消息对象
      const message: WebSocketMessage = {
        type: 'plate_recognition',
        data: imageData,
        timestamp: Date.now()
      };

      this.socket.send(JSON.stringify(message), (err) => {
        if (err) {
          hilog.error(DOMAIN, TAG, `Failed to send message: ${JSON.stringify(err)}`);

          if (this.onErrorCallback) {
            this.onErrorCallback(`发送失败: ${err.message}`);
          }
        } else {
          hilog.info(DOMAIN, TAG, 'Recognition request sent successfully');
        }
      });
    } catch (error) {
      hilog.error(DOMAIN, TAG, `Exception during send: ${error}`);

      if (this.onErrorCallback) {
        this.onErrorCallback(`发送异常: ${error}`);
      }
    }
  }

  /**
   * 处理接收到的消息
   * @param data 接收到的消息数据
   */
  private handleMessage(data: string | ArrayBuffer): void {
    try {
      let messageStr: string;

      if (typeof data === 'string') {
        messageStr = data;
      } else {
        // 将ArrayBuffer转换为字符串
        const decoder = util.TextDecoder.create('utf-8');
        const dataArray = new Uint8Array(data);
        messageStr = decoder.decode(dataArray);
      }

      // 解析消息
      const message: WebSocketMessage = JSON.parse(messageStr);

      hilog.info(DOMAIN, TAG, `Received message type: ${message.type}`);

      switch (message.type) {
        case 'plate_result':
          if (message.data && message.data.plates) {
            hilog.info(DOMAIN, TAG, `WebSocketService: Raw plates from server: ${JSON.stringify(message.data.plates)}`);

            // 数据映射：将服务器返回的结构转换为客户端的 PlateInfo[]
            const clientPlates: PlateInfo[] = message.data.plates.map(
              (serverPlate: ServerSidePlateObject): PlateInfo => {
                return {
                  text: serverPlate.plate_no,
                  confidence: serverPlate.ocr_conf,
                  box: serverPlate.rect
                };
              }
            );
            hilog.info(DOMAIN, TAG, `WebSocketService: Mapped clientPlates: ${JSON.stringify(clientPlates)}`);

            if (this.onPlateResultCallback) {
              this.onPlateResultCallback(clientPlates);
            }
          } else {
            hilog.warn(DOMAIN, TAG, 'Received plate_result message but message.data.plates is undefined or null.');
            if (this.onPlateResultCallback) { // 即使数据有问题，也通知回调一个空数组
              this.onPlateResultCallback([]);
            }
          }
          break;

        case 'system_message':
          if (this.onSystemMessageCallback && message.data && message.data.message) {
            this.onSystemMessageCallback(message.data.message);
          }
          break;

        case 'heartbeat_ack':
          hilog.info(DOMAIN, TAG, 'Received heartbeat acknowledgment');
          break;

        default:
          hilog.warn(DOMAIN, TAG, `Unknown message type: ${message.type}`);
      }
    } catch (error) {
      hilog.error(DOMAIN, TAG, `Error handling message: ${error}`);
    }
  }

  /**
   * 发送心跳消息
   */
  private sendHeartbeat(): void {
    if (!this.isConnected || !this.socket) {
      return;
    }

    try {
      // 创建心跳消息
      const message: WebSocketMessage = {
        type: 'heartbeat',
        timestamp: Date.now()
      };

      this.socket.send(JSON.stringify(message), (err) => {
        if (err) {
          hilog.error(DOMAIN, TAG, `Failed to send heartbeat: ${JSON.stringify(err)}`);
        }
      });
    } catch (error) {
      hilog.error(DOMAIN, TAG, `Exception during heartbeat: ${error}`);
    }
  }

  /**
   * 开始心跳定时器
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      this.sendHeartbeat();
    }, CommonConstants.HEARTBEAT_INTERVAL);
  }

  /**
   * 停止心跳定时器
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * 安排重新连接
   */
  private scheduleReconnect(): void {
    this.cancelReconnect();
    this.reconnectTimer = setTimeout(() => {
      hilog.info(DOMAIN, TAG, 'Attempting to reconnect...');
      this.connect();
    }, CommonConstants.RECONNECT_INTERVAL);
  }

  /**
   * 取消重新连接
   */
  private cancelReconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  /**
   * 设置连接成功回调
   * @param callback 回调函数
   */
  onConnect(callback: () => void): void {
    this.onConnectCallback = callback;
  }

  /**
   * 设置断开连接回调
   * @param callback 回调函数
   */
  onDisconnect(callback: () => void): void {
    this.onDisconnectCallback = callback;
  }

  /**
   * 设置错误回调
   * @param callback 回调函数
   */
  onError(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  /**
   * 设置车牌识别结果回调
   * @param callback 回调函数
   */
  onPlateResult(callback: (plates: PlateInfo[]) => void): void {
    this.onPlateResultCallback = callback;
  }

  /**
   * 设置系统消息回调
   * @param callback 回调函数
   */
  onSystemMessage(callback: (message: string) => void): void {
    this.onSystemMessageCallback = callback;
  }

  /**
   * 获取连接状态
   * @returns 是否已连接
   */
  isConnectedToServer(): boolean {
    return this.isConnected;
  }
}
