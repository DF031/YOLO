/**
 * 消息类型定义
 */

/**
 * 基础消息接口
 */
export interface BaseMessage {
  type: string;
  timestamp: number;
}

/**
 * 车牌识别请求数据
 */
export interface PlateRecognitionData {
  image: string;
}

/**
 * 车牌识别请求消息
 */
export interface PlateRecognitionMessage extends BaseMessage {
  type: 'plate_recognition';
  data: PlateRecognitionData;
}

/**
 * 车牌识别结果数据
 */
export interface PlateResultData {
  plates: PlateInfo[];
}

/**
 * 车牌识别结果消息
 */
export interface PlateResultMessage extends BaseMessage {
  type: 'plate_result';
  data: PlateResultData;
}

/**
 * 系统消息数据
 */
export interface SystemMessageData {
  message: string;
}

/**
 * 系统消息
 */
export interface SystemMessage extends BaseMessage {
  type: 'system_message';
  data: SystemMessageData;
}

/**
 * 心跳消息
 */
export interface HeartbeatMessage extends BaseMessage {
  type: 'heartbeat';
}

/**
 * 心跳响应消息
 */
export interface HeartbeatAckMessage extends BaseMessage {
  type: 'heartbeat_ack';
}

/**
 * 车牌信息
 */
export interface PlateInfo {
  text: string;
  confidence: number;
  box: number[];
}
