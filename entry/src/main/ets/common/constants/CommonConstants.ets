/**
 * 通用常量
 */
export class CommonConstants {
  /**
   * WebSocket服务器URL
   */
  static readonly SERVER_URL: string = 'ws://172.16.23.87:5000/ws';
  
  /**
   * 心跳间隔（毫秒）
   */
  static readonly HEARTBEAT_INTERVAL: number = 30000;
  
  /**
   * 重连间隔（毫秒）
   */
  static readonly RECONNECT_INTERVAL: number = 5000;
  
  /**
   * 页面标题
   */
  static readonly APP_TITLE: string = '智能车牌识别系统';
  
  /**
   * 页面内边距
   */
  static readonly PAGE_PADDING: number = 16;
  
  /**
   * 组件间距
   */
  static readonly COMPONENT_SPACING: number = 16;
  
  /**
   * 按钮高度
   */
  static readonly BUTTON_HEIGHT: number = 40;
  
  /**
   * 图片预览高度
   */
  static readonly IMAGE_PREVIEW_HEIGHT: number = 240;
  
  /**
   * 结果项高度
   */
  static readonly RESULT_ITEM_HEIGHT: number = 60;
  
  /**
   * 标题字体大小
   */
  static readonly TITLE_FONT_SIZE: number = 24;
  
  /**
   * 副标题字体大小
   */
  static readonly SUBTITLE_FONT_SIZE: number = 18;
  
  /**
   * 正文字体大小
   */
  static readonly BODY_FONT_SIZE: number = 16;
  
  /**
   * 小字体大小
   */
  static readonly SMALL_FONT_SIZE: number = 14;
  
  /**
   * 主色调
   */
  static readonly PRIMARY_COLOR: string = '#007DFF';
  
  /**
   * 成功颜色
   */
  static readonly SUCCESS_COLOR: string = '#4CAF50';
  
  /**
   * 警告颜色
   */
  static readonly WARNING_COLOR: string = '#FF9800';
  
  /**
   * 错误颜色
   */
  static readonly ERROR_COLOR: string = '#F44336';
  
  /**
   * 背景颜色
   */
  static readonly BACKGROUND_COLOR: string = '#F5F5F5';
  
  /**
   * 卡片背景颜色
   */
  static readonly CARD_BACKGROUND_COLOR: string = '#FFFFFF';
  
  /**
   * 边框颜色
   */
  static readonly BORDER_COLOR: string = '#E0E0E0';
  
  /**
   * 文本颜色
   */
  static readonly TEXT_COLOR: string = '#333333';
  
  /**
   * 次要文本颜色
   */
  static readonly SECONDARY_TEXT_COLOR: string = '#666666';
  
  /**
   * 禁用文本颜色
   */
  static readonly DISABLED_TEXT_COLOR: string = '#999999';

  static readonly ANNOTATION_FONT_SIZE_VP: number = 12;
  static readonly ANNOTATION_BOX_COLOR: string = '#FF0000'; // Red
  static readonly ANNOTATION_LINE_WIDTH: number = 2;    // vp
  static readonly ANNOTATION_TEXT_BACKGROUND_COLOR: string = 'rgba(255, 255, 255, 0.75)'; // Semi-transparent white
  static readonly ANNOTATION_TEXT_COLOR: string = '#FF0000';      // Red
  static readonly CANVAS_BACKGROUND_COLOR: string = '#EEEEEE';    // Light gray
  static readonly IMAGE_PREVIEW_ALT_COLOR: string = '#F0F0F0'; // 定义一个浅灰色作为图片加载失败时的背景色

  /**
   * 允许上传的最大图片大小 (字节)
   * 例如：5MB = 5 * 1024 * 1024
   */
  static readonly MAX_IMAGE_SIZE_BYTES: number = 5 * 1024 * 1024;
}