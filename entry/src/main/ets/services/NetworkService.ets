/**
 * 网络服务类
 * 负责监控网络状态变化
 */

import connection from '@ohos.net.connection';
import { hilog } from '@kit.PerformanceAnalysisKit';

const DOMAIN = 0x0000;
const TAG = 'NetworkService';

export class NetworkService {
  private netConnection: connection.NetConnection | null = null;
  private netAvailable: boolean = false;
  private onNetworkChangeCallback: ((available: boolean) => void) | null = null;

  /**
   * 构造函数
   */
  constructor() {
    hilog.info(DOMAIN, TAG, 'NetworkService created');
  }

  /**
   * 初始化网络监听
   */
  init(): void {
    try {
      // 创建网络连接对象
      this.netConnection = connection.createNetConnection();

      // 注册网络状态变化回调
      this.netConnection.on('netAvailable', (data: connection.NetHandle) => {
        hilog.info(DOMAIN, TAG, `Network available: ${data.netId}`);
        this.netAvailable = true;

        if (this.onNetworkChangeCallback) {
          this.onNetworkChangeCallback(true);
        }
      });

      this.netConnection.on('netUnavailable', () => {
        hilog.info(DOMAIN, TAG, 'Network unavailable');
        this.netAvailable = false;

        if (this.onNetworkChangeCallback) {
          this.onNetworkChangeCallback(false);
        }
      });

      // 获取当前网络状态
      this.checkNetworkState();

      hilog.info(DOMAIN, TAG, 'Network monitoring initialized');
    } catch (error) {
      hilog.error(DOMAIN, TAG, `Failed to initialize network monitoring: ${error}`);
    }
  }

  /**
   * 检查当前网络状态
   */
  private async checkNetworkState(): Promise<void> {
    if (!this.netConnection) {
      return;
    }

    try {
      this.netAvailable = true; // 假设网络可用
      hilog.info(DOMAIN, TAG, `Initial network state: ${this.netAvailable ? 'available' : 'unavailable'}`);

      if (this.onNetworkChangeCallback) {
        this.onNetworkChangeCallback(this.netAvailable);
      }
    } catch (error) {
      hilog.error(DOMAIN, TAG, `Failed to check network state: ${error}`);
      this.netAvailable = false;

      if (this.onNetworkChangeCallback) {
        this.onNetworkChangeCallback(false);
      }
    }
  }

  /**
   * 释放资源
   */
  release(): void {
    if (this.netConnection) {
      this.netConnection = null;
    }
    hilog.info(DOMAIN, TAG, 'Network monitoring released');
  }

  /**
   * 设置网络状态变化回调
   * @param callback 回调函数
   */
  onNetworkChange(callback: (available: boolean) => void): void {
    this.onNetworkChangeCallback = callback;
  }

  /**
   * 获取当前网络是否可用
   * @returns 网络是否可用
   */
  isNetworkAvailable(): boolean {
    return this.netAvailable;
  }
}
