import { hilog } from '@kit.PerformanceAnalysisKit';
import promptAction from '@ohos.promptAction';
import { WebSocketService } from '../services/WebSocketService';
import { NetworkService } from '../services/NetworkService';
import { ImageService } from '../services/ImageService';
import { PlateInfo } from '../model/MessageTypes';
import { CommonConstants } from '../common/constants/CommonConstants';
import { PlateResultItem } from '../components/PlateResultItem';

import image from '@ohos.multimedia.image';
import { PlateAnnotatorCanvas } from '../components/PlateAnnotatorCanvas'; // <--- 新增: 导入新组件

const DOMAIN = 0x0000;
const TAG = 'Index';

@Entry
@Component
struct Index {
  @State isConnected: boolean = false;
  @State isLoading: boolean = false;
  @State errorMessage: string = '';
  @State imagePath: string = ''; // Still used for Base64 conversion and potentially as a fallback
  @State recognizedPlates: PlateInfo[] = [];
  @State selectedPixelMap: image.PixelMap | null = null;
  @State selectedImageInfo: image.ImageInfo | null = null;


  private webSocketService: WebSocketService = new WebSocketService();
  private networkService: NetworkService = new NetworkService();
  private imageService: ImageService = new ImageService();


  aboutToAppear() {
    // 初始化网络监控
    this.networkService.init();
    this.networkService.onNetworkChange((available: boolean) => {
      hilog.info(DOMAIN, TAG, `Network state changed: ${available ? 'available' : 'unavailable'}`);

      if (available) {
        // 网络可用时尝试连接WebSocket
        this.connectWebSocket();
      } else {
        // 网络不可用时显示错误信息
        this.isConnected = false;
        this.errorMessage = '网络不可用，请检查网络连接';
      }
    });

    // 初始化WebSocket服务
    this.initWebSocketService();

    // 如果网络可用，连接WebSocket
    if (this.networkService.isNetworkAvailable()) {
      this.connectWebSocket();
    }
  }

  aboutToDisappear() {
    // 断开WebSocket连接
    this.webSocketService.disconnect();

    // 释放网络监控资源
    this.networkService.release();
  }

  /**
   * 初始化WebSocket服务
   */
  private initWebSocketService() {
    // 连接成功回调
    this.webSocketService.onConnect(() => {
      this.isConnected = true;
      this.errorMessage = '';
    });

    // 断开连接回调
    this.webSocketService.onDisconnect(() => {
      this.isConnected = false;
      this.errorMessage = '与服务器的连接已断开';
    });

    // 错误回调
    this.webSocketService.onError((error: string) => {
      this.errorMessage = error;
    });

    // 车牌识别结果回调
    this.webSocketService.onPlateResult((plates: PlateInfo[]) => {
      this.isLoading = false;

      hilog.info(DOMAIN, TAG, `Received plates from WebSocketService: ${JSON.stringify(plates)}`);

      if (plates && plates.length > 0) {
        plates.forEach((p, index) => {
          hilog.info(DOMAIN, TAG, `Plate ${index}: text=${p.text}, confidence=${p.confidence}, box=${JSON.stringify(p.box)}`);
        });
      } else {
        hilog.info(DOMAIN, TAG, 'Received empty plates array or plates is undefined/null.');
      }

      this.recognizedPlates = plates;

      if (this.recognizedPlates && this.recognizedPlates.length === 0 && (this.imagePath || this.selectedPixelMap)) {
        promptAction.showToast({
          message: '未检测到车牌',
          duration: 2000
        });
      } else if (this.recognizedPlates && this.recognizedPlates.length > 0) {
        promptAction.showToast({
          message: `检测到 ${this.recognizedPlates.length} 个车牌`,
          duration: 2000
        });
      }
    });

    // 系统消息回调
    this.webSocketService.onSystemMessage((message: string) => {
      this.isLoading = false;
      promptAction.showToast({
        message: message,
        duration: 3000
      });
    });
  }

  /**
   * 连接WebSocket服务器
   */
  private connectWebSocket() {
    if (!this.webSocketService.isConnectedToServer()) {
      this.webSocketService.connect();
    }
  }

  /**
   * 选择图片并发送识别请求
   */
  private async selectAndRecognize() {
    if (!this.isConnected) {
      promptAction.showToast({
        message: '未连接到服务器，请稍后再试',
        duration: 2000
      });
      return;
    }

    try {
      const tempImagePath = await this.imageService.selectImageFromGallery();
      if (!tempImagePath) {
        return;
      }

      this.imagePath = tempImagePath;
      this.isLoading = true;
      this.recognizedPlates = [];
      this.selectedPixelMap = null;
      this.selectedImageInfo = null;
      let loadedPixelMap: image.PixelMap | null = null;

      try {
        loadedPixelMap = await this.imageService.loadPixelMap(this.imagePath);
        if (loadedPixelMap) {
          const imageInfo = loadedPixelMap.getImageInfoSync();
          if (!imageInfo || !imageInfo.size || imageInfo.size.width === 0 || imageInfo.size.height === 0) {
            hilog.error(DOMAIN, TAG, `PixelMap loaded, but ImageInfo is invalid: ${JSON.stringify(imageInfo)}`);
            this.isLoading = false;
            this.errorMessage = `加载的图片信息无效`;
            promptAction.showToast({ message: this.errorMessage, duration: 3000 });
            this.selectedPixelMap = null;
            this.selectedImageInfo = null;
            return;
          }
          hilog.info(DOMAIN, TAG, `PixelMap loaded and ImageInfo obtained successfully in Index. Width: ${imageInfo.size.width}`);
          this.selectedPixelMap = loadedPixelMap;
          this.selectedImageInfo = imageInfo;
        } else {
          hilog.error(DOMAIN, TAG, 'ImageService.loadPixelMap returned null.');
          throw new Error('加载PixelMap失败 (返回null)');
        }
      } catch (pixelMapError) {
        hilog.error(DOMAIN, TAG, `Failed to load PixelMap or get ImageInfo in Index: ${pixelMapError.message || JSON.stringify(pixelMapError)}`);
        this.isLoading = false;
        this.errorMessage = `加载图片预览失败: ${pixelMapError.message || JSON.stringify(pixelMapError)}`;
        promptAction.showToast({ message: this.errorMessage, duration: 3000 });
        this.selectedPixelMap = null;
        this.selectedImageInfo = null;
        return;
      }


      if (!this.selectedPixelMap || !this.selectedImageInfo) {
        hilog.warn(DOMAIN, TAG, "selectedPixelMap or selectedImageInfo is null after load and ImageInfo check, aborting recognition request.");
        if (this.isLoading) {
            this.isLoading = false;
        }
        return;
      }

      const imageBase64 = await this.imageService.convertImageToBase64(this.imagePath);
      this.webSocketService.sendPlateRecognitionRequest(imageBase64);

    } catch (error) {
      this.isLoading = false;
      this.errorMessage = `操作失败: ${error.message || error}`;
      promptAction.showToast({ message: this.errorMessage, duration: 3000 });
    }
  }

  build() {
    Column() {
      // 标题栏
      Row() {
        Text(CommonConstants.APP_TITLE)
          .fontSize(CommonConstants.TITLE_FONT_SIZE)
          .fontWeight(FontWeight.Bold)
          .fontColor(CommonConstants.TEXT_COLOR)

        Blank()

        Text(this.isConnected ? '已连接' : '未连接')
          .fontSize(CommonConstants.SMALL_FONT_SIZE)
          .fontColor(this.isConnected ? CommonConstants.SUCCESS_COLOR : CommonConstants.ERROR_COLOR)
      }
      .width('100%')
      .padding(CommonConstants.PAGE_PADDING)

      // 错误信息
      if (this.errorMessage) {
        Text(this.errorMessage)
          .fontSize(CommonConstants.SMALL_FONT_SIZE)
          .fontColor(CommonConstants.ERROR_COLOR)
          .width('100%')
          .textAlign(TextAlign.Center)
          .padding(CommonConstants.PAGE_PADDING)
          .backgroundColor('#FFEBEE')
          .margin({ bottom: CommonConstants.COMPONENT_SPACING })
      }

      // 启用 PlateAnnotatorCanvas 进行预览和标注
      if (this.selectedPixelMap && this.selectedImageInfo) {
        Stack() {
        Image(this.selectedPixelMap)
          .width('100%')
          .height(CommonConstants.IMAGE_PREVIEW_HEIGHT)
          .objectFit(ImageFit.Contain)
            .backgroundColor(CommonConstants.CARD_BACKGROUND_COLOR) // 背景可以保留，以防图片加载有延迟
            .alt(CommonConstants.IMAGE_PREVIEW_ALT_COLOR)

          PlateAnnotatorCanvas({
            pixelMap: this.selectedPixelMap, // 仍然可以传递，虽然内部不再绘制它
            plates: this.recognizedPlates,
            canvasDisplayHeight: CommonConstants.IMAGE_PREVIEW_HEIGHT,
            imageInfo: this.selectedImageInfo
          })
          .width('100%')
        }
        .width('100%')
        .height(CommonConstants.IMAGE_PREVIEW_HEIGHT) // 确保Stack有明确的高度
          .margin({ bottom: CommonConstants.COMPONENT_SPACING })

      } else if (this.imagePath && !this.isLoading) {
        // Fallback: 如果PixelMap无效但有路径，显示普通图片
        Image(this.imagePath)
          .width('100%')
          .height(CommonConstants.IMAGE_PREVIEW_HEIGHT)
          .objectFit(ImageFit.Contain)
          .backgroundColor(CommonConstants.CARD_BACKGROUND_COLOR)
          .margin({ bottom: CommonConstants.COMPONENT_SPACING })
          .alt(CommonConstants.IMAGE_PREVIEW_ALT_COLOR)
      } else { // 默认占位符
        Column() {
          Text('未选择图片')
            .fontSize(CommonConstants.BODY_FONT_SIZE)
            .fontColor(CommonConstants.SECONDARY_TEXT_COLOR)
        }
        .width('100%')
        .height(CommonConstants.IMAGE_PREVIEW_HEIGHT)
        .backgroundColor(CommonConstants.CARD_BACKGROUND_COLOR)
        .justifyContent(FlexAlign.Center)
        .margin({ bottom: CommonConstants.COMPONENT_SPACING })
      }

      // 操作按钮
      Button('选择图片进行识别')
        .width('100%')
        .height(CommonConstants.BUTTON_HEIGHT)
        .backgroundColor(CommonConstants.PRIMARY_COLOR)
        .fontColor(Color.White)
        .fontSize(CommonConstants.BODY_FONT_SIZE)
        .enabled(!this.isLoading)
        .margin({ bottom: CommonConstants.COMPONENT_SPACING })
        .onClick(() => this.selectAndRecognize())

      // 加载指示器
      if (this.isLoading) {
        LoadingProgress()
          .width(30)
          .height(30)
          .color(CommonConstants.PRIMARY_COLOR)
          .margin({ bottom: CommonConstants.COMPONENT_SPACING })
      }

      // 识别结果
      if (this.recognizedPlates.length > 0) {
        Text('识别结果')
          .fontSize(CommonConstants.SUBTITLE_FONT_SIZE)
          .fontWeight(FontWeight.Bold)
          .fontColor(CommonConstants.TEXT_COLOR)
          .width('100%')
          .margin({ bottom: CommonConstants.COMPONENT_SPACING })

        List() {
          ForEach(this.recognizedPlates, (plate: PlateInfo) => {
            ListItem() {
              PlateResultItem({ plateInfo: plate })
            }
            .margin({ bottom: 8 })
          })
        }
        .width('100%')
        .layoutWeight(1)
      } else if (!this.isLoading && (this.imagePath || this.selectedPixelMap)) {
        Text('未检测到车牌')
          .fontSize(CommonConstants.BODY_FONT_SIZE)
          .fontColor(CommonConstants.SECONDARY_TEXT_COLOR)
          .margin({ top: CommonConstants.COMPONENT_SPACING })
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor(CommonConstants.BACKGROUND_COLOR)
    .padding(CommonConstants.PAGE_PADDING)
  }
}