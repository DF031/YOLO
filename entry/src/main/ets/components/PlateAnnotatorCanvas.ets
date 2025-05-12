import image from '@ohos.multimedia.image';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { BusinessError } from '@ohos.base';
// Removed: import { CanvasRenderingContext2D, RenderingContextSettings, Area } from '@kit.ArkUI';
// TextMetrics might be needed for measureText result
// Attempt to import TextMetrics, if this specific path is wrong, a structural type will be used.
// import { TextMetrics } from '@ohos.graphics.text'; // Or another relevant graphics module

import { PlateInfo } from '../model/MessageTypes';
import { CommonConstants } from '../common/constants/CommonConstants';

const DOMAIN = 0x0000;
const TAG = 'PlateAnnotatorCanvas';

interface ExpectedTextMetrics {
  width: number;
}

@Component
export struct PlateAnnotatorCanvas {
  @Prop @Watch('onPixelMapPropChange') pixelMap: image.PixelMap | null;
  @Prop @Watch('onPlatesPropChange') plates: PlateInfo[];
  @Prop canvasDisplayHeight: number;
  @Prop imageInfo: image.ImageInfo | null;

  private settings = new RenderingContextSettings(true);
  private context = new CanvasRenderingContext2D(this.settings);

  @State actualCanvasWidth: number = 0;
  @State actualCanvasHeight: number = 0;

  onPixelMapPropChange(): void {
    hilog.info(DOMAIN, TAG, 'pixelMap prop changed, redrawing canvas.');
    this.drawCanvas();
  }

  onPlatesPropChange(): void {
    hilog.info(DOMAIN, TAG, 'plates prop changed, redrawing canvas.');
    this.drawCanvas();
  }

  private drawCanvas() {
    if (!this.context) {
      hilog.warn(DOMAIN, TAG, 'Canvas context not ready for drawing.');
      return;
    }

    const canvasWidthVp = this.actualCanvasWidth;
    const canvasHeightVp = this.actualCanvasHeight;

    this.context.clearRect(0, 0, canvasWidthVp, canvasHeightVp);

    try {
      if (!this.imageInfo || !this.imageInfo.size) { // 仍然需要 imageInfo 来计算正确的缩放比例
        hilog.error(DOMAIN, TAG, `Received invalid imageInfo prop for annotations: ${JSON.stringify(this.imageInfo)}`);
        // 可能不需要在这里绘制错误文本，因为底下的Image组件会处理图片加载失败的情况
        return;
      }

      const imagePxWidth = this.imageInfo.size.width;
      const imagePxHeight = this.imageInfo.size.height;

      if (imagePxWidth <= 0 || imagePxHeight <= 0) {
        hilog.error(DOMAIN, TAG, `PixelMap dimensions from prop are zero or invalid for annotations: ${imagePxWidth}x${imagePxHeight}`);
        return;
      }

      const imageAspectRatio = imagePxWidth / imagePxHeight;
      // canvasWidthVp 和 canvasHeightVp 是此Canvas组件的实际VP尺寸，应与底下Image组件的图片显示区域一致
      const canvasAspectRatio = canvasWidthVp / canvasHeightVp;

      let drawnImageWidthVp: number; // 这是图片在Image组件中实际显示的VP宽度
      let drawnImageHeightVp: number; // 这是图片在Image组件中实际显示的VP高度

      if (imageAspectRatio > canvasAspectRatio) {
        drawnImageWidthVp = canvasWidthVp;
        drawnImageHeightVp = canvasWidthVp / imageAspectRatio;
      } else {
        drawnImageHeightVp = canvasHeightVp;
        drawnImageWidthVp = canvasHeightVp * imageAspectRatio;
      }

      const offsetX_vp = (canvasWidthVp - drawnImageWidthVp) / 2; // 图片在Image组件中的X偏移 (居中)
      const offsetY_vp = (canvasHeightVp - drawnImageHeightVp) / 2; // 图片在Image组件中的Y偏移 (居中)

      if (this.plates && this.plates.length > 0) {
        this.plates.forEach((plate, index) => {
          const x_min_px = plate.box[0];
          const y_min_px = plate.box[1];
          const x_max_px = plate.box[2];
          const y_max_px = plate.box[3];

          const scaleX_px_to_vp = drawnImageWidthVp / imagePxWidth;
          const scaleY_px_to_vp = drawnImageHeightVp / imagePxHeight;

          const rectX_vp = (x_min_px * scaleX_px_to_vp) + offsetX_vp;
          const rectY_vp = (y_min_px * scaleY_px_to_vp) + offsetY_vp;
          const rectWidth_vp = (x_max_px - x_min_px) * scaleX_px_to_vp;
          const rectHeight_vp = (y_max_px - y_min_px) * scaleY_px_to_vp;

          this.context.strokeStyle = CommonConstants.ANNOTATION_BOX_COLOR || '#FF0000';
          this.context.lineWidth = CommonConstants.ANNOTATION_LINE_WIDTH || 2;
          this.context.strokeRect(rectX_vp, rectY_vp, rectWidth_vp, rectHeight_vp);

          const fontSizeVp = CommonConstants.ANNOTATION_FONT_SIZE_VP || 12;
          const text = `${plate.text} (${(plate.confidence * 100).toFixed(1)}%)`;
          this.context.font = `${fontSizeVp}vp sans-serif`;

          const textMetrics = this.context.measureText(text) as ExpectedTextMetrics;
          const textWidthVp = textMetrics.width;
          const textHeightVp = fontSizeVp;
          const textBgPaddingVp = 2;

          const textY_vp = rectY_vp - textBgPaddingVp; 
          const textX_vp = rectX_vp + textBgPaddingVp; 

          this.context.fillStyle = CommonConstants.ANNOTATION_TEXT_BACKGROUND_COLOR || 'rgba(255, 255, 255, 0.75)';
          this.context.fillRect(
            rectX_vp,
            textY_vp - textHeightVp - textBgPaddingVp,
            textWidthVp + (textBgPaddingVp * 2),
            textHeightVp + (textBgPaddingVp * 2)
          );

          this.context.fillStyle = CommonConstants.ANNOTATION_TEXT_COLOR || '#FF0000';
          this.context.textAlign = 'left';
          this.context.textBaseline = 'bottom'; 
          this.context.fillText(text, textX_vp, textY_vp);
          hilog.debug(DOMAIN, TAG, `Drew plate ${index}: ${text} at [${rectX_vp.toFixed(2)}, ${rectY_vp.toFixed(2)}, ${rectWidth_vp.toFixed(2)}, ${rectHeight_vp.toFixed(2)}]vp`);
        });
      }
    } catch (e) {
      const error = e as BusinessError;
      hilog.error(DOMAIN, TAG, `Error drawing canvas: ${error.message || JSON.stringify(error)} Code: ${error.code}`);
    }
  }

  build() {
    Canvas(this.context)
      .width('100%')
      .height(this.canvasDisplayHeight)
      // .backgroundColor(CommonConstants.CANVAS_BACKGROUND_COLOR || CommonConstants.CARD_BACKGROUND_COLOR || '#EEEEEE') // <--- 改为透明
      .backgroundColor(Color.Transparent) // <--- 设置画布背景透明
      .onReady(() => {
        if (this.context) {
          this.actualCanvasWidth = this.context.width;
          this.actualCanvasHeight = this.context.height;
          hilog.info(DOMAIN, TAG, `Canvas onReady. Context dimensions: ${this.actualCanvasWidth}x${this.actualCanvasHeight}vp. CanvasDisplayHeight prop: ${this.canvasDisplayHeight}vp`);
          this.drawCanvas();
        }
      })
      .onAreaChange((_oldValue, newValue) => {
        this.actualCanvasWidth = newValue.width as number;
        this.actualCanvasHeight = newValue.height as number;
        hilog.info(DOMAIN, TAG, `Canvas onAreaChange. New size: ${this.actualCanvasWidth}x${this.actualCanvasHeight}vp`);
        this.drawCanvas();
      })
  }
} 