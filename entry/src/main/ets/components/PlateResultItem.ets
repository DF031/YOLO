/**
 * 车牌识别结果项组件
 */

import { PlateInfo } from '../model/MessageTypes';
import { CommonConstants } from '../common/constants/CommonConstants';

@Component
export struct PlateResultItem {
  plateInfo: PlateInfo = { text: '', confidence: 0, box: [0, 0, 0, 0] };

  build() {
    Row() {
      Column() {
        Text(this.plateInfo.text)
          .fontSize(CommonConstants.SUBTITLE_FONT_SIZE)
          .fontWeight(FontWeight.Bold)
          .fontColor(CommonConstants.TEXT_COLOR)
          .margin({ bottom: 4 })

        Row() {
          Text('置信度:')
            .fontSize(CommonConstants.SMALL_FONT_SIZE)
            .fontColor(CommonConstants.SECONDARY_TEXT_COLOR)

          Text(`${(this.plateInfo.confidence * 100).toFixed(1)}%`)
            .fontSize(CommonConstants.SMALL_FONT_SIZE)
            .fontColor(this.getConfidenceColor())
            .fontWeight(FontWeight.Bold)
            .margin({ left: 4 })
        }
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)

      Column() {
        Text('位置:')
          .fontSize(CommonConstants.SMALL_FONT_SIZE)
          .fontColor(CommonConstants.SECONDARY_TEXT_COLOR)
          .margin({ bottom: 4 })

        Text(`[${this.plateInfo.box.join(', ')}]`)
          .fontSize(CommonConstants.SMALL_FONT_SIZE)
          .fontColor(CommonConstants.SECONDARY_TEXT_COLOR)
          .maxLines(1)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
      }
      .alignItems(HorizontalAlign.End)
      .width('40%')
    }
    .width('100%')
    .height(CommonConstants.RESULT_ITEM_HEIGHT)
    .backgroundColor(CommonConstants.CARD_BACKGROUND_COLOR)
    .borderRadius(8)
    .padding(CommonConstants.PAGE_PADDING)
  }

  /**
   * 根据置信度获取对应的颜色
   * @returns 颜色值
   */
  private getConfidenceColor(): string {
    if (this.plateInfo.confidence >= 0.9) {
      return CommonConstants.SUCCESS_COLOR;
    } else if (this.plateInfo.confidence >= 0.7) {
      return CommonConstants.PRIMARY_COLOR;
    } else {
      return CommonConstants.WARNING_COLOR;
    }
  }
}
