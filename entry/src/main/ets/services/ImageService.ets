/**
 * 图片服务类
 * 负责图片选择和Base64转换
 */

import { hilog } from '@kit.PerformanceAnalysisKit';
import picker from '@ohos.file.picker';
import fs from '@ohos.file.fs';
import image from '@ohos.multimedia.image';
import util from '@ohos.util';
import { BusinessError } from '@ohos.base';
import { CommonConstants } from '../common/constants/CommonConstants';

// 定义图片选择器接口
interface PhotoPickerInterface {
  select(options: PhotoSelectOptions): Promise<PhotoSelectResult>;
}

// 定义图片选择选项接口
interface PhotoSelectOptions {
  MIMEType: string;
  maxSelectNumber: number;
}

// 定义图片选择结果接口
interface PhotoSelectResult {
  photoUris?: string[];
}

// 定义图片打包选项接口
interface PackingOption {
  format: string;
  quality: number;
}

// 定义图片数据接口
interface ImageData {
  byteArray: ArrayBuffer;
}

// 模拟图片选择器接口
class MockPhotoPicker implements PhotoPickerInterface {
  async select(options: PhotoSelectOptions): Promise<PhotoSelectResult> {
    // 模拟返回结果
    return {
      photoUris: ['file:///data/storage/el2/base/temp_image.jpg']
    };
  }
}

const DOMAIN = 0x0000;
const TAG = 'ImageService';

export class ImageService {
  /**
   * 从相册选择图片
   * @returns 选中图片的URI
   */
  async selectImageFromGallery(): Promise<string> {
    try {
      const photoSelectOptions = new picker.PhotoSelectOptions();
      photoSelectOptions.MIMEType = picker.PhotoViewMIMETypes.IMAGE_TYPE; // 使用 picker 模块的常量
      photoSelectOptions.maxSelectNumber = 1;

      const photoPicker = new picker.PhotoViewPicker(); // 使用实际的 PhotoViewPicker
      hilog.info(DOMAIN, TAG, 'Showing PhotoViewPicker...');
      const photoSelectResult = await photoPicker.select(photoSelectOptions);

      if (photoSelectResult && photoSelectResult.photoUris && photoSelectResult.photoUris.length > 0) {
        const photoUri = photoSelectResult.photoUris[0];
        hilog.info(DOMAIN, TAG, `Selected image URI: ${photoUri}`);
        return photoUri;
      } else {
        hilog.info(DOMAIN, TAG, 'No image selected or selection cancelled by user.');
        return ''; // 用户未选择或取消
      }
    } catch (err) {
      const error = err as BusinessError;
      hilog.error(DOMAIN, TAG, `Failed to select image, code: ${error.code}, message: ${error.message}`);
      if (error.code === 200001 || (error.message && error.message.toLowerCase()
        .includes('cancel'))) {
        hilog.warn(DOMAIN, TAG, 'User cancelled image selection.');
        return ''; // 用户取消，不应视为致命错误抛出
      }
      throw new Error(`选择图片失败: ${error.message} (代码: ${error.code})`);
    }
  }

  /**
   * 根据 URI 加载 PixelMap 对象
   * @param uri 图片的URI
   * @returns PixelMap 对象
   */
  async loadPixelMap(uri: string): Promise<image.PixelMap> {
    if (!uri) {
      hilog.error(DOMAIN, TAG, 'Image URI is empty for PixelMap loading.');
      throw new Error('图片URI为空，无法加载PixelMap');
    }
    hilog.info(DOMAIN, TAG, `Loading PixelMap for URI: ${uri}`);
    let file: fs.File | null = null;
    try {
      file = fs.openSync(uri, fs.OpenMode.READ_ONLY);
      const imageSource = image.createImageSource(file.fd);
      const pixelMap = await imageSource.createPixelMap();
      hilog.info(DOMAIN, TAG, 'PixelMap loaded successfully.');
      imageSource.release();
      return pixelMap;
    } catch (err) {
      const error = err as BusinessError;
      hilog.error(DOMAIN, TAG, `Failed to load PixelMap: ${error.message} (Code: ${error.code || 'N/A'})`);
      throw new Error(`加载PixelMap失败: ${error.message}`);
    } finally {
      if (file) {
        try {
          fs.closeSync(file); // 使用 file 对象关闭
        } catch (closeError) {
          const cError = closeError as BusinessError;
          hilog.error(DOMAIN, TAG, `Failed to close file ${uri} after PixelMap loading: ${cError.message}`);
        }
      }
    }
  }

  /**
   * 将图片URI转换为Base64编码
   * @param uri 图片URI
   * @returns Base64编码的图片数据
   */
  async convertImageToBase64(uri: string): Promise<string> {
    if (!uri) {
      hilog.error(DOMAIN, TAG, 'Image URI is empty for Base64 conversion.');
      throw new Error('图片URI为空，无法进行Base64转换');
    }
    hilog.info(DOMAIN, TAG, `Starting Base64 conversion for URI: ${uri}`);

    let file: fs.File | null = null;
    try {
      // 打开文件
      file = fs.openSync(uri, fs.OpenMode.READ_ONLY);
      hilog.info(DOMAIN, TAG, `File opened successfully: ${uri} (fd: ${file.fd})`);

      // 获取文件大小 - 使用文件描述符 file.fd
      const stat = fs.statSync(file.fd); // 使用 file.fd
      const fileSize = stat.size;
      hilog.info(DOMAIN, TAG, `File size: ${fileSize} bytes`);

      if (fileSize === 0) {
        hilog.warn(DOMAIN, TAG, 'File is empty (0 bytes).');
        // 根据需求，可以返回空字符串或特定错误，或一个表示空图片的Base64
        throw new Error('选择的文件为空 (0字节)');
      }
      if (fileSize > CommonConstants.MAX_IMAGE_SIZE_BYTES) {
        hilog.error(DOMAIN, TAG,
          `Image size ${fileSize} exceeds max allowed size ${CommonConstants.MAX_IMAGE_SIZE_BYTES}`);
        throw new Error(`图片大小超过限制 (${CommonConstants.MAX_IMAGE_SIZE_BYTES / (1024 * 1024)}MB)`);
      }

      // 读取文件内容到ArrayBuffer - 使用文件描述符 file.fd
      const buffer = new ArrayBuffer(fileSize);
      const bytesRead = fs.readSync(file.fd, buffer, { offset: 0 }); // 使用 file.fd
      hilog.info(DOMAIN, TAG, `Bytes read from file: ${bytesRead}`);

      if (bytesRead < fileSize) {
        hilog.warn(DOMAIN, TAG, `Read ${bytesRead} bytes, but file size is ${fileSize}. Using actual bytes read.`);
        // 通常意味着读取不完整，可能需要处理这种情况
      }

      // 使用Uint8Array进行Base64编码
      const uint8Array = new Uint8Array(buffer, 0, bytesRead); // 确保只编码实际读取的字节
      const base64Helper = new util.Base64Helper();
      const base64Data = base64Helper.encodeToStringSync(uint8Array);

      hilog.info(DOMAIN, TAG, 'Image converted to Base64 successfully.');
      // 在这里打印完整的 Base64 字符串，用于调试
      // 非常长的 Base64 字符串可能会被 hilog 截断或分多行显示
      // hilog.debug(DOMAIN, TAG, `Generated Base64 (raw): ${base64Data}`); // 打印原始的，不带前缀

      let mimeType = 'image/jpeg'; // 默认
      const lowerUri = uri.toLowerCase();
      if (lowerUri.endsWith('.png')) {
        mimeType = 'image/png';
      } else if (lowerUri.endsWith('.gif')) {
        mimeType = 'image/gif';
      } else if (lowerUri.endsWith('.webp')) {
        mimeType = 'image/webp';
      }
      // 对于 jpg 和 jpeg, image/jpeg 已经覆盖
      //return `data:${mimeType};base64,${base64Data}`;
      const fullBase64String = `data:${mimeType};base64,${base64Data}`;

      // 打印带前缀的完整 Base64 字符串
      hilog.info(DOMAIN, TAG,
        `Full Base64 String ready to be sent (first 100 chars): ${fullBase64String.substring(0, 100)}...`);

      return fullBase64String;

    } catch (err) {
      const error = err as BusinessError;
      hilog.error(DOMAIN, TAG, `Failed to convert image to Base64: ${error.message} (Code: ${error.code || 'N/A'})`);
      throw new Error(`图片转换为Base64失败: ${error.message}`);
    } finally {
      if (file) {
        try {
          fs.closeSync(file);
          hilog.info(DOMAIN, TAG, `File closed: ${uri}`);
        } catch (closeError) {
          const cError = closeError as BusinessError;
          hilog.error(DOMAIN, TAG, `Failed to close file ${uri}: ${cError.message}`);
        }
      }
    }
  }
}