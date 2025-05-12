import torch
import cv2
import numpy as np
import os
import logging
from PIL import Image
# from io import BytesIO # Not explicitly used after changes
import easyocr 

# 尝试从ultralytics的不同可能位置导入attempt_load_weights
try:
    from ultralytics.nn.tasks import attempt_load_weights
except ImportError:
    try:
        from ultralytics.models.yolo.model import attempt_load_weights # 较新版本可能在这里
    except ImportError:
        logging.error("Failed to import attempt_load_weights from ultralytics. Please ensure ultralytics is installed and the import path is correct.")
        attempt_load_weights = None


# 配置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- 配置常量 ---
YOLO_IMG_SIZE = 640
DETECT_MODEL_NAME = 'best11n.pt'  

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, 'weights')
DETECT_MODEL_PATH = os.path.join(MODEL_DIR, DETECT_MODEL_NAME)

CONF_THRESHOLD = 0.25  
IOU_THRESHOLD = 0.45  

# --- 全局模型变量 ---
DEVICE = None
DETECT_MODEL = None
OCR_READER = None     
MODELS_LOADED = False

# --- Helper Functions (保持不变或微调) ---
def letter_box(img, size=(640, 640)):
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    new_img = cv2.resize(img, (new_w, new_h))
    top = int((size[0] - new_h) / 2)
    bottom = size[0] - new_h - top
    left = int((size[1] - new_w) / 2)
    right = size[1] - new_w - left
    img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top

def xywh2xyxy(det):
    y = det.clone()
    y[:, 0] = det[:, 0] - det[:, 2] / 2
    y[:, 1] = det[:, 1] - det[:, 3] / 2
    y[:, 2] = det[:, 0] + det[:, 2] / 2
    y[:, 3] = det[:, 1] + det[:, 3] / 2
    return y

def my_nums(dets, iou_thresh, device):
    if dets.numel() == 0:
        return []
    y = dets.clone()
    # dets is expected to be [x1, y1, x2, y2, score, class_idx, ...other_data]
    # For NMS, we only need the first 5 (box + score)
    y_box_score = y[:, :5] 
    index = torch.argsort(y_box_score[:, -1], descending=True)
    keep = []
    while index.size()[0] > 0:
        i = index[0].item()
        keep.append(i)
        if index.size()[0] == 1:
            break
        x1 = torch.maximum(y_box_score[i, 0], y_box_score[index[1:], 0])
        y1 = torch.maximum(y_box_score[i, 1], y_box_score[index[1:], 1])
        x2 = torch.minimum(y_box_score[i, 2], y_box_score[index[1:], 2])
        y2 = torch.minimum(y_box_score[i, 3], y_box_score[index[1:], 3])
        zero_ = torch.tensor(0).to(device)
        w = torch.maximum(zero_, x2 - x1)
        h = torch.maximum(zero_, y2 - y1)
        inter_area = w * h
        union_area1 = (y_box_score[i, 2] - y_box_score[i, 0]) * (y_box_score[i, 3] - y_box_score[i, 1])
        union_area2 = (y_box_score[index[1:], 2] - y_box_score[index[1:], 0]) * \
                      (y_box_score[index[1:], 3] - y_box_score[index[1:], 1])
        iou = inter_area / (union_area1 + union_area2 - inter_area + 1e-6)
        idx = torch.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep

def restore_box(dets, r, left, top):
    # Assumes dets[:, :4] are [x1,y1,x2,y2]
    dets[:, [0, 2]] = (dets[:, [0, 2]] - left) / r
    dets[:, [1, 3]] = (dets[:, [1, 3]] - top) / r
    return dets

def post_processing(prediction, conf_thresh, iou_thresh, r, left, top, device):
    prediction = prediction.permute(0, 2, 1).squeeze(0)
    num_coords = 4

    xc = prediction[:, 4] > conf_thresh
    x = prediction[xc]

    if x.shape[1] > 5: # Check if class_idx column exists
        plate_class_idx = 0  # Assuming license plate is class 0
        x = x[x[:, 5].int() == plate_class_idx]

    if not len(x):
        return []

    boxes = x[:, :num_coords] # Box [cx, cy, w, h]
    boxes = xywh2xyxy(boxes)  # Convert to [x1, y1, x2, y2]
    
    score = x[:, 4:5] # Keep as a 2D tensor for torch.cat
    
    if x.shape[1] > 5:
        class_indices = x[:, 5:6].float() # Keep as a 2D tensor
        x_for_nms = torch.cat((boxes, score, class_indices), dim=1)
    else: # Only box and score
        x_for_nms = torch.cat((boxes, score), dim=1)

    keep = my_nums(x_for_nms, iou_thresh, device) # my_nums expects at least box and score
    final_results = x_for_nms[keep]
    final_results_restored = restore_box(final_results, r, left, top)
    return final_results_restored

def pre_processing_yolo(img_cv2, img_size, device):
    img_letterboxed, r, left, top = letter_box(img_cv2, (img_size, img_size))
    img_tensor = img_letterboxed[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, r, left, top

def load_all_models():
    global DEVICE, DETECT_MODEL, OCR_READER, MODELS_LOADED #PLATE_REC_MODEL removed
    if MODELS_LOADED:
        return True

    logger.info("Loading models...")
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {DEVICE}")

        if not os.path.exists(DETECT_MODEL_PATH):
            logger.error(f"Detection model file not found: {DETECT_MODEL_PATH}")
            raise FileNotFoundError(f"Detection model file not found: {DETECT_MODEL_PATH}")
        
        if attempt_load_weights is None:
            logger.error("attempt_load_weights function is not available. Cannot load YOLO detection model.")
            raise ImportError("attempt_load_weights is not imported correctly.")

        DETECT_MODEL = attempt_load_weights(DETECT_MODEL_PATH, device=DEVICE)
        DETECT_MODEL.eval()
        logger.info(f"YOLOv8 detection model loaded from {DETECT_MODEL_PATH}")

        logger.info("Initializing EasyOCR Reader...")
        OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=(DEVICE.type == 'cuda'))
        logger.info("EasyOCR Reader initialized.")
        
        MODELS_LOADED = True
        logger.info("All models loaded successfully.")
        return True

    except FileNotFoundError as fnf_error:
        logger.error(f"Model file not found: {str(fnf_error)}")
        MODELS_LOADED = False
    except ImportError as import_error:
        logger.error(f"Import error during model loading: {str(import_error)}")
        MODELS_LOADED = False
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        MODELS_LOADED = False
    return MODELS_LOADED

def _perform_recognition_internal(img_cv2):
    if not MODELS_LOADED:
        logger.error("Models are not loaded. Cannot perform recognition.")
        return []

    img_ori = img_cv2.copy()
    img_tensor, r, left, top = pre_processing_yolo(img_cv2, YOLO_IMG_SIZE, DEVICE)
    
    with torch.no_grad():
        predict_raw = DETECT_MODEL(img_tensor)
        logger.info(f"Raw prediction output shape: {predict_raw[0].shape if isinstance(predict_raw, (list, tuple)) and len(predict_raw) > 0 else 'Unknown shape'}")
        # logger.info(f"Raw prediction output (first few elements if large): {str(predict_raw[0][:, :, :5]) if isinstance(predict_raw, (list, tuple)) and len(predict_raw) > 0 and predict_raw[0].numel() > 0 else 'Empty or not as expected'}")
        
        if isinstance(predict_raw, (list, tuple)) and len(predict_raw) > 0:
            predict = predict_raw[0]
        else:
            predict = predict_raw 
    
    logger.info(f"Shape of tensor going into post_processing: {predict.shape}")
    outputs = post_processing(predict, CONF_THRESHOLD, IOU_THRESHOLD, r, left, top, DEVICE)
    logger.info(f"Outputs from post_processing (detected plates before OCR): {outputs}")

    result_list = []
    for output_item in outputs:
        output_numpy = output_item.cpu().numpy()
        rect = output_numpy[:4].astype(int).tolist()  # x1, y1, x2, y2
        score = float(output_numpy[4]) # Detection confidence

        h_img, w_img = img_ori.shape[:2]
        rect[0] = max(0, rect[0]); rect[1] = max(0, rect[1])
        rect[2] = min(w_img, rect[2]); rect[3] = min(h_img, rect[3])

        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            logger.warning(f"Skipping invalid ROI: {rect}")
            continue
            
        roi_img = img_ori[rect[1]:rect[3], rect[0]:rect[2]]

        if roi_img.size == 0:
            logger.warning(f"Skipping empty ROI for rect {rect}")
            continue
        
        if OCR_READER:
            # Convert ROI to RGB for EasyOCR if it's BGR
            # roi_img_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB) 
            # EasyOCR can often handle BGR numpy arrays directly.
            ocr_results = OCR_READER.readtext(roi_img)
            logger.debug(f"EasyOCR raw results for ROI {rect}: {ocr_results}")
            
            plate_text = ""
            highest_ocr_conf = 0
            if ocr_results:
                for (_, text, ocr_conf) in ocr_results:
                    plate_text += text
                    if ocr_conf > highest_ocr_conf: # Get an idea of OCR confidence
                        highest_ocr_conf = ocr_conf
                plate_text = plate_text.replace(" ", "") # Remove spaces
            else:
                 logger.warning(f"EasyOCR found no text in ROI {rect}")

            plate_color = "未知" 
            color_conf = 0.0
            plate_type = 0 # Default to single layer

            result_dict = {
                'plate_no': plate_text,
                'plate_color': plate_color, # Placeholder
                'rect': rect,
                'detect_conf': score,
                'ocr_conf': highest_ocr_conf, # Confidence from OCR (max char or average)
                'roi_height': roi_img.shape[0],
                'color_conf': color_conf, # Placeholder
                'plate_type': plate_type # Placeholder
            }
            result_list.append(result_dict)
            logger.info(f"Detected Plate: {plate_text}, Detect Conf: {score:.2f}, OCR Conf: {highest_ocr_conf:.2f}, Rect: {rect}")
        else:
            logger.error("OCR_READER is not available.")
            continue

    return result_list

def recognize_plate(pil_image: Image.Image):
    if not MODELS_LOADED:
        logger.error("recognize_plate called but models are not loaded. Attempting to load now.")
        if not load_all_models():
             logger.error("Failed to load models on demand. Cannot proceed with recognition.")
             return []

    if pil_image is None:
        logger.error("Input image is None.")
        return []

    try:
        img_cv2_rgb = np.array(pil_image.convert('RGB')) 
        img_cv2_bgr = cv2.cvtColor(img_cv2_rgb, cv2.COLOR_RGB2BGR)
        logger.info(f"Processing image with shape: {img_cv2_bgr.shape}")
        return _perform_recognition_internal(img_cv2_bgr)
    except Exception as e:
        logger.error(f"Error during plate recognition: {str(e)}", exc_info=True)
        return []

if not MODELS_LOADED:
    if not load_all_models():
        logger.critical("CRITICAL: Models failed to load on initial import. Plate recognition will not work.")
    else:
        logger.info("Models successfully loaded during initial import.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running plate_recognizer.py in test mode with new EasyOCR setup...")

    if not MODELS_LOADED:
        logger.error("Models did not load. Exiting test.")
        exit()
    
    test_image_path = os.path.join(CURRENT_DIR, '..', 'backend', 'OIP.jpg') 

    if test_image_path and os.path.exists(test_image_path):
        try:
            pil_img = Image.open(test_image_path)
            logger.info(f"Test image loaded: {test_image_path}")
            
            cpu_start_time = 0
            if DEVICE.type == 'cuda':
                import time
                cpu_start_time = time.perf_counter()
            else:
                import time
                cpu_start_time = time.perf_counter()

            results = recognize_plate(pil_img)
            
            import time # Ensure time is available
            cpu_end_time = time.perf_counter()
            elapsed_time_s = cpu_end_time - cpu_start_time
            logger.info(f"Recognition time: {elapsed_time_s*1000:.3f} ms ({DEVICE.type})")

            if results:
                logger.info("Recognition Results:")
                for res in results:
                    logger.info(f"  Plate: {res['plate_no']}, Detect Conf: {res['detect_conf']:.2f}, OCR Conf: {res.get('ocr_conf', 0):.2f}, Rect: {res['rect']}")
            else:
                logger.info("No plates recognized in the test image.")

        except FileNotFoundError:
            logger.error(f"Test image not found at {test_image_path}.")
        except Exception as e:
            logger.error(f"Error during testing: {e}", exc_info=True)
    else:
        logger.warning(f"Test image path '{test_image_path}' not found or not specified. Skipping direct recognition test.")

    logger.info("Test mode finished.")
