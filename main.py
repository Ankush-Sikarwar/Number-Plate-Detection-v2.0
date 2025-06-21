from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io
import re
from rapidfuzz import fuzz

app = FastAPI()
expected_format = "MH12AB1234" 


pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


model = YOLO("best.pt")


def ocr_by_contours(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    text = ""
    for x, y, w, h in bounding_boxes:
        if w*h < 100:  
            continue

        char_img = thresh[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (32, 64))  

        
        char_text = pytesseract.image_to_string(char_img, config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        text += char_text.strip()

    return text



def preprocess_input(image):
    image = cv2.bilateralFilter(image, 11, 17, 17)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image


def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def is_valid_plate(text):
    pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$"
    return re.match(pattern, text.replace(" ", "").upper()) is not None

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

       

        results = model(image)[0]
        detections = results.boxes.xyxy.cpu().numpy()

        output = []
        for box in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = image[y1:y2, x1:x2]

            processed_crop = preprocess_crop(cropped)
            # text = ocr_by_contours(processed_crop)
            text = pytesseract.image_to_string(processed_crop, config="--psm 7")
            clean_text = text.strip().replace("\n", "").replace(" ", "")
            print(clean_text)
            score = fuzz.ratio(clean_text, expected_format)
            print(score)

            if score > 10:
                output.append({
                    "box": [x1, y1, x2, y2],
                    "text": clean_text
                })

        return JSONResponse(content={"detections": output})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
