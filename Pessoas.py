import cv2
import cvzone
from ultralytics import YOLO
import math
import numpy as np
from sort import *

# --- WEBCAM ---

cap = cv2.VideoCapture(0)

# Configura a resolução da webcam
cap.set(3, 1280) # Largura
cap.set(4, 720) # Altura

MODEL_PATH = '../Yolo-model/yolov8l.pt'

try:
    # Tenta carregar o modelo YOLOv8l
    model = YOLO(MODEL_PATH)
    
    if model is None or not cap.isOpened():
        raise RuntimeError("Falha ao abrir a webcam ou carregar o modelo YOLOv8l.")
        
except Exception as e:
    print(f"Erro Crítico: Falha no setup. Detalhes: {e}")
    cap = None


# Mapeamento de classes
traducao = {
    "person": "pessoa",
}

# Tracking: Inicializa o rastreador SORT
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

paused = False

print("\n--- INICIANDO RASTREAMENTO DE PESSOAS (EXCLUSIVO) ---")
print("Contagem de Pessoas (Console):")

# --- LOOP PRINCIPAL DE PROCESSAMENTO ---
if cap is not None and cap.isOpened():
    while True:
        key = cv2.waitKey(1)
        if key == ord(' '):
            paused = not paused
        elif key == ord('q'):
            break

        if not paused:
            success, img = cap.read()
            if not success:
                print("Falha ao ler o frame da webcam.")
                break

            results = model.predict(img, stream=True, verbose=False) 

            detections = np.empty((0, 5))

            for r in results:
                boxes = r.boxes
                for box in boxes:

                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    conf = math.ceil((box.conf[0] * 100)) / 100

                    cls = int(box.cls[0])
                    currentClass = model.names[cls]

                    # FILTRO: Focando APENAS em "person"
                    if currentClass == "person" and conf > 0.5:
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack([detections, currentArray])

            resultsTracker = tracker.update(detections)
            
            count_pessoas = len(resultsTracker)
            
            # --- RELATÓRIO DE CONTATEM NO CONSOLE ---
            print(f"Pessoas na Tela: {count_pessoas}")
            # ---------------------------------------

            # Loop de Desenho (O(n))
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                
                # Desenho da Bounding Box
                cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(0, 255, 0))
                cvzone.putTextRect(img, f'{int(id)} - Pessoa', (max(0, x1), max(35, y1)),
                                scale=2, thickness=3, offset=3)

                cvzone.putTextRect(img, f' Pessoas: {count_pessoas}', (50, 50))


        if 'img' in locals() and success:
            cv2.imshow("Detecção de Pessoas (YOLOv8l)", img)

    cap.release()
    cv2.destroyAllWindows()
else:
    print("O loop principal não foi executado. Verifique se a webcam está conectada e se o modelo foi carregado corretamente.")