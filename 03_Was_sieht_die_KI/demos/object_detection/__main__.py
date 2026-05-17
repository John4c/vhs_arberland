"""YOLO26 Live-Stream mit USB-Kamera
Dieses Skript verwendet das YOLO26-Modell, um Objekte in Echtzeit von einer USB-Kamera zu erkennen.
Es zeigt die erkannten Objekte im Video-Stream an und berechnet die FPS (Frames per Second) für
die Performance-Überwachung.
"""
import os
import cv2
import torch
from ultralytics import YOLO


def main(cam_idx: int, path: str) -> None:
    """Startet den YOLO26 Live-Stream von einer USB-Kamera und zeigt die erkannten Objekte an.

    Args:
        cam_idx (int): Index der USB-Kamera (z.B. 0 für die erste Kamera, 1 für die zweite Kamera,
            etc.)
        path (str): Pfad zum YOLO26-Modell
    """
    # GPU-Beschleunigung aktivieren, falls verfügbar
    device = 'cpu'

    # NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        device = 0
        print(f'NVIDIA GPU erkannt: {torch.cuda.get_device_name(0)}')
    # AMD GPU (DirectML für Windows)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 0
        print('AMD GPU (via DirectML) erkannt')
    else:
        print('CPU wird verwendet')

    print(f'Verwendetes Device: {device}')
    model = YOLO(model=path)
    if isinstance(device, int):
        model.to(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    else:
        model.to(device)

    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    if cap.isOpened() is True:
        print('Starte YOLO26 Live-Stream. Mit [q] beenden.')

        while cv2.waitKey(1) & 0xFF != ord('q'):
            success, frame = cap.read()
            if success is False:
                print('Fehler: Kamerabild kann nicht gelesen werden.')
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            cv2.imshow('YOLO26 USB Kamera', annotated_frame)

        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f'Fehler: Kann die Kamera {cam_idx} nicht öffnen.')
    return

if __name__ == '__main__':
    demo_loc = os.path.dirname(__file__)
    MODEL_FILE_PATH = os.path.join(demo_loc, 'yolo26n.pt')
    main(cam_idx=0, path=MODEL_FILE_PATH)
