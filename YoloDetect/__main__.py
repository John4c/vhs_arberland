#!/home/johann/vhs_arberland/.venv/bin python3

"""YOLO26 Live-Stream mit USB-Kamera
Dieses Skript verwendet das YOLO26-Modell, um Objekte in Echtzeit von einer USB-Kamera zu erkennen.
Es zeigt die erkannten Objekte im Video-Stream an und berechnet die FPS (Frames per Second) für
die Performance-Überwachung.
"""
import os
import argparse
import json
import cv2
from ultralytics import YOLO

from obj_detect import UsbCamAttach

if __name__ == '__main__':

    demo_loc = os.path.dirname(__file__)
    with open(os.path.join(demo_loc, 'COCO_Deutsch.json'), 'r', encoding='utf-8') as f:
        class_names_list = json.load(f)
    # Klassennamen als Dictionary mit Integer-Schlüsseln (so erwartet es YOLO intern)
    class_names = {i: name for i, name in enumerate(class_names_list)}

    model_description = {
        'det': 'Detection',
        'seg': 'Instance Segmentation',
        'sem': 'Semantic Segmentation',
        'pose': 'Pose Estimation',
        'obb': 'Oriented Bounding Boxes',
        'cls': 'Classification'
    }
    parser = argparse.ArgumentParser(description='YOLO26 USB Kamera Live Detection')
    parser.add_argument('--device-index', type=int, default=0, help='Index der USB-Kamera')
    parser.add_argument('--show-orig', dest='show_orig', action='store_true',
        help='Originales Kamera Bild anzeigen')
    HELP_TEXT = 'YOLO-Aufgabe entsprechend der Modellbenennung: \n\t' +\
        '\n\t'.join([f'{key}: {value}' for key, value in model_description.items()]) +\
        '\n\nBeispiele: \n\t--task det für YOLO26 Detection Modell' +\
        '\n\t--task seg,det für YOLO26 Detection and Instance Segmentation Modell' +\
        '\n\nHinweis: Das Modell muss im "models"-Ordner vorhanden sein.'
    parser.add_argument('--task', type=str, default='det', help=HELP_TEXT)
    parser.add_argument('--german', dest='german', action='store_true',
        help='Klassenbezeichnungen auf Deutsch anzeigen, möglich für det, seg und pose.')
    parser.add_argument('--stream', dest='stream', action='store_true',
        help='Live-Stream anzeigen')
    ARGS = parser.parse_args()

    requested_tasks = ARGS.task.split(',')
    assert ARGS.task in model_description or \
        all(task in model_description for task in requested_tasks), \
        'Fehler: Ungültige YOLO-Aufgabe. Bitte wählen Sie eine der folgenden Aufgaben: ' + \
        ', '.join(model_description.keys())
    if ARGS.stream:
        print('Hinweis: Live-Stream wird angezeigt.')
        requested_window_names = {task: f'{model_description[task]} Live-Stream'
                                  for task in requested_tasks}
        DELAY_PERIOD = 1
    else:
        print('Hinweis: Standbild wird angezeigt.')
        requested_window_names = {task: f'{model_description[task]} Standbild'
                                  for task in requested_tasks}
        DELAY_PERIOD = 0
    COMPARE_WINDOW_NAME = 'Originales USB Kamera Bild'

    models_loc =  os.path.join(demo_loc, 'models')
    model_files = {file[8:-3]: os.path.join(models_loc, file) for file in os.listdir(models_loc)}
    model_files['det'] =model_files.pop('')
    requested_models = { name: YOLO(file) for name, file in model_files.items() }
    assert all(task in requested_models for task in requested_tasks), \
        'Fehler: Nicht alle angeforderten YOLO26 Modelle sind im "models"-Ordner vorhanden. ' + \
        f'Verfügbare Modelle: {list(requested_models.keys())}'

    for win_name in requested_window_names.values():
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if ARGS.show_orig:
        cv2.namedWindow(COMPARE_WINDOW_NAME, cv2.WINDOW_NORMAL)

    with UsbCamAttach(ARGS.device_index) as img_capture:
        key = 0xFF
        while key not in {ord('q'), 27}:  # Beenden mit [q] oder [ESC]
            image = img_capture.read()
            if ARGS.show_orig:
                cv2.imshow(COMPARE_WINDOW_NAME, image)
            outputs = {name: model(image) for name, model in requested_models.items()
                       if name in requested_tasks}
            for name, results in outputs.items():
                # Klassennamen in den Result-Objekten setzen (falls German-Option aktiv)
                if ARGS.german and name in ('det', 'seg', 'pose'):
                    results[0].names = class_names
                annotated_image = results[0].plot()
                cv2.imshow(requested_window_names[name], annotated_image)
            key = cv2.waitKey(DELAY_PERIOD) & 0xFF

    for win_name in requested_window_names.values():
        cv2.destroyWindow(win_name)
    if ARGS.show_orig:
        cv2.destroyWindow(COMPARE_WINDOW_NAME)
