#!/home/johann/vhs_arberland/.venv/bin python3

"""YOLO26 Live-Stream mit USB-Kamera
Dieses Skript verwendet das YOLO26-Modell, um Objekte in Echtzeit von einer USB-Kamera zu erkennen.
Es zeigt die erkannten Objekte im Video-Stream an und berechnet die FPS (Frames per Second) für
die Performance-Überwachung.
"""
import os
import argparse
import cv2
from ultralytics import YOLO

from obj_detect import UsbCamAttach

if __name__ == '__main__':

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
    orig_image_parser = parser.add_mutually_exclusive_group(required=False)
    orig_image_parser.add_argument('--show-orig-image', dest='orig_image', action='store_true', help='Originales Kamera Bild anzeigen')
    orig_image_parser.add_argument('--no-show-orig-image', dest='orig_image', action='store_false', help='Originales Kamera Bild nicht anzeigen')
    parser.set_defaults(orig_image=True)

    help_description = 'YOLO-Aufgabe entsprechend der Modellbenennung: \n\t' +\
        '\n\t'.join([f'{key}: {value}' for key, value in model_description.items()]) +\
        '\n\nBeispiele: \n\t--task det für YOLO26 Detection Modell' +\
        '\n\t--task seg,det für YOLO26 Detection and Instance Segmentation Modell' +\
        '\n\nHinweis: Das Modell muss im "models"-Ordner vorhanden sein.'
    parser.add_argument('--task', type=str, default='det', help=help_description)

    stream_parser = parser.add_mutually_exclusive_group(required=False)
    stream_parser.add_argument('--stream', dest='stream', action='store_true', 
                        help='Kamera Live-Stream zur Laufzeit anzeigen')
    stream_parser.add_argument('--no-stream', dest='stream', action='store_false',
                        help='Nur Standbild anzeigen')
    parser.set_defaults(stream=False)
    args = parser.parse_args()

    requested_tasks = args.task.split(',')
    assert args.task in model_description or \
        all(task in model_description for task in requested_tasks), \
        'Fehler: Ungültige YOLO-Aufgabe. Bitte wählen Sie eine der folgenden Aufgaben: ' + \
        ', '.join(model_description.keys())
    if args.stream:
        print('Hinweis: Live-Stream wird angezeigt.')
        requested_window_names = {task: f'{model_description[task]} Live-Stream' for task in requested_tasks}
        delayed_period = 1
    else:
        print('Hinweis: Standbild wird angezeigt.')
        requested_window_names = {task: f'{model_description[task]} Standbild' for task in requested_tasks}
        delayed_period = 0
    comparison_window_name = 'Originales USB Kamera Bild'

    models_loc =  os.path.join(os.path.dirname(__file__), 'models')
    model_files = {file[8:-3]: os.path.join(models_loc, file) for file in os.listdir(models_loc)}
    model_files['det'] =model_files.pop('')
    requested_models = { name: YOLO(file) for name, file in model_files.items() }
    assert all(task in requested_models for task in requested_tasks), \
        'Fehler: Nicht alle angeforderten YOLO26 Modelle sind im "models"-Ordner vorhanden. ' + \
        f'Verfügbare Modelle: {list(requested_models.keys())}'

    for win_name in requested_window_names.values():
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if args.orig_image:
        cv2.namedWindow(comparison_window_name, cv2.WINDOW_NORMAL)

    with UsbCamAttach(args.device_index) as img_capture:
        for _ in range(5):  # Kamera aufwärmen
            img_capture.read()
        key = 0xFF
        while key not in {ord('q'), 27}:  # Beenden mit [q] oder [ESC]
            image = img_capture.read()
            if args.orig_image:
                cv2.imshow(comparison_window_name, image)
            outputs = {name: model(image) for name, model in requested_models.items() if name in requested_tasks}
            for name, results in outputs.items():
                annotated_image = results[0].plot()
                cv2.imshow(requested_window_names[name], annotated_image)
            key = cv2.waitKey(delayed_period) & 0xFF

    for win_name in requested_window_names.values():
        cv2.destroyWindow(win_name)
    if args.orig_image:
        cv2.destroyWindow(comparison_window_name)
