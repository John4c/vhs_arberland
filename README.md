# Demo Anwendungen

Dieses Repository enthält Demo-Anwendungen, die Live-Videoausgabe und KI-Modelle kombinieren. Aktuell ist die YOLO-basierte Demo auf Ultralytics YOLO26 fokussiert.

---

# 1. YOLO26 basierend auf Ultralytics

## Übersicht

Die aktuelle Demo verwendet ein YOLO26-Modell für verschiedene Anwendungsfälle mit einer USB-Kamera. Die Anwendung unterstützt mehrere Aufgaben wie Objekterkennung, Segmentierung, Pose-Estimation und Klassifikation.

## Wichtige Use-Cases

Demo-Szenarien sind in `.vscode/launch.json` definiert:

1. **YOLO Detection + Instance Segmentation**
   
   - Debug-Konfiguration: `Detection`
   - Argumente: `--task det,seg --german`
   - Beschreibung: Führe mehrere Modelle gleichzeitig aus und zeige die Ergebnisse als Standbild.

2. **YOLO Pose Estimation**
   
   - Debug-Konfiguration: `Pose Estimation`
   - Argumente: `--stream --task pose`
   - Beschreibung: Zeige den Live-Stream mit Pose-Estimation-Ergebnissen an.

3. **YOLO Classification**
   
   - Debug-Konfiguration: `Classification`
   - Argumente: `--task cls`
   - Beschreibung: Klassifiziere das Kamerabild und zeige das Ergebnis als Standbild.

## Starten der Demo

1. Öffne ein Terminal im Repository-Stammverzeichnis.
2. Aktiviere gegebenenfalls die Python-Umgebung.
3. Starte die Anwendung mit:
   - `python ./YoloDetect --task det,seg --german`
   - `python ./YoloDetect --stream --task pose`
   - `python ./YoloDetect --task cls`

## Struktur

- `YoloDetect/__main__.py` - Einstiegspunkt der YOLO-Demo
- `YoloDetect/obj_detect.py` - Kamera-Anbindung
- `YoloDetect/models/` - Enthält die benötigten YOLO26-Modelldateien. Standardmäßig müssen diese von der Ultralytics-Seite heruntergeladen oder selbst trainiert werden.

---

# Vorbereitung weiterer Demo-Anwendungen

Dieses README ist so aufgebaut, dass weitere Demo-Module einfach ergänzt werden können. Für jede neue Demo sollte ein eigener Abschnitt mit kurzer Beschreibung, Startanleitung und wichtigen Konfigurationen ergänzt werden.
