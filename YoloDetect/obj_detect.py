"""
This module provides a class for attaching to a USB camera using OpenCV's VideoCapture.
The UsbCamAttach class is a context manager that handles the initialization and cleanup of the
camera resource and ensures that the camera is warmed up before use.
Usage:
    with UsbCamAttach(device_index) as camera:
        frame = camera.read()
        # Process the frame as needed
"""

import cv2

class UsbCamAttach(cv2.VideoCapture):
    """
    A context manager for attaching to a USB camera using OpenCV's VideoCapture.
    This class ensures that the camera is properly initialized and warmed up before use,
    and it handles the cleanup of the camera resource when done.
    """

    def __init__(self, device_index=0):
        super().__init__(device_index)
        super().set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        return

    def __enter__(self):
        if not self.isOpened():
            raise RuntimeError(f'Fehler: Kann die Kamera {self.get(cv2.CAP_PROP_POS_FRAMES)} ' +
                               'nicht öffnen.')
        for _ in range(5):  # Kamera aufwärmen
            self.read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def read(self, *args, **kwargs):
        """
        Read a frame from the camera.
        Raises a RuntimeError if the frame cannot be read.
        """
        success, frame = super().read(*args, **kwargs)
        if not success:
            raise RuntimeError('Fehler: Kamerabild kann nicht gelesen werden.')
        return frame
