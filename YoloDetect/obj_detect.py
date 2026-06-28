import cv2

class UsbCamAttach(cv2.VideoCapture):
    def __init__(self, device_index=0):
        super().__init__(device_index)
        super().set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        return
    
    def __enter__(self):
        if not self.isOpened():
            raise RuntimeError(f'Fehler: Kann die Kamera {self.get(cv2.CAP_PROP_POS_FRAMES)} nicht öffnen.')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def read(self, *args, **kwargs):
        success, frame = super().read(*args, **kwargs)
        if not success:
            raise RuntimeError('Fehler: Kamerabild kann nicht gelesen werden.')
        return frame