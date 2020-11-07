"""Virtual Background webcam desktop application.

The application use deep learning model from:
for real time portrait segmentation. That segmentation would be used to make
virtual background.
Note: To archieve real time performance, we applied OpenVino to convert ONNX model
to optimal one.
For GUI part, PyQt5 was used.
"""
import os
import sys
import ctypes
import time
import logging
import threading
from subprocess import Popen, TimeoutExpired

import cv2
import numpy as np
import yaml
import pyvirtualcam

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QFileDialog, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

from openvino.inference_engine import IENetwork, IECore
from utils import generate_input, resize_padding, softmax
from pygrabber.dshow_graph import FilterGraph


# Global variables and constants
default_bkg_img_path = 'background.jpg'
bkg_img_path = default_bkg_img_path
bkg_lock = threading.Lock()
bkg_changed_event = threading.Event()

default_video_id = 0
video_id = default_video_id
video_lock = threading.Lock()
video_changed_event = threading.Event()
exit_event = threading.Event()

OBS_VIRTUAL_CAM_DLL_PATH = os.path.join(os.getcwd(), 'OBS-VirtualCam\\bin\\32bit\\obs-virtualsource.dll')
created_virtual_cam_flag = threading.Event()

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load camera devices
cam_to_id = {}
try:
    graph = FilterGraph()
    cam_to_id = {name: id for id, name in enumerate(graph.get_input_devices())}
except Exception as e:
    logging.error(e)


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def run_as(command):
    if is_admin():
        try:
            proc = Popen(command)
        except TimeoutExpired:
            pass
    else:
        ctypes.windll.shell32.ShellExecuteW(None, u'runas', command[0], ' '.join(command[1:]), None, 1)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        global default_bkg_img_path
        global bkg_img_path
        global bkg_changed_event

        global default_video_id
        global video_id
        global video_lock

        global exit_event

        # capture from web cam
        CONFIG_FILE = 'model_mobilenetv2_with_prior_channel.yaml'
        curr_dir = os.getcwd()
        config_path = os.path.join(curr_dir, 'config', CONFIG_FILE)

        with open(config_path,'rb') as f:
            cont = f.read()
        config = yaml.safe_load(cont)

        xml_path = 'model.xml'
        bin_path = 'model.bin'
        core = IECore()
        network = core.read_network(model=xml_path, weights=bin_path)
        exec_network = core.load_network(network, 'CPU')

        ### Input shape required by model
        input_blob  = next(iter(network.inputs))
        input_shape = network.inputs[input_blob].shape
        height = input_shape[2]
        width = input_shape[3]
        
        out_shape = (960, 640)
        background_img = cv2.imread(default_bkg_img_path)
        background_img = cv2.resize(background_img, out_shape)

        # Create virtual cam
        cap = None
        prev_video_id = None
        try:
            with pyvirtualcam.Camera(width=out_shape[0], height=out_shape[1], fps=30) as cam:
                while True:
                    if exit_event.is_set():
                        break
                    
                    if cap is not None:
                        if video_changed_event.is_set():
                            if prev_video_id != video_id:
                                try:
                                    cap.release()
                                except Exception as e:
                                    del cap

                                with video_lock:
                                    cap = cv2.VideoCapture(video_id)
                                    video_changed_event.clear()
                    else:
                        cap = cv2.VideoCapture(video_id)
                    prev_video_id = video_id

                    display_img = np.zeros((*out_shape, 3), dtype='uint8')
                    if cap is not None:
                        ret, origin_image = cap.read()
                        if ret:
                            # Check background change
                            if bkg_changed_event.is_set():
                                with bkg_lock:
                                    background_img = cv2.imread(bkg_img_path)
                                    background_img = cv2.resize(background_img, out_shape)
                                    bkg_changed_event.clear()
                            
                            # Preprocessing
                            in_shape = origin_image.shape
                            image, bbx = resize_padding(origin_image,
                                                        [config['input_height'], config['input_width']],
                                                        pad_value=config['padding_color'])
                            p_image = generate_input(config, image, None)
                            p_image = p_image[None, :, :, :]

                            # Prediction
                            exec_network.infer({input_blob: p_image})
                            results = exec_network.requests[0].outputs
                            
                            # Post-processing
                            output_mask = results['output']
                            pred = softmax(output_mask, axis=1)
                            predimg = pred[0].transpose((1, 2, 0))[:,:,1]
                            alphargb = predimg[bbx[1]:bbx[3], bbx[0]:bbx[2]]
                            alphargb = cv2.resize(alphargb, out_shape)
                            alphargb = cv2.cvtColor(alphargb, cv2.COLOR_GRAY2BGR)
                            
                            # Display
                            origin_image = cv2.resize(origin_image, out_shape)
                            display_img = np.uint8(origin_image * alphargb + background_img * (1-alphargb))

                    self.change_pixmap_signal.emit(display_img)
                    display_img = cv2.flip(display_img, 1)
                    rgba_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGBA)
                    cam.send(rgba_img)
        except Exception as e:
            logging.error(e)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beecam")
        self.disply_width = 960
        self.display_height = 640

        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.cam_selection = QComboBox(self)
        devices = [device for device in cam_to_id if 'obs' not in device.lower()]
        self.cam_selection.addItems(devices)
        self.cam_selection.activated[str].connect(self.update_video_source)
        self.cam_selection.setFixedSize(200, 50)
        self.img_browser_button = QPushButton(self)
        self.img_browser_button.setFixedSize(100, 50)
        self.img_browser_button.setText('Choose image')
        self.img_browser_button.clicked.connect(self.get_files)
        self.virtual_cam_button = QPushButton(self)
        self.virtual_cam_button.setFixedSize(200, 50)
        self.virtual_cam_button.setText('Create virtual camera')
        self.virtual_cam_button.clicked.connect(self.on_virtual_cam_clicked)

        # create a vertical box layout and add the two labels
        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label)
        vbox = QVBoxLayout()
        hbox.addLayout(vbox)
        vbox.addWidget(self.cam_selection)
        vbox.addWidget(self.img_browser_button)
        vbox.addWidget(self.virtual_cam_button)
        # set the vbox layout as the widgets layout
        self.setLayout(hbox)

        # Set initial video background image
        img = np.zeros((self.display_height, self.disply_width, 3), dtype=np.uint8)
        qt_img = self.convert_cv_qt(img)
        self.image_label.setPixmap(qt_img)

        # create the video capture thread
        # connect its signal to the update_image slot
        # start the thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        if tuple(cv_img.shape[:2]) != tuple([self.display_height, self.disply_width]):
            cv_img = cv2.resize(cv_img, (self.disply_width, self.display_height))
        
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def update_video_source(self, text):
        """
        """
        global video_id
        global video_changed_event
        global video_lock
        try:
            new_video_id = cam_to_id.get(text, 0)
            with video_lock:
                video_id = new_video_id
            video_changed_event.set()
        except Exception:
            pass
    
    def get_files(self):
        global default_bkg_img_path
        global bkg_img_path
        global bkg_changed_event

        home_dir = os.path.expanduser('~')
        picture_dir = os.path.join(home_dir, 'Pictures')
        open_dir = home_dir
        if os.path.isdir(picture_dir):
            open_dir = picture_dir

        file_dialog = QFileDialog()
        file_dialog.setWindowTitle('Open background image')
        file_dialog.setDirectory(open_dir)
        file_full_path = None
        if file_dialog.exec_() == QtWidgets.QDialog.Accepted:
            file_full_path = str(file_dialog.selectedFiles()[0])
        
        with bkg_lock:
            if file_full_path is None:
                bkg_img_path = default_bkg_img_path
            else:
                bkg_img_path = file_full_path
                bkg_changed_event.set()
        return file_full_path
    
    def on_virtual_cam_clicked(self, text):
        if created_virtual_cam_flag.is_set():
            # Remove the virtual camera
            command = ['regsvr32', '/u', OBS_VIRTUAL_CAM_DLL_PATH]
            run_as(command)
            created_virtual_cam_flag.clear()
            self.virtual_cam_button.setText('Create virtual camera')
        else:
            # Create a virtual camera
            command = ['regsvr32', '/n', '/i:1', OBS_VIRTUAL_CAM_DLL_PATH]
            run_as(command)
            created_virtual_cam_flag.set()
            self.virtual_cam_button.setText('Remove virtual camera')
    
    def closeEvent(self, event):
        global exit_event
        exit_event.set()
        self.thread.quit()
        self.thread.wait()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.move(200, 100)
    a.show()
    sys.exit(app.exec_())