import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import time
import numpy as np
from AMGSRN.Other.utility_functions import str2bool
from PyQt5.QtCore import QSize, Qt, QTimer, QMutex
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QStackedLayout, \
    QComboBox, QSlider, QFileDialog, QColorDialog, QCheckBox, QGroupBox
from superqt import QRangeSlider
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QEvent, Qt
from AMGSRN.renderer import Camera, Scene, TransferFunction, RawData
from AMGSRN.UI.utils import Arcball, torch_float_to_numpy_uint8
from AMGSRN.Models.options import load_options
from AMGSRN.Models.models import load_model
from typing import List
import json
import pyqtgraph as pg
import imageio.v3 as imageio

pg.setConfigOptions(antialias=True)

def linear_to_log(y):
    """Map linear [0,1] opacity to log-scaled [0,1] for display (3 decades)."""
    return np.log10(np.clip(y, 0, 1) * 999 + 1) / 3.0

def log_to_linear(y):
    """Map log-scaled [0,1] display value back to linear [0,1] opacity."""
    return (np.power(10, np.clip(y, 0, 1) * 3.0) - 1) / 999.0

def serialize_camera_state(camera):
    """Extract all serializable state from an Arcball camera into a dict."""
    skip_keys = {
        'camera_dirs', 'mouse_start', 'mouse_curr',
        'width', 'height', 'resolution',
    }
    state = {}
    for key, val in vars(camera).items():
        if key in skip_keys:
            continue
        if isinstance(val, np.ndarray):
            state[key] = {"__ndarray__": True, "data": val.tolist(), "dtype": str(val.dtype)}
        elif isinstance(val, (float, int, bool, str)):
            state[key] = val
        elif isinstance(val, np.floating):
            state[key] = float(val)
        elif isinstance(val, np.integer):
            state[key] = int(val)
        # Skip non-serializable attributes silently
    return state

def deserialize_camera_state(state):
    """Convert a serialized camera state dict back to native types."""
    restored = {}
    for key, val in state.items():
        if isinstance(val, dict) and val.get("__ndarray__"):
            restored[key] = np.array(val["data"], dtype=val["dtype"])
        else:
            restored[key] = val
    return restored

# For locking renderer actions
render_mutex = QMutex()

# Directories for things of interest
project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.abspath(os.path.join(project_folder_path, "..", ".."))
data_folder = os.path.join(project_folder_path, "Data")
savedmodels_folder = os.path.join(project_folder_path, "SavedModels")
tf_folder = os.path.join(project_folder_path, "Colormaps")

class TransferFunctionEditor(pg.GraphItem):
    '''
    Thanks to https://stackoverflow.com/questions/45624912/draggable-line-with-multiple-break-points
    '''
    def __init__(self, parent=None):
        self.dragPoint = None
        self.dragOffset = None
        self.lastDragPointIndex = 0
        self.parent = parent
        pg.GraphItem.__init__(self)

    def setData(self, convert_to_log=True, **kwds):
        '''
        Assumes kwds['pos'] is a pre-sorted lists of tuples of control point -> opacity
        sorted by control point value. I.e.
        [[0, 0], [0.5, 1.0], [1.0, 0.0]]
        is a mountain and is legal because kwds['pos'][:,0] is strictly increasing.
        '''
        self.data = kwds
        self.data['size']=12
        self.data['pxMode']=True
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            # Normalize control point x to [0,1]
            self.data['pos'][:,0] -= self.data['pos'][0,0]
            self.data['pos'][:,0] /= self.data['pos'][-1,0]
            # Clip opacity between 0 and 1
            self.data['pos'][:,1] = np.clip(self.data['pos'][:,1], 0.0, 1.0)
            # Convert Y to log display space
            if convert_to_log:
                self.data['pos'][:,1] = linear_to_log(self.data['pos'][:,1])
            self.data['adj'] = np.column_stack((np.arange(0, npts-1), np.arange(1, npts)))
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        if(self.parent is not None):
            if "pos" in self.data.keys():
                opacity_control_points = self.data['pos'][:,0]
                # Convert Y back from log display space to linear for the renderer
                opacity_values = log_to_linear(self.data['pos'][:,1])
                if self.parent.render_worker is not None:
                    self.parent.render_worker.change_opacity_controlpoints.emit(
                        opacity_control_points, opacity_values
                    )
    def deleteLastPoint(self):
        if(self.lastDragPointIndex > 0 and 
           self.lastDragPointIndex < self.data['pos'].shape[0]-1):
            new_pos = np.concatenate(
                [self.data['pos'][0:self.lastDragPointIndex],
                self.data['pos'][self.lastDragPointIndex+1:]],
                axis=0
            )
            self.data['pos'] = new_pos
            self.setData(convert_to_log=False, **self.data)
            self.lastDragPointIndex -= 1
        
    def mouseDragEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.lastDragPointIndex = ind
            self.dragOffset = [
                self.data['pos'][ind][0] - pos[0],
                self.data['pos'][ind][1] - pos[1]
            ]
        elif ev.isFinish():       
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.data()[0]
        
        # Cannot move endpoints
        if(ind == 0 or ind == self.data['pos'].shape[0]-1):
            # only move y
            self.data['pos'][ind][1] = np.clip(ev.pos()[1] + self.dragOffset[1], 0.0, 1.0)
        # Points in between cannot move past other points to maintain ordering
        else:
            # move x
            self.data['pos'][ind][0] = np.clip(ev.pos()[0] + self.dragOffset[0], 
                                               self.data['pos'][ind-1][0]+1e-4,
                                               self.data['pos'][ind+1][0]-1e-4)
            # move y
            self.data['pos'][ind][1] = np.clip(ev.pos()[1] + self.dragOffset[1], 0.0, 1.0)

        self.updateGraph()
        ev.accept()
        
    def mouseClickEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        
        p = event.pos()
        x = np.clip(p.x(), 0.0, 1.0)
        y = np.clip(p.y(), 0.0, 1.0)
        
        pts = self.scatter.pointsAt(p)
        if len(pts) > 0:
            return
        
        if x > 0 and x < 1.0:
            ind = 0
            while x > self.data['pos'][ind][0]:
                ind += 1
            new_pos = np.concatenate(
                [self.data['pos'][0:ind],
                 [[x, y]],
                 self.data['pos'][ind:]
                 ],
                axis=0
            )
            self.data['pos'] = new_pos
            self.setData(convert_to_log=False, **self.data)
            self.lastDragPointIndex = ind
                
class MainWindow(QMainWindow):
    
    loading_model = True
    last_img = np.zeros([1,1,3],dtype=np.uint8)
    updates_per_second = pyqtSignal(float)
    frame_time = pyqtSignal(float)
    vram_use = pyqtSignal(float)
    status_text_update = pyqtSignal(str)
    timestep_max = pyqtSignal(int)
    render_status = pyqtSignal(str)
    render_worker = None
    
    def __init__(self, parent=None):
        super().__init__(parent)
        project_folder_path = os.path.dirname(os.path.abspath(__file__))
        project_folder_path = os.path.abspath(os.path.join(project_folder_path, "..", ".."))
        
        self.setWindowTitle("Neural Volume Renderer") 
            
        # Find all available models/colormaps        
        self.available_models = os.listdir(savedmodels_folder)
        self.available_tfs = os.listdir(tf_folder)
        self.available_data = os.listdir(data_folder)
        
        # Full screen layout
        layout = QHBoxLayout()        
        
        # Render area
        self.render_view = QLabel()   
        # self.render_view.setMaximumWidth(1024)
        # self.render_view.setMaximumWidth(1024)
        self.render_view.mousePressEvent = self.mouseClicked
        self.render_view.mouseReleaseEvent = self.mouseReleased
        self.render_view.mouseMoveEvent = self.mouseMove   
        self.render_view.wheelEvent = self.zoom   
        
        # Settings area
        self.settings_ui = QVBoxLayout()       
         
        self.load_box = QHBoxLayout()
        self.load_box.addWidget(QLabel("Load file or folder:"))
        self.load_button = QPushButton("Browse")
        self.load_button.clicked.connect(self.open_file_dialog)
        self.load_box.addWidget(self.load_button)

        self.file_path_label = QLabel("No file selected")
        self.load_box.addWidget(self.file_path_label)
        
        self.tf_box = QHBoxLayout()  
        self.tf_box.addWidget(QLabel("Colormap:"))
        self.tfs_dropdown = self.load_colormaps_dropdown()
        self.tfs_dropdown.currentTextChanged.connect(self.load_tf)
        self.tf_box.addWidget(self.tfs_dropdown)
        
        self.batch_slider_box = QHBoxLayout()      
        self.batch_slider_label = QLabel("Batch size (2^x): 20")  
        self.batch_slider_box.addWidget(self.batch_slider_label)
        self.batch_slider = QSlider(Qt.Horizontal)
        self.batch_slider.setMinimum(18)
        self.batch_slider.setMaximum(25)
        self.batch_slider.setValue(20)   
        self.batch_slider.setTickPosition(QSlider.TicksBelow)
        self.batch_slider.setTickInterval(1)  
        self.batch_slider.valueChanged.connect(self.change_batch_visual)
        self.batch_slider.sliderReleased.connect(self.change_batch)
        self.batch_slider_box.addWidget(self.batch_slider)   
        
        self.spp_slider_box = QHBoxLayout()      
        self.spp_slider_label = QLabel("Samples per ray: 256")  
        self.spp_slider_box.addWidget(self.spp_slider_label)
        self.spp_slider = QSlider(Qt.Horizontal)
        self.spp_slider.setMinimum(7)
        self.spp_slider.setMaximum(13)
        self.spp_slider.setValue(8)   
        self.spp_slider.setTickPosition(QSlider.TicksBelow)
        self.spp_slider.setTickInterval(1)  
        self.spp_slider.valueChanged.connect(self.change_spp_visual)
        self.spp_slider.sliderReleased.connect(self.change_spp)
        self.spp_slider_box.addWidget(self.spp_slider)   
        
        self.view_xy_button = QPushButton("reset view to xy-plane")
        self.view_xy_button.setFixedHeight(25)
        self.view_xy_button.clicked.connect(lambda: self.render_worker.view_xy.emit())
        
        self.density_toggle = QPushButton("Toggle density")
        self.density_toggle.setFixedHeight(25)
        self.density_toggle.clicked.connect(self.toggle_density)

        # === Camera save/load buttons ===
        self.camera_io_box = QHBoxLayout()
        self.save_camera_button = QPushButton("Save Camera")
        self.save_camera_button.setFixedHeight(25)
        self.save_camera_button.clicked.connect(self.save_camera)
        self.camera_io_box.addWidget(self.save_camera_button)

        self.load_camera_button = QPushButton("Load Camera")
        self.load_camera_button.setFixedHeight(25)
        self.load_camera_button.clicked.connect(self.load_camera)
        self.camera_io_box.addWidget(self.load_camera_button)
        # === End camera save/load buttons ===

        # === Aspect ratio lock ===
        self.lock_aspect_checkbox = QCheckBox("Lock 16:9 aspect ratio (for 4K renders)")
        self.lock_aspect_checkbox.setChecked(False)
        self.lock_aspect_checkbox.stateChanged.connect(self.apply_aspect_lock)
        # === End aspect ratio lock ===

        # === Lighting controls ===
        self.lighting_group = QGroupBox("Lighting")
        self.lighting_layout = QVBoxLayout()

        # Shading on/off
        self.shading_checkbox = QCheckBox("Enable shading")
        self.shading_checkbox.setChecked(True)
        self.shading_checkbox.stateChanged.connect(self.change_shading_enabled)
        self.lighting_layout.addWidget(self.shading_checkbox)

        # Light mode dropdown
        self.light_mode_box = QHBoxLayout()
        self.light_mode_box.addWidget(QLabel("Light mode:"))
        self.light_mode_dropdown = QComboBox()
        self.light_mode_dropdown.addItems(["Headlight", "Scene light"])
        self.light_mode_dropdown.currentTextChanged.connect(self.change_light_mode)
        self.light_mode_box.addWidget(self.light_mode_dropdown)
        self.lighting_layout.addLayout(self.light_mode_box)

        # Ambient slider
        self.ambient_slider_box = QHBoxLayout()
        self.ambient_slider_label = QLabel("Ambient: 0.30")
        self.ambient_slider_box.addWidget(self.ambient_slider_label)
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setMinimum(0)
        self.ambient_slider.setMaximum(100)
        self.ambient_slider.setValue(30)
        self.ambient_slider.valueChanged.connect(self.change_ambient_visual)
        self.ambient_slider.sliderReleased.connect(self.change_ambient)
        self.ambient_slider_box.addWidget(self.ambient_slider)
        self.lighting_layout.addLayout(self.ambient_slider_box)

        # Diffuse slider
        self.diffuse_slider_box = QHBoxLayout()
        self.diffuse_slider_label = QLabel("Diffuse: 0.70")
        self.diffuse_slider_box.addWidget(self.diffuse_slider_label)
        self.diffuse_slider = QSlider(Qt.Horizontal)
        self.diffuse_slider.setMinimum(0)
        self.diffuse_slider.setMaximum(100)
        self.diffuse_slider.setValue(70)
        self.diffuse_slider.valueChanged.connect(self.change_diffuse_visual)
        self.diffuse_slider.sliderReleased.connect(self.change_diffuse)
        self.diffuse_slider_box.addWidget(self.diffuse_slider)
        self.lighting_layout.addLayout(self.diffuse_slider_box)

        # Specular slider
        self.specular_slider_box = QHBoxLayout()
        self.specular_slider_label = QLabel("Specular: 0.00")
        self.specular_slider_box.addWidget(self.specular_slider_label)
        self.specular_slider = QSlider(Qt.Horizontal)
        self.specular_slider.setMinimum(0)
        self.specular_slider.setMaximum(100)
        self.specular_slider.setValue(0)
        self.specular_slider.valueChanged.connect(self.change_specular_visual)
        self.specular_slider.sliderReleased.connect(self.change_specular)
        self.specular_slider_box.addWidget(self.specular_slider)
        self.lighting_layout.addLayout(self.specular_slider_box)

        # Shininess slider
        self.shininess_slider_box = QHBoxLayout()
        self.shininess_slider_label = QLabel("Shininess: 32")
        self.shininess_slider_box.addWidget(self.shininess_slider_label)
        self.shininess_slider = QSlider(Qt.Horizontal)
        self.shininess_slider.setMinimum(1)
        self.shininess_slider.setMaximum(128)
        self.shininess_slider.setValue(32)
        self.shininess_slider.valueChanged.connect(self.change_shininess_visual)
        self.shininess_slider.sliderReleased.connect(self.change_shininess)
        self.shininess_slider_box.addWidget(self.shininess_slider)
        self.lighting_layout.addLayout(self.shininess_slider_box)

        # Scene light position sliders (X, Y, Z)
        self.light_pos_label = QLabel("Light position (scene mode):")
        self.lighting_layout.addWidget(self.light_pos_label)

        self.light_x_box = QHBoxLayout()
        self.light_x_label = QLabel("X: 0.50")
        self.light_x_box.addWidget(self.light_x_label)
        self.light_x_slider = QSlider(Qt.Horizontal)
        self.light_x_slider.setMinimum(-100)
        self.light_x_slider.setMaximum(300)
        self.light_x_slider.setValue(50)
        self.light_x_slider.valueChanged.connect(self.change_light_pos_visual)
        self.light_x_slider.sliderReleased.connect(self.change_light_position)
        self.light_x_box.addWidget(self.light_x_slider)
        self.lighting_layout.addLayout(self.light_x_box)

        self.light_y_box = QHBoxLayout()
        self.light_y_label = QLabel("Y: 0.50")
        self.light_y_box.addWidget(self.light_y_label)
        self.light_y_slider = QSlider(Qt.Horizontal)
        self.light_y_slider.setMinimum(-100)
        self.light_y_slider.setMaximum(300)
        self.light_y_slider.setValue(50)
        self.light_y_slider.valueChanged.connect(self.change_light_pos_visual)
        self.light_y_slider.sliderReleased.connect(self.change_light_position)
        self.light_y_box.addWidget(self.light_y_slider)
        self.lighting_layout.addLayout(self.light_y_box)

        self.light_z_box = QHBoxLayout()
        self.light_z_label = QLabel("Z: 0.50")
        self.light_z_box.addWidget(self.light_z_label)
        self.light_z_slider = QSlider(Qt.Horizontal)
        self.light_z_slider.setMinimum(-100)
        self.light_z_slider.setMaximum(300)
        self.light_z_slider.setValue(50)
        self.light_z_slider.valueChanged.connect(self.change_light_pos_visual)
        self.light_z_slider.sliderReleased.connect(self.change_light_position)
        self.light_z_box.addWidget(self.light_z_slider)
        self.lighting_layout.addLayout(self.light_z_box)

        self.lighting_group.setLayout(self.lighting_layout)
        # === End lighting controls ===

        self.transfer_function_box = QVBoxLayout()
        self.tf_editor = TransferFunctionEditor(self)
        self.background_color_box = QHBoxLayout()
        self.background_color_box.addWidget(QLabel("Background Color:"))
        self.background_color_button = QPushButton()
        self.background_color_button.setFixedSize(30, 30)
        self.background_color_button.setStyleSheet("background-color: white;")
        self.background_color_button.clicked.connect(self.choose_background_color)
        self.background_color_box.addWidget(self.background_color_button)
        self.transfer_function_box.addLayout(self.background_color_box)

        self.tf_rescale_slider_box = QHBoxLayout()  
        self.tf_rescale_slider_mintxt = QLabel(f"data range: {0:3}%")
        self.tf_rescale_slider_maxtxt = QLabel(f"{100}%")
        self.tf_rescale_slider = QRangeSlider(Qt.Horizontal)
        self.tf_rescale_slider.setRange(0, 1000)
        self.tf_rescale_slider.setValue((0, 1000))
        self.tf_rescale_slider.setTickInterval(1)
        self.tf_rescale_slider.valueChanged.connect(self.change_tf_range_visual)
        self.tf_rescale_slider.sliderReleased.connect(self.change_tf_range)
        self.tf_rescale_slider_box.addWidget(self.tf_rescale_slider_mintxt)
        self.tf_rescale_slider_box.addWidget(self.tf_rescale_slider)
        self.tf_rescale_slider_box.addWidget(self.tf_rescale_slider_maxtxt)

        self.timestep_selector_box = QHBoxLayout()      
        self.timestep_selector_label = QLabel("Timestep: 0")  
        self.timestep_selector_box.addWidget(self.timestep_selector_label)
        self.timestep_slider = QSlider(Qt.Horizontal)
        self.timestep_slider.setMinimum(0)
        self.timestep_slider.setMaximum(1)
        self.timestep_slider.setValue(0)   
        self.timestep_slider.setTickPosition(QSlider.TicksBelow)
        self.timestep_slider.setTickInterval(1)  
        self.timestep_slider.valueChanged.connect(self.change_timestep)
        #self.timestep_slider.sliderReleased.connect(self.change_timestep)
        self.timestep_selector_box.addWidget(self.timestep_slider)   
        
        x = np.linspace(0.0, 1.0, 4)
        pos = np.column_stack((x, x))
        win = pg.GraphicsLayoutWidget()
        win.setMinimumHeight(250)
        view = win.addViewBox(row=0, col=1, rowspan=2, colspan=2) 
        view.enableAutoRange(axis='xy', enable=False)
        view.setYRange(0, 1.0, padding=0.1, update=True)
        view.setXRange(0, 1.0, padding=0.1, update=True)
        view.setBackgroundColor([255, 255, 255, 255])
        view.setMouseEnabled(x=False,y=False)
        x_axis = pg.AxisItem("bottom", linkView=view)
        # Custom log-scale tick labels for the Y axis
        log_ticks = [
            (linear_to_log(0.0), "0"),
            (linear_to_log(0.001), "0.001"),
            (linear_to_log(0.01), "0.01"),
            (linear_to_log(0.1), "0.1"),
            (linear_to_log(0.5), "0.5"),
            (linear_to_log(1.0), "1.0"),
        ]
        y_axis = pg.AxisItem("left", linkView=view)
        y_axis.setTicks([log_ticks])
        win.addItem(x_axis, row=2, col=1, colspan=2)
        win.addItem(y_axis, row=0, col=0, rowspan=2)
        view.addItem(self.tf_editor)
        self.transfer_function_box.addWidget(win)
        #self.transfer_function_box.addWidget(x_axis)
        #self.transfer_function_box.addWidget(y_axis)

        # === Save Colormap button ===
        self.save_colormap_button = QPushButton("Save Colormap")
        self.save_colormap_button.setFixedHeight(25)
        self.save_colormap_button.clicked.connect(self.save_colormap)
        self.transfer_function_box.addWidget(self.save_colormap_button)
        # === End Save Colormap button ===
        
                        
        self.status_text = QLabel("") 
        self.memory_use_label = QLabel("VRAM use: -- GB") 
        self.update_framerate_label = QLabel("Update framerate: -- fps") 
        self.frame_time_label = QLabel("Last frame time: -- sec.") 
        self.render_status_label = QLabel("Render: idle")
        self.status_text_update.connect(self.update_status_text)
        self.vram_use.connect(self.update_vram)
        self.timestep_max.connect(self.update_timestep_max)
        self.updates_per_second.connect(self.update_updates)
        self.frame_time.connect(self.update_frame_time)
        self.render_status.connect(self.update_render_status)
        self.save_img_button = QPushButton("Save image")
        self.save_img_button.clicked.connect(self.save_img)
        
        self.settings_ui.addLayout(self.load_box)
        self.settings_ui.addLayout(self.tf_box)
        self.settings_ui.addLayout(self.batch_slider_box)
        self.settings_ui.addLayout(self.spp_slider_box)
        self.settings_ui.addWidget(self.view_xy_button)
        # self.settings_ui.addWidget(self.density_toggle)
        self.settings_ui.addLayout(self.camera_io_box)
        self.settings_ui.addWidget(self.lock_aspect_checkbox)
        self.settings_ui.addWidget(self.lighting_group)
        self.settings_ui.addLayout(self.transfer_function_box)
        self.settings_ui.addLayout(self.tf_rescale_slider_box)
        self.settings_ui.addLayout(self.timestep_selector_box)
        # self.settings_ui.addStretch()
        # self.settings_ui.addWidget(self.status_text)
        self.settings_ui.addWidget(self.memory_use_label)
        self.settings_ui.addWidget(self.update_framerate_label)
        self.settings_ui.addWidget(self.frame_time_label)
        self.settings_ui.addWidget(self.render_status_label)
        self.settings_ui.addWidget(self.save_img_button)
        
        # UI full layout        
        layout.addWidget(self.render_view, stretch=4)
        layout.addLayout(self.settings_ui, stretch=1)        
        layout.setContentsMargins(0,0,10,10)
        layout.setSpacing(20)
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(layout)
        self.setCentralWidget(self.centralWidget)
        
        # Set up render thread
        self.render_thread = QThread()
        self.render_worker = RendererThread(self)        
        self.load_renderer()
        
        # Variables to use for interaction
        self.rotating = False
        self.panning = False
        self.last_x = None
        self.last_y = None
   

    def open_file_dialog(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.Directory)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        file_dialog.setOption(QFileDialog.ShowDirsOnly, False)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedUrls()
            if selected_files:
                file_path = selected_files[0].toLocalFile()
                print(file_path)
                self.file_path_label.setText(file_path)
                self.load_file_or_folder(file_path)

    def load_file_or_folder(self, path):
        if os.path.isfile(path):
            self.load_file(path)
        elif os.path.isdir(path):
            self.load_folder(path)
        else:
            print("Invalid path selected")

    def load_file(self, file_path):
        # Implement file loading logic here
        pass

    def load_folder(self, folder_path):
        self.load_model(folder_path)
        pass 

    def choose_background_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.background_color_button.setStyleSheet(f"background-color: {color.name()};")
            c = color.getRgbF()
            self.render_worker.change_background_color.emit(c[0], c[1], c[2])
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.tf_editor.deleteLastPoint()
        event.accept()

    def save_camera(self):
        """Save current camera state to a JSON file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Camera View", "", "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.endswith(".json"):
            filepath += ".json"

        render_mutex.lock()
        state = serialize_camera_state(self.render_worker.camera)
        render_mutex.unlock()

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Camera saved to {filepath}")
        self.status_text_update.emit(f"Camera saved to {os.path.basename(filepath)}")

    def load_camera(self):
        """Load camera state from a JSON file and apply it."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Camera View", "", "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return

        with open(filepath, 'r') as f:
            state = json.load(f)

        restored = deserialize_camera_state(state)
        self.render_worker.load_camera_state.emit(restored)
        print(f"Camera loaded from {filepath}")
        self.status_text_update.emit(f"Camera loaded from {os.path.basename(filepath)}")

    def save_colormap(self):
        """Save the current color + opacity map as a ParaView-style JSON file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Colormap", "", "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return
        if not filepath.endswith(".json"):
            filepath += ".json"

        name = os.path.splitext(os.path.basename(filepath))[0]

        # --- Opacity from the TF editor (normalized 0-1) ---
        if 'pos' not in self.tf_editor.data:
            self.status_text_update.emit("No opacity data to save")
            return
        opacity_pos = self.tf_editor.data['pos'].copy()
        opacity_x = opacity_pos[:, 0]                 # control-point scalars in [0,1]
        opacity_y = log_to_linear(opacity_pos[:, 1])   # convert display-log back to linear

        # Build Points array: [scalar, opacity, midpoint, sharpness] per control point
        points = []
        for x, y in zip(opacity_x, opacity_y):
            points.extend([float(x), float(y), 0.5, 0.0])

        # --- Color from the currently-loaded colormap file ---
        current_tf_name = self.tfs_dropdown.currentText()
        tf_path = os.path.join(tf_folder, current_tf_name)
        rgb_points = []
        try:
            with open(tf_path, 'r') as f:
                tf_data = json.load(f)
            src = tf_data[0] if isinstance(tf_data, list) else tf_data
            raw = src.get('RGBPoints', [])
            # Normalise the scalar channel to [0,1] so it matches the opacity
            if len(raw) >= 8:                        # at least 2 colour stops
                scalars = raw[0::4]
                s_min, s_max = scalars[0], scalars[-1]
                rng = s_max - s_min if s_max != s_min else 1.0
                rgb_points = list(raw)                # copy
                for i in range(0, len(rgb_points), 4):
                    rgb_points[i] = (rgb_points[i] - s_min) / rng
            else:
                rgb_points = list(raw)
        except Exception as e:
            print(f"Warning: could not read colormap file: {e}")
            rgb_points = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

        output = [{
            "ColorSpace": "RGB",
            "Name": name,
            "Points": points,
            "RGBPoints": rgb_points
        }]

        with open(filepath, 'w') as f:
            json.dump(output, f, indent='\t')

        print(f"Colormap saved to {filepath}")
        self.status_text_update.emit(f"Colormap saved to {os.path.basename(filepath)}")

    def save_img(self):
        folderpath,_ = QFileDialog.getSaveFileName(self, 'Select Save Location')
        if ".jpg" not in folderpath and ".png" not in folderpath:
            folderpath = folderpath + ".png"
        
        print(f"Saving to {folderpath}")
        
        imageio.imwrite(folderpath, self.last_img)
             
    def update_status_text(self, val):
        self.status_text.setText(f"{val}")
        
    def update_vram(self, val):
        self.memory_use_label.setText(f"VRAM use: {val:0.02f} GB")
    
    def update_timestep_max(self, val):
        print(f"Updating timestep max to {val}")
        self.timestep_slider.setMaximum(val)
    
    def set_timestep(self, val):
        self.timestep_slider.setValue(val)
        self.timestep_selector_label.setText(f"Timestep: {val}")

    def update_updates(self, val):
        self.update_framerate_label.setText(f"Update framerate: {val:0.02f} fps")
    
    def toggle_density(self):
        if("AMGSRN" in self.render_worker.opt['model'] or
           "APMGSRN" in self.render_worker.opt['model']):
            self.render_worker.toggle_density.emit()
        else:
            self.density_toggle.setChecked(not self.density_toggle.isChecked())

    def update_frame_time(self, val):
        self.frame_time_label.setText(f"Last frame time: {val:0.02f} sec.")

    def update_render_status(self, val):
        self.render_status_label.setText(f"Render: {val}")
     
    def data_box_update(self, s):
        self.loading_model = "Model" in s
        if(self.loading_model):
            self.models_dropdown.clear()
            self.models_dropdown.addItems(self.available_models)
        else:            
            self.models_dropdown.clear()
            self.models_dropdown.addItems(self.available_data)

    def load_colormaps_dropdown(self):
        dropdown = QComboBox()        
        dropdown.addItems(self.available_tfs)        
        return dropdown
                    
    def load_renderer(self):    
        self.render_worker.moveToThread(self.render_thread)                
        self.render_worker.progress.connect(self.set_render_image)      
        self.render_thread.started.connect(self.render_worker.run)
                
        self.render_thread.start()
        
    def finish_worker(self):
        self.render_worker.finished.connect(self.render_thread.quit)
        self.render_worker.finished.connect(self.render_worker.deleteLater)
        self.render_thread.finished.connect(self.render_worker.deleteLater)
    
    def apply_aspect_lock(self):
        self.resizeEvent(None)

    def resizeEvent(self, event):
        w = self.render_view.frameGeometry().width()
        h = self.render_view.frameGeometry().height()
        if self.lock_aspect_checkbox.isChecked():
            target = 16 / 9
            if w / h > target:
                w = int(h * target)
            else:
                h = int(w / target)
        self.render_worker.resize.emit(w, h)
        if event is not None:
            QMainWindow.resizeEvent(self, event)
     
    def mouseClicked(self, event):
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.startRotate()
            if event.button() == Qt.RightButton:
                self.startPan()
                             
    def startPan(self):
        self.panning = True
        
    def startRotate(self):
        self.rotating = True
  
    def load_model(self, s):
        if s == "":
            return
        self.status_text_update.emit(f"Loading model {s}...")
        if(self.loading_model):
            self.render_worker.load_new_model.emit(s)
        else:
            self.render_worker.load_new_data.emit(s)
        self.status_text_update.emit("")

        
    def load_tf(self, s):
        print(f"TF changed {s}")
        self.render_worker.change_transfer_function.emit(s)
    
    def change_batch_visual(self):
        val = int(self.batch_slider.value())
        self.batch_slider_label.setText(f"Batch size (2^x): {val}")
        
    def change_batch(self):
        val = int(self.batch_slider.value())
        self.batch_slider_label.setText(f"Batch size (2^x): {val}")
        self.render_worker.change_batch_size.emit(val)
     
    def change_timestep(self):
        val = int(self.timestep_slider.value())
        self.timestep_selector_label.setText(f"Timestep: {val}")
        self.render_worker.change_timestep.emit(val)
     
    def change_spp_visual(self):
        val = int(self.spp_slider.value())
        self.spp_slider_label.setText(f"Samples per ray: {2**val}")
        
    def change_spp(self):
        val = int(self.spp_slider.value())
        self.spp_slider_label.setText(f"Samples per ray: {2**val}")
        self.render_worker.change_spp.emit(2**val)
     
    def change_tf_range_visual(self):
        dmin, dmax = [int(v) for v in self.tf_rescale_slider.value()]
        self.tf_rescale_slider_mintxt.setText(f"data range: {dmin//10:3}%")
        self.tf_rescale_slider_maxtxt.setText(f"{dmax//10}%")
    
    def change_tf_range(self):
        dmin, dmax = [int(v) for v in self.tf_rescale_slider.value()]
        self.render_worker.tf_rescale.emit(dmin/1000.0, dmax/1000.0)

    # === Lighting UI callbacks ===
    def change_shading_enabled(self, state):
        enabled = state == Qt.Checked
        self.render_worker.change_shading_enabled.emit(enabled)

    def change_light_mode(self, text):
        mode = 'headlight' if text == "Headlight" else 'scene'
        self.render_worker.change_light_mode.emit(mode)

    def change_ambient_visual(self):
        val = self.ambient_slider.value() / 100.0
        self.ambient_slider_label.setText(f"Ambient: {val:.2f}")

    def change_ambient(self):
        val = self.ambient_slider.value() / 100.0
        self.ambient_slider_label.setText(f"Ambient: {val:.2f}")
        self.render_worker.change_ambient.emit(val)

    def change_diffuse_visual(self):
        val = self.diffuse_slider.value() / 100.0
        self.diffuse_slider_label.setText(f"Diffuse: {val:.2f}")

    def change_diffuse(self):
        val = self.diffuse_slider.value() / 100.0
        self.diffuse_slider_label.setText(f"Diffuse: {val:.2f}")
        self.render_worker.change_diffuse.emit(val)

    def change_specular_visual(self):
        val = self.specular_slider.value() / 100.0
        self.specular_slider_label.setText(f"Specular: {val:.2f}")

    def change_specular(self):
        val = self.specular_slider.value() / 100.0
        self.specular_slider_label.setText(f"Specular: {val:.2f}")
        self.render_worker.change_specular.emit(val)

    def change_shininess_visual(self):
        val = self.shininess_slider.value()
        self.shininess_slider_label.setText(f"Shininess: {val}")

    def change_shininess(self):
        val = self.shininess_slider.value()
        self.shininess_slider_label.setText(f"Shininess: {val}")
        self.render_worker.change_shininess.emit(float(val))

    def change_light_pos_visual(self):
        x = self.light_x_slider.value() / 100.0
        y = self.light_y_slider.value() / 100.0
        z = self.light_z_slider.value() / 100.0
        self.light_x_label.setText(f"X: {x:.2f}")
        self.light_y_label.setText(f"Y: {y:.2f}")
        self.light_z_label.setText(f"Z: {z:.2f}")

    def change_light_position(self):
        x = self.light_x_slider.value() / 100.0
        y = self.light_y_slider.value() / 100.0
        z = self.light_z_slider.value() / 100.0
        self.render_worker.change_light_position.emit(x, y, z)
    # === End lighting UI callbacks ===

    def mouseReleased(self, event):
        if event.type() == QEvent.MouseButtonRelease:
            if(event.button()) == Qt.LeftButton:
                self.endRotate()
            if(event.button()) == Qt.RightButton:
                self.endPan()
    
    def endPan(self):
        self.panning = False
        self.last_x = None
        self.last_y = None
        
    def endRotate(self):
        self.rotating = False
        self.last_x = None
        self.last_y = None
        
    def mouseMove(self, event):
        
        w = self.render_view.frameGeometry().width()
        h = self.render_view.frameGeometry().height()
        x = (event.x() / w) * 2 - 1
        y = -((event.y() / h) * 2 - 1)
        
        if(self.last_x is None or self.last_y is None):
            self.last_x = x
            self.last_y = y
            return
        if(x is None or y is None):
            print("x or y was None!")
            return
        
        if self.rotating:
            self.render_worker.rotate.emit(self.last_x, self.last_y, x, y)
            self.last_x = x
            self.last_y = y
        if self.panning:
            self.render_worker.pan.emit(self.last_x, self.last_y, x, y)
            self.last_x = x
            self.last_y = y
    
    def zoom(self,event):
        scroll = event.angleDelta().y()/120
        self.render_worker.zoom.emit(-scroll)
         
    def set_render_image(self, img: np.ndarray):
        height, width, channel = img.shape
        self.last_img = img
        bytesPerLine = channel * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(qImg)
        self.render_view.setPixmap(pixmap.scaled(self.render_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
class RendererThread(QObject):
    progress = pyqtSignal(np.ndarray)
    rotate = pyqtSignal(float, float, float, float)
    pan = pyqtSignal(float, float, float, float)
    zoom = pyqtSignal(float)
    resize = pyqtSignal(int, int)
    change_spp = pyqtSignal(int)
    load_new_model = pyqtSignal(str)
    load_new_data = pyqtSignal(str)
    change_transfer_function = pyqtSignal(str)
    change_background_color = pyqtSignal(float, float, float)
    change_batch_size = pyqtSignal(int)
    change_timestep = pyqtSignal(int)
    change_opacity_controlpoints = pyqtSignal(np.ndarray, np.ndarray)
    view_xy = pyqtSignal()
    tf_rescale = pyqtSignal(float, float)
    toggle_density = pyqtSignal()
    # Lighting signals
    change_shading_enabled = pyqtSignal(bool)
    change_light_mode = pyqtSignal(str)
    change_ambient = pyqtSignal(float)
    change_diffuse = pyqtSignal(float)
    change_specular = pyqtSignal(float)
    change_shininess = pyqtSignal(float)
    change_light_position = pyqtSignal(float, float, float)
    # Camera state signal
    load_camera_state = pyqtSignal(dict)

    def __init__(self, parent=None):
        super(RendererThread, self).__init__()
        self.parent = parent
        
        # Local variables needed to keep track of 
        self.device = "cuda:0"
        self.spp = 256
        self.batch_size = 2**20
        self.resolution = [256,256]
        self.full_shape = [1,1,1]
        self.opt = None  
        self.model = None
        self.camera = None
        self.update_rate = []
        self.frame_rate = []
        self.tf = TransferFunction(self.device)      
        
        #self.initialize_model()  
        self.parent.status_text_update.emit("Initializing scene...") 
        self.initialize_camera()        
        self.scene = Scene(self.model, self.camera, 
                           self.full_shape, self.resolution, 
                           self.batch_size, self.spp, 
                           self.tf, self.device)
        self.do_change_transfer_function("Coolwarm.json")
        self.do_view_xy()
        self.scene.on_setting_change()
        
        # Set up events
        self.pan.connect(self.do_pan)
        self.rotate.connect(self.do_rotate)
        self.zoom.connect(self.do_zoom)
        self.resize.connect(self.do_resize)
        self.change_batch_size.connect(self.do_change_batch_size)
        self.change_timestep.connect(self.do_change_timestep)
        self.change_spp.connect(self.do_change_spp)
        self.change_transfer_function.connect(self.do_change_transfer_function)
        self.change_opacity_controlpoints.connect(self.do_change_opacities)
        self.load_new_model.connect(self.do_change_model)
        self.change_background_color.connect(self.do_change_background_color)
        self.load_new_data.connect(self.do_change_data)
        self.view_xy.connect(self.do_view_xy)
        self.tf_rescale.connect(self.do_tf_rescale)
        self.toggle_density.connect(self.do_toggle_density)
        # Lighting event connections
        self.change_shading_enabled.connect(self.do_change_shading_enabled)
        self.change_light_mode.connect(self.do_change_light_mode)
        self.change_ambient.connect(self.do_change_ambient)
        self.change_diffuse.connect(self.do_change_diffuse)
        self.change_specular.connect(self.do_change_specular)
        self.change_shininess.connect(self.do_change_shininess)
        self.change_light_position.connect(self.do_change_light_position)
        # Camera state connection
        self.load_camera_state.connect(self.do_load_camera_state)
        self.parent.status_text_update.emit(f"")
        
    def run(self):
        last_spot = 0
        current_spot = 0
        try:
            while True:
                if self.model is None:
                    time.sleep(0.1)
                    continue
                current_spot = self.scene.current_order_spot
                render_mutex.lock()
                if(self.scene.current_order_spot == 0):
                    frame_start_time = time.time()
                    self.parent.render_status.emit("Rendering...")
                update_start_time = time.time()
                self.scene.one_step_update()
                if(current_spot < len(self.scene.render_order)):
                    update_time = time.time() - update_start_time
                    self.update_rate.append(update_time)
                if(current_spot == len(self.scene.render_order) and 
                    last_spot < current_spot):
                        frame_time = time.time() - frame_start_time
                        self.frame_rate.append(frame_time)
                        last_frame_time = self.frame_rate[-1]
                        self.parent.frame_time.emit(last_frame_time)
                        self.parent.render_status.emit("Complete")
                img = torch_float_to_numpy_uint8(self.scene.temp_image)
                render_mutex.unlock()

                self.progress.emit(img)
                self.parent.vram_use.emit(self.scene.get_mem_use())
                
                if(len(self.update_rate) > 20):
                    self.update_rate.pop(0)
                if(len(self.frame_rate) > 5):
                    self.frame_rate.pop(0)
                if(len(self.update_rate) > 0):
                    average_update_fps = 1/np.array(self.update_rate).mean()
                    self.parent.updates_per_second.emit(average_update_fps)
                last_spot = current_spot
        except Exception as e:
            print(f"Exiting render thread.")
            raise e

    def do_change_background_color(self, r, g, b):
        render_mutex.lock()
        self.scene.set_background_color(r,g,b)
        self.scene.on_setting_change()
        render_mutex.unlock()

    def do_resize(self, w, h):
        render_mutex.lock()
        self.scene.image_resolution = [h,w]
        self.scene.on_resize()
        self.update_rate = []
        self.frame_rate = []
        render_mutex.unlock()
        
    def do_rotate(self, last_x, last_y, x, y):
        render_mutex.lock()
        self.camera.mouse_start = np.array([last_x, last_y])
        self.camera.mouse_curr = np.array([x, y])
        self.camera.rotate()
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_pan(self, last_x, last_y, x, y):
        render_mutex.lock()
        mouse_start = np.array([last_x, last_y])
        mouse_curr = np.array([x, y])
        mouse_delta = mouse_curr - mouse_start
        self.camera.pan(mouse_delta)
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_zoom(self, zoom):
        render_mutex.lock()
        self.camera.zoom(zoom)
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_view_xy(self):
        render_mutex.lock()
        self.camera.reset_view_xy(np.array([ 
                self.full_shape[2]/2,
                self.full_shape[1]/2,
                self.full_shape[0]/2
                ], dtype=np.float32),
                (self.full_shape[0]**2 + \
                    self.full_shape[1]**2 + \
                    self.full_shape[2]**2)**0.5   
            )
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_tf_rescale(self, dmin, dmax):
        render_mutex.lock()
        print(f"Setting tf minmax to {dmin} {dmax}")
        self.tf.set_mapping_minmax(dmin, dmax)
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_change_transfer_function(self, s):
        render_mutex.lock()
        self.tf.loadColormap(s)
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()
        data_for_tf_editor = np.stack(
            [self.scene.transfer_function.opacity_control_points.cpu(),
             self.scene.transfer_function.opacity_values.cpu()],
            axis=0
        ).transpose()
        self.parent.tf_editor.setData(pos=data_for_tf_editor)

    def do_change_opacities(self, control_points, values):  
        render_mutex.lock()
        self.scene.transfer_function.update_opacities(
            control_points, values
        )
        self.scene.on_tf_change()
        render_mutex.unlock()
    
    def do_toggle_density(self):
        render_mutex.lock()
        self.scene.toggle_density()
        self.scene.on_setting_change()
        render_mutex.unlock()

    def do_change_batch_size(self, b):
        render_mutex.lock()
        self.batch_size = 2**b
        self.scene.batch_size = 2**b
        self.scene.on_resize()
        render_mutex.unlock()

    def do_change_spp(self, b):
        render_mutex.lock()
        self.spp = b
        self.scene.spp = b
        self.scene.on_resize()
        render_mutex.unlock()

    def initialize_camera(self):
        aabb = np.array([0.0, 0.0, 0.0, 
                        self.full_shape[2]-1,
                        self.full_shape[1]-1,
                        self.full_shape[0]-1], 
                        dtype=np.float32)
        self.camera = Arcball(
            scene_aabb=aabb,
            coi=np.array(
                [aabb[5]/2, 
                aabb[4]/2, 
                aabb[3]/2], 
                dtype=np.float32), 
            dist=(aabb[3]**2 + \
                aabb[4]**2 + \
                aabb[5]**2)**0.5,
            fov=60.0
        )
    
    def initialize_model(self):
        first_model = os.listdir(savedmodels_folder)[0]
        print(f"Loading model {first_model}")
        self.parent.status_text_update.emit(f"Loading model {first_model}...")
        self.opt = load_options(os.path.abspath(os.path.join('SavedModels', first_model)))
        self.model = load_model(self.opt, self.device).to(self.device)
        self.model.eval()
        self.full_shape = self.model.get_volume_extents()
        print(f"Min/max: {self.model.min().item():0.02f}/{self.model.max().item():0.02f}")
        self.parent.set_timestep(0)
        self.parent.timestep_max.emit(self.opt['n_timesteps']-1)
        self.tf.set_minmax(self.model.min(), self.model.max())  
        self.parent.status_text_update.emit(f"")   
    
    def do_change_model(self, s):
        print("Resetting timestep to 0")
        self.parent.set_timestep(0)

        render_mutex.lock()
        p = os.path.abspath(s)
        print(f"Loading model from {p}")
        self.opt = load_options(p)
        self.model = load_model(self.opt, self.device, p).to(self.device)
        print(f"Model loaded")
        self.model.eval()
        print(f"Getting extents")
        self.full_shape = self.model.get_volume_extents()
        print(f"Setting minmax")
        self.tf.set_minmax(self.model.min(), self.model.max()) 
        print(f"Setting scene model")
        self.scene.model = self.model
        print(f"Setting aabb")
        self.scene.set_aabb([ 
                self.full_shape[0]-1,
                self.full_shape[1]-1,
                self.full_shape[2]-1
                ])
        self.camera.set_aabb(np.array([0,0,0,
                                       self.full_shape[0]-1,
                                       self.full_shape[1]-1,
                                       self.full_shape[2]-1]))
        #self.scene.precompute_occupancy_grid()
        print(f"Min/max: {self.model.min().item():0.02f}/{self.model.max().item():0.02f}")
        print(f"Setting timestep max")
        self.parent.timestep_max.emit(self.opt['n_timesteps']-1)
        print(f"Calling on_setting_change")
        self.scene.on_setting_change()
        render_mutex.unlock()
        self.do_view_xy()
    
    def do_change_timestep(self, t):
        if self.model is None:
            return
        print(f"Setting timestep to {t}")
        render_mutex.lock()
        self.model.set_default_timestep(t)
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()
    
    def do_change_data(self, s):
        render_mutex.lock()
        p = os.path.abspath(s)
        print(f"Loading data from {p}")
        self.model = RawData(p, self.device)
        different_size = (self.full_shape[0] != self.model.shape[0] or
                          self.full_shape[1] != self.model.shape[1] or
                          self.full_shape[2] != self.model.shape[2])
        self.full_shape = self.model.shape
        self.model.eval()
        self.tf.set_minmax(self.model.min(), self.model.max())        
        self.scene.model = self.model
        self.scene.set_aabb([ 
                self.full_shape[2]-1,
                self.full_shape[1]-1,
                self.full_shape[0]-1
                ])
        if(different_size and False):
            
            self.camera.update_coi(
                np.array([ 
                self.full_shape[2]/2,
                self.full_shape[1]/2,
                self.full_shape[0]/2
                ], dtype=np.float32)            
            )
            self.camera.update_dist((self.full_shape[0]**2 + \
                    self.full_shape[1]**2 + \
                    self.full_shape[2]**2)**0.5)
        #self.scene.precompute_occupancy_grid()
        print(f"Min/max: {self.model.min().item():0.02f}/{self.model.max().item():0.02f}")
        self.scene.on_setting_change()
        render_mutex.unlock()

    # === Camera state handler ===
    # Attributes that are transient or resolution-dependent and should
    # not be saved/loaded — they get regenerated by the scene or camera.
    _CAMERA_SKIP_KEYS = {
        'camera_dirs', 'mouse_start', 'mouse_curr',
        'width', 'height', 'resolution',
    }

    def do_load_camera_state(self, restored):
        """Apply a deserialized camera state dict to the current camera."""
        render_mutex.lock()
        for key, val in restored.items():
            if key in self._CAMERA_SKIP_KEYS:
                continue
            if hasattr(self.camera, key):
                setattr(self.camera, key, val)
            else:
                print(f"Warning: camera has no attribute '{key}', skipping")
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()
    # === End camera state handler ===

    # === Lighting handler methods ===
    def do_change_shading_enabled(self, enabled):
        render_mutex.lock()
        self.scene.use_shading = enabled
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_change_light_mode(self, mode):
        render_mutex.lock()
        self.scene.light_mode = mode
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_change_ambient(self, val):
        render_mutex.lock()
        self.scene.ambient = val
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_change_diffuse(self, val):
        render_mutex.lock()
        self.scene.diffuse_strength = val
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_change_specular(self, val):
        render_mutex.lock()
        self.scene.specular_strength = val
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_change_shininess(self, val):
        render_mutex.lock()
        self.scene.shininess = val
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()

    def do_change_light_position(self, x, y, z):
        render_mutex.lock()
        import torch
        # Scale normalized [0,1] slider values to AABB extent
        aabb_extent = self.scene.scene_aabb[3:]
        self.scene.light_position = torch.tensor(
            [x * aabb_extent[0].item(),
             y * aabb_extent[1].item(),
             z * aabb_extent[2].item()],
            device=self.scene.device
        )
        self.scene.on_rotate_zoom_pan()
        render_mutex.unlock()
    # === End lighting handler methods ===

def run_renderer():
    app = QApplication([])
    window = MainWindow()    
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    run_renderer()