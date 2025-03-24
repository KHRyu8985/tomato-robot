import pyzed.sl as sl
import src.zed_sdk.viewer as gl
import cv2 as cv
import numpy as np

class ZedTracker:
    def __init__(self):
        self.zed = None
        self.viewer = None
        self.zed_image = sl.Mat()
        self.objects = sl.Objects()
        self.runtime_parameters = sl.RuntimeParameters()
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    def initialize_zed(self, resolution=sl.RESOLUTION.HD720, display_viewer=True):
        # initialize zed camera
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.camera_resolution = resolution

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("[Error] failed to open zed camera:", err)
            return False

        # object detection settings
        obj_param = sl.ObjectDetectionParameters()
        obj_param.enable_tracking = True
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
        if obj_param.enable_tracking:
            self.zed.enable_positional_tracking()
        self.zed.enable_object_detection(obj_param)

        self.obj_runtime_param.detection_confidence_threshold = 10 # lower confidence threshold
        # filtering object class (fruits and vegetables)
        self.obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.FRUIT_VEGETABLE]

        # initialize viewer
        camera_info = self.zed.get_camera_information()
        self.viewer = gl.GLViewer()
        self.viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, obj_param.enable_tracking) # viewer 초기화
        return True

    def grab_frame_and_objects(self):
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.zed_image, sl.VIEW.LEFT)

            frame = self.zed_image.get_data()
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            # extract object detection results
            self.zed.retrieve_objects(self.objects, self.obj_runtime_param)
            return True, frame, self.objects
        else:
            return False, None, None

    def update_viewer(self):
        if self.viewer is not None and self.viewer.is_available(): # update viewer if viewer is not None and is available
            self.viewer.update_view(self.zed_image, self.objects)

    def get_viewer_frame(self):
        if self.viewer is not None and self.viewer.is_available():
            return self.viewer.get_current_frame()
        return None

    def close_zed(self):
        if self.zed is not None:
            self.zed.disable_object_detection()
            self.zed.disable_positional_tracking()
            self.zed.close()
        if self.viewer is not None:
            self.viewer.exit()
