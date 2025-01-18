import cv2
import mediapipe as mp
import numpy as np
from flask_socketio import SocketIO
import base64
import traceback
import os
import time
import threading
class ChangeFace:
    def __init__(self, socketio, animal_image_path='static/pic/image.png',expression_delay=0.1):
        self.socketio = socketio
        self.animal_image_path = animal_image_path
        self.last_expression_time = time.time()  # 记录上次表情检测的时间
        self.expression_delay = expression_delay  # 延时，单位为秒
        self.expression = 'neutral'
        self.lock = threading.Lock()
        
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,

        )
        
        # 初始化 MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1
        )
        
        # 检查并加载动物头像
        self.animal_expressions = {}
        base_path = 'static/pic'  # 确保这个路径正确
        image_paths = {
            'neutral': 'image.png',
            'sad': 'image_sad.png',
            'happy': 'image_happy.png'
        }
        
        for expression, filename in image_paths.items():
            full_path = os.path.join(base_path, filename)
            if not os.path.exists(full_path):
                print(f"警告: 找不到图片文件 {full_path}")
                continue
            
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"错误: 无法加载图片 {full_path}")
                continue
                
            # 如果图片是RGBA格式，转换为BGR
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            self.animal_expressions[expression] = img
        
        if not self.animal_expressions:
            raise RuntimeError("未能加载任何动物头像图片")
        
        # 表情关键点索引
        self.expression_landmarks = {
            'left_eye': [33, 133, 157, 158, 159, 160, 161, 173, 246],
            'right_eye': [362, 263, 386, 387, 388, 389, 390, 398, 466],
            'mouth': [0, 37, 39, 40,  185,61,146,91,181,84,17,314,405,321,375,291,409,270,269,267 ],
            'mouth_up': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
            'mouth_down': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
            'eyebrows': [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
        }
    def detect_expression_async(self, face_landmarks):
        """异步表情检测"""
        def detect():
            self.detect_expression(face_landmarks)
        threading.Thread(target=detect).start()
    def detect_expression(self, face_landmarks):
        """检测面部表情"""
        try:
            #current_time = time.time()
            #if current_time - self.last_expression_time < self.expression_delay:
            #    return self.expression  # 如果没到时间，直接返回默认表情

            #self.last_expression_time = current_time  # 更新上次检测时间
            # 获取嘴部关键点
            mouth_up_points = [face_landmarks.landmark[i] for i in self.expression_landmarks['mouth_up']]
            mouth_down_points = [face_landmarks.landmark[i] for i in self.expression_landmarks['mouth_down']]
            
            # 计算嘴部高度和宽度比例
            mouth_height = (mouth_up_points[5].y - mouth_up_points[0].y)
            mouth_height2 = (mouth_down_points[5].y - mouth_down_points[0].y)
            mouth_top = face_landmarks.landmark[13].y
            mouth_bottom = face_landmarks.landmark[14].y
            mouth_open=abs( mouth_top- mouth_bottom)
            mouth_width = abs(mouth_up_points[5].x - mouth_up_points[0].x)
            mouth_width2 = abs(mouth_up_points[5].x - mouth_up_points[0].x)
            mouth_ratio = mouth_height / mouth_width
            mouth_ratio2 = mouth_height2 / mouth_width2
            print('mouth_ratio',mouth_ratio)
            print('mouth_ratio2',mouth_ratio2)
            print('mouth_open', mouth_open)
            # 获取眉毛关键点
            left_eyebrow = [face_landmarks.landmark[i] for i in self.expression_landmarks['eyebrows'][:5]]
            right_eyebrow = [face_landmarks.landmark[i] for i in self.expression_landmarks['eyebrows'][5:]]
            
            # 计算眉毛高度
            eyebrow_height = (left_eyebrow[2].y + right_eyebrow[2].y) / 2
            print('eyebrow_height',eyebrow_height)
            # 根据特征判断表情
            if(mouth_open <0.01 ):
                if mouth_ratio > 0.09:  # 嘴巴张开
                    self.expression = 'happy'
                elif mouth_ratio < -0.3:  # 嘴巴张开
                    self.expression = 'sad'
                else:
                    self.expression = 'neutral'
            else:
                if mouth_ratio2 > 0.1:
                    if  mouth_ratio<-0.2:
                        self.expression = 'neutral'
                    else:
                        self.expression = 'happy'
                else:
                    self.expression = 'neutral'
            return self.expression
        except Exception as e:
            print(f"表情检测错误: {e}")
            return 'neutral'

    def swap_face_with_animal(self, frame):
        """基于表情的动态换脸"""
        try:
            if frame is None or frame.size == 0:
                return frame
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 获取面部关键点
                    h, w, _ = frame.shape
                    face_points = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
                    
                    # 检查面部点是否有效
                    if face_points.size == 0 or np.any(np.isnan(face_points)):
                        continue
                    # 绘制面部关键点
                    for i, landmark in enumerate(face_landmarks.landmark):
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        # 可以根据需要调整圆圈大小或颜色
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    # 绘制嘴巴和眉毛的关键点作为表情特征
                    mouth_points = [face_landmarks.landmark[i] for i in self.expression_landmarks['mouth']]
                    left_eyebrow = [face_landmarks.landmark[i] for i in self.expression_landmarks['eyebrows'][:5]]
                    right_eyebrow = [face_landmarks.landmark[i] for i in self.expression_landmarks['eyebrows'][5:]]

                    # 在嘴巴关键点和眉毛关键点上绘制圆圈
                    for point in mouth_points :
                        x, y = int(point.x * w), int(point.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # 红色圆圈表示表情特征关键点
                    for point in  left_eyebrow + right_eyebrow:
                        x, y = int(point.x * w), int(point.y * h)
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # 蓝色圆圈表示表情特征关键点

                    # 获取面部表情特征
                    expression_features = self.get_expression_features(face_landmarks)
                    # 检测表情
                    expression = self.detect_expression(face_landmarks)
                    print(f"识别到的表情: {expression}")

                    # 获取基础动物头像
                    base_avatar = self.animal_expressions.get(expression, self.animal_expressions['neutral'])
                    #base_avatar = self.animal_expressions['neutral']
                    if base_avatar is None or base_avatar.size == 0:
                        continue

                    # 计算面部区域和变换矩阵
                    face_bbox = self.get_face_bbox(face_points)
                    transform_matrix = self.get_transform_matrix(face_bbox, base_avatar.shape)

                    # 变形动物头像以匹配人脸表情
                    warped_avatar = self.warp_avatar_with_expression(
                        base_avatar,
                        expression_features,
                        transform_matrix
                    )

                    if warped_avatar is not None and warped_avatar.size > 0:
                        # 平滑混合
                        frame = self.blend_images(frame, warped_avatar, face_bbox)
                
            return frame
            
        except Exception as e:
            print(f"换脸处理错误: {str(e)}")
            traceback.print_exc()  # 打印详细错误信息
            return frame

    def get_expression_features(self, face_landmarks):
        """提取详细的表情特征"""
        features = {
            'mouth_open': 0,
            'smile_degree': 0,
            'eye_open': 0,
            'eyebrow_raise': 0
        }
        
        # 计算嘴部开合度
        mouth_top = face_landmarks.landmark[13].y
        mouth_bottom = face_landmarks.landmark[14].y
        features['mouth_open'] = mouth_bottom - mouth_top
        
        # 计算微笑程度
        mouth_left = face_landmarks.landmark[61]
        mouth_right = face_landmarks.landmark[291]
        mouth_center = face_landmarks.landmark[0]
        features['smile_degree'] = (mouth_left.y + mouth_right.y) / 2 - mouth_center.y
        
        # 计算眼睛开合度
        left_eye_top = face_landmarks.landmark[159].y
        left_eye_bottom = face_landmarks.landmark[145].y
        right_eye_top = face_landmarks.landmark[386].y
        right_eye_bottom = face_landmarks.landmark[374].y
        features['eye_open'] = ((left_eye_bottom - left_eye_top) + 
                               (right_eye_bottom - right_eye_top)) / 2
        
        # 计算眉毛上扬程度
        left_eyebrow = face_landmarks.landmark[107].y
        right_eyebrow = face_landmarks.landmark[336].y
        features['eyebrow_raise'] = -(left_eyebrow + right_eyebrow) / 2
        
        return features

    def get_face_bbox(self, face_points):
        """获取面部边界框"""
        x_min = int(np.min(face_points[:, 0]))
        y_min = int(np.min(face_points[:, 1]))
        x_max = int(np.max(face_points[:, 0]))
        y_max = int(np.max(face_points[:, 1]))
        
        # 扩大边界框以包含整个头部
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, x_min - int(width * 0.2))
        y_min = max(0, y_min - int(height * 0.2))
        x_max = x_max + int(width * 0.2)
        y_max = y_max + int(height * 0.2)
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def get_transform_matrix(self, face_bbox, avatar_shape):
        """计算变换矩阵"""
        src_h, src_w = avatar_shape[:2]
        dst_w = face_bbox[2]
        dst_h = face_bbox[3]
        
        # 计算缩放比例
        scale_x = dst_w / src_w
        scale_y = dst_h / src_h
        scale = min(scale_x, scale_y)
        print(scale_x,scale_y)
        # 计算变换矩阵
        matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0]
        ], dtype=np.float32)
        
        return matrix

    def warp_avatar_with_expression2(self, avatar, expression_features, transform_matrix):
        """根据表情特征变形动物头像"""
        try:
            h, w = avatar.shape[:2]
            print(f"原始头像尺寸: 高度 = {h}, 宽度 = {w}")

            # 创建变形网格（提前定义float32类型）
            grid_x = np.arange(w, dtype=np.float32)
            grid_y = np.arange(h, dtype=np.float32)

            # 计算变形参数
            mouth_center_y = h * 0.7
            eye_y = h * 0.45
            left_eye_x = w * 0.35
            right_eye_x = w * 0.65
            eyebrow_y = h * 0.35

            print(f"嘴巴中心位置 (y): {mouth_center_y}")
            print(f"眼睛中心位置 (y): {eye_y}")
            print(f"左眼位置 (x): {left_eye_x}, 右眼位置 (x): {right_eye_x}")
            print(f"眉毛中心位置 (y): {eyebrow_y}")

            # 预计算平方部分，减少重复计算
            dx_eye_left = grid_x - left_eye_x
            dx_eye_right = grid_x - right_eye_x
            dx_eyebrow_left = grid_x - left_eye_x
            dx_eyebrow_right = grid_x - right_eye_x

            dy_eye = grid_y - eye_y
            dy_eyebrow = grid_y - eyebrow_y
            dy_mouth = grid_y - mouth_center_y

            print(f"dx_eye_left 示例值: {dx_eye_left[:5]}")  # 打印前5个值
            print(f"dy_eye 示例值: {dy_eye[:5]}")  # 打印前5个值

            # 创建变形区域蒙版（向量化操作）
            mouth_region = np.exp(-0.5 * (dy_mouth / (h * 0.1)) ** 2)
            eye_region_left = np.exp(-0.5 * ((dx_eye_left / (w * 0.1)) ** 2 + (dy_eye / (h * 0.1)) ** 2))
            eye_region_right = np.exp(-0.5 * ((dx_eye_right / (w * 0.1)) ** 2 + (dy_eye / (h * 0.1)) ** 2))
            eyebrow_region_left = np.exp(-0.5 * ((dx_eyebrow_left / (w * 0.15)) ** 2 + (dy_eyebrow / (h * 0.1)) ** 2))
            eyebrow_region_right = np.exp(-0.5 * ((dx_eyebrow_right / (w * 0.15)) ** 2 + (dy_eyebrow / (h * 0.1)) ** 2))

            print(f"mouth_region 示例值: {mouth_region[:5]}")
            print(f"eye_region_left 示例值: {eye_region_left[:5]}")
            print(f"eyebrow_region_left 示例值: {eyebrow_region_left[:5]}")

            # 应用变形
            # 嘴部变形
            grid_y += expression_features['mouth_open'] * 30 * mouth_region
            print(f"嘴部变形后的 grid_y（嘴巴开合）: {grid_y[:5]}")

            # 眼睛变形
            eye_region = eye_region_left + eye_region_right
            grid_y += expression_features['eye_open'] * 15 * eye_region
            print(f"眼睛变形后的 grid_y（眼睛开合）: {grid_y[:5]}")

            # 眉毛变形
            eyebrow_region = eyebrow_region_left + eyebrow_region_right
            grid_y += expression_features['eyebrow_raise'] * 10 * eyebrow_region
            print(f"眉毛变形后的 grid_y（眉毛抬高）: {grid_y[:5]}")

            # 变形网格重新组合
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)  # 重建网格
            print(f"重建后的 grid_x 示例值: {grid_x[:5, :5]}")
            print(f"重建后的 grid_y 示例值: {grid_y[:5, :5]}")

            # 应用变换矩阵
            grid = cv2.remap(avatar, grid_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            print(f"变形后的图像尺寸: {grid.shape}")

            # 调整大小以匹配变换矩阵
            output_size = (int(transform_matrix[0, 2] + w * transform_matrix[0, 0]),
                           int(transform_matrix[1, 2] + h * transform_matrix[1, 1]))
            print(f"调整后的输出尺寸: {output_size}")

            warped = cv2.warpAffine(grid, transform_matrix, output_size,
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            print(f"最终变形后的图像尺寸: {warped.shape}")

            return warped

        except Exception as e:
            print(f"变形处理错误: {e}")
            return avatar

    def warp_avatar_with_expression(self, avatar, expression_features, transform_matrix):
        """根据表情特征变形动物头像"""
        try:
            h, w = avatar.shape[:2]

            # 创建变形网格（提前定义float32类型）
            grid_x = np.arange(w, dtype=np.float32)
            grid_y = np.arange(h, dtype=np.float32)

            # 计算变形参数
            mouth_center_y = h * 0.7
            eye_y = h * 0.45
            left_eye_x = w * 0.35
            right_eye_x = w * 0.65
            eyebrow_y = h * 0.35

            # 预计算平方部分，减少重复计算
            dx_eye_left = grid_x - left_eye_x
            dx_eye_right = grid_x - right_eye_x
            dx_eyebrow_left = grid_x - left_eye_x
            dx_eyebrow_right = grid_x - right_eye_x

            dy_eye = grid_y - eye_y
            dy_eyebrow = grid_y - eyebrow_y
            dy_mouth = grid_y - mouth_center_y

            # 创建变形区域蒙版（向量化操作）
            mouth_region = np.exp(-0.5 * (dy_mouth / (h * 0.1)) ** 2)
            eye_region_left = np.exp(-0.5 * ((dx_eye_left / (w * 0.1)) ** 2 + (dy_eye / (h * 0.1)) ** 2))
            eye_region_right = np.exp(-0.5 * ((dx_eye_right / (w * 0.1)) ** 2 + (dy_eye / (h * 0.1)) ** 2))
            eyebrow_region_left = np.exp(-0.5 * ((dx_eyebrow_left / (w * 0.15)) ** 2 + (dy_eyebrow / (h * 0.1)) ** 2))
            eyebrow_region_right = np.exp(-0.5 * ((dx_eyebrow_right / (w * 0.15)) ** 2 + (dy_eyebrow / (h * 0.1)) ** 2))

            # 应用变形
            # 嘴部变形
            grid_y += expression_features['mouth_open'] * 30 * mouth_region

            # 眼睛变形
            eye_region = eye_region_left + eye_region_right
            grid_y += expression_features['eye_open'] * 15 * eye_region

            # 眉毛变形
            eyebrow_region = eyebrow_region_left + eyebrow_region_right
            grid_y += expression_features['eyebrow_raise'] * 10 * eyebrow_region

            # 变形网格重新组合
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)  # 重建网格

            # 应用变换矩阵
            grid = cv2.remap(avatar, grid_x, grid_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

            # 调整大小以匹配变换矩阵
            output_size = (int(transform_matrix[0, 2] + w * transform_matrix[0, 0]),
                           int(transform_matrix[1, 2] + h * transform_matrix[1, 1]))
            print(f"调整后的输出尺寸: {output_size}")
            warped = cv2.warpAffine(grid, transform_matrix, output_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            print(f"最终的输出尺寸: {warped.shape}")
            return warped

        except Exception as e:
            print(f"变形处理错误: {e}")
            return avatar

    def blend_images(self, frame, warped_avatar, face_bbox):
        """平滑混合图像"""
        try:
            x, y, w, h = face_bbox

            # 确保坐标在有效范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)


            if w <= 0 or h <= 0:
                return frame

            # 提取目标区域
            roi = frame[y:y+h, x:x+w]


            # 调整warped_avatar大小以精确匹配ROI
            warped_avatar = cv2.resize(warped_avatar, (roi.shape[1], roi.shape[0]))

            # 确保warped_avatar是3通道
            if len(warped_avatar.shape) == 2:
                warped_avatar = cv2.cvtColor(warped_avatar, cv2.COLOR_GRAY2BGR)
            elif warped_avatar.shape[-1] == 4:
                warped_avatar = cv2.cvtColor(warped_avatar, cv2.COLOR_BGRA2BGR)

            # 创建精确匹配ROI大小的蒙版
            mask = np.zeros(roi.shape, dtype=np.float32)
            center = (roi.shape[1]//2, roi.shape[0]//2)
            axes = (int(roi.shape[1]*0.4), int(roi.shape[0]*0.5))
            cv2.ellipse(mask, center, axes, 0, 0, 360, (1,1,1), -1)

            # 羽化边缘
            kernel_size = min(31, roi.shape[0]-1 if roi.shape[0] % 2 == 0 else roi.shape[0],
                                roi.shape[1]-1 if roi.shape[1] % 2 == 0 else roi.shape[1])
            if kernel_size > 1:
                kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size - 1
                mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 11)

            # 确保所有数组都是float32类型进行混合
            roi = roi.astype(np.float32)
            warped_avatar = warped_avatar.astype(np.float32)

            # 混合图像
            blended = cv2.multiply(1.0 - mask, roi) + cv2.multiply(mask, warped_avatar)

            # 更新原始图像
            frame[y:y+h, x:x+w] = blended.astype(np.uint8)

            return frame

        except Exception as e:
            print(f"混合图像错误: {str(e)}")
            traceback.print_exc()  # 打印详细错误信息
            return frame
