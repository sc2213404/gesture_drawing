# gesture_backend.py

import base64
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import gevent
import time

class GestureBackend:
    def __init__(self, socketio_instance):
        self.socketio = socketio_instance

        # 初始化 Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 手指名称对应的关键点
        self.FINGER_TIPS = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }

        # 绘画状态
        self.drawing_state = {
            'drawing': False,
            'current_path': [],
            'paths': [],
            'color': (0, 0, 255),  # 默认红色 (B, G, R)
            'eraser_size': 30,      # 调整橡皮擦大小
            'zoom_level': 1.0,
            'pan': {'x': 0, 'y': 0},
            'previous_pan_pos': None  # 用于跟踪平移移动
        }

        # 线程安全的绘画命令队列
        self.command_queue = deque()

        # 启动后台线程
        self.socketio.start_background_task(target=self.process_commands)

    def count_fingers(self, hand_landmarks, hand_label):
        """
        计算伸展的手指数量
        """
        count = 0
        # 大拇指的特殊处理
        if hand_label == 'Right':
            if hand_landmarks.landmark[self.FINGER_TIPS['thumb']].x < hand_landmarks.landmark[self.FINGER_TIPS['thumb'] - 1].x:
                count += 1
        else:
            if hand_landmarks.landmark[self.FINGER_TIPS['thumb']].x > hand_landmarks.landmark[self.FINGER_TIPS['thumb'] - 1].x:
                count += 1

        # 其他手指
        for finger in ['index', 'middle', 'ring', 'pinky']:
            tip = self.FINGER_TIPS[finger]
            pip = tip - 2
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                count += 1
        return count

    def determine_gesture(self, hand_landmarks, hand_label):
        """
        根据手势确定操作
        """
        gesture = None
        # 右手
        if hand_label == 'Right':
            # 检查每个手指是否伸出
            index_up = hand_landmarks.landmark[self.FINGER_TIPS['index']].y < hand_landmarks.landmark[self.FINGER_TIPS['index'] - 2].y
            middle_up = hand_landmarks.landmark[self.FINGER_TIPS['middle']].y < hand_landmarks.landmark[self.FINGER_TIPS['middle'] - 2].y
            ring_up = hand_landmarks.landmark[self.FINGER_TIPS['ring']].y < hand_landmarks.landmark[self.FINGER_TIPS['ring'] - 2].y
            pinky_up = hand_landmarks.landmark[self.FINGER_TIPS['pinky']].y < hand_landmarks.landmark[self.FINGER_TIPS['pinky'] - 2].y
            thumb_up = hand_landmarks.landmark[self.FINGER_TIPS['thumb']].y < hand_landmarks.landmark[self.FINGER_TIPS['thumb'] - 2].y

            # 仅食指和中指伸出时的手势判断
            if index_up and middle_up and not ring_up and not pinky_up:
                # 获取食指和中指的位置
                index_tip = hand_landmarks.landmark[self.FINGER_TIPS['index']]
                middle_tip = hand_landmarks.landmark[self.FINGER_TIPS['middle']]
                
                # 计算手指的垂直和水平差异
                vertical_diff = abs(index_tip.y - middle_tip.y)
                horizontal_diff = abs(index_tip.x - middle_tip.x)
                
                # 根据手指移动方向判断手势类型
                if vertical_diff > horizontal_diff:
                    gesture = 'adjust_size'  # 上下移动调整大小
                else:
                    gesture = 'change_color'  # 左右移动切换颜色
            
            # 单指绘画
            elif index_up and not middle_up and not ring_up and not pinky_up:
                gesture = 'draw'
            # 擦除手势（四指伸出）
            elif index_up and middle_up and ring_up and pinky_up:
                gesture = 'erase'

        elif hand_label == 'Left':
            # 检查每个手指是否伸出
            index_up = hand_landmarks.landmark[self.FINGER_TIPS['index']].y < hand_landmarks.landmark[self.FINGER_TIPS['index'] - 2].y
            middle_up = hand_landmarks.landmark[self.FINGER_TIPS['middle']].y < hand_landmarks.landmark[self.FINGER_TIPS['middle'] - 2].y
            ring_up = hand_landmarks.landmark[self.FINGER_TIPS['ring']].y < hand_landmarks.landmark[self.FINGER_TIPS['ring'] - 2].y
            pinky_up = hand_landmarks.landmark[self.FINGER_TIPS['pinky']].y < hand_landmarks.landmark[self.FINGER_TIPS['pinky'] - 2].y
            thumb_up = hand_landmarks.landmark[self.FINGER_TIPS['thumb']].y < hand_landmarks.landmark[self.FINGER_TIPS['thumb'] - 2].y

            # 移动手势（左手四指伸出）
            if index_up and middle_up and ring_up and pinky_up:
                gesture = 'grab_move'
            # 缩放手势（左手中指和食指收缩）
            elif index_up and middle_up and not ring_up and not pinky_up:
                gesture = 'zoom_drag'

        return gesture

    def handle_video_frame(self, data, sid):
        """
        处理来自前端的视频帧
        """
        try:
            # 数据格式：data:image/jpeg;base64,/9j/...
            header, encoded = data.split(',', 1)
            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 翻转帧以矫正前置摄像头的镜像效应
            frame = cv2.flip(frame, 1)

            # 处理帧
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            gesture_commands = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label  # 'Left' 或 'Right'
                    gesture = self.determine_gesture(hand_landmarks, hand_label)

                    # 获取食指指尖位置
                    index_tip = hand_landmarks.landmark[self.FINGER_TIPS['index']]
                    x = int(index_tip.x * frame.shape[1])
                    y = int(index_tip.y * frame.shape[0])

                    if gesture:
                        gesture_commands.append({
                            'gesture': gesture,
                            'position': {'x': x, 'y': y},
                            'hand': hand_label
                        })

                    # 绘制手部标注
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

            # 将处理后的帧编码为 Base64
            _, buffer = cv2.imencode('.jpg', frame)
            processed_image = base64.b64encode(buffer).decode('utf-8')
            processed_data_url = f'data:image/jpeg;base64,{processed_image}'

            # 发送处理后的图像回前端，指定房间
            self.socketio.emit('processed_frame', processed_data_url, room=sid)

            # 发送手势指令回前端，指定房间
            if gesture_commands:
                self.socketio.emit('gesture', gesture_commands, room=sid)

                # 将手势命令加入队列以处理绘画状态
                for cmd in gesture_commands:
                    self.command_queue.append((cmd, sid))

        except Exception as e:
            print(f"Error processing frame: {e}")

    def process_commands(self):
        """
        后台线程处理手势命令并更新绘画状态
        """
        # 添加可用颜色列表
        available_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 0, 0), (255, 255, 0)]  # BGR格式
        current_color_index = 0
        last_color_change_time = time.time()
        last_size_change_time = time.time()
        
        while True:
            gevent.sleep(0.01)  # 10ms 轮询
            if self.command_queue:
                cmd, sid = self.command_queue.popleft()
                gesture = cmd['gesture']
                pos = cmd['position']
                current_time = time.time()
                
                if gesture == 'adjust_size':
                    if self.drawing_state['previous_pan_pos'] is not None:
                        prev_y = self.drawing_state['previous_pan_pos']['y']
                        delta_y = pos['y'] - prev_y
                        
                        # 添加时间间隔检查，避免过快变化
                        if current_time - last_size_change_time > 0.1:  # 100ms间隔
                            if delta_y < -5:  # 向上移动，增加大小
                                new_size = min(50, self.drawing_state['eraser_size'] + 2)
                                self.drawing_state['eraser_size'] = new_size
                                last_size_change_time = current_time
                                
                                # 发送更新大小事件
                                self.socketio.emit('update_size', {
                                    'size': new_size,
                                    'type': 'both'
                                }, room=sid)
                                print(f"Size increased to: {new_size}")
                                
                            elif delta_y > 5:  # 向下移动，减小大小
                                new_size = max(2, self.drawing_state['eraser_size'] - 2)
                                self.drawing_state['eraser_size'] = new_size
                                last_size_change_time = current_time
                                
                                # 发送更新大小事件
                                self.socketio.emit('update_size', {
                                    'size': new_size,
                                    'type': 'both'
                                }, room=sid)
                                print(f"Size decreased to: {new_size}")
                    
                    self.drawing_state['previous_pan_pos'] = pos
                
                elif gesture == 'change_color':
                    if self.drawing_state['previous_pan_pos'] is not None:
                        prev_x = self.drawing_state['previous_pan_pos']['x']
                        delta_x = pos['x'] - prev_x
                        
                        # 添加时间间隔检查，避免过快切换颜色
                        if current_time - last_color_change_time > 0.3:  # 300ms间隔
                            if abs(delta_x) > 30:  # 需要足够的移动距离才切换颜色
                                if delta_x > 0:  # 向右移动
                                    current_color_index = (current_color_index + 1) % len(available_colors)
                                else:  # 向左移动
                                    current_color_index = (current_color_index - 1) % len(available_colors)
                                
                                new_color = available_colors[current_color_index]
                                self.drawing_state['color'] = new_color
                                last_color_change_time = current_time
                                
                                # 发送更新颜色事件
                                self.socketio.emit('update_color', {
                                    'r': new_color[2],
                                    'g': new_color[1],
                                    'b': new_color[0]
                                }, room=sid)
                                print(f"Color changed to: RGB({new_color[2]}, {new_color[1]}, {new_color[0]})")
                    
                    self.drawing_state['previous_pan_pos'] = pos
                
                # 原有的手势处理代码继续保持不变
                elif gesture == 'draw':
                    # 开始绘画
                    if not self.drawing_state['drawing']:
                        self.drawing_state['drawing'] = True
                        self.drawing_state['current_path'] = [pos]
                    else:
                        self.drawing_state['current_path'].append(pos)
                    # 广播绘画点到指定房间
                    self.socketio.emit('draw', pos, room=sid)
                    print(f"Emitted draw at {pos} to SID: {sid}")

                elif gesture == 'erase':
                    erase_size = self.drawing_state['eraser_size']
                    erase_pos = pos
                    
                    # 更新当前路径
                    if self.drawing_state['current_path']:
                        updated_points = []
                        for point in self.drawing_state['current_path']:
                            # 计算点到擦除中心的距离
                            distance = ((point['x'] - erase_pos['x']) ** 2 + 
                                       (point['y'] - erase_pos['y']) ** 2) ** 0.5
                            # 如果点不在擦除范围内，保留该点
                            if distance > erase_size:
                                updated_points.append(point)
                        
                        # 更新当前路径
                        self.drawing_state['current_path'] = updated_points
                    
                    # 发送擦除事件和更新后的当前路径到前端
                    self.socketio.emit('erase', {
                        'x': erase_pos['x'], 
                        'y': erase_pos['y'], 
                        'size': erase_size,
                        'current_path': self.drawing_state['current_path']  # 发送更新后的当前路径
                    }, room=sid)
                    
                    print(f"Emitted erase at {erase_pos} with size {erase_size} to SID: {sid}")

                elif gesture == 'zoom_drag':
                    # 实现缩放
                    if self.drawing_state['previous_pan_pos'] is not None:
                        prev_x = self.drawing_state['previous_pan_pos']['x']
                        prev_y = self.drawing_state['previous_pan_pos']['y']
                        delta_x = pos['x'] - prev_x
                        delta_y = pos['y'] - prev_y

                        # 根据移动方向调整缩放和平移
                        self.drawing_state['pan']['x'] += delta_x * 0.5  # 提高平移因子
                        self.drawing_state['pan']['y'] += delta_y * 0.5

                        # 根据垂直移动调整缩放级别
                        if delta_y < -10:
                            self.drawing_state['zoom_level'] *= 1.2
                        elif delta_y > 10:
                            self.drawing_state['zoom_level'] /= 1.2

                        # 更新 previous_pan_pos
                        self.drawing_state['previous_pan_pos'] = pos

                        # 广播缩放和拖动事件
                        self.socketio.emit('zoom', self.drawing_state['zoom_level'], room=sid)
                        #self.socketio.emit('pan', self.drawing_state['pan'], room=sid)
                        print(f"Emitted zoom level {self.drawing_state['zoom_level']} and pan {self.drawing_state['pan']} to SID: {sid}")
                    else:
                        # 设置初始位置
                        self.drawing_state['previous_pan_pos'] = pos

                elif gesture == 'grab_move':
                    # 实现抓取并移动画布
                    if self.drawing_state['previous_pan_pos'] is not None:
                        prev_x = self.drawing_state['previous_pan_pos']['x']
                        prev_y = self.drawing_state['previous_pan_pos']['y']
                        delta_x = pos['x'] - prev_x
                        delta_y = pos['y'] - prev_y

                        # 调整平移因子
                        self.drawing_state['pan']['x'] += delta_x * 0.5  # 提高平移因子使移动更灵敏
                        self.drawing_state['pan']['y'] += delta_y * 0.5

                        # 更新 previous_pan_pos
                        self.drawing_state['previous_pan_pos'] = pos

                        # 广播平移事件
                        self.socketio.emit('pan', self.drawing_state['pan'], room=sid)
                        print(f"Emitted pan {self.drawing_state['pan']} to SID: {sid}")
                    else:
                        # 设置初始位置
                        self.drawing_state['previous_pan_pos'] = pos

                # 更新绘画状态
                if gesture == 'draw' and self.drawing_state['drawing']:
                    self.drawing_state['paths'].append({
                        'color': self.drawing_state['color'],
                        'points': list(self.drawing_state['current_path'])
                    })
                    self.drawing_state['current_path'] = [pos]

    def get_initial_state(self):
        """
        获取当前的绘画状态
        """
        return self.drawing_state
