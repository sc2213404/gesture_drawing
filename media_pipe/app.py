
from flask import Flask, render_template
from flask_socketio import SocketIO, join_room, leave_room
from gesture_backend import GestureBackend
from flask import request
import gevent
from gevent import monkey

# 使用 gevent 作为异步模式
monkey.patch_all()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# 初始化 SocketIO，确保使用最新版本
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# 初始化后端模块
gesture_backend = GestureBackend(socketio)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    room = request.sid  # 使用会话 ID 作为房间名
    join_room(room)
    initial_state = gesture_backend.get_initial_state()
    print(f"Initial state for SID {room}: {initial_state}")
    socketio.emit('init', initial_state, room=room)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    room = request.sid
    leave_room(room)

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    处理来自前端的视频帧
    """
    gesture_backend.handle_video_frame(data, request.sid)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
