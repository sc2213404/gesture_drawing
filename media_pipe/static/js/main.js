const socket = io();

const videoCapture = document.getElementById('video-capture');
const videoFloating = document.getElementById('video-floating');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const cursor = document.getElementById('cursor');

// Canvas transformations
let scale = 1.0;
let offsetX = 0;
let offsetY = 0;

// Maintain paths for redrawing
let paths = [];
let isEraserMode = false;
let brushWidth = 2;
let brushColor = 'red';

// 获取摄像头视频流
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        videoCapture.srcObject = stream;
        // 克隆流给悬浮窗口
        const clonedStream = stream.clone();
        videoFloating.srcObject = clonedStream;
        videoFloating.play();
        // 定时发送视频帧
        setInterval(() => {
            sendFrame();
        }, 100); // 每100ms发送一次
    })
    .catch(err => {
        console.error('Error accessing camera:', err);
    });

function sendFrame() {
    const canvasTemp = document.createElement('canvas');
    canvasTemp.width = videoCapture.videoWidth;
    canvasTemp.height = videoCapture.videoHeight;
    const ctxTemp = canvasTemp.getContext('2d');
    // 由于视频已通过 CSS 镜像显示，这里不需再次翻转
    ctxTemp.drawImage(videoCapture, 0, 0, canvasTemp.width, canvasTemp.height);
    const dataURL = canvasTemp.toDataURL('image/jpeg');
    socket.emit('video_frame', dataURL);
}

// 处理初始化画布
socket.on('init', (state) => {
    console.log('Received initial state:', state);
    // Initialize paths
    paths = state.paths.map(path => {
        return {
            color: `rgb(${path.color[2]}, ${path.color[1]}, ${path.color[0]})`, // Convert BGR to RGB
            points: path.points.map(p => transformPosition(p.x, p.y))
        };
    });
    // Draw paths
    redrawCanvas();
});

// 处理绘画指令
socket.on('draw', (pos) => {
    console.log('Received draw event:', pos);
    const transformedPos = transformPosition(pos.x, pos.y);
    
    if (!isEraserMode) {
        // 确保有一个当前路径
        if (paths.length === 0 || !paths[paths.length - 1].inProgress) {
            paths.push({
                color: brushColor,
                points: [transformedPos],
                inProgress: true
            });
        } else {
            // 添加点到当前路径
            const currentPath = paths[paths.length - 1];
            currentPath.points.push(transformedPos);
            
            // 绘制当前路径
            ctx.beginPath();
            ctx.strokeStyle = currentPath.color;
            ctx.lineWidth = brushWidth;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            
            drawLine(currentPath.points);
        }
        
        // 更新光标位置
        updateCursor(transformedPos.x, transformedPos.y, brushWidth);
    }
});

// 处理擦除指令
socket.on('erase', (data) => {
    console.log('Received erase event:', data);
    const { x, y, size } = data;
    // Transform position based on current zoom and pan
    const transformedX = x * scale + offsetX;
    const transformedY = y * scale + offsetY;
    const transformedSize = size * scale;

    // 擦除指定区域（圆形擦除）
    ctx.save();
    ctx.beginPath();
    ctx.arc(transformedX, transformedY, transformedSize / 2, 0, 2 * Math.PI);
    ctx.clip();
    ctx.clearRect(transformedX - transformedSize, transformedY - transformedSize, transformedSize * 2, transformedSize * 2);
    ctx.restore();

    // Optionally, draw an eraser cursor
    updateCursor(transformedX, transformedY, transformedSize);

    console.log(`Erased at (${transformedX}, ${transformedY}) with size ${transformedSize}`);
});

// 处理平移指令
socket.on('pan', (pan) => {
    console.log('Received pan event:', pan);
    // pan contains 'x' and 'y' representing the new pan position
    offsetX = pan.x * scale;
    offsetY = pan.y * scale;
    // Redraw canvas with new pan
    redrawCanvas();
});

// 处理缩放指令
socket.on('zoom', (zoomLevel) => {
    console.log('Received zoom event:', zoomLevel);
    // zoomLevel is a scalar value
    scale = zoomLevel;
    // Redraw all paths with new scale and offset
    redrawCanvas();
});

// 处理更新事件
socket.on('update', (state) => {
    console.log('Received update state:', state);
    // Initialize paths
    paths = state.paths.map(path => {
        return {
            color: `rgb(${path.color[2]}, ${path.color[1]}, ${path.color[0]})`, // Convert BGR to RGB
            points: path.points.map(p => transformPosition(p.x, p.y))
        };
    });
    // Draw paths
    redrawCanvas();
});

// 更新事件处理器
socket.on('update_size', function(data) {
    const size = data.size;
    if (data.type === 'both') {
        // 更新画笔和橡皮擦大小
        brushWidth = size;
        eraserSize = size;
        // 更新UI显示
        document.getElementById('brush-width').value = size;
        // 更新光标大小
        updateCursor(lastX, lastY, size);
    }
});

socket.on('update_color', function(color) {
    // 更新当前颜色
    const newColor = `rgb(${color.r}, ${color.g}, ${color.b})`;
    brushColor = newColor;
    
    // 高亮显示当前选中的颜色
    document.querySelectorAll('.color-option').forEach(option => {
        option.classList.remove('selected');
        if (getRGBColor(option.style.backgroundColor) === newColor) {
            option.classList.add('selected');
        }
    });
});

// 辅助函数：将颜色字符串标准化为RGB格式
function getRGBColor(color) {
    const div = document.createElement('div');
    div.style.backgroundColor = color;
    document.body.appendChild(div);
    const rgbColor = window.getComputedStyle(div).backgroundColor;
    document.body.removeChild(div);
    return rgbColor;
}

// 添加变量跟踪最后的光标位置
let lastX = 0;
let lastY = 0;

function updateCursor(x, y, size=20) {
    lastX = x;
    lastY = y;
    cursor.style.left = (x - size / 2) + 'px';
    cursor.style.top = (y - size / 2) + 'px';
    cursor.style.width = size + 'px';
    cursor.style.height = size + 'px';
    cursor.style.display = 'block';
}

function hideCursor() {
    cursor.style.display = 'none';
}

function transformPosition(x, y) {
    return {
        x: x * scale + offsetX,
        y: y * scale + offsetY
    };
}

function redrawCanvas() {
    // 清除画布
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 应用变换
    ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);
    
    // 重绘所有路径
    paths.forEach(path => {
        ctx.beginPath();
        ctx.strokeStyle = path.color;
        ctx.lineWidth = brushWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        drawLine(path.points);
    });
}

const eraserIcon = document.getElementById('eraser-icon');
const penIcon = document.getElementById('pen-icon');

eraserIcon.addEventListener('click', function() {
    if (isEraserMode) {
        isEraserMode = false;
        canvas.style.cursor = 'url("../static/img/pen-cursor.png"), auto';
    } else {
        isEraserMode = true;
        canvas.style.cursor = 'url("../static/img/eraser-cursor.png"), auto';
    }
    this.classList.toggle('active', isEraserMode);
    penIcon.classList.remove('active');
    document.getElementById('brush-settings').style.display = 'none';
});

penIcon.addEventListener('click', function() {
    if (!isEraserMode) {
        document.getElementById('brush-settings').style.display = 'block';
        this.classList.add('active');
        eraserIcon.classList.remove('active');
    } else {
        isEraserMode = false;
        canvas.style.cursor = 'url("../static/img/pen-cursor.png"), auto';
        this.classList.add('active');
        eraserIcon.classList.remove('active');
    }
});

document.getElementById('brush-width').addEventListener('change', function() {
    brushWidth = this.value;
    document.getElementById('brush-settings').style.display = 'none';
});

const colorOptions = document.querySelectorAll('#color选择 .color-option');
colorOptions.forEach(function(option) {
    option.addEventListener('click', function() {
        colorOptions.forEach(function(opt) {
            opt.classList.remove('selected');
        });
        this.classList.add('selected');
        brushColor = this.getAttribute('data-color');
        document.getElementById('brush-settings').style.display = 'none';
    });
});

// 添加绘画相关函数
function drawLine(points) {
    if (points.length < 2) return;
    
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    
    // 设置绘画样式
    ctx.strokeStyle = brushColor;
    ctx.lineWidth = brushWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // 使用贝塞尔曲线使线条更平滑
    for (let i = 1; i < points.length - 2; i++) {
        const xc = (points[i].x + points[i + 1].x) / 2;
        const yc = (points[i].y + points[i + 1].y) / 2;
        ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
    }
    
    // 处理最后两个点
    if (points.length > 2) {
        const lastPoint = points[points.length - 1];
        const secondLastPoint = points[points.length - 2];
        ctx.quadraticCurveTo(
            secondLastPoint.x,
            secondLastPoint.y,
            lastPoint.x,
            lastPoint.y
        );
    } else {
        const lastPoint = points[points.length - 1];
        ctx.lineTo(lastPoint.x, lastPoint.y);
    }
    
    ctx.stroke();
}

// 添加鼠标事件处理（用于测试）
let isDrawing = false;

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale - offsetX;
    const y = (e.clientY - rect.top) / scale - offsetY;
    
    if (!isEraserMode) {
        paths.push({
            color: brushColor,
            points: [{x, y}],
            inProgress: true
        });
    }
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / scale - offsetX;
    const y = (e.clientY - rect.top) / scale - offsetY;
    
    if (!isEraserMode) {
        const currentPath = paths[paths.length - 1];
        currentPath.points.push({x, y});
        drawLine(currentPath.points);
    } else {
        // 擦除功能
        const eraseData = {
            x: x * scale + offsetX,
            y: y * scale + offsetY,
            size: brushWidth
        };
        socket.emit('erase', eraseData);
    }
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    if (paths.length > 0) {
        paths[paths.length - 1].inProgress = false;
    }
});

canvas.addEventListener('mouseleave', () => {
    isDrawing = false;
    if (paths.length > 0) {
        paths[paths.length - 1].inProgress = false;
    }
});