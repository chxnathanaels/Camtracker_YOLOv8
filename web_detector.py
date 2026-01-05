from flask import Flask, Response, request, jsonify
import cv2
import os
import sys
from ultralytics import YOLOWorld

app = Flask(__name__)

# --- DIAGNOSTIC START ---
print("--- [1] Checking Environment ---")
base_path = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_path, "video.mp4")

if not os.path.exists(video_path):
    print(f"!!! CRITICAL ERROR: File 'video.mp4' not found in {base_path}")
    input("Press Enter to close...") # Keeps terminal open
    sys.exit()

print(f"--- [2] Loading AI Model (RTX 3500 Ada) ---")
try:
    model = YOLOWorld("yolov8s-worldv2.pt")
    current_targets = ["car", "truck"]
    model.set_classes(current_targets)
    print("--- [3] AI Ready! ---")
except Exception as e:
    print(f"!!! MODEL ERROR: {e}")
    input("Press Enter to close...")
    sys.exit()

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: OpenCV could not open the video file.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Predict
        results = model.predict(source=frame, device=0, verbose=False, conf=0.1)
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
        <body style="background:#0f172a; color:white; text-align:center; font-family:sans-serif; padding-top:30px;">
            <h1 style="color:#3b82f6;">AI CUSTOM DETECTOR</h1>
            <p id="status" style="color:#94a3b8;">Current Target: car, truck</p>
            <img src="/video_feed" style="width:800px; border:3px solid #3b82f6; border-radius:10px;">
            <div style="margin-top:20px;">
                <input type="text" id="q" placeholder="What should I find?">
                <button onclick="updateAI()">Analyze</button>
            </div>
            <script>
                function updateAI() {
                    const val = document.getElementById('q').value;
                    fetch('/set_query', {
                        method:'POST', 
                        headers:{'Content-Type':'application/json'}, 
                        body:JSON.stringify({query: val})
                    }).then(() => {
                        document.getElementById('status').innerText = "Searching for: " + val;
                    });
                }
            </script>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_query', methods=['POST'])
def set_query():
    global current_targets
    data = request.json
    new_query = data.get("query", "car")
    
    # Clean up the words you typed
    current_targets = [x.strip() for x in new_query.split(",")]
    
    # THE RE-PROGRAMMING LINE
    model.set_classes(current_targets) 
    
    print(f"\n🚀 AI RE-PROGRAMMED! Searching only for: {current_targets}")
    print(f"--- AI VULCABULARY UPDATED ---")
    print(f"The AI is now looking for: {model.names}")
    return jsonify({"status": "success", "active": current_targets})

if __name__ == "__main__":
    print("--- [4] Starting Flask Server on http://127.0.0.1:5000 ---")
    app.run(host='0.0.0.0', port=5000, debug=False)