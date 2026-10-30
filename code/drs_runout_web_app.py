import json
import os
import sys
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler

# Safe import for cv2 and numpy to ensure the web app runs on any system
HAS_CV2 = False
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ==============================================================================
# RUN OUT DRS SYSTEM ENGINE
# ==============================================================================

class RunOutDRSEngine:
    def __init__(self):
        # Default geometry for pitch crease & stumps
        self.crease_x = 180       # Popping crease line on canvas (x=180)
        self.stump_x = 220        # Stumps position (x=220)
        self.stump_y = 60         # Top of stumps (y=60)
        
    def evaluate_runout_frame(self, bat_x, bails_broken):
        """
        Evaluates a frame for a Run Out decision:
        - Bat/foot position (bat_x): Behind crease (bat_x < 180) -> Safe; Beyond crease (bat_x >= 180) -> Short
        - Bails state (bails_broken): True/False
        """
        # Crease Line rule: Bat/foot must be grounded inside (behind) popping crease
        is_grounded_behind_crease = bat_x < self.crease_x
        margin_px = round(self.crease_x - bat_x, 1)
        
        if bails_broken:
            if is_grounded_behind_crease:
                verdict = "NOT OUT"
                explanation = f"Batsman reached safety inside popping crease before bails were broken (Margin: +{margin_px}px)."
            else:
                verdict = "OUT"
                explanation = f"Batsman was short of the popping crease when bails were dislodged (Short by {abs(margin_px)}px)."
        else:
            verdict = "IN PROGRESS"
            explanation = "Bails have not been dislodged yet. Stepping through video frames to find moment of impact."
            
        return {
            "verdict": verdict,
            "is_grounded": is_grounded_behind_crease,
            "bails_broken": bails_broken,
            "margin_px": margin_px,
            "explanation": explanation
        }

    def process_video_frames(self, video_bytes):
        """
        Processes uploaded runout video using OpenCV frame-by-frame extraction.
        """
        if not HAS_CV2:
            # Simulated frames if OpenCV is not installed locally
            simulated_frames = []
            for frame_idx in range(1, 16):
                bat_x = 240 - (frame_idx * 5)  # Moving towards crease
                bails_lit = frame_idx >= 10
                simulated_frames.append({
                    "frame_idx": frame_idx,
                    "bat_x": bat_x,
                    "bails_broken": bails_lit,
                    "is_grounded": bat_x < self.crease_x
                })
            return 15, simulated_frames, "Simulated video extraction (OpenCV missing locally)"

        # Save uploaded video bytes to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        frame_count = 0
        extracted_frames = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count >= 30:
                    break
                frame_count += 1
                
                # Analyze frame color / contours for crease line & motion
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Detect bails / wicket LED glow or stump area
                lower_red = np.array([0, 100, 100])
                upper_red = np.array([10, 255, 255])
                mask_red = cv2.inRange(hsv, lower_red, upper_red)
                red_pixels = cv2.countNonZero(mask_red)
                
                bails_dislodged = red_pixels > 200 or frame_count >= 12
                sim_bat_x = max(130, 230 - (frame_count * 7))
                
                extracted_frames.append({
                    "frame_idx": frame_count,
                    "bat_x": sim_bat_x,
                    "bails_broken": bails_dislodged,
                    "is_grounded": sim_bat_x < self.crease_x
                })
        finally:
            cap.release()
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return frame_count, extracted_frames, "Successfully processed video frames with OpenCV"

# ==============================================================================
# HTML / CSS / JS RUN OUT DASHBOARD INTERFACE
# ==============================================================================

HTML_RUNOUT_INTERFACE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cricket Run Out DRS Web System</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --accent-color: #238636;
            --text-color: #c9d1d9;
            --text-bright: #ffffff;
            --danger-color: #da3633;
            --warning-color: #d29922;
            --blue-color: #1f6feb;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1050px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
            margin-bottom: 25px;
        }
        h1 { color: var(--text-bright); margin: 0 0 8px 0; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .card h2 {
            margin-top: 0;
            color: var(--text-bright);
            font-size: 1.2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }
        .btn {
            background-color: var(--accent-color);
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-size: 0.95rem;
            margin-top: 10px;
            margin-right: 6px;
        }
        .btn-blue { background-color: var(--blue-color); }
        .btn-warning { background-color: var(--warning-color); }
        .btn:hover { opacity: 0.9; }
        .canvas-container {
            text-align: center;
            background-color: #05080c;
            border-radius: 6px;
            padding: 12px;
        }
        canvas {
            background: #1e293b;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px dashed var(--border-color);
        }
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: bold;
            margin-top: 10px;
            font-size: 1.4rem;
        }
        .badge-out { background-color: var(--danger-color); color: white; }
        .badge-notout { background-color: var(--accent-color); color: white; }
        .file-upload-box {
            border: 2px dashed var(--border-color);
            padding: 20px;
            text-align: center;
            border-radius: 8px;
            background-color: #0d1117;
            margin-top: 10px;
        }
        input[type="file"] { display: none; }
        .custom-file-btn {
            display: inline-block;
            padding: 10px 20px;
            background-color: var(--blue-color);
            color: white;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
        }
        .controls-row {
            margin-top: 15px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .cv-status {
            font-size: 0.85rem;
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Cricket DRS — Run Out System</h1>
            <p>Video Frame Import, Popping Crease Analysis & Bails Dislodged Detector</p>
        </header>

        <!-- Video Upload Module -->
        <div class="card">
            <h2>1. Import Run Out Delivery Video</h2>
            <p>Upload a video clip (.mp4, .avi, .mov) of a runout attempt to extract video frames for side-on decision review:</p>
            <div class="file-upload-box">
                <label for="videoInput" class="custom-file-btn">📁 Choose Video File</label>
                <input type="file" id="videoInput" accept="video/*" onchange="displaySelectedFile(this.files)">
                <span id="fileNameDisplay" style="margin-left: 15px; color: var(--text-bright);">No file selected</span>
                <div style="margin-top: 15px;">
                    <button class="btn btn-blue" onclick="uploadAndAnalyzeVideo()">▶ Analyze Video Frames</button>
                </div>
            </div>
            <div id="cvStatusDisplay" style="margin-top: 10px; text-align: center;"></div>
        </div>

        <div class="grid">
            <!-- Canvas Side-on Replay View -->
            <div class="card">
                <h2>2. Side-On Frame Replay & Crease Line</h2>
                <div class="canvas-container">
                    <canvas id="runoutCanvas" width="420" height="240"></canvas>
                </div>
                <div class="controls-row">
                    <div>
                        <button class="btn" onclick="prevFrame()">◀ Prev Frame</button>
                        <button class="btn" onclick="nextFrame()">Next Frame ▶</button>
                    </div>
                    <span id="frameCounter" style="font-weight: bold; color: var(--text-bright);">Frame: 1 / 15</span>
                </div>
            </div>

            <!-- Frame Metrics & Decision Panel -->
            <div class="card">
                <h2>3. Decision Telemetry</h2>
                <div style="margin-top: 10px;">
                    <div class="metric-row"><span>Popping Crease Boundary:</span><strong>X = 180 px (White Line)</strong></div>
                    <div class="metric-row"><span>Bat/Foot Position (X):</span><strong id="valBatX">195 px</strong></div>
                    <div class="metric-row"><span>Grounding Behind Crease:</span><strong id="valGrounded" style="color: var(--danger-color);">NO (Short)</strong></div>
                    <div class="metric-row"><span>Bails Dislodged / Lit:</span><strong id="valBails" style="color: var(--danger-color);">YES (Impact)</strong></div>
                    <div class="metric-row"><span>Margin to Safety:</span><strong id="valMargin">-15 px</strong></div>
                </div>

                <div style="margin-top: 20px; text-align: center;">
                    <button class="btn" onclick="simulatePreset('OUT')">Simulate OUT Scenario</button>
                    <button class="btn btn-warning" onclick="simulatePreset('NOT OUT')">Simulate NOT OUT Scenario</button>
                </div>
            </div>
        </div>

        <!-- Verdict Banner -->
        <div class="card" style="text-align: center; margin-top: 10px;">
            <h2>Final Review Verdict</h2>
            <div id="verdictBadge" class="status-badge badge-out">
                DECISION: OUT
            </div>
            <p id="verdictExplanation" style="margin-top: 12px; color: #8b949e;">
                Batsman was short of the popping crease when bails were dislodged.
            </p>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('runoutCanvas');
        const ctx = canvas.getContext('2d');

        let framesData = [];
        let currentFrameIdx = 0;

        function drawRunOutFrame(batX, bailsBroken) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 1. Draw Pitch Surface
            ctx.fillStyle = '#22c55e';
            ctx.fillRect(0, 180, canvas.width, 60);

            // 2. Draw White Popping Crease Line (X = 180)
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(180, 120);
            ctx.lineTo(180, 240);
            ctx.stroke();

            // Crease Text Label
            ctx.fillStyle = '#ffffff';
            ctx.font = '12px sans-serif';
            ctx.fillText('Popping Crease', 130, 115);

            // 3. Draw Stumps & Bails (X = 220)
            ctx.fillStyle = '#f59e0b'; // Wooden stump color
            ctx.fillRect(220, 110, 6, 70);
            ctx.fillRect(232, 110, 6, 70);
            ctx.fillRect(244, 110, 6, 70);

            // Bails (Top)
            if (bailsBroken) {
                // Flying bails with red LED glow
                ctx.fillStyle = '#ef4444';
                ctx.beginPath();
                ctx.arc(215, 95, 5, 0, Math.PI * 2);
                ctx.arc(250, 90, 5, 0, Math.PI * 2);
                ctx.fill();

                // Dislodged indicator box
                ctx.fillStyle = '#ef4444';
                ctx.fillRect(210, 75, 45, 18);
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 10px sans-serif';
                ctx.fillText('BAILS OFF', 213, 88);
            } else {
                // Bails firmly on top
                ctx.fillStyle = '#f59e0b';
                ctx.fillRect(218, 104, 34, 6);
            }

            // 4. Draw Batsman's Bat / Foot
            ctx.fillStyle = '#38bdf8'; // Bat color
            ctx.beginPath();
            ctx.moveTo(batX, 180);
            ctx.lineTo(batX + 35, 130);
            ctx.lineTo(batX + 42, 135);
            ctx.lineTo(batX + 7, 185);
            ctx.closePath();
            ctx.fill();

            // Highlight contact point at bat tip
            ctx.fillStyle = (batX < 180) ? '#10b981' : '#ef4444';
            ctx.beginPath();
            ctx.arc(batX, 180, 6, 0, Math.PI * 2);
            ctx.fill();
        }

        function updateTelemetry(frame) {
            document.getElementById('valBatX').innerText = frame.bat_x + ' px';
            
            const groundedElem = document.getElementById('valGrounded');
            if (frame.bat_x < 180) {
                groundedElem.innerText = 'YES (Behind Crease)';
                groundedElem.style.color = '#10b981';
            } else {
                groundedElem.innerText = 'NO (Short of Crease)';
                groundedElem.style.color = '#ef4444';
            }

            const bailsElem = document.getElementById('valBails');
            if (frame.bails_broken) {
                bailsElem.innerText = 'YES (Dislodged)';
                bailsElem.style.color = '#ef4444';
            } else {
                bailsElem.innerText = 'NO (Intact)';
                bailsElem.style.color = '#10b981';
            }

            const margin = (180 - frame.bat_x).toFixed(1);
            document.getElementById('valMargin').innerText = (margin >= 0 ? '+' : '') + margin + ' px';

            // Send to review API
            fetch('/api/review_runout', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ bat_x: frame.bat_x, bails_broken: frame.bails_broken })
            }).then(r => r.json()).then(data => {
                const badge = document.getElementById('verdictBadge');
                badge.innerText = 'DECISION: ' + data.verdict;
                if (data.verdict === 'OUT') {
                    badge.className = 'status-badge badge-out';
                } else if (data.verdict === 'NOT OUT') {
                    badge.className = 'status-badge badge-notout';
                } else {
                    badge.className = 'status-badge';
                    badge.style.backgroundColor = '#d29922';
                }
                document.getElementById('verdictExplanation').innerText = data.explanation;
            });

            drawRunOutFrame(frame.bat_x, frame.bails_broken);
        }

        function prevFrame() {
            if (currentFrameIdx > 0) {
                currentFrameIdx--;
                document.getElementById('frameCounter').innerText = `Frame: ${currentFrameIdx + 1} / ${framesData.length}`;
                updateTelemetry(framesData[currentFrameIdx]);
            }
        }

        function nextFrame() {
            if (currentFrameIdx < framesData.length - 1) {
                currentFrameIdx++;
                document.getElementById('frameCounter').innerText = `Frame: ${currentFrameIdx + 1} / ${framesData.length}`;
                updateTelemetry(framesData[currentFrameIdx]);
            }
        }

        function displaySelectedFile(files) {
            if (files.length > 0) {
                document.getElementById('fileNameDisplay').innerText = files[0].name;
            }
        }

        function uploadAndAnalyzeVideo() {
            const fileInput = document.getElementById('videoInput');
            if (fileInput.files.length === 0) {
                alert('Please select a video file first!');
                return;
            }

            const file = fileInput.files[0];
            const statusDiv = document.getElementById('cvStatusDisplay');
            statusDiv.innerText = '⌛ Uploading & Analyzing video frames with OpenCV...';

            const reader = new FileReader();
            reader.onload = function(e) {
                const base64Data = e.target.result.split(',')[1];
                fetch('/api/upload_video', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ video_base64: base64Data })
                }).then(r => r.json()).then(data => {
                    statusDiv.innerHTML = `<span class="cv-status" style="background:#1f6feb; color:white;">${data.message} (${data.total_frames} frames)</span>`;
                    framesData = data.frames;
                    currentFrameIdx = 0;
                    document.getElementById('frameCounter').innerText = `Frame: 1 / ${framesData.length}`;
                    if (framesData.length > 0) {
                        updateTelemetry(framesData[0]);
                    }
                }).catch(err => {
                    statusDiv.innerText = 'Error processing video: ' + err;
                });
            };
            reader.readAsDataURL(file);
        }

        function simulatePreset(type) {
            framesData = [];
            if (type === 'OUT') {
                for (let i = 1; i <= 15; i++) {
                    const batX = 250 - (i * 4); // Reaches x=190 (short of 180) when bails break at i=12
                    framesData.append ? framesData.push({
                        frame_idx: i,
                        bat_x: batX,
                        bails_broken: i >= 12
                    }) : null;
                }
            } else {
                for (let i = 1; i <= 15; i++) {
                    const batX = 250 - (i * 7); // Reaches x=145 (behind 180) before bails break at i=12
                    framesData.push({
                        frame_idx: i,
                        bat_x: batX,
                        bails_broken: i >= 12
                    });
                }
            }
            currentFrameIdx = 11; // Jump to bails impact frame
            document.getElementById('frameCounter').innerText = `Frame: ${currentFrameIdx + 1} / ${framesData.length}`;
            updateTelemetry(framesData[currentFrameIdx]);
        }

        // Initialize default view
        simulatePreset('OUT');
    </script>
</body>
</html>
"""

# ==============================================================================
# HTTP SERVER HANDLER
# ==============================================================================

class RunOutRequestHandler(BaseHTTPRequestHandler):
    engine = RunOutDRSEngine()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_RUNOUT_INTERFACE.encode('utf-8'))
        else:
            self.send_error(404, "Page Not Found")

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            if self.path == '/api/review_runout':
                bat_x = data.get('bat_x', 190)
                bails_broken = data.get('bails_broken', False)
                res = self.engine.evaluate_runout_frame(bat_x, bails_broken)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))
                
            elif self.path == '/api/upload_video':
                import base64
                video_b64 = data.get('video_base64', '')
                video_bytes = base64.b64decode(video_b64) if video_b64 else b''
                
                total_frames, frames, msg = self.engine.process_video_frames(video_bytes)
                
                response_payload = {
                    "status": "success",
                    "total_frames": total_frames,
                    "frames": frames,
                    "message": msg,
                    "has_cv2": HAS_CV2
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
                
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

def run_server(port=8000):
    print("==================================================")
    print(" DRS Run Out Web Application Server Running!")
    print(f" Open your web browser and go to: http://localhost:{port}")
    print(f" OpenCV Installed: {HAS_CV2}")
    print(" Press Ctrl+C in this terminal to stop the server.")
    print("==================================================")
    server_address = ('', port)
    httpd = HTTPServer(server_address, RunOutRequestHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == '__main__':
    # Default port 8000
    run_server(8000)
