import cv2
import mediapipe as mp
import math
import numpy as np

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.current_metrics = {}
        self.alerts = []

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between 3 points (x, y)."""
        # point2 is the vertex
        v1 = (point1[0] - point2[0], point1[1] - point2[1])
        v2 = (point3[0] - point2[0], point3[1] - point2[1])
        
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 * mag2 == 0:
            return 0
            
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        angle = math.degrees(math.acos(cos_angle))
        return angle

    def calculate_2d_angle_horizontal(self, p1, p2):
        """Calculate angle of a line relative to horizontal."""
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def analyze_frame(self, frame):
        """Process single frame and update metrics."""
        self.alerts = []
        self.current_metrics = {
            "head_tilt_angle": 0.0,
            "shoulder_balance": 0.0,
            "slouch_distance": 0.0,
            "spine_angle": 180.0,
            "neck_strain": 0.0,
            "detected": False
        }

        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image.
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return

        self.current_metrics["detected"] = True
        landmarks = results.pose_landmarks.landmark
        
        # Extract required landmarks
        n = self.mp_pose.PoseLandmark.NOSE.value
        l_ear = self.mp_pose.PoseLandmark.LEFT_EAR.value
        r_ear = self.mp_pose.PoseLandmark.RIGHT_EAR.value
        l_sh = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
        r_sh = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        l_hip = self.mp_pose.PoseLandmark.LEFT_HIP.value
        r_hip = self.mp_pose.PoseLandmark.RIGHT_HIP.value

        # Coordinates
        nose = (landmarks[n].x, landmarks[n].y, landmarks[n].z)
        left_ear = (landmarks[l_ear].x, landmarks[l_ear].y)
        right_ear = (landmarks[r_ear].x, landmarks[r_ear].y)
        left_shoulder = (landmarks[l_sh].x, landmarks[l_sh].y)
        right_shoulder = (landmarks[r_sh].x, landmarks[r_sh].y)
        left_hip = (landmarks[l_hip].x, landmarks[l_hip].y)
        right_hip = (landmarks[r_hip].x, landmarks[r_hip].y)

        # 1. HEAD TILT
        # Angle of eyes/ears relative to horizontal
        eye_angle = self.calculate_2d_angle_horizontal(left_ear, right_ear)
        # normalize to tilt from vertical
        head_tilt = abs(eye_angle) if abs(eye_angle) < 90 else 180 - abs(eye_angle)
        self.current_metrics["head_tilt_angle"] = head_tilt

        # 2. SHOULDER BALANCE
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        self.current_metrics["shoulder_balance"] = shoulder_diff

        # 3. SLOUCH DETECTION
        # Distance between nose and midpoint of shoulders
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        # Simple relative slouch (nose drops closer to shoulders)
        slouch_dist = shoulder_mid_y - nose[1] 
        self.current_metrics["slouch_distance"] = slouch_dist

        # 4. SPINE ALIGNMENT
        # Using midpoint of shoulders and hips
        mid_shoulder = ((left_shoulder[0]+right_shoulder[0])/2, shoulder_mid_y)
        mid_hip = ((left_hip[0]+right_hip[0])/2, (left_hip[1]+right_hip[1])/2)
        
        # Define a vertical point above mid_hip
        vertical_point = (mid_hip[0], mid_hip[1] - 0.5)
        spine_angle = self.calculate_angle(mid_shoulder, mid_hip, vertical_point)
        # Convert to 180 based ideal
        spine_angle = 180 - spine_angle if spine_angle > 90 else 180 + spine_angle
        self.current_metrics["spine_angle"] = spine_angle

        # 5. NECK STRAIN (Approximation using z-depth of nose vs shoulders)
        # Using z-coordinate (depth). If nose z is much more negative (closer to cam) than shoulders
        shoulder_mid_z = (landmarks[l_sh].z + landmarks[r_sh].z) / 2.0
        neck_strain = shoulder_mid_z - nose[2]
        self.current_metrics["neck_strain"] = neck_strain

        self.generate_alerts()

    def generate_alerts(self):
        """List current posture issues."""
        if self.current_metrics["head_tilt_angle"] > 15:
            self.alerts.append("Head tilted - straighten up")
            
        if self.current_metrics["shoulder_balance"] > 0.08:
            self.alerts.append("Shoulders uneven - relax both sides")
            
        # If nose drops too close to shoulders (e.g. < 10% of frame height)
        if self.current_metrics["slouch_distance"] < 0.15:
            self.alerts.append("Slouching - sit straight with back against chair")
            
        if self.current_metrics["spine_angle"] < 165 or self.current_metrics["spine_angle"] > 195:
            self.alerts.append("Curved spine detected")
            
        # Z-depth strain
        if self.current_metrics["neck_strain"] > 0.3:
            self.alerts.append("Neck strain detected - align head over shoulders")

    def get_posture_score(self):
        """Composite score 0-100"""
        if not self.current_metrics.get("detected"):
            return 0
            
        score = 100
        
        # Deduct points based on metrics
        score -= min(30, (self.current_metrics["head_tilt_angle"] / 15) * 10)
        
        if self.current_metrics["shoulder_balance"] > 0.04:
            score -= min(20, (self.current_metrics["shoulder_balance"] / 0.08) * 15)
            
        if self.current_metrics["slouch_distance"] < 0.2:
            score -= 20
            
        spine_dev = abs(180 - self.current_metrics["spine_angle"])
        if spine_dev > 10:
            score -= min(30, (spine_dev / 15) * 15)
            
        if self.current_metrics["neck_strain"] > 0.2:
            score -= 10
            
        return max(0, int(score))

    def get_results(self):
        """Return formatted result."""
        score = self.get_posture_score()
        
        rec = "Good posture! Keep it up 👍"
        if score < 50:
            rec = "Poor posture detected. Please sit up straight, align ears with shoulders."
        elif score < 75:
            rec = "Posture is okay, but could be better. Relax your shoulders."
            
        return {
            "head_tilt_angle": round(self.current_metrics.get("head_tilt_angle", 0), 1),
            "shoulder_balance": round(self.current_metrics.get("shoulder_balance", 0), 3),
            "slouch_distance": round(self.current_metrics.get("slouch_distance", 0), 2),
            "spine_angle": round(self.current_metrics.get("spine_angle", 180), 1),
            "neck_strain": round(self.current_metrics.get("neck_strain", 0), 2),
            "posture_score": score,
            "alerts": self.alerts,
            "recommendation": rec
        }
