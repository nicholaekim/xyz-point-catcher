"""
Live Joint List - Shows all 26 OpenXR hand joints with XYZ values
"""
import argparse
import csv
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from pythonosc import dispatcher
from pythonosc import osc_server

# Finger connections for drawing skeleton lines
FINGER_CONNECTIONS = [
    [1, 2, 3, 4, 5],       # Thumb: wrist -> tip
    [1, 6, 7, 8, 9, 10],   # Index
    [1, 11, 12, 13, 14, 15], # Middle
    [1, 16, 17, 18, 19, 20], # Ring
    [1, 21, 22, 23, 24, 25], # Little
    [0, 1],                 # Palm to wrist
]

# Base hand skeleton positions (forms a hand shape for 3D visualization)
# These are approximate positions that form a recognizable hand
BASE_HAND_POSITIONS = np.array([
    [0.0, 0.0, 0.0],      # 0: Palm
    [0.0, -0.08, 0.0],    # 1: Wrist
    [-0.08, 0.04, 0.04],  # 2: Thumb metacarpal
    [-0.14, 0.08, 0.06],  # 3: Thumb proximal
    [-0.18, 0.12, 0.07],  # 4: Thumb distal
    [-0.22, 0.16, 0.08],  # 5: Thumb tip
    [-0.04, 0.08, 0.0],   # 6: Index metacarpal
    [-0.05, 0.18, 0.0],   # 7: Index proximal
    [-0.05, 0.26, 0.0],   # 8: Index intermediate
    [-0.05, 0.32, 0.0],   # 9: Index distal
    [-0.05, 0.38, 0.0],   # 10: Index tip
    [0.0, 0.08, 0.0],     # 11: Middle metacarpal
    [0.0, 0.20, 0.0],     # 12: Middle proximal
    [0.0, 0.28, 0.0],     # 13: Middle intermediate
    [0.0, 0.34, 0.0],     # 14: Middle distal
    [0.0, 0.42, 0.0],     # 15: Middle tip
    [0.04, 0.08, 0.0],    # 16: Ring metacarpal
    [0.05, 0.18, 0.0],    # 17: Ring proximal
    [0.05, 0.25, 0.0],    # 18: Ring intermediate
    [0.05, 0.30, 0.0],    # 19: Ring distal
    [0.05, 0.36, 0.0],    # 20: Ring tip
    [0.08, 0.07, 0.0],    # 21: Little metacarpal
    [0.10, 0.14, 0.0],    # 22: Little proximal
    [0.10, 0.20, 0.0],    # 23: Little intermediate
    [0.10, 0.24, 0.0],    # 24: Little distal
    [0.10, 0.28, 0.0],    # 25: Little tip
])


# ─────────────────────────────────────────────────────────────────────────────
# OpenXR Hand Joint Names (26 joints, exact order)
# ─────────────────────────────────────────────────────────────────────────────
JOINT_NAMES = [
    "Palm",
    "Wrist",
    "Thumb metacarpal",
    "Thumb proximal",
    "Thumb distal",
    "Thumb tip",
    "Index metacarpal",
    "Index proximal",
    "Index intermediate",
    "Index distal",
    "Index tip",
    "Middle metacarpal",
    "Middle proximal",
    "Middle intermediate",
    "Middle distal",
    "Middle tip",
    "Ring metacarpal",
    "Ring proximal",
    "Ring intermediate",
    "Ring distal",
    "Ring tip",
    "Little metacarpal",
    "Little proximal",
    "Little intermediate",
    "Little distal",
    "Little tip",
]

NUM_JOINTS = 26


# ─────────────────────────────────────────────────────────────────────────────
# Shared State
# ─────────────────────────────────────────────────────────────────────────────
class HandState:
    def __init__(self):
        self.lock = threading.Lock()
        self.positions = np.zeros((NUM_JOINTS, 3))
        self.device_name = ""
        self.packet_count = 0
        self.has_data = False
        self.calibrated = False
        self.offset = np.zeros((NUM_JOINTS, 3))

    def update(self, device_name: str, positions: np.ndarray):
        with self.lock:
            if not self.calibrated:
                self.offset = positions.copy()
                self.calibrated = True
            self.device_name = device_name
            self.positions = positions - self.offset
            self.packet_count += 1
            self.has_data = True

    def reset_calibration(self):
        with self.lock:
            self.calibrated = False
            self.positions = np.zeros((NUM_JOINTS, 3))

    def get(self):
        with self.lock:
            return {
                "positions": self.positions.copy(),
                "device_name": self.device_name,
                "packet_count": self.packet_count,
                "has_data": self.has_data
            }


left_state = HandState()
right_state = HandState()


# ─────────────────────────────────────────────────────────────────────────────
# OSC Handler
# ─────────────────────────────────────────────────────────────────────────────
def default_handler(address: str, *args):
    # Only process kinematic data (has 26 joints)
    if "/kinematic" not in address.lower():
        return
    
    if len(args) < 187:  # 5 header + 26*7 joint values
        return
    
    try:
        device_name = str(args[3]).lower()
        joint_data = args[5:]
        values_per_joint = 7  # x, y, z, qw, qx, qy, qz
        
        positions = np.zeros((NUM_JOINTS, 3))
        for i in range(NUM_JOINTS):
            base = i * values_per_joint
            # Get quaternion (indices 3,4,5,6 after xyz) and convert to euler-like values
            qw = float(joint_data[base + 3])
            qx = float(joint_data[base + 4])
            qy = float(joint_data[base + 5])
            qz = float(joint_data[base + 6])
            # Store quaternion components as "XYZ" for display (these change with movement)
            positions[i, 0] = qx
            positions[i, 1] = qy
            positions[i, 2] = qz
        
        # Route to correct hand based on device name
        # Device names are "Reality Glove (L)" and "Reality Glove (R)"
        if "(l)" in device_name or "left" in device_name:
            left_state.update(device_name, positions)
        else:
            right_state.update(device_name, positions)
    except Exception as e:
        print(f"[Error] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────
class JointListGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Live Joint List - Both Hands")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e1e")

        # Header
        header = tk.Label(root, text="OpenXR Hand Joints - Left & Right", 
                         font=("Consolas", 14, "bold"), fg="#00ccff", bg="#1e1e1e")
        header.pack(pady=5)

        # Device labels frame
        device_frame = tk.Frame(root, bg="#1e1e1e")
        device_frame.pack()
        self.left_device_label = tk.Label(device_frame, text="Left: (waiting...)",
                                     font=("Consolas", 10), fg="#ff9966", bg="#1e1e1e")
        self.left_device_label.pack(side=tk.LEFT, padx=20)
        self.right_device_label = tk.Label(device_frame, text="Right: (waiting...)",
                                     font=("Consolas", 10), fg="#66ccff", bg="#1e1e1e")
        self.right_device_label.pack(side=tk.LEFT, padx=20)

        # Create frame with scrollbar for joint list
        list_frame = tk.Frame(root, bg="#1e1e1e")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Column headers
        header_frame = tk.Frame(list_frame, bg="#2d2d2d")
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="Idx", width=4, font=("Consolas", 10, "bold"),
                fg="#888", bg="#2d2d2d", anchor="w").pack(side=tk.LEFT)
        tk.Label(header_frame, text="Joint Name", width=18, font=("Consolas", 10, "bold"),
                fg="#888", bg="#2d2d2d", anchor="w").pack(side=tk.LEFT)
        # Left hand header
        tk.Label(header_frame, text="Left X", width=8, font=("Consolas", 9, "bold"),
                fg="#ff9966", bg="#2d2d2d").pack(side=tk.LEFT)
        tk.Label(header_frame, text="Y", width=8, font=("Consolas", 9, "bold"),
                fg="#ff9966", bg="#2d2d2d").pack(side=tk.LEFT)
        tk.Label(header_frame, text="Z", width=8, font=("Consolas", 9, "bold"),
                fg="#ff9966", bg="#2d2d2d").pack(side=tk.LEFT)
        tk.Label(header_frame, text=" | ", width=2, font=("Consolas", 9),
                fg="#444", bg="#2d2d2d").pack(side=tk.LEFT)
        # Right hand header
        tk.Label(header_frame, text="Right X", width=8, font=("Consolas", 9, "bold"),
                fg="#66ccff", bg="#2d2d2d").pack(side=tk.LEFT)
        tk.Label(header_frame, text="Y", width=8, font=("Consolas", 9, "bold"),
                fg="#66ccff", bg="#2d2d2d").pack(side=tk.LEFT)
        tk.Label(header_frame, text="Z", width=8, font=("Consolas", 9, "bold"),
                fg="#66ccff", bg="#2d2d2d").pack(side=tk.LEFT)

        # Scrollable canvas for joint rows
        canvas = tk.Canvas(list_frame, bg="#1e1e1e", highlightthickness=0)
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        self.joint_frame = tk.Frame(canvas, bg="#1e1e1e")

        self.joint_frame.bind("<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.joint_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create labels for each joint (both hands)
        self.left_labels = []
        self.right_labels = []
        for i in range(NUM_JOINTS):
            row_bg = "#252525" if i % 2 == 0 else "#1e1e1e"
            row = tk.Frame(self.joint_frame, bg=row_bg)
            row.pack(fill=tk.X)

            idx_lbl = tk.Label(row, text=f"{i}", width=4, font=("Consolas", 10),
                              fg="#666", bg=row_bg, anchor="w")
            idx_lbl.pack(side=tk.LEFT)

            name_lbl = tk.Label(row, text=JOINT_NAMES[i], width=18, font=("Consolas", 10),
                               fg="#ccc", bg=row_bg, anchor="w")
            name_lbl.pack(side=tk.LEFT)

            # Left hand values
            lx_lbl = tk.Label(row, text="0.000", width=8, font=("Consolas", 9),
                            fg="#ff9966", bg=row_bg)
            lx_lbl.pack(side=tk.LEFT)
            ly_lbl = tk.Label(row, text="0.000", width=8, font=("Consolas", 9),
                            fg="#ff9966", bg=row_bg)
            ly_lbl.pack(side=tk.LEFT)
            lz_lbl = tk.Label(row, text="0.000", width=8, font=("Consolas", 9),
                            fg="#ff9966", bg=row_bg)
            lz_lbl.pack(side=tk.LEFT)
            
            tk.Label(row, text=" | ", width=2, font=("Consolas", 9),
                    fg="#444", bg=row_bg).pack(side=tk.LEFT)

            # Right hand values
            rx_lbl = tk.Label(row, text="0.000", width=8, font=("Consolas", 9),
                            fg="#66ccff", bg=row_bg)
            rx_lbl.pack(side=tk.LEFT)
            ry_lbl = tk.Label(row, text="0.000", width=8, font=("Consolas", 9),
                            fg="#66ccff", bg=row_bg)
            ry_lbl.pack(side=tk.LEFT)
            rz_lbl = tk.Label(row, text="0.000", width=8, font=("Consolas", 9),
                            fg="#66ccff", bg=row_bg)
            rz_lbl.pack(side=tk.LEFT)

            self.left_labels.append((lx_lbl, ly_lbl, lz_lbl))
            self.right_labels.append((rx_lbl, ry_lbl, rz_lbl))

        # Button frame
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=10)
        
        recal_btn = tk.Button(btn_frame, text="Recalibrate", 
                             font=("Consolas", 10), fg="#fff", bg="#444",
                             command=self._recalibrate, width=12)
        recal_btn.pack(side=tk.LEFT, padx=5)
        
        export_btn = tk.Button(btn_frame, text="Export to CSV", 
                              font=("Consolas", 10), fg="#fff", bg="#2a6e2a",
                              command=self._export_csv, width=12)
        export_btn.pack(side=tk.LEFT, padx=5)
        
        live_3d_btn = tk.Button(btn_frame, text="Live 3D View", 
                              font=("Consolas", 10), fg="#fff", bg="#6a2a6e",
                              command=self._show_live_3d, width=12)
        live_3d_btn.pack(side=tk.LEFT, padx=5)
        
        # Recording state
        self.recording = False
        self.recorded_frames = []  # List of (left_pos, right_pos) tuples
        
        self.record_btn = tk.Button(btn_frame, text="Record", 
                              font=("Consolas", 10), fg="#fff", bg="#8b0000",
                              command=self._toggle_recording, width=10)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        playback_btn = tk.Button(btn_frame, text="Playback", 
                              font=("Consolas", 10), fg="#fff", bg="#2a4a6e",
                              command=self._playback_animation, width=10)
        playback_btn.pack(side=tk.LEFT, padx=5)
        
        # Export counter
        self.export_count = 0
        
        # Live 3D view state
        self.live_3d_active = False
        self.live_3d_fig = None
        self.live_3d_ax = None

        # Status
        self.status_label = tk.Label(root, text="Packets: 0",
                                    font=("Consolas", 9), fg="#666", bg="#1e1e1e")
        self.status_label.pack(pady=5)

        # Start update loop
        self._update()

    def _recalibrate(self):
        left_state.reset_calibration()
        right_state.reset_calibration()

    def _export_csv(self):
        """Export current joint positions to CSV file."""
        left_data = left_state.get()
        right_data = right_state.get()
        
        if not left_data['has_data'] and not right_data['has_data']:
            messagebox.showwarning("No Data", "No hand data available to export.")
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"both_hands_{timestamp}.csv"
        
        # Ask user where to save
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_filename,
            title="Export Hand Poses to CSV"
        )
        
        if not filepath:
            return  # User cancelled
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Left Glove Section
                writer.writerow(["=== LEFT GLOVE ==="])
                writer.writerow(["Index", "Joint Name", "X", "Y", "Z"])
                for i in range(NUM_JOINTS):
                    lx, ly, lz = left_data['positions'][i]
                    writer.writerow([i, JOINT_NAMES[i], f"{lx:.6f}", f"{ly:.6f}", f"{lz:.6f}"])
                
                # Empty row separator
                writer.writerow([])
                
                # Right Glove Section
                writer.writerow(["=== RIGHT GLOVE ==="])
                writer.writerow(["Index", "Joint Name", "X", "Y", "Z"])
                for i in range(NUM_JOINTS):
                    rx, ry, rz = right_data['positions'][i]
                    writer.writerow([i, JOINT_NAMES[i], f"{rx:.6f}", f"{ry:.6f}", f"{rz:.6f}"])
            
            self.export_count += 1
            messagebox.showinfo("Exported", f"Saved to:\n{filepath}")
            print(f"[Export] Saved both hand poses to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export:\n{e}")

    def _toggle_recording(self):
        """Toggle recording of hand movement frames."""
        if not self.recording:
            self.recording = True
            self.recorded_frames = []
            self.record_btn.config(text="Stop", bg="#ff4444")
            print("[Recording] Started recording frames...")
        else:
            self.recording = False
            self.record_btn.config(text="Record", bg="#8b0000")
            print(f"[Recording] Stopped. Captured {len(self.recorded_frames)} frames.")
    
    def _playback_animation(self):
        """Play back recorded hand movement as 3D animation."""
        if len(self.recorded_frames) < 2:
            messagebox.showwarning("No Recording", "Record some hand movement first!\n\nClick 'Record' to start, move your hands, then click 'Stop'.")
            return
        
        print(f"[Playback] Playing {len(self.recorded_frames)} frames...")
        
        fig = plt.figure(figsize=(12, 6))
        ax_left = fig.add_subplot(121, projection='3d')
        ax_right = fig.add_subplot(122, projection='3d')
        
        fig.suptitle('Hand Movement Animation', fontsize=14, fontweight='bold')
        
        def draw_hand_skeleton(ax, positions, color, title):
            ax.clear()
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Combine base hand shape with quaternion movement data
            movement_scale = 0.3
            display_positions = BASE_HAND_POSITIONS + (positions * movement_scale)
            
            # Plot joints
            ax.scatter(display_positions[:, 0], display_positions[:, 1], display_positions[:, 2], 
                      c='red', s=50, depthshade=True)
            
            # Add joint labels
            for i in range(NUM_JOINTS):
                ax.text(display_positions[i, 0], display_positions[i, 1], display_positions[i, 2], 
                       str(i), fontsize=8, color='darkred')
            
            # Draw skeleton connections
            for finger in FINGER_CONNECTIONS:
                for j in range(len(finger) - 1):
                    idx1, idx2 = finger[j], finger[j + 1]
                    ax.plot([display_positions[idx1, 0], display_positions[idx2, 0]],
                           [display_positions[idx1, 1], display_positions[idx2, 1]],
                           [display_positions[idx1, 2], display_positions[idx2, 2]],
                           color=color, linewidth=2)
            
            # Set fixed axis limits for stable view
            ax.set_xlim([-0.35, 0.25])
            ax.set_ylim([-0.15, 0.5])
            ax.set_zlim([-0.2, 0.2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        def animate(frame_idx):
            left_pos, right_pos = self.recorded_frames[frame_idx]
            draw_hand_skeleton(ax_left, left_pos, 'green', 'Left Hand')
            draw_hand_skeleton(ax_right, right_pos, 'blue', 'Right Hand')
            fig.suptitle(f'Hand Movement Animation - Frame {frame_idx + 1}/{len(self.recorded_frames)}', 
                        fontsize=14, fontweight='bold')
            return []
        
        anim = FuncAnimation(fig, animate, frames=len(self.recorded_frames), 
                            interval=16, repeat=True, blit=False)
        plt.tight_layout()
        plt.show(block=False)
    
    def _show_live_3d(self):
        """Show live 3D hand skeleton visualization."""
        left_data = left_state.get()
        right_data = right_state.get()
        
        if not left_data['has_data'] and not right_data['has_data']:
            messagebox.showwarning("No Data", "No hand data available. Connect gloves first.")
            return
        
        self.live_3d_active = True
        self.live_3d_fig = plt.figure(figsize=(12, 6))
        self.live_3d_ax_left = self.live_3d_fig.add_subplot(121, projection='3d')
        self.live_3d_ax_right = self.live_3d_fig.add_subplot(122, projection='3d')
        
        self.live_3d_fig.suptitle('Live 3D Hand Skeleton', fontsize=14, fontweight='bold')
        
        def update_live_3d(frame):
            if not self.live_3d_active:
                return []
            
            left_data = left_state.get()
            right_data = right_state.get()
            
            # Record frame if recording
            if self.recording:
                self.recorded_frames.append((left_data['positions'].copy(), right_data['positions'].copy()))
            
            # Draw left hand
            self.live_3d_ax_left.clear()
            self.live_3d_ax_left.set_title('Left Hand', fontsize=12, fontweight='bold', color='#ff9966')
            if left_data['has_data']:
                self._draw_skeleton_3d(self.live_3d_ax_left, left_data['positions'], 'green')
            
            # Draw right hand
            self.live_3d_ax_right.clear()
            self.live_3d_ax_right.set_title('Right Hand', fontsize=12, fontweight='bold', color='#66ccff')
            if right_data['has_data']:
                self._draw_skeleton_3d(self.live_3d_ax_right, right_data['positions'], 'blue')
            
            return []
        
        self.live_anim = FuncAnimation(self.live_3d_fig, update_live_3d, interval=16, blit=False, cache_frame_data=False)
        
        def on_close(event):
            self.live_3d_active = False
        
        self.live_3d_fig.canvas.mpl_connect('close_event', on_close)
        
        # Add "Get Points" button
        btn_ax = self.live_3d_fig.add_axes([0.42, 0.02, 0.16, 0.05])
        self.get_points_btn = Button(btn_ax, 'Get Points', color='#4a4a4a', hovercolor='#6a6a6a')
        self.get_points_btn.label.set_color('white')
        self.get_points_btn.on_clicked(self._show_points_chart)
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show(block=False)
    
    def _draw_skeleton_3d(self, ax, positions, color):
        """Draw 3D hand skeleton on given axes."""
        # Combine base hand shape with quaternion movement data
        # Scale the movement data to make it visible
        movement_scale = 0.3
        display_positions = BASE_HAND_POSITIONS + (positions * movement_scale)
        
        # Plot joints as red dots
        ax.scatter(display_positions[:, 0], display_positions[:, 1], display_positions[:, 2], 
                  c='red', s=50, depthshade=True)
        
        # Add joint index labels
        for i in range(NUM_JOINTS):
            ax.text(display_positions[i, 0], display_positions[i, 1], display_positions[i, 2], 
                   str(i), fontsize=8, color='darkred')
        
        # Draw finger connections as green lines
        for finger in FINGER_CONNECTIONS:
            for j in range(len(finger) - 1):
                idx1, idx2 = finger[j], finger[j + 1]
                ax.plot([display_positions[idx1, 0], display_positions[idx2, 0]],
                       [display_positions[idx1, 1], display_positions[idx2, 1]],
                       [display_positions[idx1, 2], display_positions[idx2, 2]],
                       color=color, linewidth=2)
        
        # Set fixed axis limits for stable view
        ax.set_xlim([-0.35, 0.25])
        ax.set_ylim([-0.15, 0.5])
        ax.set_zlim([-0.2, 0.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    def _show_points_chart(self, event):
        """Show a chart with all 26 joint positions for both hands."""
        left_data = left_state.get()
        right_data = right_state.get()
        
        # Create a new figure for the points table
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 10))
        fig.suptitle('Joint Positions Snapshot', fontsize=16, fontweight='bold')
        
        # Prepare table data for left hand
        left_table_data = []
        for i in range(NUM_JOINTS):
            x, y, z = left_data['positions'][i]
            left_table_data.append([i, JOINT_NAMES[i], f"{x:+.4f}", f"{y:+.4f}", f"{z:+.4f}"])
        
        # Prepare table data for right hand
        right_table_data = []
        for i in range(NUM_JOINTS):
            x, y, z = right_data['positions'][i]
            right_table_data.append([i, JOINT_NAMES[i], f"{x:+.4f}", f"{y:+.4f}", f"{z:+.4f}"])
        
        columns = ['Idx', 'Joint Name', 'X', 'Y', 'Z']
        
        # Left hand table
        ax_left.axis('off')
        ax_left.set_title('Left Hand', fontsize=14, fontweight='bold', color='#ff9966')
        left_table = ax_left.table(
            cellText=left_table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colWidths=[0.08, 0.35, 0.19, 0.19, 0.19]
        )
        left_table.auto_set_font_size(False)
        left_table.set_fontsize(9)
        left_table.scale(1, 1.3)
        
        # Style header row
        for j, col in enumerate(columns):
            left_table[(0, j)].set_facecolor('#ff9966')
            left_table[(0, j)].set_text_props(fontweight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, NUM_JOINTS + 1):
            color = '#f5f5f5' if i % 2 == 0 else 'white'
            for j in range(len(columns)):
                left_table[(i, j)].set_facecolor(color)
        
        # Right hand table
        ax_right.axis('off')
        ax_right.set_title('Right Hand', fontsize=14, fontweight='bold', color='#66ccff')
        right_table = ax_right.table(
            cellText=right_table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            colWidths=[0.08, 0.35, 0.19, 0.19, 0.19]
        )
        right_table.auto_set_font_size(False)
        right_table.set_fontsize(9)
        right_table.scale(1, 1.3)
        
        # Style header row
        for j, col in enumerate(columns):
            right_table[(0, j)].set_facecolor('#66ccff')
            right_table[(0, j)].set_text_props(fontweight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, NUM_JOINTS + 1):
            color = '#f5f5f5' if i % 2 == 0 else 'white'
            for j in range(len(columns)):
                right_table[(i, j)].set_facecolor(color)
        
        plt.tight_layout()
        plt.show(block=False)
        print("[Get Points] Captured joint positions snapshot")

    def _update(self):
        left_data = left_state.get()
        right_data = right_state.get()
        
        if left_data['has_data']:
            self.left_device_label.config(text=f"Left: {left_data['device_name']}")
            for i in range(NUM_JOINTS):
                x, y, z = left_data['positions'][i]
                self.left_labels[i][0].config(text=f"{x:+.3f}")
                self.left_labels[i][1].config(text=f"{y:+.3f}")
                self.left_labels[i][2].config(text=f"{z:+.3f}")
        
        if right_data['has_data']:
            self.right_device_label.config(text=f"Right: {right_data['device_name']}")
            for i in range(NUM_JOINTS):
                x, y, z = right_data['positions'][i]
                self.right_labels[i][0].config(text=f"{x:+.3f}")
                self.right_labels[i][1].config(text=f"{y:+.3f}")
                self.right_labels[i][2].config(text=f"{z:+.3f}")
        
        self.status_label.config(text=f"Packets: L={left_data['packet_count']} R={right_data['packet_count']}")
        self.root.after(50, self._update)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def start_multi_osc_servers(host: str, ports: list):
    """Start OSC servers on multiple ports."""
    disp = dispatcher.Dispatcher()
    disp.set_default_handler(default_handler)
    
    servers = []
    for port in ports:
        try:
            server = osc_server.ThreadingOSCUDPServer((host, port), disp)
            servers.append(server)
            print(f"[OSC] Listening on {host}:{port}")
        except Exception as e:
            print(f"[OSC] Could not bind to port {port}: {e}")
    
    # Start all servers in threads
    for server in servers[:-1]:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    
    # Last server runs in current thread
    if servers:
        servers[-1].serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Live Joint List")
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Try common OSC ports for StretchSense gloves
    ports = [9000, 9001, 9002, 9003, 9004, 9005]
    
    osc_thread = threading.Thread(
        target=start_multi_osc_servers, 
        args=(args.host, ports), 
        daemon=True
    )
    osc_thread.start()

    root = tk.Tk()
    app = JointListGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
