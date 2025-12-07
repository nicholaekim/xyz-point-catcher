"""
Live Joint List - Shows all 26 OpenXR hand joints with XYZ values
"""
import argparse
import csv
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import numpy as np
from pythonosc import dispatcher
from pythonosc import osc_server


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
# OSC Server
# ─────────────────────────────────────────────────────────────────────────────
def start_osc_server(host: str, port: int):
    disp = dispatcher.Dispatcher()
    disp.set_default_handler(default_handler)
    server = osc_server.ThreadingOSCUDPServer((host, port), disp)
    print(f"[OSC] Listening on {host}:{port}")
    server.serve_forever()


def start_dual_osc_servers(host: str, port1: int, port2: int):
    """Start two OSC servers for left and right gloves on different ports."""
    disp = dispatcher.Dispatcher()
    disp.set_default_handler(default_handler)
    
    server1 = osc_server.ThreadingOSCUDPServer((host, port1), disp)
    server2 = osc_server.ThreadingOSCUDPServer((host, port2), disp)
    
    print(f"[OSC] Listening on {host}:{port1} and {host}:{port2}")
    
    # Run second server in a thread
    thread2 = threading.Thread(target=server2.serve_forever, daemon=True)
    thread2.start()
    
    # Run first server in main thread
    server1.serve_forever()


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
        
        # Export counter
        self.export_count = 0

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
        
        total_packets = left_data['packet_count'] + right_data['packet_count']
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
