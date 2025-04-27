import tkinter as tk
from tkinter import ttk
import subprocess
import webbrowser
import os

# Function to run prediction GUI
def run_prediction_gui():
    subprocess.Popen(["python", "prediction_gui.py"])

# Function to run main program
def run_main():
    subprocess.Popen(["python", "main.py"])

# Function to open custom gestures GUI
def open_custom_gestures():
    file_path = os.path.abspath("index.html")
    webbrowser.open(f"file://{file_path}")

# Create main window
root = tk.Tk()
root.title("Silent Cue Hand Gesture Recognition System")
root.geometry("500x300")
root.resizable(False, False)

# Apply a modern theme
style = ttk.Style()
style.configure("TNotebook", background="#f0f0f0")
style.configure("TNotebook.Tab", font=("Arial", 11, "bold"))
style.configure("TButton", font=("Arial", 10), padding=5)

# Create Notebook (Tabs)
notebook = ttk.Notebook(root)

# Create Frames for Tabs
tab1 = ttk.Frame(notebook, padding=20)
tab2 = ttk.Frame(notebook, padding=20)
tab3 = ttk.Frame(notebook, padding=20)

notebook.add(tab1, text="Hand Gesture Recognition")
notebook.add(tab2, text="Mouse & Volume Control")
notebook.add(tab3, text="Custom Hand Gestures")

notebook.pack(expand=True, fill="both", padx=10, pady=10)

# Title Label
title_label = ttk.Label(root, text="Select a Feature to Run", font=("Arial", 12, "bold"))
title_label.pack(pady=5)

# Buttons
btn1 = ttk.Button(tab1, text="Start Gesture Recognition", command=run_prediction_gui)
btn1.pack(expand=True)

btn2 = ttk.Button(tab2, text="Start Mouse & Volume Control", command=run_main)
btn2.pack(expand=True)

btn3 = ttk.Button(tab3, text="Open Custom Gestures GUI Admin", command=open_custom_gestures)
btn3.pack(expand=True)

# Run GUI
root.mainloop()
