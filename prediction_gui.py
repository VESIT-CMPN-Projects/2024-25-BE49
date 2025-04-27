import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import Label, Canvas, StringVar, Listbox, Button, filedialog, Frame
from PIL import Image, ImageTk
from nltk.corpus import words
import nltk
import time
import os
import subprocess
import sys

# Download NLTK words if not already downloaded
nltk.download('words')
word_list = [word.lower() for word in words.words() if len(word) > 2]  # Filter short words

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=2)

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# Create a Tkinter window
window = tk.Tk()
window.title("Silent Cue - Hand Gesture Recognition")
window.geometry("1366x768")

# Main container frame
main_frame = Frame(window)
main_frame.pack(fill='both', expand=True)

# Video feed frame (left side)
video_frame = Frame(main_frame, width=900, height=600)
video_frame.pack(side='left', fill='both', expand=True)
video_frame.pack_propagate(False)

# Video label
video_label = Label(video_frame)
video_label.pack(fill='both', expand=True)

# Control panel frame (right side)
control_panel = Frame(main_frame, width=466, bg='#f0f0f0')
control_panel.pack(side='right', fill='both')

# Prediction label
prediction_var = StringVar()
prediction_label = Label(control_panel, textvariable=prediction_var, font=("Arial", 20), bg='#f0f0f0')
prediction_label.pack(pady=(20, 10))

# Hand side label
hand_side_var = StringVar()
hand_side_label = Label(control_panel, textvariable=hand_side_var, font=("Arial", 18), bg='#f0f0f0')
hand_side_label.pack()

# Word formation - Modified to show both sentence and current word
word_var = StringVar()
word_label = Label(control_panel, textvariable=word_var, font=("Arial", 18), bg='#f0f0f0', justify='left', anchor='w', wraplength=400)
word_label.pack(pady=10, fill='x', padx=10)

# Canvas for hand points
canvas = Canvas(control_panel, width=250, height=250, bg='white', highlightthickness=1, highlightbackground="black")
canvas.pack(pady=10)

# Create a container for buttons and suggestions
control_container = Frame(control_panel, bg='#f0f0f0')
control_container.pack(fill='both', expand=True, padx=10, pady=10)

# Create a frame for suggestions on the right
suggestion_frame = Frame(control_container, bg='#f0f0f0')
suggestion_frame.pack(side='right', fill='both', expand=True)

# Suggestions listbox
suggestion_listbox = Listbox(suggestion_frame, height=4, font=("Arial", 14), selectbackground='lightblue')
suggestion_listbox.pack(pady=10, fill='x', padx=20)

# Button container
button_container = Frame(control_panel, bg='#f0f0f0')
button_container.pack(fill='x', padx=20, pady=10)

# Button grid configuration
button_container.grid_columnconfigure(0, weight=1)
button_container.grid_columnconfigure(1, weight=1)
button_container.grid_rowconfigure(0, weight=1)
button_container.grid_rowconfigure(1, weight=1)
button_container.grid_rowconfigure(2, weight=1)

button_container = Frame(control_container, bg='#f0f0f0')
button_container.pack(side='left', fill='y', padx=(0, 10))

# Create a frame for buttons on the left
button_frame = Frame(control_container, bg='#f0f0f0')
button_frame.pack(side='left', fill='y', padx=(0, 10))

# Create a frame for suggestions on the right
suggestion_frame = Frame(control_container, bg='#f0f0f0')
suggestion_frame.pack(side='right', fill='both', expand=True)

button_columns = Frame(control_container, bg='#f0f0f0')
button_columns.pack(side='left', fill='y', padx=(0, 10))

# Left button column
left_button_column = Frame(button_container, bg='#f0f0f0')
left_button_column.pack(side='left', fill='y', expand=True)

# Right button column
right_button_column = Frame(button_columns, bg='#f0f0f0')
right_button_column.pack(side='left', fill='y', padx=5)

# Current case state
current_case = True  # True for uppercase, False for lowercase

def toggle_case():
    """Toggle between uppercase and lowercase"""
    global current_case
    current_case = not current_case
    case_button.config(text="Uppercase" if current_case else "Lowercase")


# Button style
button_style = {
    'font': ("Arial", 12),  # Slightly larger font
    'bd': 2,
    'relief': 'ridge',
    'height': 1,
    'width': 12,  # Fixed width for consistent sizing
    'padx': 10,   # More horizontal padding
    'pady': 5     # More vertical padding
}

# Current word state
current_word = ""
current_sentence = ""
detected_letter = ""
last_prediction_time = 0
prediction_delay = 0.5  # Seconds between predictions

def update_word_display():
    """Update the word display to show both sentence and current word"""
    display_text = f"Sentence: {current_sentence}\nCurrent Word: {current_word}"
    word_var.set(display_text)

def open_gesture_guide():
    """Open the gesture guide document."""
    try:
        file_path = "custom_hand_gestures.docx"
        if os.path.exists(file_path):
            if os.name == 'nt':  # Windows
                os.startfile(file_path)
            else:  # macOS and Linux
                subprocess.run(['open', file_path] if sys.platform == 'darwin' else ['xdg-open', file_path])
        else:
            file_path = filedialog.askopenfilename(
                title="Select Gesture Guide Document",
                filetypes=[("Word Documents", "*.docx"), ("All Files", "*.*")]
            )
            if file_path:
                if os.name == 'nt':  # Windows
                    os.startfile(file_path)
                else:  # macOS and Linux
                    subprocess.run(['open', file_path] if sys.platform == 'darwin' else ['xdg-open', file_path])
    except Exception as e:
        print(f"Error opening document: {e}")

def save_to_file():
    """Save the current sentence to file."""
    global current_sentence
    if current_sentence.strip():
        with open("saved_sentences.txt", "a") as f:
            f.write(current_sentence + "\n")
        current_sentence = ""
        update_word_display()
        update_suggestions()

def backspace():
    """Remove last character from current word."""
    global current_word
    if current_word:
        current_word = current_word[:-1]
        update_word_display()
        update_suggestions()

def clear_word():
    """Clear the current word."""
    global current_word
    current_word = ""
    update_word_display()
    update_suggestions()

def clear_sentence():
    """Clear the entire sentence."""
    global current_word, current_sentence
    current_word = ""
    current_sentence = ""
    update_word_display()
    update_suggestions()

def add_space():
    """Add space to the current word/sentence."""
    global current_word, current_sentence
    if current_word:
        current_sentence += current_word + " "
        current_word = ""
    elif current_sentence:  # Allow adding space even with empty current word
        current_sentence += " "
    update_word_display()
    update_suggestions()
    window.update_idletasks()

def update_suggestions():
    """Update word suggestions based on current word."""
    suggestion_listbox.delete(0, tk.END)
    if current_word:
        suggestions = [word for word in word_list if word.startswith(current_word.lower())]
        for word in sorted(suggestions, key=len)[:10]:
            suggestion_listbox.insert(tk.END, word)

def update_frame():
    """Process video frames and update GUI."""
    global current_word, detected_letter, last_prediction_time
    
    ret, frame = cap.read()
    if not ret:
        return
    
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    canvas.delete("all")
    
    current_time = time.time()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            data_aux = []
            x_ = []
            y_ = []
            
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
            
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))
            
            try:
                prediction = int(model.predict([np.asarray(data_aux)])[0])
                hand_side = "Left" if prediction < 26 else "Right"
                hand_side_var.set(f"Hand: {hand_side}")
                
                if current_time - last_prediction_time > prediction_delay:
                    if prediction == 53:
                        detected_letter = ' '
                        prediction_var.set("Detected: SPACE")
                        add_space()
                    elif 0 <= prediction < 26:
                        detected_letter = chr(prediction + 65)
                        prediction_var.set(f"Detected: {detected_letter} (Left)")
                    elif 26 <= prediction < 52:
                        detected_letter = chr(prediction - 26 + 65)
                        prediction_var.set(f"Detected: {detected_letter} (Right)")
                    else:
                        detected_letter = ''
                        prediction_var.set("Detected: ?")
                    
                    last_prediction_time = current_time
                
                for landmark in hand_landmarks.landmark:
                    cx, cy = int(landmark.x * 300), int(landmark.y * 300)
                    canvas.create_oval(cx-3, cy-3, cx+3, cy+3, 
                                    fill='blue' if prediction < 26 else 'red', 
                                    outline='black')
                    
            except (ValueError, IndexError) as e:
                print(f"Prediction error: {e}")
                continue
    
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)
    
    window.after(10, update_frame)

def confirm_letter(event=None):
    """Add detected letter to current word."""
    global current_word, detected_letter
    if detected_letter and detected_letter != ' ':
        current_word += detected_letter
        update_word_display()
        update_suggestions()

def select_suggestion(event):
    """Select a word from suggestions."""
    global current_word, current_sentence
    if suggestion_listbox.curselection():
        selected_word = suggestion_listbox.get(suggestion_listbox.curselection())
        current_sentence += selected_word + ' '
        current_word = ""
        update_word_display()
        update_suggestions()
        
# Left column buttons - all same width
case_button = Button(left_button_column, text="Uppercase", command=toggle_case, **button_style)
case_button.pack(fill='x', pady=5, ipady=3)  # Added internal padding
Button(left_button_column, text="Space", command=add_space, **button_style).pack(fill='x', pady=5, ipady=3)
Button(left_button_column, text="Backspace", command=backspace, **button_style).pack(fill='x', pady=5, ipady=3)
Button(left_button_column, text="Gesture Guide", command=open_gesture_guide, **button_style).pack(fill='x', pady=5, ipady=3)

# Right column buttons - all same width
Button(right_button_column, text="Clear Word", command=clear_word, **button_style).pack(fill='x', pady=5, ipady=3)
Button(right_button_column, text="Clear Sentence", command=clear_sentence, **button_style).pack(fill='x', pady=5, ipady=3)
Button(right_button_column, text="Save", command=save_to_file, **button_style).pack(fill='x', pady=5, ipady=3)

# Modify the confirm_letter function to respect case
def confirm_letter(event=None):
    """Add detected letter to current word with case handling"""
    global current_word, detected_letter
    if detected_letter and detected_letter != ' ':
        if current_case:
            current_word += detected_letter.upper()
        else:
            current_word += detected_letter.lower()
        update_word_display()
        update_suggestions()

# Add keyboard input handling
def on_key_press(event):
    """Handle keyboard input with case sensitivity"""
    char = event.char
    if char.isalpha():
        if current_case:
            current_word += char.upper()
        else:
            current_word += char.lower()
        update_word_display()
        update_suggestions()
    elif char == ' ':
        add_space()
    elif event.keysym == 'BackSpace':
        backspace()

# Update the key bindings
window.bind('<Key>', on_key_press)
window.bind('<Return>', confirm_letter)
window.bind('<space>', lambda e: add_space())
window.bind('<Escape>', lambda e: window.destroy())
suggestion_listbox.bind('<Double-Button-1>', select_suggestion)
window.bind('<Down>', lambda e: suggestion_listbox.focus())

# Initialize variables
prediction_var.set("Detected: ")
hand_side_var.set("Hand: ")
update_word_display()

# Start video processing
update_frame()

# Start main loop
window.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()