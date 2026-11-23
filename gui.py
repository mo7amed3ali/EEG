import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import threading
import time

from inference import predict_eeg, EEG_CHANNELS


# ================= COLORS =================

BG = "#080D18"
CARD = "#111B2E"
CARD2 = "#16243A"
CYAN = "#00E5FF"
BLUE = "#2979FF"
PURPLE = "#9C27B0"
GREEN = "#00E676"
WHITE = "#FFFFFF"
GRAY = "#8A9BB5"
RED = "#FF5252"


# ================= MAIN WINDOW =================

root = tk.Tk()
root.title("NEUROVISION AI | EEG Classification")
root.geometry("1250x820")
root.configure(bg=BG)
root.minsize(1100, 750)


# ================= VARIABLES =================

file_path = None
epochs = []
current_epoch = None
animation_offset = 0
prediction_running = False


# ================= HELPER FUNCTIONS =================

def make_label(parent, text, size=12, color=WHITE, bold=False):
    return tk.Label(
        parent,
        text=text,
        bg=parent.cget("bg"),
        fg=color,
        font=("Segoe UI", size, "bold" if bold else "normal")
    )


def make_card(parent, width=None, height=None):
    card = tk.Frame(
        parent,
        bg=CARD,
        highlightbackground="#263A56",
        highlightthickness=1,
        width=width,
        height=height
    )
    return card


# ================= HEADER =================

header = tk.Frame(root, bg=BG)
header.pack(fill="x", padx=30, pady=(20, 10))

title = tk.Label(
    header,
    text="NEUROVISION AI",
    bg=BG,
    fg=CYAN,
    font=("Segoe UI", 27, "bold")
)
title.pack(side="left")

subtitle = tk.Label(
    header,
    text="EEG MOTOR IMAGERY CLASSIFICATION",
    bg=BG,
    fg=GRAY,
    font=("Segoe UI", 10)
)
subtitle.pack(side="left", padx=20, pady=(12, 0))

status_label = tk.Label(
    header,
    text="● MODEL READY",
    bg=BG,
    fg=GREEN,
    font=("Segoe UI", 10, "bold")
)
status_label.pack(side="right", pady=15)


# ================= TOP INFORMATION =================

info_frame = tk.Frame(root, bg=BG)
info_frame.pack(fill="x", padx=30, pady=10)


def create_stat_card(parent, title, value):
    card = make_card(parent, height=100)
    card.pack(side="left", fill="both", expand=True, padx=6)

    tk.Label(
        card,
        text=title,
        bg=CARD,
        fg=GRAY,
        font=("Segoe UI", 10)
    ).pack(pady=(17, 5))

    value_label = tk.Label(
        card,
        text=value,
        bg=CARD,
        fg=CYAN,
        font=("Segoe UI", 21, "bold")
    )
    value_label.pack()

    return value_label


epochs_value = create_stat_card(info_frame, "TOTAL EPOCHS", "--")
channels_value = create_stat_card(info_frame, "EEG CHANNELS", "22")
samples_value = create_stat_card(info_frame, "SAMPLES / EPOCH", "201")
duration_value = create_stat_card(info_frame, "EPOCH DURATION", "0.8 sec")


# ================= MAIN CONTENT =================

main_frame = tk.Frame(root, bg=BG)
main_frame.pack(fill="both", expand=True, padx=30, pady=15)

left_frame = tk.Frame(main_frame, bg=BG)
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

right_frame = tk.Frame(main_frame, bg=BG, width=390)
right_frame.pack(side="right", fill="both", padx=(10, 0))
right_frame.pack_propagate(False)


# ================= EEG SIGNAL CARD =================

signal_card = make_card(left_frame, height=310)
signal_card.pack(fill="both", expand=True, pady=(0, 12))

tk.Label(
    signal_card,
    text="EEG SIGNAL VISUALIZATION",
    bg=CARD,
    fg=CYAN,
    font=("Segoe UI", 13, "bold")
).pack(anchor="w", padx=20, pady=15)

signal_canvas = tk.Canvas(
    signal_card,
    bg="#0B1424",
    highlightthickness=0,
    height=220
)
signal_canvas.pack(fill="both", expand=True, padx=15, pady=(0, 15))


# ================= SIGNAL ANIMATION =================

wave_signal = None


def animate_signal():
    global animation_offset

    signal_canvas.delete("wave")

    width = signal_canvas.winfo_width()
    height = signal_canvas.winfo_height()

    if width < 10 or height < 10:
        root.after(50, animate_signal)
        return

    signal_canvas.create_line(
        0, height // 2,
        width, height // 2,
        fill="#263A56",
        dash=(3, 5)
    )

    if wave_signal is not None and len(wave_signal) > 1:

        points = []

        for i, value in enumerate(wave_signal):
            x = i * width / (len(wave_signal) - 1)

            y = height / 2 - value * height * 0.35

            points.extend([x, y])

        signal_canvas.create_line(
            points,
            fill=CYAN,
            width=2,
            smooth=True,
            tags="wave"
        )

    else:

        points = []

        for x in range(0, width, 3):
            y = (
                height / 2
                + np.sin((x + animation_offset) * 0.035) * 20
                + np.sin((x + animation_offset) * 0.09) * 8
            )

            points.extend([x, y])

        signal_canvas.create_line(
            points,
            fill=CYAN,
            width=2,
            smooth=True,
            tags="wave"
        )

    animation_offset += 5

    root.after(40, animate_signal)


# ================= UPLOAD CARD =================

upload_card = make_card(left_frame, height=140)
upload_card.pack(fill="x", pady=(0, 12))

tk.Label(
    upload_card,
    text="DATASET INPUT",
    bg=CARD,
    fg=CYAN,
    font=("Segoe UI", 13, "bold")
).pack(anchor="w", padx=20, pady=(15, 5))

file_label = tk.Label(
    upload_card,
    text="No dataset selected",
    bg=CARD,
    fg=GRAY,
    font=("Segoe UI", 10)
)
file_label.pack(anchor="w", padx=20, pady=5)

upload_button = tk.Button(
    upload_card,
    text="UPLOAD EEG CSV",
    bg=BLUE,
    fg=WHITE,
    activebackground=CYAN,
    activeforeground=BG,
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    cursor="hand2",
    padx=20,
    pady=8,
    command=lambda: upload_file()
)
upload_button.pack(anchor="e", padx=20, pady=5)


# ================= PREDICTION CARD =================

prediction_card = make_card(right_frame, height=230)
prediction_card.pack(fill="x", pady=(0, 12))

tk.Label(
    prediction_card,
    text="CLASSIFICATION RESULT",
    bg=CARD,
    fg=CYAN,
    font=("Segoe UI", 13, "bold")
).pack(anchor="w", padx=20, pady=15)

prediction_label = tk.Label(
    prediction_card,
    text="WAITING...",
    bg=CARD,
    fg=GRAY,
    font=("Segoe UI", 25, "bold")
)
prediction_label.pack(pady=20)

prediction_description = tk.Label(
    prediction_card,
    text="Upload EEG data to start analysis",
    bg=CARD,
    fg=GRAY,
    font=("Segoe UI", 9)
)
prediction_description.pack()


# ================= PROBABILITY CARD =================

probability_card = make_card(right_frame)
probability_card.pack(fill="both", expand=True, pady=(0, 12))

tk.Label(
    probability_card,
    text="CLASS PROBABILITIES",
    bg=CARD,
    fg=CYAN,
    font=("Segoe UI", 13, "bold")
).pack(anchor="w", padx=20, pady=15)

probability_bars = {}

for class_name in ["foot", "left", "right", "tongue"]:

    row = tk.Frame(probability_card, bg=CARD)
    row.pack(fill="x", padx=20, pady=9)

    tk.Label(
        row,
        text=class_name.upper(),
        bg=CARD,
        fg=WHITE,
        font=("Segoe UI", 9, "bold"),
        width=9,
        anchor="w"
    ).pack(side="left")

    bar_bg = tk.Frame(row, bg="#263A56", height=12)
    bar_bg.pack(side="left", fill="x", expand=True, padx=8)

    bar = tk.Frame(bar_bg, bg=CYAN, height=12, width=0)
    bar.place(x=0, y=0)

    value_label = tk.Label(
        row,
        text="0.00%",
        bg=CARD,
        fg=GRAY,
        font=("Segoe UI", 9),
        width=8
    )
    value_label.pack(side="right")

    probability_bars[class_name] = (bar_bg, bar, value_label)


# ================= PROGRESS =================

progress_card = make_card(right_frame, height=90)
progress_card.pack(fill="x")

tk.Label(
    progress_card,
    text="ANALYSIS PROGRESS",
    bg=CARD,
    fg=CYAN,
    font=("Segoe UI", 10, "bold")
).pack(anchor="w", padx=15, pady=(12, 5))

progress_canvas = tk.Canvas(
    progress_card,
    height=12,
    bg="#263A56",
    highlightthickness=0
)
progress_canvas.pack(fill="x", padx=15)

progress_text = tk.Label(
    progress_card,
    text="0%",
    bg=CARD,
    fg=GRAY,
    font=("Segoe UI", 9)
)
progress_text.pack(anchor="e", padx=15, pady=5)


def update_progress(value):
    progress_canvas.delete("all")

    width = progress_canvas.winfo_width()

    if width > 1:
        progress_canvas.create_rectangle(
            0, 0,
            width * value / 100,
            12,
            fill=CYAN,
            outline=""
        )

    progress_text.config(text=f"{value:.0f}%")


# ================= DATA PROCESSING =================

def extract_epochs(df):

    if all(col in df.columns for col in EEG_CHANNELS):

        data = df[EEG_CHANNELS].copy()

    else:

        # استخدام أول 22 عمودًا رقميًا فقط
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 22:
            raise ValueError(
                "CSV must contain at least 22 numeric EEG channels."
            )

        data = numeric_df.iloc[:, :22].copy()

        data.columns = EEG_CHANNELS

    # تقسيم البيانات باستخدام patient و epoch
    if "patient" in df.columns and "epoch" in df.columns:

        groups = df.groupby(
            ["patient", "epoch"],
            sort=False
        ).groups

        result = []

        for indexes in groups.values():

            epoch_data = data.loc[indexes].to_numpy()

            if len(epoch_data) != 201:
                raise ValueError(
                    f"An epoch contains {len(epoch_data)} samples instead of 201."
                )

            result.append(epoch_data.T)

        return result

    # تقسيم البيانات إلى Epochs كل واحدة 201 sample
    raw_data = data.to_numpy()

    if len(raw_data) % 201 != 0:
        raise ValueError(
            f"Total samples ({len(raw_data)}) "
            "must be divisible by 201."
        )

    result = []

    for start in range(0, len(raw_data), 201):

        epoch = raw_data[start:start + 201]

        result.append(epoch.T)

    return result


# ================= UPLOAD FUNCTION =================

def upload_file():

    global file_path, epochs, wave_signal, current_epoch

    path = filedialog.askopenfilename(
        title="Select EEG Dataset",
        filetypes=[
            ("CSV Files", "*.csv"),
            ("All Files", "*.*")
        ]
    )

    if not path:
        return

    try:

        df = pd.read_csv(path)

        loaded_epochs = extract_epochs(df)

        if not loaded_epochs:
            raise ValueError("No valid EEG epochs found.")

        for epoch in loaded_epochs:

            if epoch.shape != (22, 201):
                raise ValueError(
                    f"Invalid epoch shape: {epoch.shape}"
                )

            if not np.isfinite(epoch).all():
                raise ValueError(
                    "Dataset contains NaN or infinite values."
                )

        file_path = path
        epochs = loaded_epochs
        current_epoch = epochs[0]

        # عرض أول قناة من أول Epoch
        wave_signal = current_epoch[0].astype(float)

        max_value = np.max(np.abs(wave_signal))

        if max_value > 0:
            wave_signal = wave_signal / max_value

        epochs_value.config(text=str(len(epochs)))
        channels_value.config(text="22")
        samples_value.config(text="201")
        duration_value.config(text="0.8 sec")

        file_label.config(
            text=f"Loaded: {path.split('/')[-1]}",
            fg=GREEN
        )

        prediction_label.config(
            text="READY",
            fg=CYAN
        )

        prediction_description.config(
            text=f"{len(epochs)} EEG epochs loaded successfully"
        )

        update_progress(0)

        messagebox.showinfo(
            "Dataset Loaded",
            f"Successfully loaded {len(epochs)} EEG epochs."
        )

    except Exception as error:

        messagebox.showerror(
            "Dataset Error",
            str(error)
        )


# ================= PROBABILITY UPDATE =================

def update_probabilities(probabilities):

    for class_name, probability in probabilities.items():

        bar_bg, bar, value_label = probability_bars[class_name]

        width = bar_bg.winfo_width()

        bar.place(
            x=0,
            y=0,
            width=max(0, int(width * probability)),
            height=12
        )

        value_label.config(
            text=f"{probability * 100:.2f}%"
        )


# ================= RUN MODEL =================

def run_prediction():

    global prediction_running

    if prediction_running:
        return

    if not epochs:

        messagebox.showwarning(
            "No Dataset",
            "Please upload an EEG CSV file first."
        )
        return

    prediction_running = True

    analyze_button.config(
        state="disabled",
        text="ANALYZING..."
    )

    status_label.config(
        text="● ANALYZING EEG",
        fg=CYAN
    )

    prediction_label.config(
        text="PROCESSING...",
        fg=CYAN
    )

    threading.Thread(
        target=prediction_worker,
        daemon=True
    ).start()


def prediction_worker():

    try:

        all_probabilities = []

        total = len(epochs)

        for index, epoch in enumerate(epochs):

            result = predict_eeg(epoch)

            all_probabilities.append(
                result["probabilities"]
            )

            progress = ((index + 1) / total) * 100

            root.after(
                0,
                lambda p=progress: update_progress(p)
            )

        # متوسط الاحتمالات لكل Epoch
        average_probabilities = {}

        for class_name in ["foot", "left", "right", "tongue"]:

            average_probabilities[class_name] = float(
                np.mean([
                    item[class_name]
                    for item in all_probabilities
                ])
            )

        final_prediction = max(
            average_probabilities,
            key=average_probabilities.get
        )

        root.after(
            0,
            lambda: show_results(
                final_prediction,
                average_probabilities
            )
        )

    except Exception as error:

        root.after(
            0,
            lambda: prediction_error(error)
        )


# ================= SHOW RESULTS =================

def show_results(prediction, probabilities):

    global prediction_running

    prediction_label.config(
        text=prediction.upper(),
        fg=GREEN
    )

    prediction_description.config(
        text="Final prediction based on average epoch probabilities"
    )

    update_probabilities(probabilities)

    status_label.config(
        text="● ANALYSIS COMPLETE",
        fg=GREEN
    )

    analyze_button.config(
        state="normal",
        text="RUN CLASSIFICATION"
    )

    prediction_running = False

    messagebox.showinfo(
        "Analysis Complete",
        f"Final Prediction: {prediction.upper()}"
    )


def prediction_error(error):

    global prediction_running

    prediction_running = False

    analyze_button.config(
        state="normal",
        text="RUN CLASSIFICATION"
    )

    status_label.config(
        text="● ANALYSIS ERROR",
        fg=RED
    )

    messagebox.showerror(
        "Prediction Error",
        str(error)
    )


# ================= ANALYZE BUTTON =================

analyze_button = tk.Button(
    root,
    text="RUN CLASSIFICATION",
    bg=CYAN,
    fg=BG,
    activebackground=BLUE,
    activeforeground=WHITE,
    font=("Segoe UI", 12, "bold"),
    relief="flat",
    cursor="hand2",
    padx=30,
    pady=12,
    command=run_prediction
)

analyze_button.pack(pady=(0, 20))


# ================= FOOTER =================

footer = tk.Label(
    root,
    text="NEUROVISION AI  |  EEG MOTOR IMAGERY PROJECT",
    bg=BG,
    fg=GRAY,
    font=("Segoe UI", 9)
)

footer.pack(pady=(0, 12))


# ================= START APPLICATION =================

animate_signal()

root.mainloop()