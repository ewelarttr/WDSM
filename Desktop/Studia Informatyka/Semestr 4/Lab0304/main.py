import sys
import matplotlib

matplotlib.use('QtAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QGridLayout, QGroupBox, QTableWidget, QTableWidgetItem,
                               QHeaderView, QProgressBar)
from PySide6.QtCore import QThread, Signal, Qt


class SimulationThread(QThread):
    update_signal = Signal(dict)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self.is_running = True

    def run(self):
        try:
            lam = float(self.p['lambda'])
            n_avg = float(self.p['n_avg'])
            sigma = float(self.p['sigma'])
            num_channels = int(self.p['channels'])
            sim_time = int(self.p['sim_time'])
            max_q = int(self.p['max_q'])

            channels = [0.0] * num_channels
            queue = []
            stats = {'time': [], 'Q': [], 'W': [], 'Ro': []}
            total_served = 0
            total_rejected = 0

            for t in range(sim_time):
                if not self.is_running: break

                num_arrivals = np.random.poisson(lam)
                current_gauss = []
                for _ in range(num_arrivals):
                    duration = np.random.normal(n_avg, sigma)
                    duration = max(1, duration)
                    if len(queue) < max_q:
                        queue.append(duration)
                        current_gauss.append(round(duration, 2))
                    else:
                        total_rejected += 1

                for i in range(num_channels):
                    if channels[i] > 0:
                        channels[i] = max(0, channels[i] - 1)
                    if channels[i] == 0 and len(queue) > 0:
                        channels[i] = queue.pop(0)
                        total_served += 1

                load = sum(1 for c in channels if c > 0) / num_channels
                stats['time'].append(t)
                stats['Q'].append(len(queue))
                stats['Ro'].append(load)
                stats['W'].append(np.mean(queue) if queue else 0)

                self.update_signal.emit({
                    't': t,
                    'channels': list(channels),
                    'stats': stats,
                    'served': total_served,
                    'rejected': total_rejected,
                    'last_gauss': current_gauss,
                    'last_poisson': num_arrivals
                })
                self.msleep(100)
        except Exception as e:
            print(f"Błąd symulacji: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zaawansowany Symulator Stacji Bazowej v2.0")
        self.resize(1400, 900)
        self.setStyleSheet("QMainWindow { background-color: #f0f0f0; }")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()
        params_group = QGroupBox("Parametry wejściowe")
        params_layout = QGridLayout()

        self.inputs = {}
        fields = [
            ('lambda', 'Lambda (λ):', '1.0'),
            ('n_avg', 'Średni czas (N):', '20'),
            ('sigma', 'Odchylenie (σ):', '5'),
            ('channels', 'Liczba kanałów:', '10'),
            ('max_q', 'Max kolejka:', '15'),
            ('sim_time', 'Czas symulacji:', '100')
        ]

        for i, (key, label, default) in enumerate(fields):
            params_layout.addWidget(QLabel(label), i, 0)
            edit = QLineEdit(default)
            params_layout.addWidget(edit, i, 1)
            self.inputs[key] = edit

        params_group.setLayout(params_layout)
        left_panel.addWidget(params_group)

        self.btn_start = QPushButton("URUCHOM SYMULACJĘ")
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: bold; height: 50px; border-radius: 5px; } QPushButton:disabled { background-color: #95a5a6; }")
        self.btn_start.clicked.connect(self.start_sim)
        left_panel.addWidget(self.btn_start)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["T", "Poisson", "Obsłużone", "Odrzucone"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        left_panel.addWidget(QLabel("Log zdarzeń:"))
        left_panel.addWidget(self.table)

        main_layout.addLayout(left_panel, 1)

        mid_panel = QVBoxLayout()

        chan_group = QGroupBox("Status Kanałów")
        self.chan_grid = QGridLayout()
        self.chan_labels = []
        for i in range(12):
            lbl = QLabel("IDLE")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(60, 60)
            lbl.setStyleSheet("background: #bdc3c7; border-radius: 10px; font-weight: bold;")
            self.chan_grid.addWidget(lbl, i // 3, i % 3)
            self.chan_labels.append(lbl)
        chan_group.setLayout(self.chan_grid)
        mid_panel.addWidget(chan_group)

        mid_panel.addWidget(QLabel("Obciążenie kolejki:"))
        self.q_bar = QProgressBar()
        self.q_bar.setStyleSheet("QProgressBar::chunk { background-color: #e67e22; }")
        mid_panel.addWidget(self.q_bar)

        self.status_lbl = QLabel("Gotowy...")
        self.status_lbl.setStyleSheet("font-size: 14px; color: #2c3e50;")
        mid_panel.addWidget(self.status_lbl)
        mid_panel.addStretch()

        main_layout.addLayout(mid_panel, 1)

        self.fig, (self.ax_q, self.ax_w, self.ax_ro) = plt.subplots(3, 1, figsize=(6, 10))
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, 2)

    def start_sim(self):
        params = {k: v.text() for k, v in self.inputs.items()}
        self.table.setRowCount(0)
        self.btn_start.setEnabled(False)
        self.thread = SimulationThread(params)
        self.thread.update_signal.connect(self.update_ui)
        self.thread.finished.connect(lambda: self.btn_start.setEnabled(True))
        self.thread.start()

    def update_ui(self, data):
        self.ax_q.clear()
        self.ax_q.plot(data['stats']['time'], data['stats']['Q'], 'r', label='Kolejka (Q)')
        self.ax_q.legend()

        self.ax_w.clear()
        self.ax_w.plot(data['stats']['time'], data['stats']['W'], 'b', label='Czas oczek. (W)')
        self.ax_w.legend()

        self.ax_ro.clear()
        self.ax_ro.fill_between(data['stats']['time'], data['stats']['Ro'], color='green', alpha=0.3,
                                label='Obciążenie (Ro)')
        self.ax_ro.set_ylim(0, 1.1)
        self.ax_ro.legend()

        self.canvas.draw()

        for i, val in enumerate(data['channels']):
            if i < len(self.chan_labels):
                if val > 0:
                    self.chan_labels[i].setText(f"{int(val)}s")
                    self.chan_labels[i].setStyleSheet(
                        "background: #e74c3c; color: white; border-radius: 10px; font-weight: bold;")
                else:
                    self.chan_labels[i].setText("FREE")
                    self.chan_labels[i].setStyleSheet(
                        "background: #2ecc71; color: white; border-radius: 10px; font-weight: bold;")

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(data['t'])))
        self.table.setItem(row, 1, QTableWidgetItem(str(data['last_poisson'])))
        self.table.setItem(row, 2, QTableWidgetItem(str(data['served'])))
        self.table.setItem(row, 3, QTableWidgetItem(str(data['rejected'])))
        self.table.scrollToBottom()

        max_q = int(self.inputs['max_q'].text())
        self.q_bar.setMaximum(max_q)
        self.q_bar.setValue(len(data['stats']['Q']))

        self.status_lbl.setText(f"Czas: {data['t']}s | Obsłużono: {data['served']} | Odrzucono: {data['rejected']}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())