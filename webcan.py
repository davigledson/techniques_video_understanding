import cv2
import numpy as np
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import logging

# Configure logging to suppress unnecessary messages
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class OpticalFlowApp:
    def __init__(self, master):
        self.master = master
        master.title("Análise de Fluxo Óptico com Seleção de Câmera")
        master.geometry("1400x800")

        # Variáveis de estado
        self.cap = None
        self.running = False
        self.prev_gray = None
        self.prev_frame = None
        self.mask = None
        self.prev_pts = None
        self.track_len = 10
        self.tracks = []
        self.detect_interval = 5
        self.frame_idx = 0
        self.is_closing = False
        self.available_cameras = []

        # Interface
        self.create_widgets()
        self.refresh_cameras()

        # Configura o handler para fechamento
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Frame de controle superior
        control_frame = Frame(self.master)
        control_frame.grid(row=0, column=0, columnspan=4, pady=10, sticky='ew')

        # Dropdown de câmeras
        self.camera_var = StringVar()
        ttk.Label(control_frame, text="Câmera:").pack(side=LEFT, padx=5)
        self.camera_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.camera_var,
            state="readonly",
            width=15
        )
        self.camera_dropdown.pack(side=LEFT, padx=5)

        # Botão para atualizar câmeras
        ttk.Button(
            control_frame,
            text="Atualizar Câmeras",
            command=self.refresh_cameras
        ).pack(side=LEFT, padx=5)

        # Dropdown de técnicas
        self.tech_var = StringVar()
        ttk.Label(control_frame, text="Técnica:").pack(side=LEFT, padx=5)
        self.tech_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.tech_var,
            values=[
                "1. Lucas-Kanade - Rastreia Pontos Chave",
                "2. Farneback - Mapa de Movimento Denso",
                "3. Block Matching - Correspondência de Blocos",
                "4. Compensação - Filtra Movimento Global",
                "5. Diferença - Detecta Áreas em Movimento"
            ],
            state="readonly",
            width=35
        )
        self.tech_dropdown.pack(side=LEFT, padx=5)
        self.tech_dropdown.current(0)

        # Botões de controle
        self.start_btn = ttk.Button(
            control_frame,
            text="Iniciar Webcam",
            command=self.start_webcam
        )
        self.start_btn.pack(side=LEFT, padx=5)

        self.stop_btn = ttk.Button(
            control_frame,
            text="Parar Webcam",
            command=self.stop_webcam,
            state=DISABLED
        )
        self.stop_btn.pack(side=LEFT, padx=5)

        # Visualização
        self.canvas_orig = Canvas(self.master, width=640, height=480, bg='black')
        self.canvas_orig.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.canvas_processed = Canvas(self.master, width=640, height=480, bg='black')
        self.canvas_processed.grid(row=1, column=2, columnspan=2, padx=10, pady=10)

        # Legenda
        self.legend = Label(self.master, text="Lucas-Kanade: Rastreia pontos-chave com trajetórias",
                            font=('Arial', 10))
        self.legend.grid(row=2, column=0, columnspan=4, pady=10)

    def refresh_cameras(self):
        """Detecta e atualiza a lista de câmeras disponíveis"""
        self.available_cameras = self.detect_available_cameras()
        self.camera_dropdown['values'] = self.available_cameras
        if self.available_cameras:
            self.camera_dropdown.current(0)
        else:
            messagebox.showwarning("Aviso", "Nenhuma câmera encontrada!")

    def detect_available_cameras(self, max_test=5):
        """Retorna lista de câmeras disponíveis"""
        available = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(f"Câmera {i}")
                cap.release()
            else:
                cap.release()
                break
        return available if available else ["Câmera 0"]

    def start_webcam(self):
        """Inicia a captura da câmera selecionada"""
        if self.cap or self.is_closing:
            return

        try:
            # Obtém o índice da câmera selecionada
            selected_cam = self.camera_var.get()
            if not selected_cam:
                messagebox.showerror("Erro", "Selecione uma câmera primeiro!")
                return

            cam_index = int(selected_cam.split()[-1])
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                messagebox.showerror("Erro", f"Não foi possível acessar {selected_cam}!")
                self.cap = None
                return

            # Configurações recomendadas
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.running = True
            self.start_btn.config(state=DISABLED)
            self.stop_btn.config(state=NORMAL)

            # Inicializa máscara para trajetórias
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Erro", "Não foi possível ler o frame da webcam!")
                self.stop_webcam()
                return

            self.mask = np.zeros_like(frame)
            self.prev_frame = frame.copy()
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.tracks = []
            self.frame_idx = 0

            self.update_frame()

        except Exception as e:
            logger.error(f"Erro ao iniciar webcam: {str(e)}")
            messagebox.showerror("Erro", f"Falha ao iniciar câmera: {str(e)}")
            if self.cap:
                self.cap.release()
                self.cap = None

    def stop_webcam(self):
        """Para a captura da webcam"""
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None

        if not self.is_closing:
            self.start_btn.config(state=NORMAL)
            self.stop_btn.config(state=DISABLED)

    def update_frame(self):
        """Atualiza o frame da webcam e aplica a técnica selecionada"""
        if not self.running or self.is_closing or not self.cap:
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.master.after(30, self.update_frame)
                return

            frame = cv2.flip(frame, 1)  # Espelha a imagem
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tech = self.tech_var.get()

            # Mostra frame original
            self.show_frame(self.canvas_orig, frame, "Original")

            # Aplica a técnica selecionada
            processed = self.process_optical_flow(frame, gray, tech)

            # Exibe resultado processado
            self.show_frame(self.canvas_processed, processed, tech.split("-")[0])

            # Atualiza frame anterior
            self.prev_gray = gray.copy()
            self.prev_frame = frame.copy()
            self.frame_idx += 1

            self.master.after(30, self.update_frame)

        except Exception as e:
            logger.error(f"Erro ao atualizar frame: {str(e)}")
            self.stop_webcam()

    def process_optical_flow(self, frame, gray, tech):
        """Aplica a técnica de fluxo óptico selecionada"""
        try:
            if "Lucas-Kanade" in tech:
                self.legend.config(text="Lucas-Kanade: Rastreia pontos-chave com trajetórias (verde)")
                return self.apply_lucas_kanade(frame, gray)
            elif "Farneback" in tech:
                self.legend.config(
                    text="Farneback: Mapa de cores mostra direção (matiz) e intensidade (brilho) do movimento")
                return self.apply_farneback(gray)
            elif "Block Matching" in tech:
                self.legend.config(text="Block Matching: Setas mostram movimento entre blocos de pixels")
                return self.apply_block_matching(frame, gray)
            elif "Compensação" in tech:
                self.legend.config(
                    text="Compensação: Áreas verdes mostram movimento relativo após remoção do movimento médio")
                return self.apply_compensation(frame, gray)
            elif "Diferença" in tech:
                self.legend.config(text="Diferença: Retângulos verdes destacam áreas com movimento significativo")
                return self.apply_motion_tracking(frame)
            else:
                return frame.copy()
        except Exception as e:
            logger.error(f"Erro ao processar técnica {tech}: {str(e)}")
            return frame.copy()

    def apply_lucas_kanade(self, frame, gray):
        vis = frame.copy()

        # Detecta novos pontos a cada N frames
        if len(self.tracks) == 0 or self.frame_idx % self.detect_interval == 0:
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            corners = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            if corners is not None:
                for x, y in corners.reshape(-1, 2):
                    self.tracks.append([(x, y)])

        if len(self.tracks) > 0:
            # Prepara pontos para rastreamento
            img0, img1 = self.prev_gray, gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

            # Calcula fluxo óptico
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                img0, img1, p0, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # Verificação reversa para consistência
            p0r, st, _ = cv2.calcOpticalFlowPyrLK(
                img1, img0, p1, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # Filtra pontos inconsistentes
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1

            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

            self.tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

        return vis

    def apply_farneback(self, gray):
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )

        # Visualização em HSV
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def apply_block_matching(self, frame, gray):
        vis = frame.copy()
        h, w = gray.shape
        block_size = 32
        search_range = 16

        for y in range(0, h - block_size, block_size * 2):
            for x in range(0, w - block_size, block_size * 2):
                block = self.prev_gray[y:y + block_size, x:x + block_size]
                min_diff = float('inf')
                best_dx, best_dy = 0, 0

                for dy in range(-search_range, search_range + 1, 2):
                    for dx in range(-search_range, search_range + 1, 2):
                        if 0 <= x + dx < w - block_size and 0 <= y + dy < h - block_size:
                            curr_block = gray[y + dy:y + dy + block_size, x + dx:x + dx + block_size]
                            diff = np.sum(np.abs(block - curr_block))
                            if diff < min_diff:
                                min_diff = diff
                                best_dx, best_dy = dx, dy

                if abs(best_dx) > 2 or abs(best_dy) > 2:
                    center_x = x + block_size // 2
                    center_y = y + block_size // 2
                    cv2.arrowedLine(vis,
                                    (center_x, center_y),
                                    (center_x + best_dx, center_y + best_dy),
                                    (0, 255, 0), 1, tipLength=0.3)
        return vis

    def apply_compensation(self, frame, gray):
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        mean_flow = np.mean(flow, axis=(0, 1))
        h, w = gray.shape
        M = np.float32([[1, 0, -mean_flow[0]], [0, 1, -mean_flow[1]]])
        compensated = cv2.warpAffine(frame, M, (w, h))

        diff = cv2.absdiff(frame, compensated)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        result = frame.copy()
        result[mask > 0] = (0, 255, 0)
        return result

    def apply_motion_tracking(self, frame):
        diff = cv2.absdiff(self.prev_frame, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis, f"Movimento: {area:.0f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis

    def show_frame(self, canvas, img, title=""):
        try:
            img = cv2.resize(img, (640, 480))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(img))

            canvas.delete("all")
            canvas.create_image(0, 0, anchor=NW, image=photo)
            canvas.image = photo
            canvas.create_text(320, 20, text=title, fill="white",
                               font=("Arial", 14, "bold"), anchor=N)
        except Exception as e:
            logger.error(f"Erro ao exibir frame: {str(e)}")

    def on_closing(self):
        """Handler para fechamento da janela"""
        self.is_closing = True
        self.stop_webcam()
        self.master.destroy()

    def __del__(self):
        """Destrutor - libera recursos"""
        if not self.is_closing:
            self.stop_webcam()


if __name__ == "__main__":
    root = Tk()
    app = OpticalFlowApp(root)
    root.mainloop()