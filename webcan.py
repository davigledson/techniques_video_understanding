import cv2
import numpy as np
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


class OpticalFlowApp:
    def __init__(self, master):
        self.master = master
        master.title("Análise de Fluxo Óptico")
        master.geometry("1200x800")

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

        # Interface
        self.create_widgets()

    def create_widgets(self):
        # Cabeçalho
        Label(self.master, text="Técnica de Fluxo Óptico:", font=('Arial', 12)).grid(
            row=0, column=0, padx=10, pady=10, sticky='w')

        # Dropdown de técnicas
        self.tech_var = StringVar()
        self.tech_choices = [
            "1. Lucas-Kanade - Rastreia Pontos Chave",
            "2. Farneback - Mapa de Movimento Denso",
            "3. Block Matching - Correspondência de Blocos",
            "4. Compensação - Filtra Movimento Global",
            "5. Diferença - Detecta Áreas em Movimento"
        ]
        self.tech_dropdown = ttk.Combobox(
            self.master, textvariable=self.tech_var, values=self.tech_choices)
        self.tech_dropdown.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
        self.tech_dropdown.current(0)

        # Controles
        self.start_btn = Button(self.master, text="Iniciar Webcam", command=self.start_webcam)
        self.start_btn.grid(row=0, column=2, padx=10, pady=10)

        self.stop_btn = Button(self.master, text="Parar Webcam", command=self.stop_webcam, state=DISABLED)
        self.stop_btn.grid(row=0, column=3, padx=10, pady=10)

        # Visualização
        self.canvas_orig = Canvas(self.master, width=640, height=480, bg='black')
        self.canvas_orig.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.canvas_processed = Canvas(self.master, width=640, height=480, bg='black')
        self.canvas_processed.grid(row=1, column=2, columnspan=2, padx=10, pady=10)

        # Legenda
        self.legend = Label(self.master, text="Lucas-Kanade: Rastreia pontos-chave com trajetórias",
                            font=('Arial', 10))
        self.legend.grid(row=2, column=0, columnspan=4, pady=10)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Webcam não encontrada!")
            return

        self.running = True
        self.start_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)

        # Inicializa máscara para trajetórias
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Erro", "Não foi possível ler o frame da webcam!")
            return

        self.mask = np.zeros_like(frame)
        self.prev_frame = frame.copy()
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.tracks = []
        self.frame_idx = 0

        self.update_frame()

    def stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.master.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)  # Espelha a imagem para parecer mais natural
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tech = self.tech_var.get()

        # Mostra frame original
        self.show_frame(self.canvas_orig, frame, "Original")

        # Aplica a técnica selecionada
        try:
            if "Lucas-Kanade" in tech:
                self.legend.config(text="Lucas-Kanade: Rastreia pontos-chave com trajetórias (verde)")
                processed = self.apply_lucas_kanade(frame, gray)
            elif "Farneback" in tech:
                self.legend.config(
                    text="Farneback: Mapa de cores mostra direção (matiz) e intensidade (brilho) do movimento")
                processed = self.apply_farneback(gray)
            elif "Block Matching" in tech:
                self.legend.config(text="Block Matching: Setas mostram movimento entre blocos de pixels")
                processed = self.apply_block_matching(frame, gray)
            elif "Compensação" in tech:
                self.legend.config(
                    text="Compensação: Áreas verdes mostram movimento relativo após remoção do movimento médio")
                processed = self.apply_compensation(frame, gray)
            elif "Diferença" in tech:
                self.legend.config(text="Diferença: Retângulos verdes destacam áreas com movimento significativo")
                processed = self.apply_motion_tracking(frame)
            else:
                processed = frame.copy()
        except Exception as e:
            print(f"Erro ao processar frame: {e}")
            processed = frame.copy()

        self.show_frame(self.canvas_processed, processed, tech.split("-")[0])

        # Atualiza frame anterior
        self.prev_gray = gray.copy()
        self.prev_frame = frame.copy()
        self.frame_idx += 1

        self.master.after(30, self.update_frame)

    def apply_lucas_kanade(self, frame, gray):
        vis = frame.copy()

        # Detecta novos pontos a cada N frames
        if len(self.tracks) == 0 or self.frame_idx % self.detect_interval == 0:
            # Usar o detector de cantos Shi-Tomasi
            feature_params = dict(maxCorners=100,
                                  qualityLevel=0.3,
                                  minDistance=7,
                                  blockSize=7)

            corners = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            if corners is not None:
                for x, y in corners.reshape(-1, 2):
                    self.tracks.append([(x, y)])

        if len(self.tracks) > 0:
            # Prepara pontos para rastreamento
            img0, img1 = self.prev_gray, gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

            # Calcula fluxo óptico
            p1, st, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                 winSize=(15, 15),
                                                 maxLevel=2,
                                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            # Rastreia pontos reversos para verificação
            p0r, st, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                  winSize=(15, 15),
                                                  maxLevel=2,
                                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            # Verifica a consistência dos pontos
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

                # Desenha o ponto atual
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

            self.tracks = new_tracks

            # Desenha as trajetórias
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

        # Visualização em HSV (direção = matiz, magnitude = brilho)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Direção (matiz)
        hsv[..., 1] = 255  # Saturação máxima
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude (brilho)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def apply_block_matching(self, frame, gray):
        vis = frame.copy()
        h, w = gray.shape
        block_size = 32  # Aumentei o tamanho do bloco para melhor desempenho
        search_range = 16  # Aumentei o alcance da busca

        # Reduz a quantidade de blocos para melhorar o desempenho
        for y in range(0, h - block_size, block_size * 2):
            for x in range(0, w - block_size, block_size * 2):
                block = self.prev_gray[y:y + block_size, x:x + block_size]
                min_diff = float('inf')
                best_dx, best_dy = 0, 0

                # Busca na vizinhança
                for dy in range(-search_range, search_range + 1, 2):  # Passo de 2 para melhorar desempenho
                    for dx in range(-search_range, search_range + 1, 2):
                        if 0 <= x + dx < w - block_size and 0 <= y + dy < h - block_size:
                            curr_block = gray[y + dy:y + dy + block_size, x + dx:x + dx + block_size]
                            diff = np.sum(np.abs(block - curr_block))

                            if diff < min_diff:
                                min_diff = diff
                                best_dx, best_dy = dx, dy

                # Desenha seta do movimento apenas se houver movimento significativo
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

        # Calcula movimento médio global
        mean_flow = np.mean(flow, axis=(0, 1))

        # Compensa o movimento médio
        h, w = gray.shape
        M = np.float32([[1, 0, -mean_flow[0]], [0, 1, -mean_flow[1]]])
        compensated = cv2.warpAffine(frame, M, (w, h))

        # Destaque do movimento residual
        diff = cv2.absdiff(frame, compensated)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Aplica máscara
        result = frame.copy()
        result[mask > 0] = (0, 255, 0)  # Destaca áreas com movimento residual
        return result

    def apply_motion_tracking(self, frame):
        # Calcula diferença absoluta entre frames
        diff = cv2.absdiff(self.prev_frame, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Limiarização e processamento morfológico
        _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Encontra e desenha contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Filtra pequenos contornos
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis, f"Movimento: {area:.0f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis

    def show_frame(self, canvas, frame, title=""):
        try:
            frame = cv2.resize(frame, (640, 480))
            if len(frame.shape) == 2:  # Se for imagem em tons de cinza
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # Se tiver canal alpha
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)

            canvas.delete("all")
            canvas.create_image(0, 0, anchor=NW, image=img)
            canvas.image = img  # Mantém referência
            canvas.create_text(320, 20, text=title, fill="white",
                               font=("Arial", 14, "bold"))
        except Exception as e:
            print(f"Erro ao exibir frame: {e}")

    def __del__(self):
        self.stop_webcam()


if __name__ == "__main__":
    root = Tk()
    app = OpticalFlowApp(root)
    root.mainloop()