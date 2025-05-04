import cv2
import numpy as np
import random
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class MotionApp:
    def __init__(self, master):
        self.master = master
        master.title("Estimação e Compensação de Movimento")
        master.geometry("1280x720")

        # Canvases para exibição
        self.canvas1 = Canvas(master, width=400, height=300, bg='black')
        self.canvas2 = Canvas(master, width=400, height=300, bg='black')
        self.canvas3 = Canvas(master, width=400, height=300, bg='black')
        for i, c in enumerate((self.canvas1, self.canvas2, self.canvas3)):
            c.grid(row=0, column=i, padx=10, pady=10)

        # Opções de técnica
        self.techniques = [
            "Lucas-Kanade (esparso)",
            "Farneback (denso)",
            "Block Matching (manual)",
            "Compensação (média de vetores)"
        ]
        self.var_tech = StringVar(master)
        self.var_tech.set(self.techniques[0])
        OptionMenu(master, self.var_tech, *self.techniques).grid(row=1, column=0, pady=10)
        Button(master, text="Load Video", command=self.load_video).grid(row=1, column=1)
        Button(master, text="Run", command=self.run).grid(row=1, column=2)

        self.frame1 = None
        self.frame2 = None
        self.frame_idx = None

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.mp4 *.avi")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 3:
            messagebox.showerror("Erro", "Vídeo muito curto.")
            cap.release()
            return
        idx = random.randint(1, total - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret1, f1 = cap.read()
        ret2, f2 = cap.read()
        cap.release()
        if not (ret1 and ret2):
            messagebox.showerror("Erro", "Não foi possível ler os quadros.")
            return
        self.frame1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        self.frame2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        self.frame_idx = idx
        self._show(self.canvas1, self.frame1, f"Frame {idx}")

    def run(self):
        if self.frame1 is None or self.frame2 is None:
            messagebox.showwarning("Atenção", "Carregue um vídeo primeiro.")
            return
        tech = self.var_tech.get()
        f1, f2 = self.frame1, self.frame2
        h, w = f1.shape

        if tech == "Lucas-Kanade (esparso)":
            p0 = cv2.goodFeaturesToTrack(f1, maxCorners=200, qualityLevel=0.01, minDistance=5)
            if p0 is None:
                messagebox.showinfo("Info", "Nenhum ponto encontrado.")
                return
            p0 = p0.astype(np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(f1, f2, p0, None)
            if p1 is None or st is None:
                messagebox.showinfo("Info", "Falha na estimação.")
                return
            mask = st.ravel() == 1
            good0 = p0.reshape(-1, 2)[mask]
            good1 = p1.reshape(-1, 2)[mask]
            if len(good0) < 1:
                messagebox.showinfo("Info", "Pontos insuficientes para exibir.")
                return
            vis = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
            for (x0, y0), (x1, y1) in zip(good0, good1):
                cv2.circle(vis, (int(x0), int(y0)), 4, (0, 255, 0), -1)  # ponto inicial
                cv2.arrowedLine(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 3, tipLength=0.3)
            self._show(self.canvas2, vis, "LK Vetores")
            self._show(self.canvas3, f2, f"Frame {self.frame_idx + 1}")

        elif tech == "Farneback (denso)":
            flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((*f1.shape, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self._show(self.canvas2, bgr, "Farneback")
            self._show(self.canvas3, f2, f"Frame {self.frame_idx + 1}")

        elif tech == "Block Matching (manual)":
            vis = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
            bs, sr = 16, 8
            for y in range(0, h - bs, bs):
                for x in range(0, w - bs, bs):
                    block = f1[y:y+bs, x:x+bs]
                    best, mind = (0, 0), np.inf
                    for dy in range(-sr, sr + 1):
                        for dx in range(-sr, sr + 1):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < h-bs and 0 <= xx < w-bs:
                                diff = np.sum((block - f2[yy:yy+bs, xx:xx+bs])**2)
                                if diff < mind:
                                    mind, best = diff, (dx, dy)
                    dx, dy = best
                    cv2.arrowedLine(vis, (x+bs//2, y+bs//2), (x+bs//2+dx, y+bs//2+dy), (0,255,0), 2, tipLength=0.3)
            self._show(self.canvas2, vis, "Block Matching")
            self._show(self.canvas3, f2, f"Frame {self.frame_idx + 1}")

        elif tech == "Compensação (média de vetores)":
            p0 = cv2.goodFeaturesToTrack(f1, maxCorners=200, qualityLevel=0.01, minDistance=5)
            if p0 is None:
                messagebox.showinfo("Info", "Nenhum ponto encontrado.")
                return
            p0 = p0.astype(np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(f1, f2, p0, None)
            if p1 is None or st is None:
                messagebox.showinfo("Info", "Falha na estimação.")
                return
            mask = st.ravel() == 1
            pts0 = p0.reshape(-1,2)[mask]
            pts1 = p1.reshape(-1,2)[mask]
            if pts0.size == 0:
                messagebox.showinfo("Info", "Pontos insuficientes.")
                return
            vecs = pts1 - pts0
            dx, dy = vecs.mean(axis=0).astype(int)
            M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
            compensated = cv2.warpAffine(f1, M, (w, h))
            self._show(self.canvas2, compensated, "Compensado")
            self._show(self.canvas3, f2, f"Frame {self.frame_idx + 1}")

    def _show(self, canvas, img, title=""):
        if img.ndim == 2:
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        disp = cv2.resize(disp, (400, 300))
        photo = ImageTk.PhotoImage(Image.fromarray(disp))
        canvas.delete("all")
        canvas.create_image(0,0,anchor=NW, image=photo)
        canvas.image = photo
        canvas.create_text(200, 10, text=title, fill="white", font=("Arial",14,"bold"))

if __name__ == "__main__":
    root = Tk()
    app = MotionApp(root)
    root.mainloop()
