import cv2
import numpy as np
import mediapipe as mp
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


class MotionAppLive:
    def __init__(self, master):
        self.master = master
        master.title("Detecção ao Vivo - Webcam com MediaPipe")
        master.geometry("1200x800")

        # Configuração do tema
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TCombobox', font=('Arial', 10))

        # Canvases (original e processado)
        self.canvas_orig = Canvas(master, width=640, height=480, bg='black')
        self.canvas_tech = Canvas(master, width=640, height=480, bg='black')
        self.canvas_orig.grid(row=0, column=0, padx=10, pady=10)
        self.canvas_tech.grid(row=0, column=1, padx=10, pady=10)

        # Frame de controles
        self.control_frame = Frame(master)
        self.control_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # Dropdown de detecções
        self.techniques = [
            "Hands - Detecção de Mãos",
            "Face Mesh - Malha Facial",
            "Face Detection - Detecção de Rostos",
            "Pose - Postura Corporal",
            "Selfie Segmentation - Segmentação",
            "Objectron - Detecção de Objetos"
        ]
        self.var_tech = StringVar(master)
        self.var_tech.set(self.techniques[0])

        self.tech_dropdown = ttk.Combobox(
            self.control_frame,
            textvariable=self.var_tech,
            values=self.techniques,
            state="readonly",
            width=30
        )
        self.tech_dropdown.pack(side=LEFT, padx=10)

        # Botões
        self.start_btn = ttk.Button(
            self.control_frame,
            text="Iniciar Webcam",
            command=self.start_webcam
        )
        self.start_btn.pack(side=LEFT, padx=5)

        self.stop_btn = ttk.Button(
            self.control_frame,
            text="Parar Webcam",
            command=self.stop_webcam,
            state=DISABLED
        )
        self.stop_btn.pack(side=LEFT, padx=5)

        # Estado
        self.cap = None
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0

        # Inicializa MediaPipe
        self.mp = mp  # Mantém referência ao módulo mediapipe
        self.initialize_mediapipe()

    def initialize_mediapipe(self):
        """Inicializa todos os módulos do MediaPipe"""
        self.mp_draw = self.mp.solutions.drawing_utils
        self.mp_drawing_styles = self.mp.solutions.drawing_styles

        # Configurações dos modelos
        self.hands = self.mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.face_mesh = self.mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_det = self.mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.6
        )

        self.pose = self.mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.selfie = self.mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

        self.objectron = self.mp.solutions.objectron.Objectron(
            static_image_mode=False,
            max_num_objects=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.8,
            model_name='Cup'
        )

    def start_webcam(self):
        """Inicia a captura da webcam"""
        if self.cap:
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Webcam não acessível")
            self.cap = None
        else:
            self.running = True
            self.start_btn.config(state=DISABLED)
            self.stop_btn.config(state=NORMAL)
            self.frame_count = 0
            self.start_time = cv2.getTickCount()
            self.update_frame()

    def stop_webcam(self):
        """Para a captura da webcam"""
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None

        # Verifica se os widgets ainda existem antes de acessá-los
        if hasattr(self, 'start_btn') and self.start_btn.winfo_exists():
            self.start_btn.config(state=NORMAL)
        if hasattr(self, 'stop_btn') and self.stop_btn.winfo_exists():
            self.stop_btn.config(state=DISABLED)

    def update_frame(self):
        """Atualiza o frame da webcam e aplica a detecção selecionada"""
        if not self.running or not self.cap:
            return

        # Captura do frame
        ret, frame = self.cap.read()
        if not ret:
            self.master.after(30, self.update_frame)
            return

        # Pré-processamento
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tech = self.var_tech.get().split(" - ")[0]

        # Cálculo do FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            end_time = cv2.getTickCount()
            self.fps = 30 / ((end_time - self.start_time) / cv2.getTickFrequency())
            self.start_time = end_time

        # Exibe frame original
        self._show(self.canvas_orig, frame, f"Original - FPS: {self.fps:.1f}")

        # Processa de acordo com a técnica selecionada
        out = self.process_detection(frame, rgb, tech)

        # Exibe resultado processado
        self._show(self.canvas_tech, out, tech)

        self.master.after(30, self.update_frame)

    def process_detection(self, frame, rgb, tech):
        """Aplica a técnica de detecção selecionada"""
        out = frame.copy()

        try:
            if tech == 'Hands':
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    for hand_landmarks in res.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            out,
                            hand_landmarks,
                            self.mp.solutions.hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

            elif tech == 'Face Mesh':
                res = self.face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    for face_landmarks in res.multi_face_landmarks:
                        self.mp_draw.draw_landmarks(
                            out,
                            face_landmarks,
                            self.mp.solutions.face_mesh.FACEMESH_CONTOURS,
                            None,
                            self.mp_drawing_styles.get_default_face_mesh_contours_style()
                        )

            elif tech == 'Face Detection':
                res = self.face_det.process(rgb)
                if res.detections:
                    for detection in res.detections:
                        self.mp_draw.draw_detection(out, detection)

            elif tech == 'Pose':
                res = self.pose.process(rgb)
                if res.pose_landmarks:
                    self.mp_draw.draw_landmarks(
                        out,
                        res.pose_landmarks,
                        self.mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )

            elif tech == 'Selfie Segmentation':
                res = self.selfie.process(rgb)
                if res.segmentation_mask is not None:
                    mask = res.segmentation_mask > 0.1
                    bg = np.zeros(frame.shape, dtype=np.uint8)
                    bg[:] = (70, 130, 180)  # Cor de fundo azul
                    out = np.where(mask[..., None], frame, bg)

            elif tech == 'Objectron':
                res = self.objectron.process(rgb)
                if res.detected_objects:
                    for obj in res.detected_objects:
                        self.mp_draw.draw_landmarks(
                            out,
                            obj.landmarks_2d,
                            self.mp.solutions.objectron.BOX_CONNECTIONS,
                            self.mp_drawing_styles.get_default_objectron_landmarks_style(),
                            self.mp_drawing_styles.get_default_objectron_connections_style()
                        )

        except Exception as e:
            print(f"Erro em {tech}: {e}")

        return out

    def _show(self, canvas, img, title):
        """Exibe a imagem no canvas especificado"""
        try:
            # Redimensiona mantendo aspect ratio
            h, w = img.shape[:2]
            aspect = w / h
            new_h = 480
            new_w = int(new_h * aspect)

            disp = cv2.resize(img, (new_w, new_h))

            # Converte para formato adequado
            if len(disp.shape) == 2 or disp.shape[2] == 1:
                disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
            elif disp.shape[2] == 4:
                disp = cv2.cvtColor(disp, cv2.COLOR_BGRA2RGB)
            else:
                disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

            # Converte para PhotoImage e exibe
            photo = ImageTk.PhotoImage(Image.fromarray(disp))
            canvas.delete('all')
            canvas.create_image(320, 240, image=photo, anchor=CENTER)
            canvas.image = photo  # Mantém referência
            canvas.create_text(
                320, 20,
                text=title,
                fill='white',
                font=('Arial', 14, 'bold'),
                anchor=N
            )
        except Exception as e:
            print(f"Erro ao exibir imagem: {e}")

    def cleanup(self):
        """Libera todos os recursos antes de fechar"""
        self.stop_webcam()

        # Fecha todos os modelos MediaPipe
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'selfie'):
            self.selfie.close()
        if hasattr(self, 'objectron'):
            self.objectron.close()

    def __del__(self):
        """Destrutor - libera recursos"""
        self.cleanup()


if __name__ == '__main__':
    root = Tk()
    app = MotionAppLive(root)


    def on_closing():
        app.cleanup()
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()