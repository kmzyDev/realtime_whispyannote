import ctypes
import queue

import numpy as np
import torch
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from pyannote.audio import Model as EmbeddingModel, Inference, Pipeline
from pyannote.core import Segment
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor

from whispyannote.front import Ui_MainWindow

cable_dll = ctypes.cdll.LoadLibrary('./src/whispyannote/cable.dll')
select_dll = ctypes.cdll.LoadLibrary('./src/whispyannote/device_select.dll')

# Prototype
cable_dll.StartLoopRecording.argtypes = [ctypes.c_char_p]
cable_dll.StartLoopRecording.restype = ctypes.c_int
cable_dll.StartMicRecording.argtypes = [ctypes.c_char_p]
cable_dll.StartMicRecording.restype = ctypes.c_int
cable_dll.GetLoopAudio.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int)]
cable_dll.GetLoopAudio.restype = ctypes.c_int
cable_dll.GetMicAudio.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_int)]
cable_dll.GetMicAudio.restype = ctypes.c_int
cable_dll.StopLoopRecording.argtypes = []
cable_dll.StopLoopRecording.restype = None
cable_dll.StopMicRecording.argtypes = []
cable_dll.StopMicRecording.restype = None
select_dll.RefreshDevices.argtypes = [ctypes.c_int]
select_dll.RefreshDevices.restype = ctypes.c_int
select_dll.GetDeviceCount.argtypes = []
select_dll.GetDeviceCount.restype = ctypes.c_int
select_dll.GetDeviceName.argtypes = [ctypes.c_int]
select_dll.GetDeviceName.restype = ctypes.c_char_p
select_dll.GetDeviceId.argtypes = [ctypes.c_int]
select_dll.GetDeviceId.restype = ctypes.c_char_p

# Param
CHANNELS = 1  # Whisperがモノラルに最適化されてるのでモノラルにする
SAMPLE_RATE = 16000  # サンプルレートもWhisperに合わせる
VAD_THRESHOLD = 0.01  # 無音検知の閾値（要調整）
BEST_SCORE = 0.20  # 別人判定の閾値（要調整）
LANGUAGE = 'ja'

# Model
whisper_model = OVModelForSpeechSeq2Seq.from_pretrained('./assets/whisper_turbo').to('GPU')
whisper_model.compile()
processor = AutoProcessor.from_pretrained('./assets/whisper_turbo')
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', cache_dir='./assets')
embedding_model = EmbeddingModel.from_pretrained('pyannote/embedding', cache_dir='./assets')
embedding_inference = Inference(embedding_model, window='whole')

known_speakers = {}


def identify_speaker(embedding_tensor, max_speakers):
    best_score = 0
    best_id = None
    for speaker, embeddings in known_speakers.items():
        for emb in embeddings:
            score = cosine_similarity([embedding_tensor], [emb])[0][0]
            if score > best_score:
                best_score = score
                best_id = speaker
    if best_score > BEST_SCORE:
        known_speakers[best_id].append(embedding_tensor)
        return best_id
    if len(known_speakers) < max_speakers:
        new_id = f'Speaker_{len(known_speakers) + 1}'
        known_speakers[new_id] = [embedding_tensor]
        return new_id
    if best_id is not None:
        known_speakers[best_id].append(embedding_tensor)
        return best_id


class RecordingThread(QThread):

    def __init__(self, audio_queue, device_type):
        super().__init__(None)
        self.audio_queue = audio_queue
        self.running = False
        self.device_id = None
        self.device_type = device_type
        if device_type == 'loop':
            self.start_recording = cable_dll.StartLoopRecording
            self.get_audio = cable_dll.GetLoopAudio
            self.stop_recording = cable_dll.StopLoopRecording
        else:
            self.start_recording = cable_dll.StartMicRecording
            self.get_audio = cable_dll.GetMicAudio
            self.stop_recording = cable_dll.StopMicRecording
    
    def set_device(self, device_id):
        self.device_id = device_id
        
    def run(self):
        self.start_recording(self.device_id)
        self.running = True
        while self.running:
            buffer_ptr = ctypes.POINTER(ctypes.c_ubyte)()
            length = ctypes.c_int(0)
            ret = self.get_audio(ctypes.byref(buffer_ptr), ctypes.byref(length))
            if ret != 1 or length.value <= 0:
                continue
            
            raw_bytes = ctypes.string_at(buffer_ptr, length.value)
            raw_ndarray = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, 1)
            self.audio_queue.put((raw_ndarray.astype(np.float32) / 32768.0))
    
    def stop(self):
        self.running = False
        self.stop_recording()
        self.wait()


class SpeechToTextThread(QThread):

    send_text = Signal(str)

    def __init__(self, audio_queue, max_speakers):
        super().__init__(None)
        self.audio_queue = audio_queue
        self.max_speakers = max_speakers
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            try:
                audio_ndarray = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue
            if np.abs(audio_ndarray).mean() < VAD_THRESHOLD:
                continue
            waveform_pttensor = torch.from_numpy(np.squeeze(audio_ndarray)).unsqueeze(0)
            formated_wav = {'waveform': waveform_pttensor, 'sample_rate': SAMPLE_RATE}
            diarization = pipeline(formated_wav, min_speakers=1, max_speakers=self.max_speakers)
            for seg, _, _ in diarization.itertracks(yield_label=True):
                seg_start = seg.start
                seg_end = min(seg.end, waveform_pttensor.shape[1] / SAMPLE_RATE)
                segment_audio = waveform_pttensor[:, int(seg_start * SAMPLE_RATE) : int(seg_end * SAMPLE_RATE)]
                if np.abs(segment_audio.numpy()).mean() < VAD_THRESHOLD:
                    continue
                try:
                    bounded_seg = Segment(seg_start, seg_end)
                    embedding = embedding_inference.crop(formated_wav, bounded_seg).squeeze()
                    speaker_id = identify_speaker(embedding, self.max_speakers)
                except Exception:
                    continue
                inputs = processor(segment_audio.squeeze(), sampling_rate=SAMPLE_RATE, return_tensors='pt')
                with torch.no_grad():
                    generated_ids = whisper_model.generate(inputs['input_features'], language=LANGUAGE)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                self.send_text.emit(f'{speaker_id}: {text}')

    def stop(self):
        self.running = False


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        select_dll.RefreshDevices(0)
        for i in range(select_dll.GetDeviceCount()):
            device_name = select_dll.GetDeviceName(i)
            device_id = select_dll.GetDeviceId(i)
            if device_name and device_id:
                self.ui.loopSelect.addItem(device_name.decode('utf-8'), device_id)
        select_dll.RefreshDevices(1)
        for i in range(select_dll.GetDeviceCount()):
            device_name = select_dll.GetDeviceName(i)
            device_id = select_dll.GetDeviceId(i)
            if device_name and device_id:
                self.ui.micSelect.addItem(device_name.decode('utf-8'), device_id)
        self.max_speakers = self.get_max_speakers()
        self.ui.toggle.clicked.connect(self.toggle_clicked)
        self.audio_queue = queue.Queue()
        self.mic_recording_thread = RecordingThread(audio_queue=self.audio_queue, device_type='mic')
        self.loop_recording_thread = RecordingThread(audio_queue=self.audio_queue, device_type='loop')
        self.transcription_thread = SpeechToTextThread(audio_queue=self.audio_queue, max_speakers=self.max_speakers)
        self.transcription_thread.send_text.connect(self.update_text)

    def get_max_speakers(self):
        dialog = SpeakerInputDialog(self)
        dialog.exec()
        return dialog.max_speakers

    def toggle_clicked(self):
        if not self.mic_recording_thread.isRunning():
            self.ui.toggle.setText('□')
            self.ui.label.setText('🎤 録音中...')
            self.ui.micSelect.setEnabled(False)
            self.ui.loopSelect.setEnabled(False)
            self.mic_recording_thread.set_device(self.ui.micSelect.currentData())
            self.loop_recording_thread.set_device(self.ui.loopSelect.currentData())
            self.mic_recording_thread.start()
            self.loop_recording_thread.start()
            self.transcription_thread.start()
        else:
            self.ui.toggle.setText('▷')
            self.ui.label.setText('')
            self.ui.micSelect.setEnabled(True)
            self.ui.loopSelect.setEnabled(True)
            self.mic_recording_thread.stop()
            self.loop_recording_thread.stop()
            self.transcription_thread.stop()

    def update_text(self, text):
        self.ui.textBrowser.append(text)

    def closeEvent(self, event):
        self.mic_recording_thread.stop()
        self.loop_recording_thread.stop()
        self.transcription_thread.stop()
        self.mic_recording_thread.wait()
        self.loop_recording_thread.wait()
        self.transcription_thread.wait()
        event.accept()


class SpeakerInputDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('whispyannote')
        self.setFixedSize(200, 100)
        layout = QVBoxLayout()
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel('参加人数：'))
        self.input_field = QLineEdit(self)
        h_layout.addWidget(self.input_field)
        layout.addLayout(h_layout)
        ok_button = QPushButton('OK')
        ok_button.clicked.connect(self.ok_clicked)
        layout.addWidget(ok_button)
        self.setLayout(layout)

    def ok_clicked(self):
        value = self.input_field.text().strip()
        if not value.isdigit() or int(value) < 1:
            QMessageBox.warning(self, '入力エラー', '参加人数は自然数にしてください。')
            return
        self.max_speakers = int(value)
        self.accept()

    def closeEvent(self, event):
        event.ignore()


def main():
    app = QApplication([])
    window = MyApp()
    window.show()
    app.exec_()
