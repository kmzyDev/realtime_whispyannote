from PySide6.QtWidgets import QApplication, QMainWindow, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout, QMessageBox
from PySide6.QtCore import QThread, Signal, Slot, QObject
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline
from pyannote.audio import Model as EmbeddingModel, Pipeline, Inference
from pyannote.core import Segment
from sklearn.metrics.pairwise import cosine_similarity
import pyaudio
import torch
import numpy as np

import datetime
import threading
import time

from whispyannote.front import Ui_MainWindow

# Param
DEVICE_ID = 1               # uv run device_list.pyで取得したID
FORMAT = pyaudio.paInt16    # 16bit
CHUNK_SIZE = 4096 * 9       # データのチャンクサイズ（要調整）
CHANNELS = 1                # Whisperがモノラルに最適化されてるのでモノラルにする
SAMPLE_RATE = 16000         # サンプルレートもWhisperに合わせる
VAD_THRESHOLD = 0.01        # 無音検知の閾値（要調整）
BEST_SCORE = 0.20           # 別人判定の閾値（要調整）
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


class SpeechToTextThread(QThread):

    # 音声テンソルと参加人数を投げるシグナル
    send_tensor_speakers = Signal(np.ndarray, int)

    def __init__(self, max_speakers=None):
        super().__init__(None)
        self.running    = False
        self.max_speakers = max_speakers

    def run(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=DEVICE_ID,
            frames_per_buffer=CHUNK_SIZE
        )
        stream.start_stream()
        self.running = True
        while self.running:
            audio_bytes = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            # レシーバにシグナル送信
            self.send_tensor_speakers.emit(
                np.frombuffer(audio_bytes, dtype=np.int16).reshape(-1, 1),
                self.max_speakers
            )

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def stop(self):
        self.running = False


class Receiver(QObject):

    send_text = Signal(str)

    @Slot(str)
    def handle_data(self, audio_ndarray, max_speakers):
        if np.abs(audio_ndarray).mean() < VAD_THRESHOLD:
            return
        waveform_pttensor = torch.from_numpy(np.squeeze(audio_ndarray).astype(np.float32) / 32768.0).unsqueeze(0)
        formated_wav = {'waveform': waveform_pttensor, 'sample_rate': SAMPLE_RATE}
        diarization = pipeline(formated_wav, min_speakers=1, max_speakers=max_speakers)
        for seg, _, _ in diarization.itertracks(yield_label=True):
            try:
                bounded_seg = Segment(seg.start, min(seg.end, waveform_pttensor.shape[1] / SAMPLE_RATE))
                embedding = embedding_inference.crop(formated_wav, bounded_seg).squeeze()
                speaker_id = identify_speaker(embedding, max_speakers)
            except Exception as e:
                speaker_id = 'Unknown'
            inputs = processor(waveform_pttensor.squeeze(), sampling_rate=SAMPLE_RATE, return_tensors='pt')
            with torch.no_grad():
                generated_ids = whisper_model.generate(inputs['input_features'], language=LANGUAGE)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.send_text.emit(f'{speaker_id}: {text}')


class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.max_speakers = self.get_max_speakers()
        self.ui.toggle.clicked.connect(self.toggle_clicked)
        self.transcription_thread = SpeechToTextThread(max_speakers=self.max_speakers)
        self.receiver = Receiver()
        self.transcription_thread.send_tensor_speakers.connect(self.receiver.handle_data)
        self.receiver.send_text.connect(self.update_text)

    def get_max_speakers(self):
        dialog = SpeakerInputDialog(self)
        dialog.exec()
        return dialog.max_speakers

    def toggle_clicked(self):
        if not self.transcription_thread.isRunning():
            self.ui.toggle.setText('□')
            self.ui.label.setText('🎤 録音中...')
            self.transcription_thread.start()
        else:
            self.ui.toggle.setText('▷')
            self.ui.label.setText('')
            self.transcription_thread.stop()

    def update_text(self, text):
        self.ui.textBrowser.append(text)

    def closeEvent(self, event):
        self.transcription_thread.stop()
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
