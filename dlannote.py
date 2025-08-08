from pyannote.audio import Model, Pipeline

TOKEN = 'your_hf_token'

pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=TOKEN, cache_dir='./assets')

embedding_model = Model.from_pretrained('pyannote/embedding', use_auth_token=TOKEN, cache_dir='./assets')
