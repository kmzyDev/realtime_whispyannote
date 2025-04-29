from optimum.intel.openvino import OVModelForSpeechSeq2Seq 
from transformers import AutoProcessor 

model_id = "openai/whisper-large-v3-turbo"
model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
model.save_pretrained('./assets/whisper_turbo')
processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained('./assets/whisper_turbo')
