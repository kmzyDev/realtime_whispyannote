import pyaudio

p = pyaudio.PyAudio()

def main():
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"デバイス {i}: {device_info['name']} (入力: {device_info['maxInputChannels']}, 出力: {device_info['maxOutputChannels']})")
    p.terminate()

if __name__ == '__main__':
    main()
