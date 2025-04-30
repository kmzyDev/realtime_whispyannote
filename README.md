## 前置き
Whisperによる音声認識とpyannoteによる話者分離を行います  
リアルタイムでやってる事例はなかったので作りました  
Intel Core Ultraシリーズ搭載のPCを想定しています

## このレポジトリをクローンしたらやること
1. **ライブラリのバグ対応**  
uv sync  
.venv\Lib\site-packages\torchaudio\lib\\_torchaudio.pyd
の先頭の_を削除してください  

> [!NOTE]
> - 3.10以降のPythonで発生する事象のようです  
https://github.com/neonbjb/tortoise-tts/issues/298#issuecomment-2174432438

2. **モデル配置**  
uv run dlwhisper.py  
uv run dlannote.py  
※dlannoteはHugging Faceのトークンと同意が必要になります

3. **起動**  
uv run whispyannote  
※モデルロードにつきアプリ起動まで1分程度かかります

**（おまけ1）UI編集のについて**  
.venv/Lib/site-packages/PySide6/designer.exe を使ってfront.uiを編集  
uv run pyside6-uic front.ui -o front.py　でfront.uiからfront.pyが出来上がります 

**（おまけ2）ビルド**  
ビルドもできるようになっています  
uv build  
uv tool install dist\whispyannote-0.1.0-py3-none-any.whl  
（.exeファイルから起動する時はassets以下も同一階層に置いてください）

## 音声が認識されない時は
uv run device_list.py  
↑音声デバイスの一覧が表示されます  
お使いの入力デバイスのIDを控えてsrc/whispyannote/main.pyの19行目(DEVICE_ID)に記述してください
