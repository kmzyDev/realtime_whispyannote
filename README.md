## 前置き
Whisperによる音声認識とpyannoteによる話者分離を行います  
リアルタイムでやってる事例はなかったので作りました  
Intel Core Ultraシリーズ搭載のPCを想定しています

## このレポジトリをクローンしたらやること
1. **ライブラリのバグ対応**  
```
uv sync
```
`.venv\Lib\site-packages\torchaudio\lib\\_torchaudio.pyd`の先頭の_を削除してください  

> [!NOTE]
> - 3.10以降のPythonで発生する事象のようです  
https://github.com/neonbjb/tortoise-tts/issues/298#issuecomment-2174432438

2. **モデル配置**  
```
uv run dlmodel/dlwhisper.py
uv run dlmodel/dlannote.py
```
※dlannoteはHugging Faceのトークンと同意が必要になります

4. **DLLビルド**  
```
docker compose up -d
docker exec -it build_env bash
x86_64-w64-mingw32-g++ -Wall -Wextra -static -static-libgcc -static-libstdc++ -shared -o cable.dll cable.cpp -lole32
x86_64-w64-mingw32-g++ -Wall -Wextra -static -static-libgcc -static-libstdc++ -shared -o device_select.dll device_select.cpp -lole32
```
※ビルド完了後はコンテナから出てください

4. **起動**  
```
uv run whispyannote
```
※モデルロードにつきアプリ起動まで1分程度かかります

**（おまけ1）UI編集のについて**  
`.venv/Lib/site-packages/PySide6/designer.exe` を使ってfront.uiを編集  
`uv run pyside6-uic front.ui -o front.py`でfront.uiからfront.pyが出来上がります 

**（おまけ2）ビルド**  
ビルドもできるようになっています  
```
uv build  
uv tool install dist\whispyannote-0.1.0-py3-none-any.whl  
```
（.exeファイルから起動する時はassets以下も同一階層に置いてください）
