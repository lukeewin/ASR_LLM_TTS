# 0. 环境说明
python

nginx

ffmpeg

# 1. 使用说明
本项目是在 python = 3.8 环境中开发

其中这里看到的是后端接口部分，前端部分可以到 https://github.com/lukeewin/ASR_LLM_TTS_Front.git 中查看

本项目完全在内网离线环境中可以使用，没有调用任何云 API 接口。

ASR引擎：openai whisper 和 faster whisper

LLM: ollama 支持的任意大模型，代码内部使用的是 qwen:1.8b 大模型，如果想要更换其它大模型，可以在源码 app.py 中找到 qwen:1.8b 更换为自己部署的大模型

TTS: MeloTTS（中国大陆地区由于网络问题，导致从 huggingface 中无法下载模型文件的可以设置使用中国大陆 huggingface 镜像）

目录结构：
```shell
| model | openai_whisper | 模型
        | faster_whisper | 模型
| app.py
```

# 2. 配置反向代理
修改 nginx,conf 文件为下面的内容：
```shell
server {
        listen       80;
        server_name  localhost;

        location / {
            root   D:\\Works\\Web_Projects\\ASR_LLM_TTS\\top\\lukeewin\\;
            index  login.html index.htm;
        }

		location /api {
				proxy_pass http://localhost:8000;
				proxy_set_header Host $host;
				proxy_set_header X-Real-IP $remote_addr;
		}
		
		location /tts {
				proxy_pass http://localhost:8001;
				proxy_set_header Host $host;
				proxy_set_header X-Real-IP $remote_addr;
		}
}
```
# 3. TTS
```python
import io
import logging
from typing import Optional, Any
from melo.api import TTS
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from enum import Enum

# https://github.com/myshell-ai/MeloTTS/pull/56/files

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI()

device = 'auto'
models = {
    'EN': TTS(language='EN', device=device),
    'ES': TTS(language='ES', device=device),
    'FR': TTS(language='FR', device=device),
    'ZH': TTS(language='ZH', device=device),
    'JP': TTS(language='JP', device=device),
    'KR': TTS(language='KR', device=device),
}


class RequestJson(BaseModel):
    text: str = 'Ahoy there matey! There she blows!'
    language: str = 'EN'
    speaker: str = 'EN-US'
    speed: float = 1.0


class ResponseJson(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class ErrorCode(Enum):
    TEXT_NOT_FOUND = 100


@app.post("/tts/stream")
async def tts_stream(payload: RequestJson):
    language = payload.language
    text = payload.text
    speaker = payload.speaker or list(models[language].hps.data.spk2id.keys())[0]
    speed = payload.speed

    logging.info(f"language: {language}")
    logging.info(f"text: {text}")
    logging.info(f"speaker: {speaker}")
    logging.info(f"speed: {speed}")

    def audio_stream():
        bio = io.BytesIO()
        if speaker == 'None':
            models[language].tts_to_file(text, models[language].hps.data.spk2id[language], bio, speed=speed, format='wav')
        else:
            models[language].tts_to_file(text, models[language].hps.data.spk2id[speaker], bio, speed=speed, format='wav')
        audio_data = bio.getvalue()
        yield audio_data

    return StreamingResponse(audio_stream(), media_type="audio/wav")


if __name__ in '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)
```
在下载完成 MeloTTS 之后，进入到 melo 目录中，添加上面的 python 代码，主要用于流式输出合成的音频。

具体如何安装 MeloTTS，可以到 https://github.com/myshell-ai/MeloTTS.git 查看。

我们只需要运行上面代码即可。

我搭建 TTS 的过程文档：https://blog.lukeewin.top/archives/melotts

# 4. LLM
默认使用 qwen:1.8b 模型，如果你要替换其它模型，可以修改下面代码：
```python
@app.post("/api/llm")
async def llm(question: str = Form()):
    dialogue_history = []
    llm_url = "http://localhost:11434/api/chat"
    headers = {
        "Content-Type": "application/json"
    }
    dialogue_history.append({
        "role": "user",
        "content": question
    })
    data = {
        "model": "qwen:1.8b",
        "messages": dialogue_history,
        "stream": False
    }
    response = requests.post(llm_url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        response_json = response.json()
        ai_answer = response_json["message"]["content"]
        dialogue_history.append({
            "role": "assistant",
            "content": ai_answer
        })
        return ResponseJsonData(
            code=200,
            message="成功",
            data={
                "text": ai_answer
            }
        )
    else:
        return ResponseJsonData(
            code=ErrorCode.LLM_FAIL,
            message="大模型生成内容失败"
        )
```
找到 "model": "qwen:1.8b" 修改为 ollama 支持的模型，比如修改为 "model": "llama3.2"

当然如果你想要对接云端的大模型，你需要自己改造代码。

# 5. ASR模型下载
OpenAI Whisper 模型可以到我的博客中查看如何下载 https://blog.lukeewin.top/archives/openai-whisper-offline-install#toc-head-15

Faster Whisper 模型可以到 https://huggingface.co/Systran 下载

# 6. 其它
视频演示：

博客：https://blog.lukeewin.top
