# 0. 环境说明
python
nginx
ffmpeg

# 1. 使用说明
本项目是在 python = 3.8 环境中开发
可运行在 Windows MacOS Linux 等系统上
其中这里看到的是后端接口部分，前端部分可以到 https://github.com/lukeewin/ASR_LLM_TTS_Front.git 中查看
本项目完全在内网离线环境中可以使用，没有调用任何云 API 接口。
ASR引擎：openai whisper 和 faster whisper
LLM: ollama 支持的任意大模型，代码内部使用的是 qwen:1.8b 大模型，如果想要更换其它大模型，可以在源码 app.py 中找到 qwen:1.8b 更换为自己部署的大模型
TTS: MeloTTS（中国大陆地区由于网络问题，导致从 huggingface 中无法下载模型文件的可以设置使用中国大陆 huggingface 镜像）
