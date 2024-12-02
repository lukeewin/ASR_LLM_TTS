import datetime
import json
import os
import uuid
from enum import Enum

import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from opencc import OpenCC
from pydantic import BaseModel
from faster_whisper import WhisperModel
from typing import Any, Optional
import whisper
import logging
from fastapi.middleware.cors import CORSMiddleware
import pymysql
from dbutils.pooled_db import PooledDB
import jwt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
converter = OpenCC('t2s')

app = FastAPI()
model = {}

SECRET_KEY = "abc123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

base_model_path = os.path.join(os.getcwd(), 'model')
faster_whisper_base_model_path = os.path.join(base_model_path, 'faster_whisper')
openai_whisper_base_model_path = os.path.join(base_model_path, 'openai_whisper')

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:49849",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ErrorCode(Enum):
    ASR_ENGINE_NOT_FOUND = 100
    NOT_AUDIO = 300
    LLM_FAIL = 400
    NOT_LOGIN = 500
    USER_OR_PASSWORD_ERROR = 501
    PASSWORD_ERROR = 502
    USER_ERROR = 503
    TOKEN_ERROR = 504


class MySQLPool:
    def __init__(self, host, port, user, password, database, max_connections=10, block=True):
        self.pool = PooledDB(
            creator=pymysql,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            maxconnections=max_connections,
            blocking=block,
            setsession=['SET AUTOCOMMIT = 1']
        )

    def get_connection(self):
        """
        获取数据库连接

        :return: 数据库连接对象
        """
        return self.pool.connection()

    def execute_query(self, sql, params=None):
        """
        执行查询语句

        :param sql: SQL查询语句
        :param params: SQL参数
        :return: 查询结果
        """
        connection = None
        cursor = None
        try:
            # 从连接池获取连接
            connection = self.get_connection()
            cursor = connection.cursor(pymysql.cursors.DictCursor)

            # 执行查询
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            # 返回查询结果
            return cursor.fetchall()

        except Exception as e:
            print(f"数据库查询错误: {e}")
            return None


db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'asr_llm_tts'
}

mysql_pool = MySQLPool(**db_config)


class FasterWhisperConfig:
    def __init__(self,
                 model_name,
                 device='auto',
                 local_files_only=True,
                 cpu_threads=os.cpu_count(),
                 num_works=os.cpu_count()):
        self.model_name = os.path.join(faster_whisper_base_model_path, model_name)
        self.device = device
        self.local_files_only = local_files_only
        self.cpu_threads = cpu_threads
        self.num_works = num_works

    def init_model(self):
        faster_whisper_model = WhisperModel(model_size_or_path=self.model_name, device=self.device,
                                            local_files_only=self.local_files_only, num_workers=self.num_works)
        return faster_whisper_model


class OpenAiWhisperConfig:
    def __init__(self,
                 model_name,
                 device='auto',
                 in_memory=False):
        self.model_name = os.path.join(openai_whisper_base_model_path, model_name + '.pt')
        self.device = device
        self.in_memory = in_memory

    def init_model(self):
        whisper_model = whisper.load_model(name=self.model_name, device=self.device, in_memory=self.in_memory)
        return whisper_model


class InitModelParameter(BaseModel):
    asr_engine: str
    model_name: str
    device: str
    cpu_threads: str
    num_works: str


class ResponseJsonData(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


def create_access_token(data: dict, expires_delta: datetime.timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


user_token = {}


@app.post("/api/login")
async def login(username: str = Form(), password: str = Form()):
    if username.strip() != '' and password.strip() != '':
        sql = "select username, password from user where username=%s"
        result = mysql_pool.execute_query(sql, (username,))
        if result:
            tmpPassword = result[0]['password']
            if password == tmpPassword:
                access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                token = create_access_token(data={"sub": username}, expires_delta=access_token_expires)
                logging.info(f"token: {token}")
                user_token["username"] = token
                return ResponseJsonData(
                    code=200,
                    message="登录成功",
                    data={
                        "token": token
                    }
                )
            else:
                return ResponseJsonData(
                    code=ErrorCode.PASSWORD_ERROR,
                    message="密码不对"
                )
        else:
            return ResponseJsonData(
                code=ErrorCode.USER_ERROR,
                message="用户名不存在"
            )
    else:
        return ResponseJsonData(
            code=ErrorCode.USER_OR_PASSWORD_ERROR,
            message="用户名或密码错误"
        )


# @app.middleware("http")
# async def filter_token(request: Request, call_next):
#     if request.url.path == '/api/login':
#         response = await call_next(request)
#         return response
#     headers = request.headers
#     token = headers.get("token")
#     logging.info(f"token: {token}")
#     if token == user_token["username"]:
#         response = await call_next(request)
#         return response
#     else:
#         return JSONResponse(
#                 status_code=401,
#                 content={
#                     "code": 504,
#                     "message": "token错误"
#                 })


@app.post("/api/init", response_model=ResponseJsonData)
async def init_model(res: Request, request: InitModelParameter):
    token = res.headers.get("token")
    if user_token["username"] != token:
        return ResponseJsonData(
            code=ErrorCode.TOKEN_ERROR,
            message="token错误"
        )
    asr_engine = ""
    if request.asr_engine == 'faster_whisper':
        faster_model_config = FasterWhisperConfig(model_name=request.model_name, device=request.device,
                                                  cpu_threads=int(request.cpu_threads),
                                                  num_works=int(request.num_works))
        faster_model_asr = faster_model_config.init_model()
        model['faster_whisper'] = faster_model_asr
        asr_engine = 'faster_whisper'
        logging.info("使用的 ASR 是 faster whisper")
        logging.info(
            f"初始化模型信息：\n model_name：{request.model_name} \n device: {request.device} \n cpu_threads: {request.cpu_threads} \n num_works: {request.num_works}")
    elif request.asr_engine == 'openai_whisper':
        whisper_model_config = OpenAiWhisperConfig(model_name=request.model_name, device=request.device,
                                                   in_memory=True)
        whisper_model_asr = whisper_model_config.init_model()
        model['openai_whisper'] = whisper_model_asr
        asr_engine = 'openai_whisper'
        logging.info("使用的 ASR 是 openai whisper")
        logging.info(
            f"初始化模型信息：\n model_name: {request.model_name} \n device: {request.device}")
    else:
        logging.warning("没有传递 asr_engine 参数")
        return ResponseJsonData(
            code=ErrorCode.ASR_ENGINE_NOT_FOUND,
            message="asr参数错误，只支持openai_whisper和faster_whisper"
        )
    return ResponseJsonData(
        code=200,
        message="模型初始化成功",
        data={
            "asr_engine": asr_engine
        }
    )


@app.post("/api/transcript", response_model=ResponseJsonData)
async def transcript(asr_engine: str = Form(),
                     task: str = Form(),
                     temperature: str = Form(),
                     best_of: str = Form(),
                     beam_size: str = Form(),
                     language: str = Form(),
                     vad_filter: str = Form(),
                     min_silence_duration_ms: str = Form(),
                     word_timestamps: str = Form(),
                     hallucination_silence_threshold: str = Form(),
                     file: UploadFile = File()):
    if asr_engine.strip() is None or asr_engine.strip() == '':
        return ResponseJsonData(
            code=ErrorCode.ASR_ENGINE_NOT_FOUND,
            message="没有传递 asr_engine 参数"
        )
    if not file.content_type.startswith('audio/'):
        return ResponseJsonData(
            code=ErrorCode.NOT_AUDIO,
            message="上传的文件非音频",
        )
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    upload_directory = os.path.join(os.getcwd(), 'upload', current_date)
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory, exist_ok=True)
    filename = str(uuid.uuid4()).replace("-", "") + '.wav'
    audio = os.path.join(upload_directory, filename)
    with open(audio, 'wb') as f:
        content = await file.read()
        f.write(content)
    asr = model.get(asr_engine)
    result = ""
    if asr_engine == 'faster_whisper':
        segments, info = asr.transcribe(task=task, audio=str(audio), beam_size=int(beam_size), language=language, without_timestamps=True,
                                        vad_filter=bool(vad_filter),
                                        vad_parameters=dict(min_silence_duration_ms=int(min_silence_duration_ms)))
        for segment in segments:
            if language == 'zh' or info.language == 'zh':
                s_text = converter.convert(segment.text)
                result += s_text
            else:
                result += segment.text
    elif asr_engine == 'openai_whisper':
        asr_result = asr.transcribe(task=task, audio=str(audio), verbose=False, language=language, temperature=float(temperature), best_of=int(best_of),
                                    beam_size=int(beam_size), word_timestamps=bool(word_timestamps),
                                    hallucination_silence_threshold=int(hallucination_silence_threshold))
        _language = asr_result["language"]
        for segment in asr_result["segments"]:
            if language == 'zh' or _language == "zh":
                s_text = converter.convert(segment.get('text'))
                result += s_text
            else:
                result += segment.get('text')
    return ResponseJsonData(
        code=200,
        message="转写成功",
        data={
            "text": result
        }
    )


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


@app.post("/api/tts")
async def tts(text: str = Form()):
    pass


if __name__ in '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
