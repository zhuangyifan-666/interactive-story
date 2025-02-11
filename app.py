from flask import Flask, request, jsonify, send_from_directory, Response
import uuid
import httpx
import requests
import json
import random
import os
import sys
from queue import Queue
import asyncio
import re
import datetime
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# 全局内存存储游戏会话（仅用于演示，不适合生产环境）
sessions: Dict[str, Dict[str, Any]] = {}

# 定义用于调试日志的队列
debug_queue = Queue()

def debug_log(message: str, msg_type: str = "log") -> None:
    """
    发送调试日志到前端，支持不同类型的日志
    msg_type可以是: log(普通日志), error(错误), intermediate(中间推理), answer(生成结果), final(最终结果)
    """
    try:
        # 添加时间戳
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = json.dumps({
            "type": msg_type,
            "log": message,
            "timestamp": timestamp
        })
        debug_queue.put(payload)
    except Exception as e:
        print(f"调试日志发送失败: {str(e)}")

# SSE 调试日志流，实时推送 debug 消息到前端
@app.route("/debug_stream")
def debug_stream():
    def event_stream():
        while True:
            try:
                # 消息已经是 JSON 格式字符串，直接发送给客户端
                message = debug_queue.get()
                yield f"data: {message}\n\n"
            except Exception as e:
                debug_log(f"调试流发送失败: {str(e)}", "error")
    return Response(event_stream(), mimetype="text/event-stream")

# 配置 API Token 和各 API URL
API_TOKEN = os.environ.get("API_TOKEN") or "sk-vkgmcrkvldamhenfivsukhxkdlceiyonzhcnntnjjqfqsnkd"
if not API_TOKEN:
    print("错误: API_TOKEN 环境变量未设置或未正确配置。请设置 API_TOKEN 环境变量！")
    sys.exit(1)

# 引入官方 DeepSeek R1 API 的密钥
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or "sk-e6a58f57642348afbd40bc79c6cb267b"
if not DEEPSEEK_API_KEY:
    print("错误: DEEPSEEK_API_KEY 环境变量未设置或未正确配置。请设置 DEEPSEEK_API_KEY 环境变量！")
    sys.exit(1)

TRANSCRIPTION_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
TTS_URL = "https://api.siliconflow.cn/v1/audio/speech"
IMAGE_GEN_URL = "https://api.siliconflow.cn/v1/images/generations"
TEXT_GEN_URL = "https://api.siliconflow.cn/v1/chat/completions"

def robust_json_parse(text: str) -> Any:
    """
    健壮的 JSON 解析函数，支持多种格式的 JSON 字符串解析
    """
    try:
        return json.loads(text)
    except Exception as e:
        # 尝试使用正则表达式提取 markdown code block 中的 JSON 内容
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if m:
            json_text = m.group(1)
            try:
                return json.loads(json_text)
            except Exception as e2:
                debug_log(f"Failed parsing after extracting code block: {e2}", "error")
        # 若正则提取不成功，则尝试手动查找 JSON 数组或对象
        stripped = text.strip()
        if stripped.startswith("```"):
            const_lines = stripped.splitlines()
            if len(const_lines) >= 3:
                stripped = "\n".join(const_lines[1:-1])
        for char in ['{', '[']:
            if char in stripped:
                try:
                    end_char = "}" if char == "{" else "]"
                    start = stripped.index(char)
                    end = stripped.rindex(end_char) + 1
                    possible_json = stripped[start:end]
                    return json.loads(possible_json)
                except Exception as e3:
                    continue
        debug_log(f"robust_json_parse failed: {e}", "error")
        raise e

# 加载 stories.json 中的故事数据
stories_path = os.path.join(os.path.dirname(__file__), "stories.json")
try:
    with open(stories_path, "r", encoding="utf-8") as f:
        stories_data = json.load(f)
except Exception as e:
    debug_log(f"加载 stories.json 失败: {e}", "error")
    stories_data = {"stories": []}

async def generate_text_async_stream(prompt: str, model: str) -> AsyncGenerator[str, None]:
    """Streaming version that yields tokens"""
    if model == "deepseek-ai/DeepSeek-R1":
        providers = ["doubao", "siliconflow", "deepseek_official"]
        last_exception = None
        for provider in providers:
            try:
                if provider == "doubao":
                    async for token in try_provider_doubao_stream(prompt, model="ep-20250206131705-gtthc"):
                        yield token
                    return
                elif provider == "siliconflow":
                    payload = {
                        "model": "deepseek-ai/DeepSeek-R1",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                        "max_tokens": 8192,
                    }
                    headers = {
                        "Authorization": f"Bearer {API_TOKEN}",
                        "Content-Type": "application/json"
                    }
                    async for token in try_provider_http_stream(TEXT_GEN_URL, payload, headers):
                        yield token
                    return
                elif provider == "deepseek_official":
                    payload = {
                        "model": "deepseek-reasoner",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                        "max_tokens": 8192,
                    }
                    headers = {
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    async for token in try_provider_http_stream("https://api.deepseek.com/v1/chat/completions", payload, headers):
                        yield token
                    return
            except Exception as e:
                last_exception = e
                debug_log(f"Provider {provider} failed: {e}", "error")
        raise Exception(f"All providers failed for deepseek-ai/DeepSeek-R1: {last_exception}")
    elif model == "deepseek-ai/DeepSeek-V3":
        try:
            async for token in try_provider_doubao_stream(prompt, model="ep-20250206203431-bql9h"):
                yield token
            return
        except Exception as e:
            debug_log(f"Doubao provider failed for DeepSeek V3: {e}", "error")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "max_tokens": 4096,
            }
            headers = {
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            }
            async for token in try_provider_http_stream(TEXT_GEN_URL, payload, headers):
                yield token
    else:
        raise ValueError(f"不支持的模型: {model}")

async def generate_text_async(prompt: str, model: str) -> Dict[str, str]:
    """Non-streaming version that returns the complete response"""
    final_result = ""
    async for token in generate_text_async_stream(prompt, model):
        final_result += token
    return {"content": final_result.strip(), "intermediate_reasoning": ""}

def generate_text(prompt: str, model: Optional[str] = None) -> Dict[str, str]:
    if model is None:
        model = "deepseek-ai/DeepSeek-V3"
    return asyncio.run(generate_text_async(prompt, model))

async def try_provider_http_stream(api_url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> AsyncGenerator[str, None]:
    """Stream version that yields tokens"""
    final_result = ""
    accumulated_intermediate = ""
    first_chunk_received = False
    start_time = asyncio.get_running_loop().time()
    first_token_time = None
    token_count = 0

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", api_url, json=payload, headers=headers) as response:
                debug_log(f"HTTP响应状态码: {response.status_code} {api_url}")
                response.raise_for_status()
                
                async for chunk in response.aiter_text():
                    if not first_chunk_received:
                        elapsed = asyncio.get_running_loop().time() - start_time
                        if elapsed > 10:
                            raise TimeoutError("TTFT > 10 seconds")
                        first_chunk_received = True
                    if not chunk:
                        continue
                    for line in chunk.splitlines():
                        line = line.strip()
                        if line.startswith("data:"):
                            line = line[len("data:"):].strip()
                        if not line or line == "[DONE]" or "keep-alive" in line.lower():
                            continue
                        try:
                            delta = robust_json_parse(line)
                            for choice in delta.get("choices", []):
                                message_delta = choice.get("delta", {})
                                if message_delta.get("reasoning_content"):
                                    accumulated_intermediate += message_delta["reasoning_content"]
                                    debug_log(accumulated_intermediate, "intermediate")
                                if message_delta.get("content"):
                                    token = message_delta["content"]
                                    final_result += token
                                    token_count += 1

                                    # Record first token time and TTFT
                                    if first_token_time is None:
                                        first_token_time = asyncio.get_running_loop().time()
                                        ttft = first_token_time - start_time
                                        debug_log(f"TTFT: {ttft:.3f}s", "log")

                                    debug_log(final_result, "answer")
                                    yield token
                        except Exception as e:
                            debug_log(f"Error parsing delta: {e}. Full response: {line}", "error")

        # Calculate and log metrics
        end_time = asyncio.get_running_loop().time()
        total_latency = end_time - start_time
        if first_token_time and token_count > 0:
            tpot = (end_time - first_token_time) / token_count
            debug_log(
                f"Metrics - Total tokens: {token_count}, TPOT: {tpot:.3f}s/token, Total latency: {total_latency:.3f}s",
                "log"
            )
        else:
            debug_log(f"Metrics - Total latency: {total_latency:.3f}s", "log")
    except Exception as e:
        debug_log(f"Stream error: {str(e)}", "error")
        raise

async def try_provider_http(api_url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, str]:
    """Non-stream version that returns the complete response"""
    final_result = ""
    accumulated_intermediate = ""
    async for token in try_provider_http_stream(api_url, payload, headers):
        final_result += token
    return {"content": final_result.strip(), "intermediate_reasoning": accumulated_intermediate}

async def try_provider_doubao_stream(prompt: str, model: str) -> AsyncGenerator[str, None]:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "max_tokens": 8192,
    }
    headers = {
        "Authorization": f"Bearer {os.environ.get('ARK_API_KEY')}",
        "Content-Type": "application/json"
    }
    api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    async for token in try_provider_http_stream(api_url, payload, headers):
        yield token

async def try_provider_doubao(prompt: str, model: str) -> Dict[str, str]:
    """Non-stream version that returns the complete response"""
    final_result = ""
    async for token in try_provider_doubao_stream(prompt, model):
        final_result += token
    return {"content": final_result.strip(), "intermediate_reasoning": ""}

def generate_image(prompt: str) -> str:
    """
    调用图片生成 API，根据生成响应提取第一张图片的 URL。
    """
    try:
        payload = {
            "model": "deepseek-ai/Janus-Pro-7B",
            "prompt": prompt,
            "seed": random.randint(0, 9999999999)
        }
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(IMAGE_GEN_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        images = result.get("images", [])
        if images and isinstance(images, list) and images[0].get("url"):
            return images[0]["url"]
        else:
            debug_log("图片生成响应格式错误", "error")
            return "暂无图片"
    except Exception as e:
        debug_log(f"图片生成失败: {str(e)}", "error")
        return "暂无图片"

def text_to_speech(text: str, voice: str = "fishaudio/fish-speech-1.5:alex") -> str:
    try:
        payload = {
            "model": "fishaudio/fish-speech-1.5",
            "input": text,
            "voice": voice,
            "response_format": "mp3",
            "sample_rate": 32000,
            "stream": True,
            "speed": 1,
            "gain": 0
        }
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(TTS_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        debug_log(f"文字转语音失败: {str(e)}", "error")
        return "语音输出失败"

def transcribe_audio(file_path: str) -> str:
    try:
        files = {
            'file': ('audio.wav', open(file_path, 'rb'), 'audio/wav'),
            'model': (None, 'FunAudioLLM/SenseVoiceSmall')
        }
        headers = {
            "Authorization": f"Bearer {API_TOKEN}"
        }
        response = requests.post(TRANSCRIPTION_URL, files=files, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        debug_log(f"语音转文字失败: {str(e)}", "error")
        return ""

def extract_novel_info(chapter_text: str) -> Dict[str, Any]:
    # 检查缓存
    for story in stories_data.get("stories", []):
        if story.get("content") == chapter_text and "extracted_info" in story:
            debug_log("加载缓存提取信息", "log")
            return story["extracted_info"]

    # 未找到缓存，则调用 AI 生成提取信息
    prompt = (
        "请从下面的章节内容中提取主要剧情背景和角色名称以及角色特征。"
        "请严格以 JSON 格式返回，不包含任何额外的说明文字。返回的 JSON 对象必须包含键 "
        "\"background\" 和 \"characters\"，其中 characters 为一个列表，每个元素包含 \"name\" 和 \"description\"。\n章节内容：\n"
        + chapter_text
    )
    try:
        result = generate_text(prompt, model="deepseek-ai/DeepSeek-R1")
        info = robust_json_parse(result["content"])
        # 保存角色提取过程的中间推理
        info["extracted_intermediate_reasoning"] = result["intermediate_reasoning"]

        # 保存到 stories_data
        found = False
        for story in stories_data.get("stories", []):
            if story.get("content") == chapter_text:
                story["extracted_info"] = info
                found = True
                break
        if not found:
            new_story = {"content": chapter_text, "extracted_info": info}
            stories_data.setdefault("stories", []).append(new_story)

        with open(stories_path, "w", encoding="utf-8") as f:
            json.dump(stories_data, f, ensure_ascii=False, indent=2)

        return info
    except Exception as e:
        debug_log(f"提取小说信息失败: {str(e)}", "error")
        return {
            "background": "",
            "characters": [],
            "extracted_intermediate_reasoning": ""
        }

def generate_levels(chapter_text: str, extracted_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    # 检查缓存
    for story in stories_data.get("stories", []):
        if story.get("content") == chapter_text and "generated_levels" in story:
            debug_log("加载缓存关卡信息", "log")
            return story["generated_levels"]

    debug_log("开始生成关卡", "log")

    # 获取角色信息
    characters_info = ""
    if extracted_info and extracted_info.get("characters"):
        characters_info = "角色信息：" + json.dumps(extracted_info.get("characters"), ensure_ascii=False) + "\n"

    prompt = (
        "请根据下面的章节内容以及提供的角色信息设计出若干个关卡，每个关卡包含关卡描述和通关条件，每个关卡都用一段话描述。描述时给用户2~3个选项供其选择"
        "请严格以 JSON 数组格式返回，不包含任何额外的说明文字。数组中的每个元素应为一个对象，格式为 "
        "{\"level\": <数字>, \"description\": \"关卡剧情描述\", \"pass_condition\": \"通关条件描述\"}。\n" +
        characters_info +
        "章节内容：\n" + chapter_text
    )

    try:
        result = generate_text(prompt, model="deepseek-ai/DeepSeek-R1")
        levels = robust_json_parse(result["content"])
        if not isinstance(levels, list):
            debug_log("生成的关卡格式错误", "error")
            levels = []

        # 保存到 stories_data
        for story in stories_data.get("stories", []):
            if story.get("content") == chapter_text:
                story["generated_levels"] = levels
                break

        with open(stories_path, "w", encoding="utf-8") as f:
            json.dump(stories_data, f, ensure_ascii=False, indent=2)

        debug_log("关卡生成结果: " + json.dumps(levels, ensure_ascii=False), "final")
        return levels
    except Exception as e:
        debug_log(f"关卡生成失败: {str(e)}", "error")
        return []

def evaluate_level(pass_condition: str, user_response: str, chat_history: str, overall_plot: str) -> Tuple[bool, str]:
    prompt = (
        f"请根据以下关卡通关条件判断用户的回答是否满足要求。用户的回答只要与通关条件中的描述相似即可通关。\n"
        f"关卡通关条件：{pass_condition}\n"
        f"用户回答：{user_response}\n"
        f"整体剧情：{overall_plot}\n"
        f"聊天记录：{chat_history}\n"
        "请分析用户回答是否大概完成了通关条件要求的任务。\n"
        "请直接回复\"通过\"或\"未通过\"，不要包含其他内容。"
    )
    try:
        result = generate_text(prompt)
        # 严格匹配完整的"通过"二字
        return result["content"].strip() == "通过", result["content"]
    except Exception as e:
        debug_log(f"关卡评估失败: {str(e)}", "error")
        return False, "评估失败"

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/create_game", methods=["POST"])
def create_game():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400
            
        chapter_text = data.get("chapter_text", "").strip()
        if not chapter_text:
            return jsonify({"error": "章节内容为空"}), 400

        # 提取小说信息
        extracted_info = extract_novel_info(chapter_text)
        if not extracted_info:
            return jsonify({"error": "小说信息提取失败"}), 500

        # 生成关卡信息
        levels = generate_levels(chapter_text, extracted_info)
        if not levels:
            return jsonify({"error": "关卡生成失败"}), 500

        # 创建游戏会话
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "extracted_info": extracted_info,
            "characters": extracted_info.get("characters", []),
            "levels": levels,
            "current_level_index": 0,
            "chat_history": "",
            "overall_plot": extracted_info.get("background", ""),
            "created_at": datetime.datetime.now().isoformat()
        }
        debug_log(f"游戏创建成功，会话ID: {session_id}", "log")

        # 检查是否已生成
        story_generated = any(
            story.get("content") == chapter_text and "extracted_info" in story 
            for story in stories_data.get("stories", [])
        )

        return jsonify({
            "session_id": session_id,
            "characters": extracted_info.get("characters", []),
            "story_generated": story_generated,
            "message": "游戏创建成功"
        })
    except Exception as e:
        debug_log(f"游戏创建失败: {str(e)}", "error")
        return jsonify({"error": f"游戏创建失败: {str(e)}"}), 500

@app.route("/select_character", methods=["POST"])
def select_character():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400

        session_id = data.get("session_id")
        if not session_id or session_id not in sessions:
            return jsonify({"error": "无效的 session_id"}), 400

        character_index = data.get("character_index")
        session = sessions[session_id]
        characters = session["characters"]
        
        if character_index is None or not isinstance(character_index, int) or character_index < 0 or character_index >= len(characters):
            return jsonify({"error": "无效的角色选择"}), 400

        session["user_role"] = characters[character_index]["name"]
        return jsonify({"message": f"你选择的角色是 {session['user_role']}"})
    except Exception as e:
        debug_log(f"角色选择失败: {str(e)}", "error")
        return jsonify({"error": f"角色选择失败: {str(e)}"}), 500

@app.route("/get_level", methods=["POST"])
def get_level():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400

        session_id = data.get("session_id")
        if not session_id or session_id not in sessions:
            return jsonify({"error": "无效的 session_id"}), 400

        session = sessions[session_id]
        current_index = session["current_level_index"]
        levels = session["levels"]

        if current_index >= len(levels):
            return jsonify({"message": "游戏结束", "game_over": True})

        level = levels[current_index]

        # 获取 AI 角色
        user_role = session["user_role"]
        available_roles = [c for c in session["characters"] if c['name'] != user_role]
        ai_role = random.choice(available_roles)["name"] if available_roles else "旁白"

        # 检查图片缓存
        if "level_images" not in session:
            session["level_images"] = {}

        # 检查 stories.json 中的图片缓存
        for story in stories_data.get("stories", []):
            if "generated_levels" in story and len(story["generated_levels"]) > current_index:
                cached_level = story["generated_levels"][current_index]
                if "level_image" in cached_level and cached_level["level_image"].startswith("http"):
                    level["level_image"] = cached_level["level_image"]
                    session["level_images"][current_index] = cached_level["level_image"]
                    return jsonify({
                        "level_number": level.get("level"),
                        "description": level.get("description"),
                        "pass_condition": level.get("pass_condition"),
                        "level_image": level["level_image"],
                        "ai_role": ai_role,
                        "game_over": False
                    })

        # 检查会话中的图片缓存
        if current_index in session["level_images"]:
            level["level_image"] = session["level_images"][current_index]
        else:
            # 异步生成图片
            level_image_prompt = f"根据关卡描述生成一张背景图片的描述。描述：{level.get('description')}"
            level["level_image"] = "图片生成中..."
            
            def generate_image_background(level, prompt, session, current_index):
                try:
                    image_url = generate_image(prompt)
                    level["level_image"] = image_url
                    # 保存到会话缓存
                    session["level_images"][current_index] = image_url
                    # 保存到 stories.json
                    for story in stories_data.get("stories", []):
                        if "generated_levels" in story and len(story["generated_levels"]) > current_index:
                            story["generated_levels"][current_index]["level_image"] = image_url
                            with open(stories_path, "w", encoding="utf-8") as f:
                                json.dump(stories_data, f, ensure_ascii=False, indent=2)
                    debug_log(f"图片生成完成，URL: {image_url}")
                except Exception as e:
                    debug_log(f"图片生成失败: {str(e)}", "error")
                    level["level_image"] = "图片生成失败"
            
            import threading
            threading.Thread(
                target=generate_image_background,
                args=(level, level_image_prompt, session, current_index),
                daemon=True
            ).start()

        return jsonify({
            "level_number": level.get("level"),
            "description": level.get("description"),
            "pass_condition": level.get("pass_condition"),
            "level_image": level["level_image"],
            "ai_role": ai_role,
            "game_over": False
        })
    except Exception as e:
        debug_log(f"获取关卡失败: {str(e)}", "error")
        return jsonify({"error": f"获取关卡失败: {str(e)}"}), 500

@app.route("/submit_response", methods=["POST"])
def submit_response():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400

        session_id = data.get("session_id")
        if not session_id or session_id not in sessions:
            return jsonify({"error": "无效的 session_id"}), 400

        user_response = data.get("user_response", "").strip()
        if not user_response:
            return jsonify({"error": "回应内容为空"}), 400

        session = sessions[session_id]
        current_index = session["current_level_index"]
        levels = session["levels"]

        if current_index >= len(levels):
            return jsonify({"message": "游戏已经结束"}), 400

        level = levels[current_index]
        overall_plot = session["overall_plot"]
        chat_history = session["chat_history"]

        passed, evaluation_feedback = evaluate_level(
            level.get("pass_condition"),
            user_response,
            chat_history,
            overall_plot
        )

        # 记录用户回应到聊天历史
        session["chat_history"] += f"\n用户：{user_response}\n"
        session["chat_history"] += f"系统评价：{evaluation_feedback}\n"

        if passed:
            session["current_level_index"] += 1
            message = f"恭喜，你通过了关卡 {level.get('level')}！"
        else:
            message = "关卡未通过，请继续尝试。"

        return jsonify({
            "passed": passed,
            "evaluation_feedback": evaluation_feedback,
            "message": message,
            "current_level_index": session["current_level_index"],
            "total_levels": len(levels)
        })
    except Exception as e:
        debug_log(f"提交回应失败: {str(e)}", "error")
        return jsonify({"error": f"提交回应失败: {str(e)}"}), 500

@app.route("/random_story", methods=["GET"])
def random_story():
    try:
        if "stories" in stories_data and stories_data["stories"]:
            story = random.choice(stories_data["stories"])
            return jsonify(story)
        else:
            return jsonify({"error": "没有找到故事"}), 404
    except Exception as e:
        debug_log(f"获取随机故事失败: {str(e)}", "error")
        return jsonify({"error": f"获取随机故事失败: {str(e)}"}), 500

@app.route("/stream_level_dialogue", methods=["GET"])
def stream_level_dialogue():
    """
    流式返回当前关卡的 AI 对话，每个 token 以 SSE 格式发送。
    调用文本生成 API，并实时提取对话 token。
    """
    try:
        session_id = request.args.get("session_id")
        if not session_id or session_id not in sessions:
            return Response("data: 无效的 session_id\n\n", mimetype="text/event-stream")

        session = sessions[session_id]
        current_index = session["current_level_index"]
        levels = session["levels"]

        if current_index >= len(levels):
            return Response("data: 游戏结束\n\n", mimetype="text/event-stream")

        level = levels[current_index]
        overall_plot = session["overall_plot"]
        chat_history = session["chat_history"]
        user_role = session["user_role"]
        available_roles = [c for c in session["characters"] if c['name'] != user_role]
        ai_role = random.choice(available_roles)["name"] if available_roles else "旁白"

        dialogue_prompt = (
            f"请以{ai_role}的身份，根据整体剧情、关卡描述和聊天历史，对用户的回答进行回应并引导用户继续尝试。\n"
            f"整体剧情：{overall_plot}\n"
            f"关卡描述：{level.get('description')}\n"
            f"通关条件：{level.get('pass_condition')}\n"
            f"聊天历史：=== BEGIN CHAT HISTORY ===\n{chat_history if chat_history else '无'}\n=== END CHAT HISTORY ===\n"
            "请发表一句话。"
        )

        debug_log("stream_level_dialogue:\n" + dialogue_prompt, "log")

        async def generate_stream():
            try:
                async for token in generate_text_async_stream(dialogue_prompt, "deepseek-ai/DeepSeek-V3"):
                    yield f"data: {token}\n\n"
            except Exception as e:
                debug_log(f"对话生成失败: {str(e)}", "error")
                yield f"data: 对话生成失败\n\n"

        return Response(
            sync_generate_stream(generate_stream()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
    except Exception as e:
        debug_log(f"对话流处理失败: {str(e)}", "error")
        return Response(f"data: 对话流处理失败: {str(e)}\n\n", mimetype="text/event-stream")

@app.route("/list_stories", methods=["GET"])
def list_stories():
    try:
        stories_list = []
        for story in stories_data.get("stories", []):
            content = story.get("content", "")
            title = story.get("title", "").strip() or "无标题"
            author = story.get("author", "").strip() or "未知作者"
            excerpt = content[:50] + ("..." if len(content) > 50 else "")
            stories_list.append({
                "id": len(stories_list),  # Use current index as id
                "title": title,
                "author": author,
                "excerpt": excerpt,
                "content": content,
                "generated": "extracted_info" in story
            })
        return jsonify(stories_list)
    except Exception as e:
        debug_log(f"获取故事列表失败: {str(e)}", "error")
        return jsonify({"error": f"获取故事列表失败: {str(e)}"}), 500

@app.route("/update_chat_history", methods=["POST"])
def update_chat_history():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "无效的请求数据"}), 400

        session_id = data.get("session_id")
        if not session_id or session_id not in sessions:
            return jsonify({"error": "无效的 session_id"}), 400

        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "消息不能为空"}), 400

        session = sessions[session_id]
        session["chat_history"] += f"\n{message}\n"

        return jsonify({"message": "聊天历史更新成功"})
    except Exception as e:
        debug_log(f"更新聊天历史失败: {str(e)}", "error")
        return jsonify({"error": f"更新聊天历史失败: {str(e)}"}), 500

def sync_generate_stream(async_gen):
    """
    将异步生成器 async_gen 转换为同步生成器，以便通过 WSGI 服务器流式发送。
    """
    from queue import Queue
    import threading
    q = Queue()

    def run():
        async def main():
            try:
                async for token in async_gen:
                    q.put(token)
            except Exception as e:
                debug_log(f"生成器运行失败: {str(e)}", "error")
                q.put(f"data: 生成器运行失败: {str(e)}\n\n")
            finally:
                q.put(None)
        asyncio.run(main())

    threading.Thread(target=run, daemon=True).start()
    while True:
        token = q.get()
        if token is None:
            break
        yield token

if __name__ == "__main__":
    app.run(debug=True, port=8888, threaded=True)
