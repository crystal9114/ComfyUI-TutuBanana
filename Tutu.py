import os
import io
import math
import random
import torch
import requests
import time
import numpy as np
from PIL import Image
from io import BytesIO
import json
import comfy.utils
import re
import aiohttp
import asyncio
import base64
import uuid
import folder_paths
import mimetypes
import cv2
import shutil
from .utils import pil2tensor, tensor2pil
from comfy.utils import common_upscale
from comfy.comfy_types import IO


def get_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Tutuapi.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Tutuapi.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


# ===== 预设管理系统 =====
def get_presets_file():
    """获取预设文件路径"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'presets.json')

def load_presets():
    """加载预设配置"""
    try:
        with open(get_presets_file(), 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 如果文件不存在，创建默认结构
        default_presets = {
            "gemini": []
        }
        save_all_presets(default_presets)
        return default_presets
    except json.JSONDecodeError:
        print("[Tutu] 预设文件格式错误，使用默认配置")
        return {"gemini": []}

def save_all_presets(presets):
    """保存所有预设到文件"""
    with open(get_presets_file(), 'w', encoding='utf-8') as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)

def save_preset(category, name, config, description=""):
    """保存单个预设"""
    if not name.strip():
        raise ValueError("预设名称不能为空")
        
    presets = load_presets()
    if category not in presets:
        presets[category] = []
    
    # 检查是否已存在同名预设
    existing_names = [p["name"] for p in presets[category]]
    if name in existing_names:
        # 如果存在同名，添加时间戳后缀
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{timestamp}"
    
    preset = {
        "id": str(uuid.uuid4()),
        "name": name,
        "description": description,
        "config": config,
        "created_time": time.time(),
        "created_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    presets[category].append(preset)
    save_all_presets(presets)
    return preset["id"]

def delete_preset(category, preset_id):
    """删除指定预设"""
    presets = load_presets()
    if category not in presets:
        return False
    
    original_count = len(presets[category])
    presets[category] = [p for p in presets[category] if p["id"] != preset_id]
    
    if len(presets[category]) < original_count:
        save_all_presets(presets)
        return True
    return False

def get_preset_by_name(category, name):
    """根据名称获取预设"""
    presets = load_presets()
    if category not in presets:
        return None
    
    for preset in presets[category]:
        if preset["name"] == name:
            return preset
    return None

def get_preset_by_id(category, preset_id):
    """根据ID获取预设"""
    presets = load_presets()
    if category not in presets:
        return None
    
    for preset in presets[category]:
        if preset["id"] == preset_id:
            return preset
    return None

def get_preset_names(category):
    """获取指定分类的所有预设名称"""
    presets = load_presets()
    if category not in presets:
        return []
    return [p["name"] for p in presets[category]]

def update_preset(category, preset_id, new_config=None, new_name=None, new_description=None):
    """更新现有预设"""
    presets = load_presets()
    if category not in presets:
        return False
    
    for preset in presets[category]:
        if preset["id"] == preset_id:
            if new_config is not None:
                preset["config"] = new_config
            if new_name is not None:
                preset["name"] = new_name
            if new_description is not None:
                preset["description"] = new_description
            preset["updated_time"] = time.time()
            preset["updated_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            save_all_presets(presets)
            return True
    return False

# ===== 预设管理系统结束 =====

# ===== 基础视频适配器类 =====
class ComflyVideoAdapter:
    def __init__(self, url):
        self.url = url if url else ""
        
    def __str__(self):
        return self.url


############################# Gemini ###########################

class TutuGeminiAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "api_provider": (
                    [
                        "ai.comfly.chat",
                        "OpenRouter"
                    ],
                    {"default": "ai.comfly.chat"}
                ),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子，改变此值会强制重新生成图片"
                }),
            },
            "optional": {
                "comfly_api_key": ("STRING", {
                    "default": "", 
                    "placeholder": "ai.comfly.chat API Key (optional, leave blank to use config)"
                }),
                "openrouter_api_key": ("STRING", {
                    "default": "", 
                    "placeholder": "OpenRouter API Key (optional, leave blank to use config)"
                }),
                "input_image_1": ("IMAGE",),  
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "response")
    FUNCTION = "process"
    CATEGORY = "Tutu"

    def __init__(self):
        config = get_config()
        self.comfly_api_key = config.get('comfly_api_key', config.get('api_key', ''))  # 向后兼容
        self.openrouter_api_key = config.get('openrouter_api_key', '')
        self.timeout = 120
    
    def add_random_variation(self, prompt, seed=0):
        """
        在提示词末尾添加隐藏的随机标识
        确保每次运行都能得到不同结果
        """
        if seed == 0:
            random_id = random.randint(10000, 99999)
        else:
            rng = random.Random(seed)
            random_id = rng.randint(10000, 99999)
        
        return f"{prompt} [variation-{random_id}]"
    
    def _truncate_base64_in_response(self, text, max_base64_len=100):
        """截断响应文本中的base64内容以避免刷屏"""
        import re
        
        def replace_base64(match):
            full_base64 = match.group(0)
            prefix = full_base64.split(',')[0] + ','  # 保留 data:image/xxx;base64, 部分
            base64_data = full_base64[len(prefix):]
            
            if len(base64_data) > max_base64_len:
                truncated = base64_data[:max_base64_len] + f"... [truncated {len(base64_data) - max_base64_len} chars]"
                return prefix + truncated
            return full_base64
        
        # 匹配 data:image/xxx;base64,xxxxxx 格式
        pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
        result = re.sub(pattern, replace_base64, text)
        
        return result
    
    def _sanitize_content_for_debug(self, content):
        """为调试输出清理内容（移除敏感数据）"""
        if isinstance(content, str):
            # 如果内容包含base64图片，截断显示
            if 'data:image/' in content:
                parts = content.split('data:image/')
                if len(parts) > 1:
                    # 只显示第一部分文本 + base64开头
                    base64_start = parts[1][:50] + "..." if len(parts[1]) > 50 else parts[1]
                    return parts[0] + f"data:image/{base64_start}"
            return content[:200] + "..." if len(content) > 200 else content
        elif isinstance(content, list):
            return [self._sanitize_content_for_debug(item) for item in content]
        elif isinstance(content, dict):
            return {k: self._sanitize_content_for_debug(v) for k, v in content.items()}
        else:
            return content

    def get_current_api_key(self, api_provider):
        """根据API提供商获取对应的API key"""
        if api_provider == "OpenRouter":
            # 节点里没填就用环境里的入口 key（密匣 openrouter.env 的 OPENROUTER_API_KEY）
            return self.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        else:
            return self.comfly_api_key
            
    def display_preset_list(self):
        """显示所有预设的详细信息"""
        print(f"\n[Tutu] 📋 ======== 预设列表 ========")
        
        try:
            presets = load_presets()
            gemini_presets = presets.get("gemini", [])
            
            if not gemini_presets:
                print(f"[Tutu] ⚪ 当前没有保存的预设")
                print(f"[Tutu] 💡 提示：在 'save_as_preset' 中输入名称来保存预设")
                return
            
            print(f"[Tutu] 📊 总共 {len(gemini_presets)} 个预设:")
            print(f"[Tutu] " + "-" * 50)
            
            for i, preset in enumerate(gemini_presets, 1):
                name = preset.get("name", "未知名称")
                description = preset.get("description", "无描述")
                created_date = preset.get("created_date", "未知时间")
                
                print(f"[Tutu] {i}. 名称: {name}")
                print(f"[Tutu]    描述: {description}")
                print(f"[Tutu]    创建时间: {created_date}")
                
                # 显示提示词模板（如果有）
                config = preset.get("config", {})
                if "prompt_template" in config:
                    template = config["prompt_template"]
                    # 截断长模板以便显示
                    if len(template) > 100:
                        template_preview = template[:100] + "..."
                    else:
                        template_preview = template
                    print(f"[Tutu]    模板: {template_preview}")
                
                print(f"[Tutu] " + "-" * 30)
                
        except Exception as e:
            print(f"[Tutu] ❌ 获取预设列表时出错: {str(e)}")
        
        print(f"[Tutu] 📋 ======== 预设列表结束 ========\n")

    def get_headers(self, api_provider="ai.comfly.chat"):
        current_api_key = self.get_current_api_key(api_provider)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {current_api_key}"
        }
        
        # OpenRouter需要额外的headers
        if api_provider == "OpenRouter":
            headers.update({
                "HTTP-Referer": "https://comfyui.com",
                "X-Title": "ComfyUI Tutu Nano Banana"
            })
        
        return headers
        return headers

    def image_to_base64(self, image):
        """将图片转换为base64，保持原始质量"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def upload_image(self, image, max_retries=3):
        """上传图像到临时托管服务，支持多个备选服务"""
        
        # 准备图像数据
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        # 备选上传服务列表（按优先级排序，使用最简单可靠的服务）
        upload_services = [
            {
                "name": "0x0.st",
                "url": "https://0x0.st",
                "method": "POST",
                "files_key": "file", 
                "response_key": "url"
            },
            {
                "name": "tmpfiles.org", 
                "url": "https://tmpfiles.org/api/v1/upload",
                "method": "POST", 
                "files_key": "file",
                "response_key": "data.url"
            },
            {
                "name": "uguu.se",
                "url": "https://uguu.se/upload",
                "method": "POST",
                "files_key": "files[]",
                "response_key": "url"
            },
            {
                "name": "x0.at",
                "url": "https://x0.at",
                "method": "POST",
                "files_key": "file",
                "response_key": "url"
            }
        ]
        
        for service in upload_services:
            for attempt in range(max_retries):
                try:
                    print(f"[Tutu DEBUG] 尝试上传到 {service['name']} (尝试 {attempt + 1}/{max_retries})...")
                    
                    # 重置buffer位置
                    buffered.seek(0)
                    
                    # 准备文件上传
                    files = {service['files_key']: ('image.png', buffered.getvalue(), 'image/png')}
                    
                    # 准备额外数据（如果需要）
                    data = service.get('extra_data', {})
                    
                    # 发送上传请求 - 使用独立session避免代理连接复用问题
                    session = requests.Session()
                    session.trust_env = True
                    try:
                        response = session.post(
                            service['url'], 
                            files=files,
                            data=data,
                            timeout=30,
                            headers={'User-Agent': 'ComfyUI-Tutu/1.0'}
                        )
                    finally:
                        session.close()
                    
                    if response.status_code == 200:
                        # 根据服务类型提取URL
                        if service['name'] in ["0x0.st", "x0.at"]:
                            # 这些服务返回纯文本URL
                            image_url = response.text.strip()
                        elif service['name'] == "uguu.se":
                            # uguu.se 返回JSON数组
                            try:
                                result = response.json()
                                if isinstance(result, list) and len(result) > 0:
                                    image_url = result[0].get('url', '')
                                else:
                                    image_url = result.get('url', '')
                            except:
                                image_url = response.text.strip()
                        else:
                            # 其他服务返回JSON
                            try:
                                result = response.json()
                                if service['name'] == "tmpfiles.org" and 'data' in result:
                                    image_url = result['data'].get('url', '')
                                else:
                                    # 通用解析
                                    keys = service['response_key'].split('.')
                                    image_url = result
                                    for key in keys:
                                        if isinstance(image_url, dict):
                                            image_url = image_url.get(key, '')
                                        else:
                                            image_url = ''
                                            break
                                        if not image_url:
                                            break
                            except Exception as e:
                                print(f"[Tutu DEBUG] JSON解析失败: {str(e)}")
                                # JSON解析失败，尝试纯文本
                                image_url = response.text.strip()
                        
                        if image_url and image_url.startswith('http'):
                            print(f"[Tutu DEBUG] 成功上传到 {service['name']}: {image_url}")
                            return image_url
                        else:
                            print(f"[Tutu DEBUG] {service['name']} 响应格式异常: {result}")
                    else:
                        print(f"[Tutu DEBUG] {service['name']} 上传失败，状态码: {response.status_code}")
                        
                except Exception as e:
                    print(f"[Tutu DEBUG] {service['name']} 上传出错 (尝试 {attempt + 1}): {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # 等待1秒后重试
                    continue
                    
        # 所有服务都失败，返回None
        print(f"[Tutu DEBUG] 所有上传服务都失败，将使用压缩的base64格式")
        return None

    def process_sse_stream(self, response, api_provider="ai.comfly.chat"):
        """Process Server-Sent Events (SSE) stream from the API with provider-specific handling"""
        accumulated_content = ""
        chunk_count = 0
        raw_response_parts = []
        current_json_buffer = ""
        
        print(f"[Tutu DEBUG] 开始处理SSE流 (API: {api_provider})...")
        
        # Different APIs might have different response structures
        is_comfly = api_provider == "ai.comfly.chat"
        is_openrouter = api_provider == "OpenRouter"
        
        try:
            for line in response.iter_lines(decode_unicode=True, chunk_size=None):
                if line:
                    print(f"[Tutu DEBUG] SSE原始行: {repr(line[:100])}")
                    
                if line and line.startswith('data: '):
                    chunk_count += 1
                    data_content = line[6:]  # Remove 'data: ' prefix
                    
                    print(f"[Tutu DEBUG] 处理第{chunk_count}个数据块...")
                    
                    if data_content.strip() == '[DONE]':
                        print(f"[Tutu DEBUG] 收到结束信号[DONE]")
                        break
                    
                    # 累积可能被分割的JSON数据
                    current_json_buffer += data_content
                    
                    try:
                        # 尝试解析累积的JSON
                        chunk_data = json.loads(current_json_buffer)
                        print(f"[Tutu DEBUG] JSON解析成功: {list(chunk_data.keys())}")
                        
                        # 清空缓冲区，因为JSON解析成功了
                        current_json_buffer = ""
                        
                        # Extract content from the chunk
                        if 'choices' in chunk_data and chunk_data['choices']:
                            choice = chunk_data['choices'][0]
                            print(f"[Tutu DEBUG] 完整Choice结构: {choice}")
                            
                            # 检查delta中的所有字段
                            if 'delta' in choice:
                                delta = choice['delta']
                                print(f"[Tutu DEBUG] Delta所有字段: {list(delta.keys())}")
                                
                                # 检查content字段
                                if 'content' in delta:
                                    content = delta['content']
                                    print(f"[Tutu DEBUG] Delta.content: {repr(content[:200]) if content else 'None/Empty'}")
                                    if content:
                                        # 修复编码问题
                                        try:
                                            if isinstance(content, str):
                                                content = content.encode('latin1').decode('utf-8')
                                        except (UnicodeDecodeError, UnicodeEncodeError):
                                            pass
                                        accumulated_content += content
                                        print(f"[Tutu DEBUG] 添加delta.content: {repr(content[:100])}")
                                
                                # 检查是否有其他包含图片数据的字段
                                for key, value in delta.items():
                                    if key != 'content' and isinstance(value, str):
                                        print(f"[Tutu DEBUG] Delta.{key}: {repr(value[:200]) if len(str(value)) > 200 else repr(value)}")
                                        # 检查是否是图片数据
                                        if 'data:image/' in str(value) or 'base64,' in str(value):
                                            print(f"[Tutu DEBUG] 🎯找到图片数据在delta.{key}中!")
                                            accumulated_content += str(value)
                                            print(f"[Tutu DEBUG] 添加图片数据: {len(str(value))}字符")
                                    
                            # 检查message中的内容
                            elif 'message' in choice:
                                message = choice['message']
                                print(f"[Tutu DEBUG] Message所有字段: {list(message.keys())}")
                                
                                if 'content' in message:
                                    content = message['content']
                                    print(f"[Tutu DEBUG] Message.content: {repr(content[:200]) if content else 'None/Empty'}")
                                    if content:
                                        try:
                                            if isinstance(content, str):
                                                content = content.encode('latin1').decode('utf-8')
                                        except (UnicodeDecodeError, UnicodeEncodeError):
                                            pass
                                        accumulated_content += content
                                        print(f"[Tutu DEBUG] 添加message.content: {repr(content[:100])}")
                                
                                # 检查message中的其他字段
                                for key, value in message.items():
                                    if key != 'content' and isinstance(value, str):
                                        print(f"[Tutu DEBUG] Message.{key}: {repr(value[:200]) if len(str(value)) > 200 else repr(value)}")
                                        # 检查是否是图片数据
                                        if 'data:image/' in str(value) or 'base64,' in str(value):
                                            print(f"[Tutu DEBUG] 🎯找到图片数据在message.{key}中!")
                                            accumulated_content += str(value)
                                            print(f"[Tutu DEBUG] 添加图片数据: {len(str(value))}字符")
                            
                            # 检查choice的其他字段，可能图片数据在别处
                            for key, value in choice.items():
                                if key not in ['delta', 'message', 'index', 'finish_reason', 'native_finish_reason', 'logprobs']:
                                    if isinstance(value, str) and ('data:image/' in value or 'base64,' in value):
                                        print(f"[Tutu DEBUG] 🎯找到图片数据在choice.{key}中!")
                                        accumulated_content += value
                                        print(f"[Tutu DEBUG] 添加图片数据: {len(value)}字符")
                                    elif value:
                                        print(f"[Tutu DEBUG] Choice.{key}: {repr(str(value)[:200])}")
                        
                        # 检查整个chunk中是否有图片数据 - 针对不同API提供商
                        chunk_str = json.dumps(chunk_data)
                        
                        if is_comfly:
                            # comfly可能把图片数据放在不同的位置
                            print(f"[Tutu DEBUG] 🔍 comfly专用检查: 搜索整个响应块")
                            
                            # 检查是否有任何图片相关的字段
                            for key, value in chunk_data.items():
                                if key not in ['id', 'object', 'created', 'model', 'system_fingerprint', 'choices', 'usage']:
                                    if isinstance(value, str) and ('data:image/' in value or 'http' in value):
                                        print(f"[Tutu DEBUG] 🎯 comfly在{key}字段发现可能的图片数据!")
                                        accumulated_content += " " + value
                                    elif value:
                                        print(f"[Tutu DEBUG] comfly额外字段{key}: {repr(str(value)[:100])}")
                            
                            # 检查choices之外的图片数据
                            if 'data:image/' in chunk_str or 'generated_image' in chunk_str or 'image_url' in chunk_str:
                                print(f"[Tutu DEBUG] 🎯 comfly JSON中发现图片相关数据!")
                                print(f"[Tutu DEBUG] 完整chunk (前500字符): {chunk_str[:500]}")
                                
                                # 尝试提取所有可能的图片URL
                                import re
                                patterns = [
                                    r'data:image/[^",\s]+',  # base64 图片
                                    r'https?://[^",\s]+\.(?:png|jpg|jpeg|gif|webp)',  # 图片URL
                                    r'"image_url":\s*"([^"]+)"',  # JSON中的image_url字段
                                    r'"generated_image":\s*"([^"]+)"'  # 生成图片字段
                                ]
                                
                                for pattern in patterns:
                                    urls = re.findall(pattern, chunk_str)
                                    if urls:
                                        print(f"[Tutu DEBUG] 🎯 comfly用模式 {pattern} 找到: {len(urls)}个URL")
                                        for url in urls:
                                            if url.startswith('data:image/'):
                                                print(f"[Tutu DEBUG] 🎯 comfly提取base64图片")
                                            else:
                                                print(f"[Tutu DEBUG] 🎯 comfly提取URL: {url[:50]}...") 
                                            accumulated_content += " " + url
                                            
                        elif is_openrouter:
                            # OpenRouter的原有处理逻辑
                            if 'data:image/' in chunk_str:
                                print(f"[Tutu DEBUG] 🎯 OpenRouter在JSON中发现图片数据!")
                                import re
                                image_urls_in_chunk = re.findall(r'data:image/[^"]+', chunk_str)
                                if image_urls_in_chunk:
                                    for url in image_urls_in_chunk:
                                        if url.startswith('data:image/'):
                                            print(f"[Tutu DEBUG] 🎯 OpenRouter提取base64图片")
                                        else:
                                            print(f"[Tutu DEBUG] 🎯 OpenRouter提取URL: {url[:50]}...")
                                        accumulated_content += " " + url
                        
                        # 保存完整的响应数据用于调试
                        raw_response_parts.append(chunk_data)
                                
                    except json.JSONDecodeError as e:
                        print(f"[Tutu DEBUG] JSON解析失败: {e}")
                        print(f"[Tutu DEBUG] 当前缓冲区内容: {repr(current_json_buffer[:200])}")
                        # 不要清空缓冲区，可能还有更多数据到来
                        
                elif line:
                    # 处理不以"data: "开头的行，它们可能是JSON的续行
                    print(f"[Tutu DEBUG] 非data行: {repr(line[:100])}")
                    if current_json_buffer:
                        # 如果有未完成的JSON，尝试添加这行
                        # 先尝试修复编码问题
                        try:
                            # 如果line包含二进制数据，尝试解码
                            if isinstance(line, str) and '\\x' in repr(line):
                                # 尝试修复UTF-8编码问题
                                fixed_line = line.encode('latin1').decode('utf-8')
                                print(f"[Tutu DEBUG] 编码修复后: {repr(fixed_line)}")
                            else:
                                fixed_line = line
                        except (UnicodeDecodeError, UnicodeEncodeError):
                            fixed_line = line
                        
                        current_json_buffer += fixed_line
                        try:
                            chunk_data = json.loads(current_json_buffer)
                            print(f"[Tutu DEBUG] 续行JSON解析成功: {list(chunk_data.keys())}")
                            
                            # 清空缓冲区
                            current_json_buffer = ""
                            
                            # 处理这个合并后的chunk_data（重要！）
                            if 'choices' in chunk_data and chunk_data['choices']:
                                choice = chunk_data['choices'][0]
                                print(f"[Tutu DEBUG] 续行完整Choice结构: {choice}")
                                
                                # 检查delta中的所有字段
                                if 'delta' in choice:
                                    delta = choice['delta']
                                    print(f"[Tutu DEBUG] 续行Delta所有字段: {list(delta.keys())}")
                                    
                                    # 检查content字段
                                    if 'content' in delta:
                                        content = delta['content']
                                        print(f"[Tutu DEBUG] 续行Delta.content: {repr(content[:200]) if content else 'None/Empty'}")
                                        if content:
                                            try:
                                                if isinstance(content, str):
                                                    content = content.encode('latin1').decode('utf-8')
                                            except (UnicodeDecodeError, UnicodeEncodeError):
                                                pass
                                            accumulated_content += content
                                            print(f"[Tutu DEBUG] 从续行添加delta.content: {repr(content[:100])}")
                                    
                                    # 检查其他字段中的图片数据
                                    for key, value in delta.items():
                                        if key != 'content' and isinstance(value, str):
                                            print(f"[Tutu DEBUG] 续行Delta.{key}: {repr(value[:200]) if len(str(value)) > 200 else repr(value)}")
                                            if 'data:image/' in str(value) or 'base64,' in str(value):
                                                print(f"[Tutu DEBUG] 🎯续行中找到图片数据在delta.{key}!")
                                                accumulated_content += str(value)
                                                print(f"[Tutu DEBUG] 从续行添加图片数据: {len(str(value))}字符")
                                        
                                # 检查message中的内容
                                elif 'message' in choice:
                                    message = choice['message']
                                    print(f"[Tutu DEBUG] 续行Message所有字段: {list(message.keys())}")
                                    
                                    if 'content' in message:
                                        content = message['content']
                                        print(f"[Tutu DEBUG] 续行Message.content: {repr(content[:200]) if content else 'None/Empty'}")
                                        if content:
                                            try:
                                                if isinstance(content, str):
                                                    content = content.encode('latin1').decode('utf-8')
                                            except (UnicodeDecodeError, UnicodeEncodeError):
                                                pass
                                            accumulated_content += content
                                            print(f"[Tutu DEBUG] 从续行添加message.content: {repr(content[:100])}")
                                    
                                    # 检查message中的其他字段
                                    for key, value in message.items():
                                        if key != 'content' and isinstance(value, str):
                                            if 'data:image/' in str(value) or 'base64,' in str(value):
                                                print(f"[Tutu DEBUG] 🎯续行中找到图片数据在message.{key}!")
                                                accumulated_content += str(value)
                                                print(f"[Tutu DEBUG] 从续行添加图片数据: {len(str(value))}字符")
                                
                                # 检查choice中的其他字段
                                for key, value in choice.items():
                                    if key not in ['delta', 'message', 'index', 'finish_reason', 'native_finish_reason', 'logprobs']:
                                        if isinstance(value, str) and ('data:image/' in value or 'base64,' in value):
                                            print(f"[Tutu DEBUG] 🎯续行中找到图片数据在choice.{key}!")
                                            accumulated_content += value
                                            print(f"[Tutu DEBUG] 从续行添加图片数据: {len(value)}字符")
                            
                            # 续行中的图片数据检查 - 针对不同API提供商
                            chunk_str = json.dumps(chunk_data)
                            
                            if is_comfly:
                                # comfly续行处理
                                print(f"[Tutu DEBUG] 🔍 comfly续行检查: 搜索图片数据")
                                
                                # 检查顶级字段中的图片数据
                                for key, value in chunk_data.items():
                                    if key not in ['id', 'object', 'created', 'model', 'system_fingerprint', 'choices', 'usage']:
                                        if isinstance(value, str) and ('data:image/' in value or 'http' in value):
                                            print(f"[Tutu DEBUG] 🎯 comfly续行在{key}发现图片数据!")
                                            accumulated_content += " " + value
                                
                                # 全面搜索续行中的图片数据
                                if 'data:image/' in chunk_str or 'generated_image' in chunk_str or 'image_url' in chunk_str:
                                    print(f"[Tutu DEBUG] 🎯 comfly续行JSON中发现图片相关数据!")
                                    import re
                                    patterns = [
                                        r'data:image/[^",\s]+',
                                        r'https?://[^",\s]+\.(?:png|jpg|jpeg|gif|webp)',
                                        r'"image_url":\s*"([^"]+)"',
                                        r'"generated_image":\s*"([^"]+)"'
                                    ]
                                    
                                    for pattern in patterns:
                                        urls = re.findall(pattern, chunk_str)
                                        if urls:
                                            print(f"[Tutu DEBUG] 🎯 comfly续行用模式找到: {len(urls)}个URL")
                                            for url in urls:
                                                if url.startswith('data:image/'):
                                                    print(f"[Tutu DEBUG] 🎯 comfly续行提取base64图片")
                                                else:
                                                    print(f"[Tutu DEBUG] 🎯 comfly续行提取URL: {url[:50]}...")
                                                accumulated_content += " " + url
                                                
                            elif is_openrouter:
                                # OpenRouter续行处理
                                if 'data:image/' in chunk_str:
                                    print(f"[Tutu DEBUG] 🎯 OpenRouter续行中发现图片数据!")
                                    import re
                                    image_urls_in_chunk = re.findall(r'data:image/[^"]+', chunk_str)
                                    if image_urls_in_chunk:
                                        for url in image_urls_in_chunk:
                                            if url.startswith('data:image/'):
                                                print(f"[Tutu DEBUG] 🎯 OpenRouter续行提取base64图片")
                                            else:
                                                print(f"[Tutu DEBUG] 🎯 OpenRouter续行提取URL: {url[:50]}...")
                                            accumulated_content += " " + url
                            
                            # 保存完整的响应数据用于调试
                            raw_response_parts.append(chunk_data)
                            
                        except json.JSONDecodeError as e:
                            print(f"[Tutu DEBUG] 续行JSON仍然解析失败: {e}")
                            # 仍然不完整，继续等待
                            pass
                        
        except Exception as e:
            print(f"[Tutu ERROR] SSE流处理错误: {e}")
            
        print(f"[Tutu DEBUG] SSE处理完成:")
        print(f"[Tutu DEBUG] - 总共处理了{chunk_count}个数据块")
        print(f"[Tutu DEBUG] - 累积内容长度: {len(accumulated_content)}")
        
        # 简单截断长内容，避免base64刷屏
        if 'data:image/' in accumulated_content:
            base64_count = accumulated_content.count('data:image/')
            print(f"[Tutu DEBUG] - 累积内容: 包含{base64_count}个base64图片 + 文本({len(accumulated_content)}字符)")
        elif len(accumulated_content) > 200:
            print(f"[Tutu DEBUG] - 累积内容: {repr(accumulated_content[:200])}...")
        else:
            print(f"[Tutu DEBUG] - 累积内容: {repr(accumulated_content)}")
        
        print(f"[Tutu DEBUG] - 完整响应块数: {len(raw_response_parts)}")
            
        return accumulated_content

    def parse_chat_response(self, response_json, api_provider="ai.comfly.chat"):
        """
        解析非流式Chat Completions响应
        参考TutuNanoBananaPro的稳妥解析策略
        """
        print(f"[Tutu] 开始解析响应 (API: {api_provider})...")
        
        try:
            # 1. 检查基本结构
            if "choices" not in response_json or not response_json["choices"]:
                print(f"[Tutu] ⚠️ 响应中没有choices字段")
                print(f"[Tutu] 完整响应: {json.dumps(response_json, indent=2, ensure_ascii=False)[:500]}")
                return ""
            
            choice = response_json["choices"][0]
            print(f"[Tutu] Choice结构: {list(choice.keys())}")
            
            # 2. 检查finish_reason（安全过滤检测）
            finish_reason = choice.get("finish_reason")
            native_finish_reason = choice.get("native_finish_reason")
            
            if native_finish_reason == "IMAGE_SAFETY":
                print(f"[Tutu] ⚠️ 检测到安全过滤: IMAGE_SAFETY")
                raise Exception("❌ 内容被安全过滤拦截\n\n可能原因：\n1. 提示词包含敏感词汇（如'女孩'、'男孩'等人物描述）\n2. 图片内容涉及人物合成\n3. OpenRouter的安全策略更严格\n\n建议：\n1. 修改提示词：将'女孩'改为'角色'、'人物'\n2. 简化人物描述，避免详细特征\n3. 添加艺术风格描述（'卡通风格'、'插画风格'）\n4. 或尝试使用Google官方API（TutuNanoBananaPro节点）")
            
            if finish_reason and finish_reason not in ["stop", "length"]:
                print(f"[Tutu] ⚠️ 异常结束原因: {finish_reason}")
            
            # 3. 提取内容 - 支持多种格式
            content = ""
            
            # 优先从message中获取（完整响应）
            if "message" in choice:
                message = choice["message"]
                print(f"[Tutu] Message字段: {list(message.keys())}")
                
                # 🎯 优先检查images字段（OpenRouter Gemini图片生成格式）
                if "images" in message and message["images"]:
                    images_data = message["images"]
                    print(f"[Tutu] 🎯 在message.images中找到图片数据: {type(images_data).__name__}")
                    
                    # 处理images数组
                    image_parts = []
                    if isinstance(images_data, list):
                        print(f"[Tutu]   images是数组，包含 {len(images_data)} 个元素")
                        for idx, img in enumerate(images_data, 1):
                            if isinstance(img, dict):
                                # 可能的格式：{"url": "data:image/..."} 或 {"data": "base64..."}
                                if "url" in img:
                                    image_parts.append(img["url"])
                                    url_preview = img["url"][:50] if len(img["url"]) > 50 else img["url"]
                                    print(f"[Tutu]     图片{idx}: 从url提取 - {url_preview}...")
                                elif "data" in img:
                                    # 构建data URI
                                    mime_type = img.get("mime_type", "image/png")
                                    data_uri = f"data:{mime_type};base64,{img['data']}"
                                    image_parts.append(data_uri)
                                    print(f"[Tutu]     图片{idx}: 从data构建URI ({len(img['data'])} 字符)")
                                else:
                                    # 尝试整个对象转JSON
                                    print(f"[Tutu]     图片{idx}: 未知dict格式 - {list(img.keys())}")
                            elif isinstance(img, str):
                                # 直接是URL字符串
                                image_parts.append(img)
                                url_preview = img[:50] if len(img) > 50 else img
                                print(f"[Tutu]     图片{idx}: 字符串 - {url_preview}...")
                    elif isinstance(images_data, str):
                        # 单个图片字符串
                        image_parts.append(images_data)
                        url_preview = images_data[:50] if len(images_data) > 50 else images_data
                        print(f"[Tutu]   单个字符串 - {url_preview}...")
                    
                    if image_parts:
                        print(f"[Tutu] ✓ 从message.images提取了 {len(image_parts)} 个图片URL")
                        return "\n".join(image_parts)
                
                # 检查content字段
                if "content" in message:
                    content = message["content"]
                    print(f"[Tutu] 从message.content提取: {len(str(content))} 字符")
            
            # 如果message为空，尝试从delta获取（某些API）
            elif "delta" in choice:
                delta = choice["delta"]
                print(f"[Tutu] Delta字段: {list(delta.keys())}")
                
                if "content" in delta:
                    content = delta["content"]
                    print(f"[Tutu] 从delta.content提取: {len(str(content))} 字符")
            
            # 4. 处理不同类型的content
            if isinstance(content, str):
                # 字符串格式（可能包含markdown图片链接或base64）
                return content
            elif isinstance(content, list):
                # 数组格式（OpenAI标准格式）
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # 提取图片URL
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = str(image_url)
                            text_parts.append(url)
                return "\n".join(text_parts)
            else:
                print(f"[Tutu] ⚠️ 未知content类型: {type(content)}")
                return str(content) if content else ""
            
        except Exception as e:
            # 如果是我们自己抛出的安全过滤异常，直接传递
            if "安全过滤拦截" in str(e):
                raise
            
            print(f"[Tutu] ❌ 解析响应时出错: {str(e)}")
            # 打印部分响应用于调试
            try:
                response_preview = json.dumps(response_json, indent=2, ensure_ascii=False)[:1000]
                print(f"[Tutu] 响应预览: {response_preview}")
            except:
                pass
            raise

    def extract_image_urls(self, response_text):
        """提取图片URL - 支持多种格式"""
        print(f"[Tutu DEBUG] 开始提取图片URL...")
        print(f"[Tutu DEBUG] 响应文本长度: {len(response_text)}")
        
        # 简化日志输出
        if 'data:image/' in response_text:
            base64_count = response_text.count('data:image/')
            print(f"[Tutu DEBUG] 响应包含 {base64_count} 个base64图片")
        elif len(response_text) > 200:
            print(f"[Tutu DEBUG] 响应文本: {response_text[:200]}...")
        else:
            print(f"[Tutu DEBUG] 响应文本: {response_text}")
        
        image_urls = []
        
        # 1. Base64 data URLs（最常见）
        print(f"[Tutu DEBUG] 1. 检查base64数据URL...")
        base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
        matches = re.findall(base64_pattern, response_text)
        if matches:
            print(f"[Tutu DEBUG] ✓ 找到 {len(matches)} 个base64图片")
            image_urls.extend(matches)
        
        # 2. Markdown图片格式 ![](url)
        if not image_urls:
            print(f"[Tutu DEBUG] 2. 检查markdown图片格式...")
            markdown_pattern = r'!\[.*?\]\((data:image/[^)]+|https?://[^)]+)\)'
            matches = re.findall(markdown_pattern, response_text)
            if matches:
                print(f"[Tutu DEBUG] ✓ 找到 {len(matches)} 个markdown图片")
                image_urls.extend(matches)
        
        # 3. 直接HTTP图片URL
        if not image_urls:
            print(f"[Tutu DEBUG] 3. 检查HTTP图片URL...")
            url_pattern = r'https?://[^\s<>"]+\.(?:jpg|jpeg|png|gif|webp|bmp)'
            matches = re.findall(url_pattern, response_text, re.IGNORECASE)
            if matches:
                print(f"[Tutu DEBUG] ✓ 找到 {len(matches)} 个HTTP图片")
                image_urls.extend(matches)
        
        # 4. JSON中的图片字段
        if not image_urls:
            print(f"[Tutu DEBUG] 4. 尝试解析JSON格式...")
            try:
                json_data = json.loads(response_text)
                # 递归搜索JSON中的图片URL
                def find_images_in_json(obj):
                    urls = []
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if key in ['image', 'image_url', 'url', 'data'] and isinstance(value, str):
                                if value.startswith('data:image/') or value.startswith('http'):
                                    urls.append(value)
                            else:
                                urls.extend(find_images_in_json(value))
                    elif isinstance(obj, list):
                        for item in obj:
                            urls.extend(find_images_in_json(item))
                    return urls
                
                json_images = find_images_in_json(json_data)
                if json_images:
                    print(f"[Tutu DEBUG] ✓ 从JSON中找到 {len(json_images)} 个图片")
                    image_urls.extend(json_images)
            except:
                pass
        
        if not image_urls:
            print(f"[Tutu DEBUG] ❌ 未找到任何图片URL")
        
        return image_urls

    def resize_to_target_size(self, image, target_size):
        """Resize image to target size while preserving aspect ratio with padding"""

        img_width, img_height = image.size
        target_width, target_height = target_size

        width_ratio = target_width / img_width
        height_ratio = target_height / img_height
        scale = min(width_ratio, height_ratio)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        resized_img = image.resize((new_width, new_height), Image.LANCZOS)

        new_img = Image.new("RGB", (target_width, target_height), (255, 255, 255))

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
 
        new_img.paste(resized_img, (paste_x, paste_y))
        
        return new_img

    def parse_resolution(self, resolution_str):
        """Parse resolution string (e.g., '1024x1024') to width and height"""
        width, height = map(int, resolution_str.split('x'))
        return (width, height)

    def _sanitize_content_for_debug(self, content):
        """Sanitize content for debug logging"""
        if isinstance(content, str):
            # String format (comfly)
            return content[:200] + ('...' if len(content) > 200 else '')
        elif isinstance(content, list):
            # Array format (OpenRouter)
            sanitized = []
            for item in content:
                if item.get('type') == 'text':
                    text = item.get('text', '')[:100]
                    sanitized.append({
                        'type': 'text',
                        'text': text + ('...' if len(item.get('text', '')) > 100 else '')
                    })
                elif item.get('type') == 'image_url':
                    sanitized.append({
                        'type': 'image_url',
                        'image_url': '[IMAGE_DATA]'
                    })
            return sanitized
        else:
            return '[UNKNOWN_CONTENT_TYPE]'

    def process(self, prompt, api_provider, seed, 
                input_image_1=None, input_image_2=None, input_image_3=None, input_image_4=None, input_image_5=None, 
                comfly_api_key="", openrouter_api_key=""):

        print(f"\n[Tutu] ========== 🍌 Nano Banana 开始处理 ==========")
        print(f"[Tutu] API提供商: {api_provider}")
        
        # 根据API提供商硬编码模型选择
        if api_provider == "OpenRouter":
            model = "google/gemini-2.5-flash-image-preview"
        else:  # ai.comfly.chat
            model = "gemini-2.5-flash-image-preview"
        
        print(f"[Tutu] 模型: {model}")
        print(f"[Tutu] 提示词长度: {len(prompt)} 字符")
        print(f"[Tutu] 随机种子: {seed}")
        
        # 准备输入图片列表 - 保持索引对应
        input_images = [input_image_1, input_image_2, input_image_3, input_image_4, input_image_5]
        non_none_count = len([img for img in input_images if img is not None])
        connected_ports = [i+1 for i, img in enumerate(input_images) if img is not None]
        
        if connected_ports:
            print(f"[Tutu] 输入图片: {non_none_count} 张")
            print(f"[Tutu] 已连接的输入端口: {connected_ports}")
            
            # 添加图片索引映射提示
            print(f"[Tutu] 🔍 图片索引映射（用于提示词）:")
            api_idx = 0
            for port_idx, img in enumerate(input_images, 1):
                if img is not None:
                    api_idx += 1
                    print(f"[Tutu]    - 端口{port_idx} → 提示词中应写'图片{api_idx}'或'第{api_idx}张图'")
            print(f"[Tutu] ⚠️ 重要：提示词中引用图片时，请使用'图片X'编号（从1开始），而不是端口号！")
        
        # 根据API提供商设置端点
        if api_provider == "OpenRouter":
            # 走统一入口 llm-router（出图先反代、不行再 OpenRouter），不直连 openrouter.ai；
            # 入口地址与 key 可用环境变量 OPENROUTER_BASE_URL / OPENROUTER_API_KEY（密匣 openrouter.env）覆盖
            api_endpoint = (os.environ.get("OPENROUTER_BASE_URL") or "https://ag.crystalwind.fun/v1").rstrip("/") + "/chat/completions"
        else:
            api_endpoint = "https://ai.comfly.chat/v1/chat/completions"

        # 添加随机变化因子到提示词
        varied_prompt = self.add_random_variation(prompt, seed)
        
        # Save original prompt for processing
        original_prompt = prompt
        
        # 处理API Key更新和保存
        config_changed = False
        config = get_config()
        
        # 处理 comfly API key
        if comfly_api_key.strip():
            self.comfly_api_key = comfly_api_key
            config['comfly_api_key'] = comfly_api_key
            config_changed = True
            
        # 处理 OpenRouter API key
        if openrouter_api_key.strip():
            self.openrouter_api_key = openrouter_api_key
            config['openrouter_api_key'] = openrouter_api_key
            config_changed = True
            
        # 保存配置
        if config_changed:
            save_config(config)
            
        # 显示当前使用的API key
        current_api_key = self.get_current_api_key(api_provider)
        print(f"[Tutu] API Key: {current_api_key[:10] if current_api_key else 'None'}***")

        try:

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # 检查是否有输入图片
            has_images = non_none_count > 0

            # 使用标准OpenAI格式（数组）- 适用于所有API提供商
            content = []
            
            if has_images:
                # 构建端口号到数组索引的映射
                port_to_array_map = {}  # 端口号 -> 数组索引
                array_idx = 0
                for port_idx, img in enumerate(input_images, 1):
                    if img is not None:
                        array_idx += 1
                        port_to_array_map[port_idx] = array_idx
                
                # 自动转换提示词中的图片引用（端口号 -> 数组索引）
                import re
                original_varied_prompt = varied_prompt
                for port_num, array_num in port_to_array_map.items():
                    # 替换各种可能的引用格式
                    patterns = [
                        (rf'图{port_num}(?![0-9])', f'图{array_num}'),  # 图2 -> 图1
                        (rf'图片{port_num}(?![0-9])', f'图片{array_num}'),  # 图片2 -> 图片1
                        (rf'第{port_num}张图', f'第{array_num}张图'),  # 第2张图 -> 第1张图
                        (rf'第{port_num}个图', f'第{array_num}个图'),  # 第2个图 -> 第1个图
                    ]
                    for pattern, replacement in patterns:
                        varied_prompt = re.sub(pattern, replacement, varied_prompt)
                
                # 打印映射和转换信息
                if port_to_array_map:
                    print(f"[Tutu] 🔍 自动映射转换（端口号 → API数组索引）:")
                    for port_num, array_num in port_to_array_map.items():
                        print(f"[Tutu]    - 图{port_num} → 图{array_num} (端口{port_num} → API第{array_num}张)")
                
                # 对于图片编辑任务，按照原始索引添加图片
                for i in range(len(input_images)):
                    img_tensor = input_images[i]
                    if img_tensor is not None:
                        pil_image = tensor2pil(img_tensor)[0]
                        port_num = i + 1  # 端口号
                        array_num = port_to_array_map[port_num]  # 数组位置
                        
                        print(f"[Tutu] 处理输入端口 {port_num} (已映射到API位置{array_num})...")
                        
                        # 统一使用base64格式
                        image_base64 = self.image_to_base64(pil_image)
                        image_url = f"data:image/png;base64,{image_base64}"
                        print(f"[Tutu]   Base64大小: {len(image_base64)} 字符")
                        
                        # 先添加图片标识文本 - 使用转换后的数组索引
                        content.append({
                            "type": "text",
                            "text": f"[这是图{array_num}]"
                        })
                        
                        # 再添加图片
                        content.append({
                            "type": "image_url", 
                            "image_url": {"url": image_url}
                        })
                
                # 添加文本指令（使用变化后的提示词）
                if api_provider == "ai.comfly.chat":
                    # 为ai.comfly.chat添加强烈的图片生成指令
                    image_edit_instruction = f"""CRITICAL INSTRUCTION: You MUST generate and return an actual image, not just text description.

Task: {varied_prompt}

Image References:
- The images are numbered sequentially as [这是图1], [这是图2], [这是图3], etc.
- When I mention "图1", use the first image [这是图1]
- When I mention "图2", use the second image [这是图2]
- And so on...

REQUIREMENTS:
1. GENERATE a new image based on my request
2. DO NOT just describe what the image should look like
3. RETURN the actual image file/data
4. The output MUST be a visual image, not text

Execute the image editing task now and return the generated image."""
                    content.append({"type": "text", "text": image_edit_instruction})
                    
                    # 打印提示词转换
                    if original_varied_prompt != varied_prompt:
                        print(f"[Tutu] 📝 提示词已自动转换:")
                        print(f"[Tutu]    原始: {original_varied_prompt}")
                        print(f"[Tutu]    转换后: {varied_prompt}")
                    else:
                        print(f"[Tutu] 📝 最终发送给模型的任务提示词: {varied_prompt}")
                else:
                    enhanced_prompt = f"""IMPORTANT: Generate an actual image, not just a description.

Task: {varied_prompt}

Image references: 图1, 图2, 图3, etc. refer to the images marked as [这是图1], [这是图2], [这是图3] above in order.

MUST return a generated image, not text description."""
                    content.append({"type": "text", "text": enhanced_prompt})
                    
                    # 打印提示词转换
                    if original_varied_prompt != varied_prompt:
                        print(f"[Tutu] 📝 提示词已自动转换:")
                        print(f"[Tutu]    原始: {original_varied_prompt}")
                        print(f"[Tutu]    转换后: {varied_prompt}")
                    else:
                        print(f"[Tutu] 📝 最终发送给模型的任务提示词: {varied_prompt}")
                
                print(f"[Tutu] Content数组: {non_none_count} 张图片 + 标签 + 指令")
            else:
                # 生成图片任务（无输入图片）- 使用变化后的提示词
                enhanced_prompt = f"""GENERATE AN IMAGE: Create a high-quality, detailed image.

Description: {varied_prompt}

CRITICAL: You MUST return an actual image, not just text description. Use your image generation capabilities to create the visual content."""
                
                content.append({"type": "text", "text": enhanced_prompt})
                
                # 打印最终发送的提示词
                print(f"[Tutu] 📝 最终发送给模型的完整指令:")
                print(f"[Tutu]    {enhanced_prompt}")

            messages = [{
                "role": "user",
                "content": content
            }]

            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 8192,
                "stream": False  # 使用非流式处理，更稳定
            }

            # 简化日志输出
            print(f"[Tutu] API端点: {api_endpoint}")
            print(f"[Tutu] 开始请求...")
            
            # 检查API Key
            headers = self.get_headers(api_provider)

            if not current_api_key or len(current_api_key) < 10:
                print(f"[Tutu] ⚠️ API Key无效")

            pbar = comfy.utils.ProgressBar(100)
            pbar.update_absolute(10)

            # 使用独立session避免代理连接复用问题
            session = requests.Session()
            session.trust_env = True
            try:
                response = session.post(
                    api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=False  # 非流式处理
                )
                
                print(f"[Tutu] 响应状态: {response.status_code}")
                
                # 检查HTTP错误
                if response.status_code != 200:
                    try:
                        error_text = response.text[:1000]
                        print(f"[Tutu] 错误响应: {error_text}")
                    except:
                        print(f"[Tutu] 无法读取错误响应")
                
                response.raise_for_status()
                
                # 直接解析完整JSON响应（非流式）
                response_json = response.json()
                response_text = self.parse_chat_response(response_json, api_provider)
                print(f"[Tutu] 响应处理完成，文本长度: {len(response_text)}")
                
            except requests.exceptions.Timeout:
                print(f"[Tutu] ❌ 请求超时 ({self.timeout}秒)")
                raise TimeoutError(f"API request timed out after {self.timeout} seconds")
            except requests.exceptions.HTTPError as e:
                print(f"[Tutu] ❌ HTTP错误: {e.response.status_code}")
                try:
                    error_detail = e.response.text[:500]
                    print(f"[Tutu] 错误详情: {error_detail}")
                    
                    # 特殊处理404错误（模型不存在）
                    if e.response.status_code == 404 and "No endpoints found" in error_detail:
                        model_error = f"""❌ **模型不存在错误**

**当前选择的模型**: `{model}`
**API提供商**: {api_provider}
**错误**: 此模型在 {api_provider} 上不可用

**解决方案**:
1. 检查API密钥是否正确
2. 确认 {api_provider} 账户有权限使用此模型
3. 检查 {api_provider} 官方文档获取最新支持的模型列表"""
                        raise Exception(model_error)
                    else:
                        raise Exception(f"HTTP {e.response.status_code} Error: {error_detail}")
                except:
                    raise Exception(f"HTTP Error: {str(e)}")
            except requests.exceptions.RequestException as e:
                print(f"[Tutu] ❌ 请求异常: {str(e)}")
                raise Exception(f"API request failed: {str(e)}")
            finally:
                session.close()
            
            pbar.update_absolute(40)

            # 简化响应格式
            formatted_response = f"**提示词**: {original_prompt}\n\n**响应时间**: {timestamp}\n\n**种子**: {seed}"
            
            print(f"[Tutu] 提取图片URL...")
            image_urls = self.extract_image_urls(response_text)
            print(f"[Tutu] 找到 {len(image_urls)} 个图片URL")
            
            if image_urls:
                try:
                    images = []
                    
                    for i, url in enumerate(image_urls):
                        pbar.update_absolute(40 + (i+1) * 50 // len(image_urls))
                        
                        try:
                            if url.startswith('data:image/'):
                                # Handle base64 data URL
                                base64_data = url.split(',', 1)[1]
                                image_data = base64.b64decode(base64_data)
                                pil_image = Image.open(BytesIO(image_data))
                            else:
                                # Handle HTTP URL - 使用独立session避免代理连接复用问题
                                img_session = requests.Session()
                                img_session.trust_env = True
                                try:
                                    img_response = img_session.get(url, timeout=self.timeout)
                                    img_response.raise_for_status()
                                    pil_image = Image.open(BytesIO(img_response.content))
                                finally:
                                    img_session.close()

                            # 直接使用生成的原图
                            img_tensor = pil2tensor(pil_image)
                            images.append(img_tensor)
                            print(f"[Tutu] 图片 {i+1} 处理成功: {pil_image.size}")
                            
                        except Exception as img_error:
                            print(f"[Tutu] ⚠️ 图片 {i+1} 处理失败: {str(img_error)}")
                            continue
                    
                    if images:
                        try:
                            combined_tensor = torch.cat(images, dim=0)
                        except RuntimeError:
                            combined_tensor = images[0] if images else None
                            
                        pbar.update_absolute(100)
                        print(f"[Tutu] ========== ✓ 处理完成 ==========\n")
                        return (combined_tensor, formatted_response)
                    else:
                        raise Exception("No images could be processed successfully")
                    
                except Exception as e:
                    print(f"[Tutu] ❌ 图片处理错误: {str(e)}")

            # No image URLs found in response
            print(f"[Tutu] ⚠️ 响应中未找到图片URL")
            if 'data:image/' in response_text:
                base64_count = response_text.count('data:image/')
                print(f"[Tutu] 响应包含 {base64_count} 个base64图片标识")
            
            pbar.update_absolute(100)

            reference_image = None
            for img in input_images:
                if img is not None:
                    reference_image = img
                    break
                
            # 添加调试信息到响应中
            debug_info = f"\n\n## 调试信息\n**状态**: 响应解析可能不完整\n**请检查控制台日志获取详细信息**"
            formatted_response += debug_info
                
            if reference_image is not None:
                print(f"[Tutu] ========== ⚠️ 处理完成(无图片) ==========\n")
                return (reference_image, formatted_response)
            else:
                default_image = Image.new('RGB', (1024, 1024), color='white')
                default_tensor = pil2tensor(default_image)
                print(f"[Tutu] ========== ⚠️ 处理完成(无图片) ==========\n")
                return (default_tensor, formatted_response)
            
        except TimeoutError as e:
            error_message = f"API timeout error: {str(e)}"
            print(f"[Tutu] ❌ 超时错误: {error_message}")
            return self.handle_error(input_images, error_message)
            
        except Exception as e:
            error_message = f"Error calling Gemini API: {str(e)}"
            print(f"[Tutu] ❌ 异常:")
            print(f"[Tutu]   类型: {type(e).__name__}")
            print(f"[Tutu]   消息: {str(e)}")
            
            return self.handle_error(input_images, error_message)
    
    def handle_error(self, input_images, error_message):
        """Handle errors with appropriate image output"""
        # 按优先级返回第一个可用的图片
        for img in input_images:
            if img is not None:
                return (img, error_message)
        
        # 如果没有输入图片，创建默认图片
        default_image = Image.new('RGB', (1024, 1024), color='white')
        default_tensor = pil2tensor(default_image)
        return (default_tensor, error_message)


WEB_DIRECTORY = "./web"    
        
NODE_CLASS_MAPPINGS = {
    "TutuGeminiAPI": TutuGeminiAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TutuGeminiAPI": "🍌 Tutu 图图的香蕉模型(OpenRouter / Comfly)",
}