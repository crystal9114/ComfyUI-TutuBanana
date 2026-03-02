import os
import io
import time
import random
import torch
import requests
import base64
import json
import re
from PIL import Image
from io import BytesIO
from .utils import pil2tensor, tensor2pil


def get_config():
    """获取配置文件"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Tutuapi.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except:
        return {}


def save_config(config):
    """保存配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Tutuapi.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


class TutuNanoBananaPro:
    """
    Tutu 香蕉模型专业版 - Gemini 3 Pro Image Preview / T8Star Nano-banana
    (Nano Banana Pro / Gemini 3 Pro)
    
    支持两种API提供商：
    1. Google官方 Gemini API
    2. T8Star Nano-banana API
    
    支持文生图、图生图、多图合成、搜索接地等功能
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # API提供商选择
                "api_provider": (
                    ["Gemini 3 Pro Image Preview", "Gemini 3.1 Flash Image Preview", "T8Star"],
                    {"default": "Gemini 3 Pro Image Preview"}
                ),
                
                # 提示词 - 从外部输入
                "prompt": ("STRING", {"forceInput": True}),
                
                # 图像设置
                "aspect_ratio": (
                    ["Auto", "1:1", "9:16", "16:9", "3:4", "4:3", "3:2", "2:3", "5:4", "4:5", "21:9", "4:1", "1:4", "8:1", "1:8", "2:1", "1:2"],
                    {"default": "1:1"}
                ),
                "image_size": (
                    ["1K", "2K", "4K"],
                    {"default": "2K"}
                ),
                
                # Google API密钥
                "google_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入你的 Google API Key (选择Google官方时使用)"
                }),
                
                # T8Star API密钥
                "t8star_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入你的 T8Star API Key (选择T8Star时使用)"
                }),
                
                # 随机种子 - 控制重新生成
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子，改变此值会强制重新生成图片"
                }),
            },
            "optional": {
                # Google搜索增强 (仅Google官方支持)
                "enable_google_search": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用搜索增强",
                    "label_off": "关闭搜索增强"
                }),
                # 14个图片输入端口
                "input_image_1": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),
                "input_image_6": ("IMAGE",),
                "input_image_7": ("IMAGE",),
                "input_image_8": ("IMAGE",),
                "input_image_9": ("IMAGE",),
                "input_image_10": ("IMAGE",),
                "input_image_11": ("IMAGE",),
                "input_image_12": ("IMAGE",),
                "input_image_13": ("IMAGE",),
                "input_image_14": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "response")
    FUNCTION = "generate"
    CATEGORY = "Tutu"
    
    def __init__(self):
        config = get_config()
        self.google_api_key = config.get('google_api_key', '')
        self.t8star_api_key = config.get('t8star_api_key', '')
    
    def get_api_config(self, api_provider):
        """获取API配置"""
        if api_provider == "Gemini 3 Pro Image Preview" or api_provider == "Google官方":
            return {
                "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent",
                "model": "gemini-3-pro-image-preview",
                "provider": "google"
            }
        elif api_provider == "Gemini 3.1 Flash Image Preview":
            return {
                "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent",
                "model": "gemini-3.1-flash-image-preview",
                "provider": "google"
            }
        else:  # T8Star
            return {
                "endpoint": "https://ai.t8star.cn/v1/images/generations",
                "model": "nano-banana-2",
                "provider": "t8star"
            }
    
    def save_api_key(self, google_key=None, t8star_key=None):
        """保存API密钥到配置文件"""
        config = get_config()
        if google_key is not None:
            config['google_api_key'] = google_key
            self.google_api_key = google_key
            print(f"[Tutu] Google API密钥已保存")
        if t8star_key is not None:
            config['t8star_api_key'] = t8star_key
            self.t8star_api_key = t8star_key
            print(f"[Tutu] T8Star API密钥已保存")
        save_config(config)
    
    def add_random_variation(self, prompt, seed=0):
        """
        在提示词末尾添加隐藏的随机标识
        用户每次运行都会得到不同结果（抽卡功能）
        结合种子使用，确保可控的随机性
        """
        # 如果seed为0，使用当前时间作为随机源
        if seed == 0:
            random_id = random.randint(10000, 99999)
        else:
            # 基于seed生成确定性的随机数
            rng = random.Random(seed)
            random_id = rng.randint(10000, 99999)
        
        return f"{prompt} [variation-{random_id}]"
    
    def build_request_payload(self, prompt, input_images, enable_google_search, aspect_ratio, image_size, seed, provider):
        """构建API请求 - 根据provider选择格式"""
        if provider == "google":
            return self.build_google_payload(prompt, input_images, enable_google_search, aspect_ratio, image_size, seed)
        else:  # t8star
            return self.build_t8star_payload(prompt, input_images, aspect_ratio, image_size, seed)
    
    def build_google_payload(self, prompt, input_images, enable_google_search, aspect_ratio, image_size, seed):
        """构建谷歌官方 Gemini API 格式的请求"""
        # 添加随机变化因子
        varied_prompt = self.add_random_variation(prompt, seed)
        
        # 构建端口号到数组索引的映射
        port_to_array_map = {}  # 端口号 -> 数组索引
        array_idx = 0
        for port_idx, img in enumerate(input_images, 1):
            if img is not None:
                array_idx += 1
                port_to_array_map[port_idx] = array_idx
        
        # 自动转换提示词中的图片引用（端口号 -> 数组索引）
        import re
        original_prompt = varied_prompt
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
        
        # 构建 contents 数组（Google官方格式）
        parts = []
        
        # 添加所有输入图片 - 保持原始索引位置
        array_position = 0  # 追踪在API数组中的实际位置
        for i in range(len(input_images)):
            img_tensor = input_images[i]
            if img_tensor is not None:
                # 转换为PIL图片
                pil_image = tensor2pil(img_tensor)[0]
                
                # 转换为base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG", optimize=True, quality=95)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # 添加图片到parts
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_base64
                    }
                })
                
                # 输出时显示真实的图片编号（i+1 对应 input_image_1 到 input_image_14）
                array_position += 1
                print(f"[Tutu] 已添加输入端口 {i+1} 的图片, Base64大小: {len(img_base64)} 字符")
        
        # 添加文本提示词
        parts.append({
            "text": varied_prompt
        })
        
        # 构建完整的payload
        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "imageSize": image_size
                }
            }
        }
        
        if aspect_ratio != "Auto":
            payload["generationConfig"]["imageConfig"]["aspectRatio"] = aspect_ratio
        
        # 如果启用搜索增强，添加tools
        if enable_google_search:
            payload["tools"] = [{"google_search": {}}]
            print(f"[Tutu] 已启用Google搜索增强")
        
        print(f"[Tutu] 图像配置: {aspect_ratio} @ {image_size}")
        print(f"[Tutu] 输入图片数: {len([img for img in input_images if img is not None])}")
        
        # 添加图片索引映射提示
        if array_position > 0:
            print(f"[Tutu] 🔍 自动映射转换（端口号 → API数组索引）:")
            for port_num, array_num in port_to_array_map.items():
                print(f"[Tutu]    - 图{port_num} → 图{array_num} (端口{port_num} → API第{array_num}张)")
        
        # 打印提示词转换
        if original_prompt != varied_prompt:
            print(f"[Tutu] 📝 提示词已自动转换:")
            print(f"[Tutu]    原始: {original_prompt}")
            print(f"[Tutu]    转换后: {varied_prompt}")
        else:
            print(f"[Tutu] 📝 最终发送给模型的提示词: {varied_prompt}")
        
        return payload
    
    def build_t8star_payload(self, prompt, input_images, aspect_ratio, image_size, seed):
        """构建T8Star API格式的请求 (OpenAI Dall-e 格式)"""
        # 添加随机变化因子
        varied_prompt = self.add_random_variation(prompt, seed)
        
        # 构建端口号到数组索引的映射
        port_to_array_map = {}  # 端口号 -> 数组索引
        array_idx = 0
        for port_idx, img in enumerate(input_images, 1):
            if img is not None:
                array_idx += 1
                port_to_array_map[port_idx] = array_idx
        
        # 自动转换提示词中的图片引用（端口号 -> 数组索引）
        import re
        original_prompt = varied_prompt
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
        
        # 构建payload - T8Star固定使用 nano-banana-2 (香蕉2/gemini-3-pro-image-preview)
        payload = {
            "model": "nano-banana-2",
            "prompt": varied_prompt,
            "image_size": image_size,
            "response_format": "url"  # 使用URL格式返回
        }
        
        if aspect_ratio != "Auto":
            payload["aspect_ratio"] = aspect_ratio
        
        # 添加参考图片（如果有）
        image_array = []
        for i in range(len(input_images)):
            img_tensor = input_images[i]
            if img_tensor is not None:
                # 转换为PIL图片
                pil_image = tensor2pil(img_tensor)[0]
                
                # 转换为base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG", optimize=True, quality=95)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # T8Star使用data URI格式
                data_uri = f"data:image/png;base64,{img_base64}"
                image_array.append(data_uri)
                
                print(f"[Tutu] 已添加输入端口 {i+1} 的图片, Base64大小: {len(img_base64)} 字符")
        
        if image_array:
            payload["image"] = image_array
        
        print(f"[Tutu] 图像配置: {aspect_ratio} @ {image_size}")
        print(f"[Tutu] 输入图片数: {len(image_array)}")
        
        # 添加图片索引映射提示
        if image_array:
            print(f"[Tutu] 🔍 自动映射转换（端口号 → API数组索引）:")
            for port_num, array_num in port_to_array_map.items():
                print(f"[Tutu]    - 图{port_num} → 图{array_num} (端口{port_num} → API第{array_num}张)")
        
        # 打印提示词转换
        if original_prompt != varied_prompt:
            print(f"[Tutu] 📝 提示词已自动转换:")
            print(f"[Tutu]    原始: {original_prompt}")
            print(f"[Tutu]    转换后: {varied_prompt}")
        else:
            print(f"[Tutu] 📝 最终发送给模型的提示词: {varied_prompt}")
        
        return payload
    
    def parse_response(self, response_json, provider):
        """解析API响应 - 根据provider选择格式"""
        if provider == "google":
            return self.parse_google_response(response_json)
        else:  # t8star
            return self.parse_t8star_response(response_json)
    
    def parse_google_response(self, response_json):
        """
        解析谷歌官方 Gemini API 响应
        {
          "candidates": [{
            "content": {
              "parts": [
                {"text": "..."},
                {"inlineData": {"mimeType": "image/png", "data": "base64..."}}
              ]
            }
          }]
        }
        """
        try:
            if "candidates" not in response_json or not response_json["candidates"]:
                raise Exception("响应中没有candidates数据")
            
            candidate = response_json["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"]:
                raise Exception("响应格式错误")
            
            parts = candidate["content"]["parts"]
            images = []
            text_parts = []
            
            for part in parts:
                # 跳过thought部分
                if part.get("thought", False):
                    continue
                    
                if "inlineData" in part:
                    # 图片数据
                    inline_data = part["inlineData"]
                    if "data" in inline_data:
                        # Base64格式
                        image_url = f"data:{inline_data.get('mimeType', 'image/png')};base64,{inline_data['data']}"
                        images.append(image_url)
                elif "text" in part:
                    # 文本数据
                    text_parts.append(part["text"])
            
            print(f"[Tutu] 解析到 {len(images)} 张图片, {len(text_parts)} 段文本")
            
            return {
                'images': images,
                'text': '\n'.join(text_parts),
                'success': len(images) > 0
            }
            
        except Exception as e:
            print(f"[Tutu] 响应解析错误: {str(e)}")
            print(f"[Tutu] 响应内容: {json.dumps(response_json, indent=2, ensure_ascii=False)[:500]}")
            raise Exception(f"响应解析失败: {str(e)}")
    
    def parse_t8star_response(self, response_json):
        """
        解析T8Star API响应 (OpenAI Dall-e 格式)
        {
          "data": [
            {"url": "https://..."},
            ...
          ]
        }
        """
        try:
            if "data" not in response_json:
                raise Exception("响应中没有data字段")
            
            images = []
            for item in response_json["data"]:
                if "url" in item:
                    images.append(item["url"])
                elif "b64_json" in item:
                    # 如果返回base64格式
                    image_url = f"data:image/png;base64,{item['b64_json']}"
                    images.append(image_url)
            
            print(f"[Tutu] 解析到 {len(images)} 张图片")
            
            return {
                'images': images,
                'text': '',  # T8Star不返回文本
                'success': len(images) > 0
            }
            
        except Exception as e:
            print(f"[Tutu] 响应解析错误: {str(e)}")
            print(f"[Tutu] 响应内容: {json.dumps(response_json, indent=2, ensure_ascii=False)[:500]}")
            raise Exception(f"响应解析失败: {str(e)}")
    
    def decode_image(self, image_url):
        """下载或解码图片"""
        try:
            if image_url.startswith('data:image/'):
                # Base64图片
                base64_data = image_url.split(',', 1)[1]
                image_data = base64.b64decode(base64_data)
                pil_image = Image.open(BytesIO(image_data))
            else:
                # HTTP URL图片 - 使用独立session避免代理连接复用问题
                session = requests.Session()
                session.trust_env = True
                try:
                    response = session.get(image_url, timeout=60)
                    response.raise_for_status()
                    pil_image = Image.open(BytesIO(response.content))
                finally:
                    session.close()
            
            # 转换为RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            print(f"[Tutu] 图片解码成功: {pil_image.size}")
            return pil2tensor(pil_image)
            
        except Exception as e:
            print(f"[Tutu] 图片解码失败: {str(e)}")
            raise
    
    def create_default_image(self, aspect_ratio, image_size):
        """创建默认占位图"""
        # 宽高比映射
        ratio_map = {
            "1:1": (1, 1), "2:3": (2, 3), "3:2": (3, 2),
            "3:4": (3, 4), "4:3": (4, 3), "4:5": (4, 5),
            "5:4": (5, 4), "9:16": (9, 16), "16:9": (16, 9),
            "21:9": (21, 9), "2:1": (2, 1), "1:2": (1, 2),
            "4:1": (4, 1), "1:4": (1, 4), "8:1": (8, 1), "1:8": (1, 8),
            "Auto": (1, 1)
        }
        
        # 分辨率映射
        size_map = {"1K": 1024, "2K": 2048, "4K": 4096}
        
        w_ratio, h_ratio = ratio_map.get(aspect_ratio, (1, 1))
        base_size = size_map.get(image_size, 1024)
        
        # 计算实际尺寸
        if w_ratio >= h_ratio:
            width = base_size
            height = int(base_size * h_ratio / w_ratio)
        else:
            height = base_size
            width = int(base_size * w_ratio / h_ratio)
        
        # 创建白色图片
        img = Image.new('RGB', (width, height), color='white')
        return pil2tensor(img)
    
    def generate(self, api_provider, prompt, aspect_ratio, image_size,
                 google_api_key, t8star_api_key, seed, 
                 enable_google_search=False,
                 input_image_1=None, input_image_2=None, input_image_3=None,
                 input_image_4=None, input_image_5=None, input_image_6=None,
                 input_image_7=None, input_image_8=None, input_image_9=None,
                 input_image_10=None, input_image_11=None, input_image_12=None,
                 input_image_13=None, input_image_14=None):
        """
        主处理函数 - 支持多种API提供商
        """
        print(f"\n[Tutu] ========== 🍌 香蕉模型专业版开始处理 ==========")
        print(f"[Tutu] API提供商: {api_provider}")
        print(f"[Tutu] 分辨率: {image_size} @ {aspect_ratio}")
        print(f"[Tutu] 提示词长度: {len(prompt)} 字符")
        print(f"[Tutu] 随机种子: {seed}")
        
        try:
            # 1. 准备输入图片 - 保持为完整数组，不过滤None以保持索引对应
            input_images = [
                input_image_1, input_image_2, input_image_3, input_image_4,
                input_image_5, input_image_6, input_image_7, input_image_8,
                input_image_9, input_image_10, input_image_11, input_image_12,
                input_image_13, input_image_14
            ]
            
            # 统计非None图片数量
            non_none_count = len([img for img in input_images if img is not None])
            print(f"[Tutu] 输入图片: {non_none_count} 张")
            
            # 显示具体连接了哪些端口
            connected_ports = [i+1 for i, img in enumerate(input_images) if img is not None]
            if connected_ports:
                print(f"[Tutu] 已连接的输入端口: {connected_ports}")
            
            if non_none_count > 14:
                print(f"[Tutu] ⚠️ 警告: 输入图片超过14张，只使用前14张")
            
            # 2. 获取API配置
            config = self.get_api_config(api_provider)
            provider = config['provider']
            
            # 3. 确定使用哪个API Key
            if provider == "google":
                api_key = google_api_key.strip() or self.google_api_key
                if not api_key or len(api_key) < 10:
                    raise Exception(f"❌ 请提供有效的 Google API Key！\n\n请在节点中输入API密钥，或在Tutuapi.json配置文件中设置。")
                # 保存API Key到配置
                if google_api_key.strip():
                    self.save_api_key(google_key=google_api_key)
            else:  # t8star
                api_key = t8star_api_key.strip() or self.t8star_api_key
                if not api_key or len(api_key) < 10:
                    raise Exception(f"❌ 请提供有效的 T8Star API Key！\n\n请在节点中输入API密钥，或在Tutuapi.json配置文件中设置。")
                # 保存API Key到配置
                if t8star_api_key.strip():
                    self.save_api_key(t8star_key=t8star_api_key)
            
            print(f"[Tutu] API Key: {api_key[:10]}***")
            
            # 4. 构建请求
            payload = self.build_request_payload(
                prompt, input_images, enable_google_search, aspect_ratio, image_size, seed, provider
            )
            
            # 5. 构建headers
            if provider == "google":
                headers = {
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json"
                }
            else:  # t8star
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            
            # 6. 发送请求
            print(f"[Tutu] 发送请求到: {config['endpoint']}")
            print(f"[Tutu] 模型: {config['model']}")
            print(f"[Tutu] 模式: {'img2img' if non_none_count > 0 else 'text2img'}")
            
            start_time = time.time()
            
            # 使用独立session避免代理连接复用问题
            session = requests.Session()
            session.trust_env = True
            try:
                response = session.post(
                    config['endpoint'],
                    headers=headers,
                    json=payload,
                    timeout=180
                )
            finally:
                session.close()
            
            elapsed = time.time() - start_time
            print(f"[Tutu] 响应状态: {response.status_code} (耗时: {elapsed:.1f}秒)")
            
            # 检查HTTP错误
            if response.status_code != 200:
                error_text = response.text[:500]
                print(f"[Tutu] 错误响应: {error_text}")
                raise Exception(f"API错误 ({response.status_code}): {error_text}")
            
            # 7. 解析响应
            result = self.parse_response(response.json(), provider)
            
            if not result['success'] or not result['images']:
                print(f"[Tutu] ⚠️ 未生成图片")
                print(f"[Tutu] 响应文本: {result['text'][:200]}")
                raise Exception("未生成图片。可能原因：\n1. 提示词不够清晰\n2. 模型理解为纯文本任务\n3. API限制\n\n请调整提示词后重试。")
            
            # 8. 下载/解码所有图片，选择分辨率最大的
            print(f"[Tutu] 开始解码图片 (共 {len(result['images'])} 张)...")
            decoded_images = []
            
            for idx, img_url in enumerate(result['images'], 1):
                try:
                    tensor = self.decode_image(img_url)
                    # 获取图片尺寸 (batch, height, width, channels)
                    h, w = tensor.shape[1:3]
                    resolution = h * w
                    decoded_images.append((tensor, w, h, resolution, idx))
                    print(f"[Tutu] 图片 {idx}: {w}x{h} (像素总数: {resolution:,})")
                except Exception as e:
                    print(f"[Tutu] ⚠️ 图片 {idx} 解码失败: {str(e)}")
            
            if not decoded_images:
                raise Exception("所有图片解码失败")
            
            # 按分辨率排序，选择最大的
            decoded_images.sort(key=lambda x: x[3], reverse=True)
            image_tensor, final_w, final_h, final_res, selected_idx = decoded_images[0]
            print(f"[Tutu] ✓ 已选择图片 {selected_idx}: {final_w}x{final_h} (最高分辨率)")
            
            # 如果有多张图片，显示未选择的图片信息
            if len(decoded_images) > 1:
                print(f"[Tutu] 其他图片已忽略:")
                for tensor, w, h, res, idx in decoded_images[1:]:
                    print(f"[Tutu]   - 图片 {idx}: {w}x{h}")
            
            # 9. 格式化响应文本
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            provider_name = "谷歌官方 Gemini API" if provider == "google" else "T8Star API"
            formatted_response = f"""🍌 **香蕉模型专业版生成结果** ({timestamp})

**API提供商**: {provider_name}
**模型**: {config['model']}
**模式**: {'img2img' if non_none_count > 0 else 'text2img'}
**请求分辨率**: {image_size} @ {aspect_ratio}
**实际输出**: {final_w}x{final_h} (从 {len(result['images'])} 张中选择最高分辨率)
**输入图片**: {non_none_count} 张 (端口: {connected_ports})"""
            
            if provider == "google":
                formatted_response += f"\n**搜索增强**: {'是' if enable_google_search else '否'}"
            
            formatted_response += f"\n**生成时间**: {elapsed:.1f} 秒\n\n✓ 生成成功"
            
            # 如果有返回的文本，添加到响应中
            if result['text'].strip():
                formatted_response += f"\n\n**模型返回文本**:\n{result['text']}"
            
            print(f"[Tutu] ========== ✓ 处理完成 ==========\n")
            
            return (image_tensor, formatted_response)
            
        except requests.exceptions.Timeout:
            error_msg = "❌ 请求超时（180秒）\n\n可能原因：\n1. 网络连接不稳定\n2. 图片太多/太大\n3. API服务响应慢\n\n建议：减少输入图片数量或稍后重试"
            print(f"[Tutu] {error_msg}")
            default_image = self.create_default_image(aspect_ratio, image_size)
            return (default_image, error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ 网络请求错误: {str(e)}\n\n请检查：\n1. 网络连接\n2. API端点是否可访问\n3. API密钥是否正确"
            print(f"[Tutu] {error_msg}")
            default_image = self.create_default_image(aspect_ratio, image_size)
            return (default_image, error_msg)
            
        except Exception as e:
            error_msg = f"❌ 错误: {str(e)}"
            print(f"[Tutu] {error_msg}")
            print(f"[Tutu] 详细错误: {repr(e)}")
            
            # 返回默认图和错误信息
            default_image = self.create_default_image(aspect_ratio, image_size)
            return (default_image, error_msg)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "TutuNanoBananaPro": TutuNanoBananaPro,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TutuNanoBananaPro": "🍌 Tutu 图图的香蕉模型专业版/香蕉2 (Google官方 / T8Star)",
}

