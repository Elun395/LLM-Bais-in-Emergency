# api.py
import os
import time
from typing import List, Optional

class LLMClient:
    """Minimal non-stream client for OpenAI-compatible Chat Completions."""

    def __init__(
        self,
        model: str = "deepseek",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        mock: bool = False,
    ):
        self.model = model

        if api_key:
            os.environ["deepseek_API_KEY"] = api_key
        if base_url:
            # 兼容多数代理要求 /v1，也避免重复拼接
            b = base_url.rstrip("/")
            if not b.endswith("/v1"):
                b += "/v1"
            os.environ["deepseek_BASE_URL"] = b

        self.mock = bool(mock or not bool(os.getenv("deepseek_API_KEY")))
        self._openai = None

    def _ensure_openai(self):
        if self._openai is None:
            from openai import OpenAI
            api_key = os.getenv("deepseek_API_KEY")
            base_url = os.getenv("deepseek_BASE_URL")
            self._openai = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        return self._openai

    @staticmethod
    def _flatten_parts(parts) -> str:
        out = []
        for p in parts or []:
            if isinstance(p, dict):
                if p.get("type") == "text" and isinstance(p.get("text"), str):
                    out.append(p["text"])
                elif isinstance(p.get("text"), str):
                    out.append(p["text"])
                elif isinstance(p.get("content"), str):
                    out.append(p["content"])
            elif hasattr(p, "text") and isinstance(getattr(p, "text"), str):
                out.append(p.text)
            elif isinstance(p, str):
                out.append(p)
        return "".join(out).strip()

    @classmethod
    def _extract_chat_content(cls, resp) -> str:
        # 兼容 OpenAI SDK 对象、dict、str
        if resp is None:
            return ""
        if hasattr(resp, "choices"):
            try:
                choice = resp.choices[0]
            except Exception:
                return ""
            msg = getattr(choice, "message", None)
            if msg is not None:
                c = getattr(msg, "content", None)
                if isinstance(c, str) and c.strip():
                    return c.strip()
                if isinstance(c, (list, tuple)):
                    txt = cls._flatten_parts(c)
                    if txt:
                        return txt
            # 一些服务把内容放在 choice.text
            t = getattr(choice, "text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
            # 兜底：转成 dict 再找
            try:
                import json
                data = json.loads(resp.model_dump_json())
                ch0 = (data.get("choices") or [{}])[0]
                msgd = ch0.get("message") or {}
                c = msgd.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
                if isinstance(c, list):
                    txt = cls._flatten_parts(c)
                    if txt:
                        return txt
                t = ch0.get("text")
                if isinstance(t, str) and t.strip():
                    return t.strip()
            except Exception:
                pass
            return ""
        if isinstance(resp, dict):
            ch0 = (resp.get("choices") or [{}])[0]
            msg = ch0.get("message") or {}
            c = msg.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, list):
                return cls._flatten_parts(c)
            for k in ("text", "content"):
                v = ch0.get(k) or resp.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""
        if isinstance(resp, str):
            return resp.strip()
        return str(resp)

    def chat(self, prompt: str, system: Optional[str] = None, max_tokens: int = 1200, timeout: int = 120, max_retries: int = 3) -> str:
        if self.mock:
            return "mocked output"

        # 重试逻辑
        for attempt in range(max_retries):
            try:
                client = self._ensure_openai()
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                # 改为纯字符串，提升兼容性
                messages.append({"role": "user", "content": prompt})

                # 使用 Chat Completions 正确参数名：max_tokens，添加超时设置
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    n=1,
                    max_tokens=max_tokens,
                    timeout=timeout  # 添加超时参数
                )
                return self._extract_chat_content(resp)
            except Exception as e:
                # 检查是否为内容安全错误
                if 'data_inspection_failed' in str(e) or 'inappropriate content' in str(e).lower():
                    print(f"警告: 内容安全检查失败，返回空响应: {str(e)[:100]}...")
                    return ""  # 返回空字符串表示安全响应
                else:
                    print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    # 如果不是最后一次尝试，等待一段时间后重试
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 指数退避
                        print(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"达到最大重试次数，返回空响应")
                        return ""  # 所有重试都失败，返回空响应而不终止程序
        
        # 理论上不会到达这里
        return ""

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        if self.mock:
            return [[0.0] * 8 for _ in texts]
        client = self._ensure_openai()
        embed_model = model or "text-embedding-3-small"
        out = client.embeddings.create(model=embed_model, input=texts)
        return [d.embedding for d in out.data]


# ---------------- example main ----------------

if __name__ == "__main__":
    """
    运行示例：
    export OPENAI_API_KEY=sk-...
    # 如直连官方：默认 BASE_URL 省略；如代理：
    export OPENAI_BASE_URL=https://your-proxy.example.com/v1
    python api.py
    """
    cli = LLMClient(model="gpt-5")
    ans = cli.chat("用一句话解释费马大定理。")
    print("模型回复：", ans or "<空>")