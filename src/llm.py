"""Thin wrappers for Claude (Anthropic API) and a local Ollama model. Every call is logged."""
import json, time, hashlib
import requests
from config import (CLAUDE_MODEL, OLLAMA_MODEL, OLLAMA_HOST, LLM_LOGS, ANTHROPIC_API_KEY, require_key,
                    OPENAI_API_KEY, OPENAI_MODEL, require_openai)

_client = None
_claude_temp = {}   # model -> whether it accepts temperature
_oai = None


def _log(kind, model, prompt, out):
    h = hashlib.md5((model + prompt).encode()).hexdigest()[:12]
    with open(LLM_LOGS / f"{kind}.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), "model": model, "hash": h, "prompt": prompt, "output": out}) + "\n")


def claude(prompt, system=None, max_tokens=2000, kind="claude", model=None):
    # Claude 5.5 models: adaptive thinking is on by default (effort "high") and thinking tokens count toward
    # max_tokens, so a small limit can cut the answer off. temperature/top_p/top_k are not accepted.
    global _client
    require_key()
    m_ = model or CLAUDE_MODEL
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    use_temp = _claude_temp.get(m_, True)
    for attempt in range(6):
        try:
            kw = dict(model=model or CLAUDE_MODEL, max_tokens=max_tokens,
                      system=system or "You are a careful research assistant.",
                      messages=[{"role": "user", "content": prompt}])
            if use_temp:
                kw["temperature"] = 0
            msg = _client.messages.create(**kw)
            out = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            _log(kind, model or CLAUDE_MODEL, prompt, out)
            return out
        except Exception as e:
            text = str(e)
            status = getattr(e, "status_code", None)
            if use_temp and ("temperature" in text.lower() or isinstance(e, TypeError)):
                print(f"  {m_} does not accept temperature; using the model default from now on")
                use_temp = False
                _claude_temp[m_] = False
                continue
            if status in (400, 401, 403, 404) or e.__class__.__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError"):
                raise SystemExit(f"Claude request failed ({e.__class__.__name__}, model={model or CLAUDE_MODEL}): {text[:500]}")
            wait = 2 ** attempt
            print(f"  claude error ({e.__class__.__name__}: {text[:200]}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Claude failed repeatedly")


def ollama(prompt, system=None, kind="ollama", json_mode=False, num_predict=400):
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "num_predict": num_predict, "seed": 42}}
    if system:
        body["system"] = system
    if json_mode:
        body["format"] = "json"
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=body, timeout=300)
        r.raise_for_status()
    except requests.ConnectionError:
        raise SystemExit("Cannot reach Ollama. Start the Ollama app (or run `ollama serve`) and retry.")
    out = r.json()["response"]
    _log(kind, OLLAMA_MODEL, prompt, out)
    return out


def gpt(prompt, system=None, kind="gpt", json_mode=False, max_tokens=600, model=None):
    """OpenAI chat completion at temperature 0 (falls back if a model rejects temperature/seed)."""
    global _oai
    require_openai()
    if _oai is None:
        from openai import OpenAI
        _oai = OpenAI(api_key=OPENAI_API_KEY)
    m = model or OPENAI_MODEL
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kw = dict(model=m, messages=msgs, temperature=0, seed=42, max_tokens=max_tokens)
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    for attempt in range(6):
        try:
            r = _oai.chat.completions.create(**kw)
            out = r.choices[0].message.content or ""
            _log(kind, m, prompt, out)
            return out
        except Exception as e:
            msg = str(e)
            if "temperature" in msg or "seed" in msg:
                kw.pop("temperature", None); kw.pop("seed", None); continue
            if "max_tokens" in msg:
                kw["max_completion_tokens"] = kw.pop("max_tokens", max_tokens); continue
            if "model" in msg and ("not found" in msg or "does not exist" in msg):
                raise SystemExit(f"OpenAI model '{m}' not available to your key. Set OPENAI_MODEL in .env (see `make check`).")
            wait = 2 ** attempt
            print(f"  gpt error ({e.__class__.__name__}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("GPT failed repeatedly")
