# ================== IMPORTS & ENV ==================
import os, re, json, time
from io import BytesIO
from typing import List, Tuple

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

st.set_page_config(page_title="OCRchestra", page_icon="🎼", layout="wide")
load_dotenv()

def get_secret(name: str, default: str = "") -> str:
    try:
        if "secrets" in dir(st) and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


# ================== STATE ==================
if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""
if "history" not in st.session_state:
    # cada item: {ts, proveedor, modelo, tarea, params, input_len, latency_ms, output}
    st.session_state["history"] = []
if "auto_ocr" not in st.session_state:
    st.session_state["auto_ocr"] = True

# ================== STYLE (CSS: fondo blanco, UI limpia) ==================
st.markdown("""
<style>
:root{
  --bg:#ffffff; --text:#161616; --muted:#5f6470; --card:#ffffff; --border:#e8e8ee;
  --accent:#6c63ff; --accent2:#00c2ff; --ok:#22c55e; --warn:#f59e0b; --err:#ef4444;
}
html, body, .stApp { background: var(--bg); color: var(--text); }
h1 span.gradient{
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
div.block-container{ padding-top: 1.0rem; }

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px; padding: 18px 20px;
  box-shadow: 0 8px 24px rgba(17, 12, 46, 0.06);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.card:hover { transform: translateY(-2px);
  box-shadow: 0 12px 28px rgba(17,12,46,.10); border-color:#d9d9e6; }

.small { font-size: 0.9rem; color: var(--muted);}

.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:.75rem; margin-right:6px;
  border:1px solid #e3e4ee; background: #f7f8ff;}
.badge.ok{ background:#effaf3; border-color:#daf0e2; color:#1c7c3b;}
.badge.warn{ background:#fff6e9; border-color:#ffe6bf; color:#8a5a00;}
.badge.err{ background:#ffefef; border-color:#ffd3d3; color:#9b1c1c;}

.stButton>button { border-radius: 12px; padding: .65rem 1rem; font-weight:600; }
.stButton>button:hover { box-shadow: 0 8px 18px rgba(108, 99, 255, .22); }

textarea, .stTextInput>div>div>input { border-radius: 12px !important; }
hr.hr {border:none; height:1px; background: linear-gradient(90deg,#fff,#ececf3,#fff); margin: 18px 0;}

 /* Dropzone del uploader más fluida */
[data-testid="stFileUploader"] > div:first-child{
  border: 2px dashed #d8d8ea !important; border-radius: 16px !important;
  transition: border-color .2s ease, background .2s ease;
  background: linear-gradient(180deg,#ffffff, #fbfbff);
}
[data-testid="stFileUploader"] > div:first-child:hover{
  border-color: #bdbdf5 !important; background: #f8f8ff;
  box-shadow: 0 8px 16px rgba(108,99,255,.08) inset;
}
</style>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
THINK_PATTERNS = [r"<think>.*?</think>", r"<reasoning>.*?</reasoning>", r"<analysis>.*?</analysis>"]
ROLE_ECHO = [
    r"^\s*\[?(USER|USR|SYSTEM|SYS|ASSISTANT|ASST)\]?:?.*$",  # líneas tipo [USR]: ..., ASSISTANT:
    r"^###\s*(User|Assistant|System).*$",                    # markdown roles
]
def clean_output(text: str) -> str:
    """Quita <think> y líneas de eco de roles, recorta espacios dobles."""
    if not text: return text
    for pat in THINK_PATTERNS:
        text = re.sub(pat, "", text, flags=re.DOTALL|re.IGNORECASE)
    # quita líneas con [USR]/[ASST]/etc
    lines = []
    for line in text.splitlines():
        if any(re.match(pat, line.strip(), flags=re.IGNORECASE) for pat in ROLE_ECHO):
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    # colapsa espacios múltiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def has_key(name: str) -> bool:
    try:
        if "secrets" in dir(st) and name in st.secrets:
            return bool(str(st.secrets[name]).strip())
    except Exception:
        pass
    return bool(os.getenv(name, "").strip())


def provider_options() -> list:
    opts = []
    if has_key("GROQ_API_KEY"): opts.append("Groq")
    if has_key("HUGGINGFACE_API_KEY") or has_key("HuggingFace_API_KEY"): opts.append("Hugging Face")
    return opts or ["Groq","Hugging Face"]

def guard_text(txt: str, min_chars: int = 10, max_chars: int = 8000) -> str:
    t = (txt or "").strip()
    if len(t) < min_chars: raise ValueError(f"Texto muy corto (≥{min_chars} chars).")
    if len(t) > max_chars: raise ValueError(f"Texto demasiado largo (≤{max_chars} chars).")
    return t

def chip(text, kind=""):  # badge helper
    cls = "badge"
    if kind=="ok": cls += " ok"
    if kind=="warn": cls += " warn"
    if kind=="err": cls += " err"
    st.markdown(f"<span class='{cls}'>{text}</span>", unsafe_allow_html=True)

# ================== OCR (EasyOCR) ==================
@st.cache_resource(show_spinner=False)
def get_reader():
    import easyocr
    use_gpu = False
    try:
        import torch
        use_gpu = bool(torch.cuda.is_available())
    except Exception:
        use_gpu = False
    return easyocr.Reader(["es","en"], gpu=use_gpu)

def sort_by_topleft(bbox: List[Tuple[float, float]]):
    ys = [p[1] for p in bbox]; xs = [p[0] for p in bbox]
    return (min(ys), min(xs))

def run_ocr(pil_img: Image.Image) -> str:
    import numpy as np
    reader = get_reader()
    arr = np.array(pil_img)
    results = reader.readtext(arr, detail=1, paragraph=False)
    if not results: return ""
    results_sorted = sorted(results, key=lambda r: sort_by_topleft(r[0]))
    lines = [r[1] for r in results_sorted if r[1]]
    return "\n".join(lines).strip()

# ================== PROMPTS ==================
TASKS = {
    "resumen": ("Eres un asistente que resume el texto del usuario en 5–7 líneas claras. "
                "No inventes contenido. Responde SOLO el resultado final; no muestres pasos ni <think>."),
    "entidades": ("Extrae entidades nombradas del texto del usuario y responde SOLO un JSON compacto con este formato: "
                  '{"PERSONA": [], "ORGANIZACIÓN": [], "LUGAR": [], "FECHA": [], "OTRO": []}. '
                  "No inventes. No muestres razonamiento."),
    "traducción": ("Traduce fielmente el texto del usuario al inglés. Mantén formato cuando sea útil. "
                   "Responde SOLO la traducción; sin razonamiento."),
}
def build_prompt(task_key: str, user_text: str) -> str:
    system = TASKS.get(task_key, TASKS["resumen"])
    return f"{system}\n\n---\nTexto del usuario:\n{user_text}"

# ================== GROQ ==================
GROQ_KEY = get_secret("GROQ_API_KEY")
@st.cache_resource(show_spinner=False)
def get_groq_client():
    from groq import Groq
    return Groq(api_key=GROQ_KEY) if GROQ_KEY else None

@st.cache_data(show_spinner=False, ttl=600)
def list_groq_models() -> list:
    client = get_groq_client()
    if not client: return []
    try:
        data = client.models.list()
        raw = getattr(data, "data", data)
        ids = [m.id for m in raw]
        allow = [i for i in ids if any(k in i.lower() for k in ["llama","mixtral","gemma","qwen","phi"])
                 and "guard" not in i.lower() and "embed" not in i.lower()]
        return sorted(allow)
    except Exception as e:
        st.warning(f"No se pudo listar modelos de Groq ({e}). Usando fallback.")
        return ["llama-3.1-70b-versatile","llama-3.1-8b-instant"]

def call_groq(model: str, task: str, user_text: str, temperature: float, max_tokens: int) -> str:
    client = get_groq_client()
    if client is None: raise RuntimeError("No hay GROQ_API_KEY en .env")
    system_msg = TASKS.get(task, TASKS["resumen"])
    resp = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role":"system","content":system_msg},{"role":"user","content":user_text}]
    )
    out = resp.choices[0].message.content if resp and resp.choices else ""
    return clean_output(out)

# ================== HUGGING FACE ==================
HF_KEY   = get_secret("HUGGINGFACE_API_KEY") or get_secret("HuggingFace_API_KEY")
@st.cache_resource(show_spinner=False)
def get_hf_client():
    if not HF_KEY: return None
    from huggingface_hub import InferenceClient
    return InferenceClient(api_key=HF_KEY)

HF_DEFAULT_MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]

def call_hf(model_id: str, task: str, user_text: str, temperature: float, max_new_tokens: int) -> str:
    """
    A) chat_completion (conversational); B) fallback a text_generation serverless si 403 Providers.
    Además: limpiamos eco de roles/prompt.
    """
    client = get_hf_client()
    if client is None: raise RuntimeError("No hay HUGGINGFACE_API_KEY en .env")

    system_msg = TASKS.get(task, TASKS["resumen"])
    messages = [{"role":"system","content":system_msg},{"role":"user","content":user_text}]
    # --- A) Chat
    try:
        resp = client.chat_completion(model=model_id, messages=messages,
                                      temperature=float(temperature), max_tokens=int(max_new_tokens))
        out = None
        if hasattr(resp,"choices") and resp.choices:
            msg = getattr(resp.choices[0], "message", None) or getattr(resp.choices[0], "delta", None)
            if msg: out = getattr(msg, "content", None) or (isinstance(msg, dict) and msg.get("content"))
        if out is None and isinstance(resp, dict):
            ch = resp.get("choices") or []
            if ch and isinstance(ch[0], dict):
                msg = ch[0].get("message") or {}
                out = msg.get("content")
        if out:
            return clean_output(out)
        raise RuntimeError("HF chat devolvió vacío.")
    except Exception as e:
        if "403" in str(e) and ("Inference Providers" in str(e) or "Forbidden" in str(e)):
            # --- B) Fallback: text_generation con stop_sequences para evitar eco
            prompt = build_prompt(task, user_text)
            try:
                out = client.text_generation(
                    prompt, model=model_id, max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature), do_sample=True, return_full_text=False,
                    stop_sequences=["[USR]", "[ASSISTANT]", "### User", "### Assistant", "\n---\n"]
                )
                return clean_output(out)
            except Exception as e2:
                raise RuntimeError(f"HF fallback (text_generation) falló: {e2}")
        raise RuntimeError(f"HF error (chat): {e}")

# ================== HEADER ==================
st.markdown("<h1>🎼 <span class='gradient'>OCRchestra</span></h1>", unsafe_allow_html=True)
st.caption("Convierte imágenes en texto (OCR) y analiza con Groq o Hugging Face: resumen, entidades y traducción.")

# ================== SIDEBAR (Proveedor / Modelos / Parámetros) ==================
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    prov_opts = provider_options()
    provider = st.radio("Proveedor LLM", prov_opts, horizontal=False)

    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
    task = st.selectbox("Tarea", list(TASKS.keys()), index=0)
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("max tokens", 64, 2048, 512, 32)

    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
    if provider == "Groq":
        models = list_groq_models()
        selected_model = st.selectbox("Modelo Groq", models or ["<sin modelos>"], index=0, disabled=not bool(models))
        if st.button("↻ Recargar modelos", use_container_width=True):
            st.cache_data.clear(); st.rerun()
        if not has_key("GROQ_API_KEY"): st.warning("Falta GROQ_API_KEY en .env", icon="⚠️")
    else:
        hf_model = st.selectbox("Modelo HF (sugeridos)", HF_DEFAULT_MODELS, index=0)
        hf_model = st.text_input("o escribe otro modelo HF", value=hf_model, help="Ej: mistralai/Mistral-7B-Instruct-v0.3")
        if not (has_key("HUGGINGFACE_API_KEY") or has_key("HuggingFace_API_KEY")):
            st.warning("Falta HUGGINGFACE_API_KEY en .env", icon="⚠️")

    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
    st.toggle("OCR automático al subir imagen", value=st.session_state["auto_ocr"], key="auto_ocr", help="Ejecuta OCR en cuanto subes la imagen.")

    # Export / limpiar historial
    if st.button("🧹 Limpiar historial", use_container_width=True):
        st.session_state["history"].clear()
        st.toast("Historial limpiado", icon="🗑️")
    if st.session_state["history"]:
        export_json = json.dumps(st.session_state["history"], ensure_ascii=False, indent=2)
        st.download_button("⬇️ Exportar historial (JSON)", export_json, "historial.json", "application/json", use_container_width=True)
        last_md = st.session_state["history"][-1]["output"]
        st.download_button("⬇️ Exportar último resultado (MD)", last_md, "resultado.md", "text/markdown", use_container_width=True)

# ================== MAIN: CARGA & OCR ==================
col_upload, col_text = st.columns([1.05, 1])
with col_upload:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🖼️ Imagen")
    uploaded = st.file_uploader("Arrastra y suelta una imagen (PNG/JPG/JPEG)", type=["png","jpg","jpeg"], accept_multiple_files=False,
                                help="Se usará para OCR y luego análisis con LLM.")
    pil_image = None
    if uploaded:
        image_bytes = uploaded.read()
        try:
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            st.image(pil_image, caption=uploaded.name, use_container_width=True)
            run_now = st.session_state["auto_ocr"]
            if st.button("🔎 Extraer texto (OCR)", type="primary", use_container_width=True) or run_now:
                with st.spinner("Ejecutando OCR…"):
                    t0 = time.time()
                    text = run_ocr(pil_image)
                    t1 = time.time()
                    st.session_state["ocr_text"] = text
                    if text: st.success(f"OCR completado en {int((t1-t0)*1000)} ms")
                    else: st.warning("No se detectó texto. Prueba con otra imagen o ajusta idiomas.")
        except Exception as e:
            st.error(f"No se pudo abrir la imagen. Detalle: {e}")
    else:
        st.caption("Tip: también puedes pegar una captura recortada para mejores resultados.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_text:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📝 Texto")
    st.text_area("Puedes editar el texto aquí antes de analizar", key="ocr_text", height=270,
                 placeholder="Aquí verás el texto reconocido…")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

# ================== ACCIÓN: ANALIZAR ==================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 Análisis")

# chips de contexto
with st.container():
    chip(f"Proveedor: {provider}")
    chip(f"Tarea: {task}")
    if provider=="Groq": chip(f"Modelo: {selected_model}")
    else: chip(f"Modelo: {hf_model}")
    chip(f"T={temperature:.2f}")
    chip(f"max={int(max_tokens)}")

st.write("")  # spacing

# botón siempre visible
clicked = st.button("✨ Analizar", type="primary", use_container_width=True)
if clicked:
    try:
        text_in = guard_text(st.session_state.get("ocr_text",""), min_chars=10, max_chars=8000)
        t0 = time.time()
        if provider == "Groq":
            models = list_groq_models()
            if not models:
                st.error("No hay modelos Groq disponibles. Pulsa ↻ en la barra lateral o revisa tu clave.")
                out = ""
            else:
                out = call_groq(selected_model, task, text_in, float(temperature), int(max_tokens))
        else:
            out = call_hf(hf_model, task, text_in, float(temperature), int(max_tokens))
        t1 = time.time()

        if not out:
            st.warning("La respuesta llegó vacía. Prueba con otro modelo, menos texto o menos tokens.")
        else:
            st.success("¡Análisis completado!")
            st.session_state["history"].append({
                "ts": int(time.time()),
                "proveedor": provider,
                "modelo": selected_model if provider=="Groq" else hf_model,
                "tarea": task,
                "params": {"temperature": float(temperature), "max_tokens": int(max_tokens)},
                "input_len": len(text_in),
                "latency_ms": int((t1-t0)*1000),
                "output": out,
            })
    except ValueError as ve:
        st.warning(str(ve))
    except Exception as e:
        st.error(f"Error al analizar: {e}")

# Resultado inmediato (último)
if st.session_state["history"]:
    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
    last = st.session_state["history"][-1]
    st.markdown("#### Resultado (último)")
    st.markdown(f"<span class='small'>latencia: {last['latency_ms']} ms · input: {last['input_len']} chars</span>", unsafe_allow_html=True)
    st.markdown(last["output"])
st.markdown("</div>", unsafe_allow_html=True)

# ================== HISTORIAL ==================
st.markdown("### 📚 Historial")
if not st.session_state["history"]:
    st.info("Sin resultados aún.")
else:
    for i, item in enumerate(reversed(st.session_state["history"]), start=1):
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**#{i}**", unsafe_allow_html=True)
            chip(item["proveedor"])
            chip(item["modelo"])
            chip(item["tarea"])
            chip(f"T={item['params']['temperature']}")
            chip(f"max={item['params']['max_tokens']}")
            st.markdown(f"<span class='small'>latencia: {item['latency_ms']} ms · input: {item['input_len']} chars</span>",
                        unsafe_allow_html=True)
            st.code(item["output"], language="markdown")
            st.markdown("</div>", unsafe_allow_html=True)

# ================== ESTADO .ENV ==================
with st.expander("🔒 Estado de variables (.env)"):
    st.write("GROQ_API_KEY cargada:", bool(os.getenv("GROQ_API_KEY")))
    st.write("HUGGINGFACE_API_KEY cargada:", bool(os.getenv("HUGGINGFACE_API_KEY")) or bool(os.getenv("HuggingFace_API_KEY")))
