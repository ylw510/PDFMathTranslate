import asyncio
import cgi
import json
import os
import shutil
import socket
import uuid
from asyncio import CancelledError
from pathlib import Path
import typing as T

import gradio as gr
import requests
import tqdm
from gradio_pdf import PDF
from string import Template
import logging

from pdf2zh import __version__
from pdf2zh.high_level import translate
from pdf2zh.doclayout import ModelInstance
from pdf2zh.config import ConfigManager
from pdf2zh.translator import (
    AnythingLLMTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DifyTranslator,
    ArgosTranslator,
    GeminiTranslator,
    GoogleTranslator,
    MiniMaxTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAITranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    GrokTranslator,
    GroqTranslator,
    DeepseekTranslator,
    OpenAIlikedTranslator,
    QwenMtTranslator,
    X302AITranslator,
)
from babeldoc.docvision.doclayout import OnnxModel
from babeldoc import __version__ as babeldoc_version

logger = logging.getLogger(__name__)


class _LazyModel:
    """Defers model loading until first access so the GUI starts instantly."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = OnnxModel.load_available()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self._ensure_loaded()
        return getattr(self._model, name)


BABELDOC_MODEL = _LazyModel()
# The following variables associate strings with translators
service_map: dict[str, BaseTranslator] = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "MiniMax": MiniMaxTranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
    "302.AI": X302AITranslator,
}

# The following variables associate strings with specific languages
lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}

# The following variable associate strings with page ranges
page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}

# Check if this is a public demo, which has resource limits
flag_demo = False

# Limit resources
if ConfigManager.get("PDF2ZH_DEMO"):
    flag_demo = True
    service_map = {
        "Google": GoogleTranslator,
    }
    page_map = {
        "First": [0],
        "First 20 pages": list(range(0, 20)),
    }
    client_key = ConfigManager.get("PDF2ZH_CLIENT_KEY")
    server_key = ConfigManager.get("PDF2ZH_SERVER_KEY")


# Limit Enabled Services
enabled_services: T.Optional[T.List[str]] = ConfigManager.get("ENABLED_SERVICES")
if isinstance(enabled_services, list):
    default_services = ["Google", "Bing"]
    enabled_services_names = [str(_).lower().strip() for _ in enabled_services]
    enabled_services = [
        k
        for k in service_map.keys()
        if str(k).lower().strip() in enabled_services_names
    ]
    if len(enabled_services) == 0:
        raise RuntimeError("No services available.")
    enabled_services = default_services + enabled_services
else:
    enabled_services = list(service_map.keys())


# Configure about Gradio show keys
hidden_gradio_details: bool = bool(ConfigManager.get("HIDDEN_GRADIO_DETAILS"))


# Public demo control
def verify_recaptcha(response):
    """
    This function verifies the reCAPTCHA response.
    """
    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": server_key, "response": response}
    result = requests.post(recaptcha_url, data=data).json()
    return result.get("success")


class PreviewInputError(Exception):
    """The link or file cannot be shown in the preview."""


def download_with_limit(url: str, save_path: str, size_limit: int) -> str:
    """
    This function downloads a file from a URL and saves it to a specified path.

    Inputs:
        - url: The URL to download the file from
        - save_path: The path to save the file to
        - size_limit: The maximum size of the file to download

    Returns:
        - The path of the downloaded file
    """
    chunk_size = 1024
    total_size = 0
    try:
        with requests.get(url, stream=True, timeout=20) as response:
            response.raise_for_status()
            content = response.headers.get("Content-Disposition")
            try:  # filename from header
                _, params = cgi.parse_header(content)
                filename = params["filename"]
            except Exception:  # filename from url
                filename = os.path.basename(url)
            filename = os.path.splitext(os.path.basename(filename))[0] + ".pdf"
            saved = save_path / filename
            with open(saved, "wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    total_size += len(chunk)
                    if size_limit and total_size > size_limit:
                        raise gr.Error("Exceeds file size limit")
                    file.write(chunk)
    except requests.RequestException as exc:
        raise PreviewInputError(
            "链接下载失败。请换一个 PDF 直链，或改用 File 上传。"
        ) from exc
    with open(saved, "rb") as file:
        if file.read(5) != b"%PDF-":
            saved.unlink(missing_ok=True)
            raise PreviewInputError(
                "这个链接不是 PDF。请使用 PDF 文件地址，或改用 File 上传。"
            )
    return saved


def unchanged_preview(message: str):
    """Keep the current preview and show why this attempt failed."""
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(value=message, visible=True),
    )


def stop_translate_file(state: dict) -> None:
    """
    This function stops the translation process.

    Inputs:
        - state: The state of the translation process

    Returns:- None
    """
    session_id = state["session_id"]
    if session_id is None:
        return
    if session_id in cancellation_event_map:
        logger.info(f"Stopping translation for session {session_id}")
        cancellation_event_map[session_id].set()


def build_bilingual_preview(
    dual_pdf: Path, dest_dir: Path, filename: str
) -> tuple[str, str]:
    """Split interleaved dual pages into one original PDF and one translation PDF.

    Each result keeps the source page size, so the preview can fit every page to
    its own column instead of scaling one double-width sheet.
    """
    import pymupdf

    original_dest = dest_dir / f"{filename}-preview-original.pdf"
    translated_dest = dest_dir / f"{filename}-preview-translated.pdf"
    with pymupdf.open(dual_pdf) as dual:
        original = pymupdf.open()
        translated = pymupdf.open()
        pair_count = dual.page_count // 2
        for i in range(pair_count):
            original.insert_pdf(dual, from_page=i * 2, to_page=i * 2)
            translated.insert_pdf(dual, from_page=i * 2 + 1, to_page=i * 2 + 1)
        original.save(original_dest, deflate=True, garbage=3)
        translated.save(translated_dest, deflate=True, garbage=3)
        original.close()
        translated.close()
    return str(original_dest), str(translated_dest)


def paper_pages_for_preview(selected_page: list[int] | None, pair_count: int) -> list[int]:
    """Map each preview slot to the 1-based page number in the source PDF."""
    if pair_count <= 0:
        return []
    if not selected_page:
        return list(range(1, pair_count + 1))
    return [page + 1 for page in selected_page[:pair_count]]


def render_sentence_map(map_path: str | None) -> str:
    """Preview reads the sidecar written at typeset time. It does not re-segment the PDF."""
    pages: list = []
    if map_path and os.path.exists(map_path):
        with open(map_path, encoding="utf-8") as handle:
            pages = json.load(handle)
    payload = json.dumps(pages, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    return f'<pre id="sentence-map" style="display:none">{payload}</pre>'


def render_paper_pager(pages: list[int]) -> str:
    """Shared pager. The number typed is the source paper page, not the pair index."""
    if not pages:
        return ""
    joined = ",".join(str(page) for page in pages)
    return (
        f'<div id="paper-pager" data-pages="{joined}">'
        f'<button type="button" class="paper-page-btn" data-dir="-1">上一页</button>'
        f'<span class="paper-page-label">第 '
        f'<input id="paper-page-input" type="number" min="{min(pages)}" max="{max(pages)}" value="{pages[0]}">'
        f" 页 / 共 {len(pages)} 页</span>"
        f'<button type="button" class="paper-page-btn" data-dir="1">下一页</button>'
        f"</div>"
    )


def translate_file(
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    page_input,
    prompt,
    threads,
    skip_subset_fonts,
    ignore_cache,
    vfont,
    mode_choice,
    recaptcha_response,
    state,
    progress=gr.Progress(),
    *envs,
):
    """
    This function translates a PDF file from one language to another.

    Inputs:
        - file_type: The type of file to translate
        - file_input: The file to translate
        - link_input: The link to the file to translate
        - service: The translation service to use
        - lang_from: The language to translate from
        - lang_to: The language to translate to
        - page_range: The range of pages to translate
        - page_input: The input for the page range
        - prompt: The custom prompt for the llm
        - threads: The number of threads to use
        - recaptcha_response: The reCAPTCHA response
        - state: The state of the translation process
        - progress: The progress bar
        - envs: The environment variables

    Returns:
        - The translated file
        - The translated file
        - The translated file
        - The progress bar
        - The progress bar
        - The progress bar
    """
    session_id = uuid.uuid4()
    state["session_id"] = session_id
    cancellation_event_map[session_id] = asyncio.Event()
    # Translate PDF content using selected service.
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise gr.Error("reCAPTCHA fail")

    progress(0, desc="Starting translation...")

    output = Path("pdf2zh_files")
    output.mkdir(parents=True, exist_ok=True)

    if file_type == "File":
        if not file_input:
            raise gr.Error("No input")
        file_path = shutil.copy(file_input, output)
    else:
        if not link_input:
            raise gr.Error("No input")
        try:
            file_path = download_with_limit(
                link_input,
                output,
                5 * 1024 * 1024 if flag_demo else None,
            )
        except PreviewInputError as exc:
            return unchanged_preview(str(exc))

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = output / f"{filename}.pdf"
    file_mono = output / f"{filename}-mono.pdf"
    file_dual = output / f"{filename}-dual.pdf"

    translator = service_map[service]
    if page_range != "Others":
        selected_page = page_map[page_range]
    else:
        selected_page = []
        for p in page_input.split(","):
            if "-" in p:
                start, end = p.split("-")
                selected_page.extend(range(int(start) - 1, int(end)))
            else:
                selected_page.append(int(p) - 1)
    lang_from = lang_map[lang_from]
    lang_to = lang_map[lang_to]

    _envs = {}
    for i, env in enumerate(translator.envs.items()):
        _envs[env[0]] = envs[i]
    for k, v in _envs.items():
        if str(k).upper().endswith("API_KEY") and str(v) == "***":
            # Load Real API_KEYs from local configure file
            real_keys: str = ConfigManager.get_env_by_translatername(
                translator, k, None
            )
            _envs[k] = real_keys

    print(f"Files before translation: {os.listdir(output)}")

    def progress_bar(t):
        if isinstance(t, dict):
            raw = float(t.get("overall_progress") or 0)
            frac = raw / 100.0 if raw > 1 else raw
            desc = t.get("stage") or "Translating..."
        else:
            total = getattr(t, "total", None) or 0
            done = getattr(t, "n", 0) or 0
            frac = done / total if total else 0
            desc = getattr(t, "desc", None) or "Translating..."
        progress(min(1.0, max(0.0, float(frac))), desc=str(desc))

    try:
        threads = int(threads)
    except ValueError:
        threads = 1

    param = {
        "files": [str(file_raw)],
        "pages": selected_page,
        "lang_in": lang_from,
        "lang_out": lang_to,
        "service": f"{translator.name}",
        "output": output,
        "thread": int(threads),
        "callback": progress_bar,
        "cancellation_event": cancellation_event_map[session_id],
        "envs": _envs,
        "prompt": Template(prompt) if prompt else None,
        "skip_subset_fonts": skip_subset_fonts,
        "ignore_cache": ignore_cache,
        "vfont": vfont,  # 添加自定义公式字体正则表达式
        "model": ModelInstance.value,
    }

    try:
        from pdf2zh.kernel import KernelRegistry
        from pdf2zh.kernel.protocol import TranslateRequest

        KernelRegistry.switch(mode_choice)
        kernel = KernelRegistry.get()
        request = TranslateRequest(
            files=[str(file_raw)],
            output=str(output),
            pages=selected_page,
            lang_in=lang_from,
            lang_out=lang_to,
            service=f"{translator.name}",
            thread=int(threads),
            envs=_envs,
            prompt=str(prompt) if prompt else None,
            skip_subset_fonts=skip_subset_fonts,
            ignore_cache=ignore_cache,
            vfont=vfont,
        )
        kernel.translate(
            request,
            callback=progress_bar,
            cancellation_event=cancellation_event_map[session_id],
        )
    except CancelledError:
        del cancellation_event_map[session_id]
        raise gr.Error("Translation cancelled")
    except Exception as exc:
        from pdf2zh.converter import TranslationServiceError

        if isinstance(exc, TranslationServiceError):
            return unchanged_preview(str(exc))
        raise
    print(f"Files after translation: {os.listdir(output)}")

    if not file_mono.exists() or not file_dual.exists():
        raise gr.Error("No output")

    original_preview, translated_preview = build_bilingual_preview(
        file_dual, output, filename
    )
    import pymupdf

    with pymupdf.open(file_dual) as dual:
        pair_count = dual.page_count // 2
    paper_pages = paper_pages_for_preview(selected_page, pair_count)

    progress(1.0, desc="Translation complete!")

    return (
        str(file_mono),
        gr.update(visible=False),
        str(file_dual),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        original_preview,
        translated_preview,
        gr.update(visible=True),
        render_paper_pager(paper_pages),
        render_sentence_map(
            str(output / f"{filename}-preview-sentence-map.json")
        ),
        gr.update(visible=False),
    )


def babeldoc_translate_file(**kwargs):
    from babeldoc.high_level import init as babeldoc_init

    babeldoc_init()
    from babeldoc.high_level import async_translate as babeldoc_translate
    from babeldoc.translation_config import TranslationConfig as YadtConfig

    for translator in [
        GoogleTranslator,
        BingTranslator,
        DeepLTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        XinferenceTranslator,
        AzureOpenAITranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        AnythingLLMTranslator,
        ArgosTranslator,
        GrokTranslator,
        GroqTranslator,
        DeepseekTranslator,
        OpenAIlikedTranslator,
        QwenMtTranslator,
        X302AITranslator,
    ]:
        if kwargs["service"] == translator.name:
            translator = translator(
                kwargs["lang_in"],
                kwargs["lang_out"],
                "",
                envs=kwargs["envs"],
                prompt=kwargs["prompt"],
                ignore_cache=kwargs["ignore_cache"],
            )
            break
    else:
        raise ValueError("Unsupported translation service")
    import asyncio
    from babeldoc.main import create_progress_handler

    for file in kwargs["files"]:
        file = file.strip("\"'")
        yadt_config = YadtConfig(
            input_file=file,
            font=None,
            pages=",".join((str(x) for x in getattr(kwargs, "raw_pages", []))),
            output_dir=kwargs["output"],
            doc_layout_model=BABELDOC_MODEL,
            translator=translator,
            debug=False,
            lang_in=kwargs["lang_in"],
            lang_out=kwargs["lang_out"],
            no_dual=False,
            no_mono=False,
            qps=kwargs["thread"],
            use_rich_pbar=False,
            disable_rich_text_translate=not isinstance(translator, OpenAITranslator),
            skip_clean=kwargs["skip_subset_fonts"],
            report_interval=0.5,
        )

        async def yadt_translate_coro(yadt_config):
            progress_context, progress_handler = create_progress_handler(yadt_config)

            # 开始翻译
            with progress_context:
                async for event in babeldoc_translate(yadt_config):
                    progress_handler(event)
                    if yadt_config.debug:
                        logger.debug(event)
                    kwargs["callback"](progress_context)
                    if kwargs["cancellation_event"].is_set():
                        yadt_config.cancel_translation()
                        raise CancelledError
                    if event["type"] == "finish":
                        result = event["translate_result"]
                        logger.info("Translation Result:")
                        logger.info(f"  Original PDF: {result.original_pdf_path}")
                        logger.info(f"  Time Cost: {result.total_seconds:.2f}s")
                        logger.info(f"  Mono PDF: {result.mono_pdf_path or 'None'}")
                        logger.info(f"  Dual PDF: {result.dual_pdf_path or 'None'}")
                        file_mono = result.mono_pdf_path
                        file_dual = result.dual_pdf_path
                        break
            import gc

            gc.collect()
            return (
                str(file_mono),
                str(file_mono),
                str(file_dual),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        return asyncio.run(yadt_translate_coro(yadt_config))


# Global setup
custom_blue = gr.themes.Color(
    c50="#E8F3FF",
    c100="#BEDAFF",
    c200="#94BFFF",
    c300="#6AA1FF",
    c400="#4080FF",
    c500="#165DFF",  # Primary color
    c600="#0E42D2",
    c700="#0A2BA6",
    c800="#061D79",
    c900="#03114D",
    c950="#020B33",
)

custom_css = """
    .secondary-text {color: #999 !important;}
    footer {visibility: hidden}
    .env-warning {color: #dd5500 !important;}
    .env-success {color: #559900 !important;}

    /* Add dashed border to input-file class */
    .input-file {
        border: 1.2px dashed #165DFF !important;
        border-radius: 6px !important;
    }

    .progress-bar-wrap {
        border-radius: 8px !important;
    }

    .progress-bar {
        border-radius: 8px !important;
    }

    .pdf-canvas {
        display: block !important;
        height: auto !important;
        overflow: visible !important;
        position: relative !important;
    }

    .sent-layer {
        position: absolute;
        pointer-events: none;
    }

    .sent-hit {
        position: absolute;
        pointer-events: auto;
        border-radius: 3px;
    }

    .sent-hit.on-src.active {
        background: rgba(22, 93, 255, 0.16);
    }

    .sent-hit.on-dst.active {
        background: rgba(0, 180, 42, 0.30);
        box-shadow: inset 0 0 0 2px rgba(0, 128, 38, 0.55);
    }

    .pdf-canvas canvas {
        width: 100% !important;
        height: auto !important;
        max-width: 100%;
    }

    #bilingual-pages {
        align-items: flex-start !important;
        flex-wrap: nowrap !important;
        gap: 16px !important;
        width: 100%;
    }

    #bilingual-pages > div {
        min-width: 0 !important;
        flex: 1 1 0% !important;
    }

    #bilingual-pages .button-row {
        display: none !important;
    }

    .preview-side-label {
        display: block;
        margin: 0 0 8px;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .preview-side-original {
        color: #0E42D2;
        background: #E8F3FF;
        border: 1px solid #94BFFF;
    }

    .preview-side-translated {
        color: #008026;
        background: #E8FFEA;
        border: 1px solid #7BE188;
    }

    #bilingual-preview,
    #paper-page-bar,
    #bilingual-pages {
        width: 100%;
    }

    #preview-error {
        color: #CB272D;
        background: #FFECE8;
        border: 1px solid #FBACA3;
        border-radius: 8px;
        padding: 10px 14px;
    }

    #paper-pager {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: nowrap;
        gap: 12px;
        width: 100%;
        box-sizing: border-box;
        margin: 0 0 12px;
        padding: 8px 12px;
        border-radius: 8px;
        background: #F7F8FA;
    }

    #paper-pager .paper-page-btn {
        border: 1px solid #C9CDD4;
        background: #fff;
        border-radius: 6px;
        padding: 6px 14px;
        cursor: pointer;
        font-size: 14px;
        white-space: nowrap;
    }

    #paper-pager .paper-page-label {
        font-size: 15px;
        font-weight: 600;
        white-space: nowrap;
    }

    #paper-page-input {
        width: 72px;
        margin: 0 4px;
        padding: 4px 8px;
        border: 1px solid #C9CDD4;
        border-radius: 6px;
        font-size: 15px;
        text-align: center;
    }

    #workspace {
        flex-wrap: nowrap !important;
        align-items: stretch !important;
    }

    #sidebar-shell {
        position: relative !important;
        overflow: hidden !important;
        min-width: 320px !important;
        flex: 1 1 0% !important;
        transition: min-width 0.28s ease, max-width 0.28s ease, flex-basis 0.28s ease;
    }

    #sidebar-shell.collapsed {
        min-width: 36px !important;
        max-width: 36px !important;
        flex: 0 0 36px !important;
    }

    #sidebar-shell.collapsed > :not(#sidebar-handle) {
        display: none !important;
    }

    #sidebar-handle {
        position: absolute !important;
        z-index: 20;
        top: 8px;
        right: 0;
        width: 28px !important;
        min-width: 28px !important;
        max-width: 28px !important;
        height: 64px !important;
        margin: 0 !important;
        padding: 0 !important;
        border-radius: 8px 0 0 8px !important;
    }

    #sidebar-shell.collapsed #sidebar-handle {
        right: 4px;
        border-radius: 8px !important;
    }
    """

demo_recaptcha = """
    <script src="https://www.google.com/recaptcha/api.js?render=explicit" async defer></script>
    <script type="text/javascript">
        var onVerify = function(token) {
            el=document.getElementById('verify').getElementsByTagName('textarea')[0];
            el.value=token;
            el.dispatchEvent(new Event('input'));
        };
    </script>
    """

tech_details_string = f"""
                    <summary>Technical details</summary>
                    - GitHub: <a href="https://github.com/Byaidu/PDFMathTranslate">Byaidu/PDFMathTranslate</a><br>
                    - BabelDOC: <a href="https://github.com/funstory-ai/BabelDOC">funstory-ai/BabelDOC</a><br>
                    - GUI by: <a href="https://github.com/reycn">Rongxin</a><br>
                    - pdf2zh Version: {__version__} <br>
                    - BabelDOC Version: {babeldoc_version}
                """
cancellation_event_map = {}


# The following code creates the GUI
with gr.Blocks(
    title="PDFMathTranslate - PDF Translation with preserved formats",
    theme=gr.themes.Default(
        primary_hue=custom_blue, spacing_size="md", radius_size="lg"
    ),
    css=custom_css,
    head=demo_recaptcha if flag_demo else "",
) as demo:
    gr.Markdown(
        "# [PDFMathTranslate @ GitHub](https://github.com/Byaidu/PDFMathTranslate)"
    )

    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=1, min_width=0, elem_id="sidebar-shell"):
            toggle_sidebar_btn = gr.Button(
                "◀",
                elem_id="sidebar-handle",
            )
            gr.Markdown("## File | < 5 MB" if flag_demo else "## File")
            file_type = gr.Radio(
                choices=["File", "Link"],
                label="Type",
                value="File",
            )
            file_input = gr.File(
                label="File",
                file_count="single",
                file_types=[".pdf", ".doc", ".docx"],
                type="filepath",
                elem_classes=["input-file"],
            )
            link_input = gr.Textbox(
                label="Link",
                visible=False,
                interactive=True,
            )
            gr.Markdown("## Option")
            service = gr.Dropdown(
                label="Service",
                choices=enabled_services,
                value=enabled_services[0],
            )
            envs = []
            for i in range(3):
                envs.append(
                    gr.Textbox(
                        visible=False,
                        interactive=True,
                    )
                )
            with gr.Row():
                lang_from = gr.Dropdown(
                    label="Translate from",
                    choices=lang_map.keys(),
                    value=ConfigManager.get("PDF2ZH_LANG_FROM", "English"),
                )
                lang_to = gr.Dropdown(
                    label="Translate to",
                    choices=lang_map.keys(),
                    value=ConfigManager.get("PDF2ZH_LANG_TO", "Simplified Chinese"),
                )
            page_range = gr.Radio(
                choices=page_map.keys(),
                label="Pages",
                value=list(page_map.keys())[0],
            )

            page_input = gr.Textbox(
                label="Page range",
                visible=False,
                interactive=True,
            )

            with gr.Accordion("Open for More Experimental Options!", open=False):
                gr.Markdown("#### Experimental")
                threads = gr.Textbox(
                    label="number of threads", interactive=True, value="4"
                )
                skip_subset_fonts = gr.Checkbox(
                    label="Skip font subsetting", interactive=True, value=False
                )
                ignore_cache = gr.Checkbox(
                    label="Ignore cache", interactive=True, value=False
                )
                vfont = gr.Textbox(
                    label="Custom formula font regex (vfont)",
                    interactive=True,
                    value=ConfigManager.get("PDF2ZH_VFONT", ""),
                )
                prompt = gr.Textbox(
                    label="Custom Prompt for llm", interactive=True, visible=False
                )
                mode_choice = gr.Dropdown(
                    label="Translation Mode",
                    choices=["fast", "precise"],
                    value="fast",
                    interactive=True,
                )
                envs.append(prompt)

            def on_select_service(service, evt: gr.EventData):
                translator = service_map[service]
                _envs = []
                for i in range(4):
                    _envs.append(gr.update(visible=False, value=""))
                for i, env in enumerate(translator.envs.items()):
                    label = env[0]
                    value = ConfigManager.get_env_by_translatername(
                        translator, env[0], env[1]
                    )
                    visible = True
                    if hidden_gradio_details:
                        if (
                            "MODEL" not in str(label).upper()
                            and value
                            and hidden_gradio_details
                        ):
                            visible = False
                        # Hidden Keys From Gradio
                        if "API_KEY" in label.upper():
                            value = "***"  # We use "***" Present Real API_KEY
                    _envs[i] = gr.update(
                        visible=visible,
                        label=label,
                        value=value,
                    )
                _envs[-1] = gr.update(visible=translator.CustomPrompt)
                return _envs

            def on_select_filetype(file_type):
                return (
                    gr.update(visible=file_type == "File"),
                    gr.update(visible=file_type == "Link"),
                )

            def on_select_page(choice):
                if choice == "Others":
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)

            def on_vfont_change(value):
                ConfigManager.set("PDF2ZH_VFONT", value)
                return value

            output_title = gr.Markdown("## Translated", visible=False)
            output_file_mono = gr.File(
                label="Download Translation (Mono)", visible=False
            )
            output_file_dual = gr.File(
                label="Download Translation (Dual)", visible=False
            )
            recaptcha_response = gr.Textbox(
                label="reCAPTCHA Response", elem_id="verify", visible=False
            )
            recaptcha_box = gr.HTML('<div id="recaptcha-box"></div>')
            translate_btn = gr.Button("Translate", variant="primary")
            cancellation_btn = gr.Button("Cancel", variant="secondary")
            tech_details_tog = gr.Markdown(
                tech_details_string,
                elem_classes=["secondary-text"],
            )
            page_range.select(on_select_page, page_range, page_input)
            service.select(
                on_select_service,
                service,
                envs,
            )
            vfont.change(on_vfont_change, inputs=vfont, outputs=None)
            file_type.select(
                on_select_filetype,
                file_type,
                [file_input, link_input],
                js=(
                    f"""
                    (a,b)=>{{
                        try{{
                            grecaptcha.render('recaptcha-box',{{
                                'sitekey':'{client_key}',
                                'callback':'onVerify'
                            }});
                        }}catch(error){{}}
                        return [a];
                    }}
                    """
                    if flag_demo
                    else ""
                ),
            )

        with gr.Column(scale=2, min_width=0, elem_id="preview-panel"):
            gr.Markdown("## Preview")
            preview_error = gr.Markdown(visible=False, elem_id="preview-error")
            preview = PDF(label="Document Preview", visible=True, height=1400)
            with gr.Column(visible=False, elem_id="bilingual-preview") as bilingual_preview:
                paper_page_bar = gr.HTML(value="", elem_id="paper-page-bar", padding=False)
                sentence_map = gr.HTML(value="", elem_id="sentence-map-host", padding=False)
                with gr.Row(elem_id="bilingual-pages"):
                    with gr.Column(scale=1, min_width=0):
                        gr.HTML(
                            '<div class="preview-side-label preview-side-original">原文</div>',
                            padding=False,
                        )
                        preview_original = PDF(
                            show_label=False,
                            visible=True,
                            height=1400,
                            interactive=False,
                        )
                    with gr.Column(scale=1, min_width=0):
                        gr.HTML(
                            '<div class="preview-side-label preview-side-translated">译文</div>',
                            padding=False,
                        )
                        preview_translated = PDF(
                            show_label=False,
                            visible=True,
                            height=1400,
                            interactive=False,
                        )

    # Event handlers
    file_input.upload(
        lambda file_path: (
            gr.update(value=file_path, visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        inputs=file_input,
        outputs=[preview, bilingual_preview, preview_error],
        js=(
            f"""
            (a,b)=>{{
                try{{
                    grecaptcha.render('recaptcha-box',{{
                        'sitekey':'{client_key}',
                        'callback':'onVerify'
                    }});
                }}catch(error){{}}
                return [a];
            }}
            """
            if flag_demo
            else ""
        ),
    )

    state = gr.State({"session_id": None})

    toggle_sidebar_btn.click(
        lambda: None,
        js="""
            () => {
                const shell = document.getElementById('sidebar-shell');
                if (!shell) return;
                const collapsed = shell.classList.toggle('collapsed');
                const btn = document.getElementById('sidebar-handle');
                if (!btn) return;
                const label = collapsed ? '▶' : '◀';
                const nodes = btn.querySelectorAll('span');
                const target = nodes.length ? nodes[nodes.length - 1] : btn;
                target.textContent = label;
            }
        """,
        show_progress="hidden",
    )

    translate_btn.click(
        translate_file,
        inputs=[
            file_type,
            file_input,
            link_input,
            service,
            lang_from,
            lang_to,
            page_range,
            page_input,
            prompt,
            threads,
            skip_subset_fonts,
            ignore_cache,
            vfont,
            mode_choice,
            recaptcha_response,
            state,
            *envs,
        ],
        outputs=[
            output_file_mono,
            preview,
            output_file_dual,
            output_file_mono,
            output_file_dual,
            output_title,
            preview_original,
            preview_translated,
            bilingual_preview,
            paper_page_bar,
            sentence_map,
            preview_error,
        ],
    ).then(
        lambda: None,
        js="""
            () => {
                try { grecaptcha.reset(); } catch (e) {}
                const shell = document.getElementById('sidebar-shell');
                const btn = document.getElementById('sidebar-handle');
                const setGlyph = (glyph) => {
                    if (!btn) return;
                    const nodes = btn.querySelectorAll('span');
                    const target = nodes.length ? nodes[nodes.length - 1] : btn;
                    target.textContent = glyph;
                };
                const error = document.getElementById('preview-error');
                const failed = error && error.offsetParent !== null;
                if (!shell) return;
                if (failed) {
                    shell.classList.remove('collapsed');
                    setGlyph('◀');
                    return;
                }
                const bilingual = document.getElementById('bilingual-preview');
                if (!bilingual || bilingual.offsetParent === null) return;
                if (shell.classList.contains('collapsed')) return;
                shell.classList.add('collapsed');
                setGlyph('▶');
            }
        """,
        show_progress="hidden",
    )

    cancellation_btn.click(
        stop_translate_file,
        inputs=[state],
    )

    demo.load(
        lambda: None,
        js="""
            () => {
                if (window.__pdf2zhPageSync) return;
                window.__pdf2zhPageSync = true;
                let last = [null, null];
                let lock = false;

                const pagesOf = () => {
                    const el = document.querySelector('#paper-pager');
                    if (!el || !el.dataset.pages) return [];
                    return el.dataset.pages.split(',')
                        .map((part) => parseInt(part, 10))
                        .filter((page) => Number.isFinite(page));
                };
                const inputsOf = () => [...document.querySelectorAll('#bilingual-pages .page-count input')];
                const currentSlot = () => {
                    const inputs = inputsOf();
                    const slot = inputs.length ? parseInt(inputs[0].value, 10) : 1;
                    return Number.isFinite(slot) && slot > 0 ? slot : 1;
                };
                const showPaperPage = (slot) => {
                    const pages = pagesOf();
                    const box = document.getElementById('paper-page-input');
                    if (!box || !pages.length || document.activeElement === box) return;
                    const paper = pages[Math.min(Math.max(slot, 1), pages.length) - 1];
                    if (box.value !== String(paper)) box.value = String(paper);
                };
                const setSlot = (slot) => {
                    const pages = pagesOf();
                    const inputs = inputsOf();
                    if (!pages.length || inputs.length < 2) return;
                    const next = Math.min(Math.max(slot, 1), pages.length);
                    lock = true;
                    inputs.forEach((input) => {
                        if (String(input.value) === String(next)) return;
                        input.value = String(next);
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                    });
                    lock = false;
                    showPaperPage(next);
                };
                const jumpToPaperPage = (paperPage) => {
                    const pages = pagesOf();
                    const box = document.getElementById('paper-page-input');
                    const index = pages.indexOf(paperPage);
                    if (index < 0) {
                        const paper = pages[currentSlot() - 1];
                        if (box && paper !== undefined) box.value = String(paper);
                        if (box) box.blur();
                        return;
                    }
                    if (box) box.blur();
                    setSlot(index + 1);
                };

                document.addEventListener('click', (event) => {
                    const btn = event.target.closest('#paper-pager .paper-page-btn');
                    if (!btn) return;
                    event.preventDefault();
                    const dir = parseInt(btn.dataset.dir, 10) || 0;
                    setSlot(currentSlot() + dir);
                });
                document.addEventListener('change', (event) => {
                    if (event.target.id !== 'paper-page-input') return;
                    jumpToPaperPage(parseInt(event.target.value, 10));
                });
                document.addEventListener('keydown', (event) => {
                    if (event.target.id !== 'paper-page-input' || event.key !== 'Enter') return;
                    event.preventDefault();
                    jumpToPaperPage(parseInt(event.target.value, 10));
                });

                setInterval(() => {
                    if (lock) return;
                    const root = document.getElementById('bilingual-preview');
                    if (!root || root.offsetParent === null) return;
                    const inputs = inputsOf();
                    if (inputs.length < 2) return;
                    const vals = inputs.map((el) => el.value);
                    if (vals[0] !== vals[1]) {
                        let src = 0;
                        if (vals[0] === last[0] && vals[1] !== last[1]) src = 1;
                        const dst = 1 - src;
                        lock = true;
                        inputs[dst].value = vals[src];
                        inputs[dst].dispatchEvent(new Event('input', { bubbles: true }));
                        lock = false;
                    }
                    last = [inputs[0].value, inputs[1].value];
                    const slot = parseInt(inputs[0].value, 10);
                    if (Number.isFinite(slot)) {
                        showPaperPage(slot);
                        mountSentences(slot);
                    }
                }, 200);

                const mountSentences = (slot) => {
                    const pre = document.getElementById('sentence-map');
                    const hosts = [...document.querySelectorAll('#bilingual-pages .pdf-canvas')];
                    if (!pre || hosts.length < 2) return;
                    const canvas = hosts[0].querySelector('canvas');
                    if (!canvas || canvas.clientWidth < 20) return;
                    let data;
                    try { data = JSON.parse(pre.textContent); } catch (e) { return; }
                    const page = data[slot - 1];
                    if (!page) return;
                    const key = slot + ':' + canvas.clientWidth + 'x' + canvas.clientHeight + ':' + pre.textContent.length;
                    if (hosts[0].dataset.sentKey === key) return;
                    const setActive = (index) => {
                        document.querySelectorAll('#bilingual-pages .sent-hit').forEach((el) => {
                            el.classList.toggle('active', el.dataset.pair === String(index));
                        });
                    };
                    hosts.forEach((host, side) => {
                        host.dataset.sentKey = key;
                        host.querySelectorAll('.sent-layer').forEach((el) => el.remove());
                        const canvas = host.querySelector('canvas');
                        if (!canvas) return;
                        const layer = document.createElement('div');
                        layer.className = 'sent-layer';
                        const hostRect = host.getBoundingClientRect();
                        const canvasRect = canvas.getBoundingClientRect();
                        layer.style.left = (canvasRect.left - hostRect.left) + 'px';
                        layer.style.top = (canvasRect.top - hostRect.top) + 'px';
                        layer.style.width = canvasRect.width + 'px';
                        layer.style.height = canvasRect.height + 'px';
                        const rectsOf = (value) => {
                            if (!value || !value.length) return [];
                            return Array.isArray(value[0]) ? value : [value];
                        };
                        page.pairs.forEach((pair, index) => {
                            rectsOf(side === 0 ? pair.src : pair.dst).forEach((box) => {
                                const hit = document.createElement('div');
                                hit.className = 'sent-hit ' + (side === 0 ? 'on-src' : 'on-dst');
                                hit.dataset.pair = String(index);
                                hit.style.left = (box[0] * 100) + '%';
                                hit.style.top = (box[1] * 100) + '%';
                                hit.style.width = (box[2] * 100) + '%';
                                hit.style.height = (box[3] * 100) + '%';
                                hit.addEventListener('mouseenter', () => setActive(index));
                                hit.addEventListener('mouseleave', (event) => {
                                    const next = event.relatedTarget;
                                    if (next && next.dataset && next.dataset.pair === String(index)) return;
                                    setActive(-1);
                                });
                                layer.appendChild(hit);
                            });
                        });
                        host.appendChild(layer);
                    });
                };
            }
        """,
        show_progress="hidden",
    )


def parse_user_passwd(file_path: str) -> tuple:
    """
    Parse the user name and password from the file.

    Inputs:
        - file_path: The file path to read.
    Outputs:
        - tuple_list: The list of tuples of user name and password.
        - content: The content of the file
    """
    tuple_list = []
    content = ""
    if not file_path:
        return tuple_list, content
    if len(file_path) == 2:
        try:
            with open(file_path[1], "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path[1]}' not found.")
    try:
        with open(file_path[0], "r", encoding="utf-8") as file:
            tuple_list = [
                tuple(line.strip().split(",")) for line in file if line.strip()
            ]
    except FileNotFoundError:
        print(f"Error: File '{file_path[0]}' not found.")
    return tuple_list, content


def _has_ipv6() -> bool:
    """Check whether the system can bind an IPv6 socket."""
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.close()
        return True
    except OSError:
        return False


def setup_gui(
    share: bool = False, auth_file: list = ["", ""], server_port=7860
) -> None:
    """
    Setup the GUI with the given parameters.

    Inputs:
        - share: Whether to share the GUI.
        - auth_file: The file path to read the user name and password.

    Outputs:
        - None
    """
    user_list, html = parse_user_passwd(auth_file)

    auth_kwargs = {}
    if len(user_list) > 0:
        auth_kwargs = {"auth": user_list, "auth_message": html}

    if flag_demo:
        demo.launch(server_name="0.0.0.0", max_file_size="5mb", inbrowser=True)
        return

    # Try binding addresses in order: "::" accepts both IPv4+IPv6 on most
    # dual-stack systems, "0.0.0.0" is IPv4-only, "127.0.0.1" is loopback,
    # and finally fall back to Gradio's share mode.
    bind_addresses = []
    if _has_ipv6():
        bind_addresses.append("[::]")
    bind_addresses.append("0.0.0.0")
    bind_addresses.append("127.0.0.1")

    for addr in bind_addresses:
        try:
            demo.launch(
                server_name=addr,
                debug=True,
                inbrowser=True,
                share=share,
                server_port=server_port,
                **auth_kwargs,
            )
            return
        except Exception:
            print(
                f"Error launching GUI using {addr}.\n"
                "This may be caused by global mode of proxy software."
            )

    # Last resort: let Gradio create a share link
    demo.launch(
        debug=True,
        inbrowser=True,
        share=True,
        server_port=server_port,
        **auth_kwargs,
    )


# For auto-reloading while developing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_gui()
