import gradio as gr
import whisper
import librosa
import soundfile as sf
import torch
import spaces
from deep_translator import GoogleTranslator
from gtts import gTTS
from fpdf import FPDF
from datetime import datetime
import os

# --- 1. MODELL DEFINIÁLÁSA ---
# Nem töltjük be itt, csak nevet adunk neki
model = None

# --- 2. ZERO-GPU OPTIMALIZÁLT FORDÍTÁS ---
@spaces.GPU(duration=120) # 120 másodperc, hogy a betöltés is beleférjen
def translate_speech(audio_path, source_lang, target_lang):
    global model # Jelezzük, hogy a kinti változót használjuk
    
    if audio_path is None: return "No audio! / Nincs hang!", None, ""

    # Csak most töltjük be, amikor már van GPU...
    if model is None:
        print("Loading Large-v3 model on GPU...")
        model = whisper.load_model("large-v3", device="cuda")
        
    lang_codes = {"Magyar": "hu", "English": "en", "Deutsch": "de"}
    src, dst = lang_codes[source_lang], lang_codes[target_lang]
    
    audio_res = None
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
        fixed_path = "fixed.wav"
        sf.write(fixed_path, audio_data, 16000)
        
        # Átírás Large-v3-mal
        result = model.transcribe(
            fixed_path, 
            language="hu" if source_lang == "Magyar" else src,
            beam_size=5
        )
        original = str(result.get("text", "")).strip()
        if not original: return "I didn't hear anything... / Nem hallottam semmit...", None, ""

        # Fordítás
        translated = GoogleTranslator(source=src, target=dst).translate(original)
        
        # TTS
        tts = gTTS(text=translated, lang=dst)
        audio_res = "result.mp3"
        tts.save(audio_res)
        
        report = f"[{datetime.now().strftime('%H:%M')}]\n({source_lang}->{target_lang})\nORIGINAL: {original}\nTRANSLATION: {translated}\n"
        return translated, audio_res, report
    except Exception as e: 
        return f"Error / Hiba: {str(e)}", None, ""

def clean_text_for_pdf(text):
    replacements = {'ő': 'ö', 'ű': 'ü', 'Ő': 'Ö', 'Ű': 'Ü', '’': "'", '–': '-', '„': '"', '”': '"'}
    for k, v in replacements.items(): text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

def save_to_pdf(history):
    if not history: return None
    try:
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, txt="WARRIOR TRANSLATION LOG", ln=True, align='C'); pdf.ln(10)
        pdf.set_font("Helvetica", size=11); clean_history = clean_text_for_pdf(history)
        pdf.multi_cell(0, 10, txt=clean_history)
        fname = "Warrior_Log.pdf"; pdf.output(fname); return fname
    except: return None

# --- 3. UI (VÁLTOZATLAN) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Warrior Translator v5.1 - ZeroGPU Pro Edition")
    with gr.Row():
        with gr.Column():
            src_lang = gr.Dropdown(choices=["Magyar", "English", "Deutsch"], value="Magyar", label="Source Language / Forrásnyelv")
            audio_in = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Record / Hang rögzítése")
            target_lang = gr.Dropdown(choices=["Magyar", "English", "Deutsch"], value="English", label="Target Language / Célnyelv")
            start_btn = gr.Button("⚔️ START TRANSLATION / FORDÍTÁS", variant="primary")
        with gr.Column():
            text_out = gr.Textbox(label="Translated Text / Lefordított szöveg")
            audio_out = gr.Audio(label="🔈 AI Voice / AI Hang")
            hist_out = gr.Textbox(label="📜 History Log / Előzmények", lines=8)
            pdf_btn = gr.Button("📄 EXPORT PDF / MENTÉS")
            pdf_file = gr.File(label="Download / Letöltés")

    start_btn.click(translate_speech, [audio_in, src_lang, target_lang], [text_out, audio_out, hist_out])
    pdf_btn.click(save_to_pdf, hist_out, pdf_file)

demo.queue().launch()
