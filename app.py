import gradio as gr
import whisper
import librosa
import soundfile as sf
from deep_translator import GoogleTranslator
from gtts import gTTS
from fpdf import FPDF
from datetime import datetime
import os

# --- MODELL BETÖLTÉSE (Base modell a 24/7 stabilitáshoz) ---
model = whisper.load_model("base") 

def translate_speech(audio_path, source_lang, target_lang):
    if audio_path is None: return "Nincs hang!", None, ""
    lang_codes = {"Magyar": "hu", "English": "en", "Deutsch": "de"}
    src, dst = lang_codes[source_lang], lang_codes[target_lang]
    audio_res = None
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
        fixed_path = "fixed.wav"
        sf.write(fixed_path, audio_data, 16000)
        result = model.transcribe(fixed_path, language=src)
        original = str(result.get("text", "")).strip()
        if not original: return "Nem hallottam semmit...", None, ""
        translated = GoogleTranslator(source=src, target=dst).translate(original)
        tts = gTTS(text=translated, lang=dst)
        audio_res = "result.mp3"
        tts.save(audio_res)
        report = f"[{datetime.now().strftime('%H:%M')}]\n({source_lang}->{target_lang})\nEREDETI: {original}\nFORDÍTÁS: {translated}\n"
        return translated, audio_res, report
    except Exception as e: 
        return f"Hiba: {str(e)}", None, ""

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

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Warrior Translator v4.1 - 24/7 Online Edition")
    with gr.Row():
        with gr.Column():
            src_lang = gr.Dropdown(choices=["Magyar", "English", "Deutsch"], value="Magyar", label="Source Language / Forrásnyelv")
            audio_in = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Record / Hang rögzítése")
            target_lang = gr.Dropdown(choices=["Magyar", "English", "Deutsch"], value="English", label="Target Language / Célnyelv")
            start_btn = gr.Button("⚔️ TRANSLATE / FORDÍTÁS", variant="primary")
        with gr.Column():
            text_out = gr.Textbox(label="Translated Text / Fordítás")
            audio_out = gr.Audio(label="🔈 AI Voice / Hang")
            hist_out = gr.Textbox(label="📜 History / Előzmények", lines=8)
            pdf_btn = gr.Button("📄 EXPORT PDF")
            pdf_file = gr.File(label="Download / Letöltés")
    start_btn.click(translate_speech, [audio_in, src_lang, target_lang], [text_out, audio_out, hist_out])
    pdf_btn.click(save_to_pdf, hist_out, pdf_file)

demo.launch()
