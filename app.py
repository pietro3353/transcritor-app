import streamlit as st
import os
import torch
import ffmpeg
import tempfile
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

# --- Configuração da Página ---
st.set_page_config(
    page_title="Transcrição de Áudio com Whisper",
    page_icon="🎙️",
    layout="centered"
)

# --- Cache do Modelo ---
# O cache do Streamlit evita recarregar o modelo pesado toda vez.
@st.cache_resource
def load_model():
    """Carrega e armazena em cache o modelo e o pipeline do Whisper."""
    # Mudar para um modelo maior se estiver usando uma GPU no Hugging Face Spaces
    # model_id = "openai/whisper-large-v3" 
    model_id = "openai/whisper-small" # Ótimo para começar com CPU

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device="cpu",  # Forçar CPU para o nível gratuito do HF Spaces
            torch_dtype=torch.float32,
        )
        st.success("Modelo de IA carregado com sucesso!")
        return pipe
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# --- Interface do Usuário ---
st.title("🎙️ Transcrição de Áudio com Whisper")
st.markdown("Faça o upload de um arquivo de áudio (MP3, WAV, M4A) e obtenha a transcrição em segundos.")

# Carrega o pipeline de IA
transcription_pipe = load_model()

if transcription_pipe:
    uploaded_file = st.file_uploader(
        "Escolha um arquivo de áudio", 
        type=['mp3', 'wav', 'm4a', 'ogg', 'flac']
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        
        if st.button("Transcrever Áudio Agora"):
            with st.spinner("O processo de transcrição foi iniciado. Por favor, aguarde..."):
                try:
                    # Salva o arquivo em um local temporário
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_audio_path = tmp_file.name
                    
                    # Executa a transcrição
                    result = transcription_pipe(
                        temp_audio_path,
                        chunk_length_s=30, # Processa em pedaços de 30s
                        batch_size=4,
                        return_timestamps=True
                    )
                    transcription_text = result["text"]
                    
                    st.divider()
                    st.subheader("✅ Transcrição Concluída:")
                    st.text_area("Resultado", transcription_text, height=250)
                    
                    # Botão de download
                    st.download_button(
                        label="Baixar Transcrição (.txt)",
                        data=transcription_text.encode("utf-8"),
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcricao.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a transcrição: {e}")
                finally:
                    # Garante que o arquivo temporário seja removido
                    if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
else:
    st.warning("O pipeline de transcrição não pôde ser carregado. A aplicação não pode funcionar.")