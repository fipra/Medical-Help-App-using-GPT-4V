import streamlit as st
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
import tempfile

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

sample_prompt = """Sei un cuoco esperto e hai un vasto repertorio di ricette. Ti verranno fornite delle immagini di ingredienti e dovrai generare una ricetta basata su quegli ingredienti. Assicurati di includere dettagli come gli ingredienti necessari, le istruzioni per la preparazione e il tempo di cottura. Rispondi solo se puoi creare una ricetta con gli ingredienti forniti. Se non sei sicuro, puoi dire "Non sono in grado di creare una ricetta con questi ingredienti"."""

# Inizializza le variabili di stato della sessione
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_gpt4_model_for_recipe(filename: str, sample_prompt=sample_prompt):
    base64_image = encode_image(filename)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": sample_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=1500
    )

    return response.choices[0].message.content


def chat_eli(query):
    eli5_prompt = "Devi spiegare quanto segue a un bambino di cinque anni. \n" + query
    messages = [
        {
            "role": "user",
            "content": eli5_prompt
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1500
    )

    return response.choices[0].message.content


st.title("Ricette basate su ingredienti fotografati con GPT-4")

with st.expander("Informazioni sull'App"):
    st.write("Carica un'immagine di ingredienti per ottenere una ricetta generata da GPT-4.")

uploaded_file = st.file_uploader("Carica un'Immagine", type=["jpg", "jpeg", "png"])

# Gestione del file temporaneo
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state['filename'] = tmp_file.name

    st.image(uploaded_file, caption='Immagine Caricata')

# Pulsante per generare la ricetta
if st.button('Genera Ricetta'):
    if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
        st.session_state['result'] = call_gpt4_model_for_recipe(st.session_state['filename'])
        st.markdown(st.session_state['result'], unsafe_allow_html=True)
        os.unlink(st.session_state['filename'])  # Elimina il file temporaneo dopo l'elaborazione

# Spiegazione ELI5
if 'result' in st.session_state and st.session_state['result']:
    st.info("Di seguito hai la possibilità di ottenere una spiegazione in termini più semplici (ELI5).")
    if st.button('Spiegazione ELI5'):
        simplified_explanation = chat_eli(st.session_state['result'])
        st.markdown(simplified_explanation, unsafe_allow_html=True)
