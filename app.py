import streamlit as st
import torch
from transformers import AutoImageProcessor, EfficientNetForImageClassification, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image

# 1. Configuração da Página
st.set_page_config(page_title="Painel de Marketing Automático", page_icon="🚀", layout="wide")
st.title("🎯 Painel de Marketing Automatizado - Case 6")

# 2. Carregamento com Cache
@st.cache_resource
def carregar_modelos():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor_visao = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
    modelo_visao = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0").to(device)
    pipeline_bert = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0 if device == "cuda" else -1)
    tokenizer_t5 = AutoTokenizer.from_pretrained("google/flan-t5-small")
    modelo_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
    return processor_visao, modelo_visao, pipeline_bert, tokenizer_t5, modelo_t5, device

processor_visao, modelo_visao, pipeline_bert, tokenizer_t5, modelo_t5, device = carregar_modelos()

# 3. Funções de IA
def analisar_imagem(imagem):
    inputs = processor_visao(imagem, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = modelo_visao(**inputs).logits
    return modelo_visao.config.id2label[logits.argmax(-1).item()]

def analisar_review_bert(review_texto):
    resultado = pipeline_bert(review_texto)[0]
    estrelas = int(resultado['label'].split()[0])
    return "Qualidade premium validada." if estrelas >= 4 else "Equilíbrio perfeito de design."

def gerar_post_publicitario(nome_produto, contexto_sentimento, palavras_chave):
    prompt = f"Write a short benefit phrase for: {palavras_chave}."
    inputs = tokenizer_t5(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = modelo_t5.generate(**inputs, max_length=30)
    return tokenizer_t5.decode(outputs[0], skip_special_tokens=True)

# 4. Interface (Com inicialização segura do session_state)
if 'keywords' not in st.session_state: st.session_state['keywords'] = ""
if 'review' not in st.session_state: st.session_state['review'] = ""

col1, col2 = st.columns(2)

with col1:
    imagem_carregada = st.file_uploader("1. Suba a imagem", type=["jpg", "png"])
    nome_produto = st.text_input("2. Nome do produto", value="Produto")
    palavras_chave = st.text_input("3. Keywords", key='keywords')
    review_cliente = st.text_area("4. Avaliação", key='review')
    
    if st.button("✨ Sugerir com IA"):
        if imagem_carregada:
            img = Image.open(imagem_carregada).convert("RGB")
            detccao = analisar_imagem(img)
            st.session_state['keywords'] = f"uso de {detccao} com estilo"
            st.session_state['review'] = f"Este {detccao} é excelente!"
            st.rerun()

    botao_gerar = st.button("🚀 Gerar Campanha")

with col2:
    if botao_gerar and imagem_carregada:
        img = Image.open(imagem_carregada).convert("RGB")
        st.image(img, use_column_width=True)
        res_vis = analisar_imagem(img)
        res_tex = analisar_review_bert(st.session_state['review'])
        post = gerar_post_publicitario(nome_produto, res_tex, st.session_state['keywords'])
        st.info(f"Copy: {post}")
