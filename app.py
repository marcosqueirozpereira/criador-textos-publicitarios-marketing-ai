import streamlit as st
import torch
from transformers import AutoImageProcessor, EfficientNetForImageClassification, pipeline, AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# 1. Configuração da Página
st.set_page_config(page_title="Painel de Marketing", page_icon="🚀", layout="wide")
st.title("🎯 Painel de Marketing Automatizado - Case 6")

# 2. Carregamento Otimizado
@st.cache_resource
def carregar_modelos():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Visão
    processor_visao = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
    modelo_visao = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0").to(device)
    # Sentimento
    pipeline_bert = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0 if device == "cuda" else -1)
    # Geração (DistilGPT-2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token # Necessário para o GPT-2
    modelo_gerador = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    
    return processor_visao, modelo_visao, pipeline_bert, tokenizer, modelo_gerador, device

processor_visao, modelo_visao, pipeline_bert, tokenizer, modelo_gerador, device = carregar_modelos()

# 3. Funções de IA
def analisar_imagem(imagem):
    inputs = processor_visao(imagem, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = modelo_visao(**inputs).logits
    return modelo_visao.config.id2label[logits.argmax(-1).item()]

def analisar_review_bert(review_texto):
    resultado = pipeline_bert(review_texto)[0]
    estrelas = int(resultado['label'].split()[0])
    return "Qualidade premium validada." if estrelas >= 4 else "Design e custo-benefício."

def gerar_post_publicitario(nome_produto, contexto, palavras_chave):
    prompt = f"Product: {nome_produto}. Style: {palavras_chave}. Review: {contexto}. Marketing ad:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = modelo_gerador.generate(
            **inputs, 
            max_new_tokens=40, 
            do_sample=True, 
            temperature=0.7, 
            top_k=50,
            no_repeat_ngram_size=2
        )
    
    copy = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Filtra apenas o conteúdo gerado após o prompt
    final_copy = copy.split("Marketing ad:")[-1].strip()
    return f"✨ Viva a experiência {nome_produto}! {final_copy.capitalize()}. 🚀 Garanta a sua agora!"

# 4. Interface Segura
if 'keywords' not in st.session_state: st.session_state['keywords'] = ""
if 'review' not in st.session_state: st.session_state['review'] = ""

col1, col2 = st.columns(2)

with col1:
    imagem_carregada = st.file_uploader("1. Suba a imagem", type=["jpg", "png"])
    nome_produto = st.text_input("2. Nome do produto", value="Produto")
    palavras_chave = st.text_input("3. Keywords", value=st.session_state['keywords'])
    review_cliente = st.text_area("4. Avaliação", value=st.session_state['review'])
    
    if st.button("✨ Sugerir com IA"):
        if imagem_carregada:
            img = Image.open(imagem_carregada).convert("RGB")
            detccao = analisar_imagem(img)
            st.session_state['keywords'] = f"uso de {detccao} com estilo moderno"
            st.session_state['review'] = f"Este {detccao} é excelente e muito resistente."
            st.rerun()

    botao_gerar = st.button("🚀 Gerar Campanha")

with col2:
    if botao_gerar and imagem_carregada:
        img = Image.open(imagem_carregada).convert("RGB")
        st.image(img, use_column_width=True)
        res_tex = analisar_review_bert(review_cliente)
        post = gerar_post_publicitario(nome_produto, res_tex, palavras_chave)
        st.info(f"Copy: {post}")
