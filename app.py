import streamlit as st
import torch
from transformers import AutoImageProcessor, EfficientNetForImageClassification, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Painel de Marketing Automático", page_icon="🚀", layout="wide")
st.title("🎯 Painel de Marketing Automatizado - Case 6")
st.markdown("Faça o upload de uma imagem e gere copys publicitários de alta conversão focados na experiência do usuário.")

# ==========================================
# 2. CARREGAMENTO DOS MODELOS (COM CACHE)
# ==========================================
@st.cache_resource
def carregar_modelos():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cérebro 1: Visão
    processor_visao = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
    modelo_visao = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0").to(device)
    
    # Cérebro 2: Sentimento
    pipeline_bert = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0 if device == "cuda" else -1)
    
    # Cérebro 3: Texto Generativo
    tokenizer_t5 = AutoTokenizer.from_pretrained("google/flan-t5-small")
    modelo_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
    
    return processor_visao, modelo_visao, pipeline_bert, tokenizer_t5, modelo_t5, device

with st.spinner("Carregando o motor de Inteligência Artificial... (Isso pode levar alguns minutos na primeira vez)"):
    processor_visao, modelo_visao, pipeline_bert, tokenizer_t5, modelo_t5, device = carregar_modelos()

# ==========================================
# 3. FUNÇÕES DO PIPELINE
# ==========================================
def analisar_imagem(imagem):
    inputs = processor_visao(imagem, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = modelo_visao(**inputs).logits
    return modelo_visao.config.id2label[logits.argmax(-1).item()]

def analisar_review_bert(review_texto):
    resultado = pipeline_bert(review_texto)[0]
    estrelas = int(resultado['label'].split()[0])
    if estrelas >= 4:
        return "Qualidade premium validada por centenas de clientes satisfeitos."
    elif estrelas == 3:
        return "O equilíbrio perfeito entre design e o melhor custo-benefício."
    else:
        return "Tecnologia totalmente renovada para atender ao seu estilo."

def gerar_post_publicitario(nome_produto, contexto_sentimento, palavras_chave):
    prompt = f"Write a short phrase about the benefit of this product: {palavras_chave}."
    inputs = tokenizer_t5(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = modelo_t5.generate(
            **inputs,
            max_length=35,          
            num_beams=4,            
            temperature=0.7,        
            do_sample=True,
            repetition_penalty=2.0  
        )
        
    beneficio_gerado = tokenizer_t5.decode(outputs[0], skip_special_tokens=True).strip().lower()
    
    if len(beneficio_gerado) < 5 or "write" in beneficio_gerado:
         beneficio_gerado = f"ter mais praticidade no seu dia a dia"

    postagem_final = (
        f"✨ Você já pensou em {beneficio_gerado}?\n\n"
        f"Conheça nosso(a) novo(a) {nome_produto} e transforme sua experiência.\n\n"
        f"💡 {contexto_sentimento}\n\n"
        f"👉 Clique no link da bio e garanta o(a) seu(sua)!\n\n"
        f"#Estilo #Inovacao #Experiencia #Lifestyle"
    )
    return postagem_final

# ==========================================
# 4. INTERFACE DE USUÁRIO (FRONTEND)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Entrada de Dados")
    imagem_carregada = st.file_uploader("1. Faça o upload da imagem", type=["jpg", "jpeg", "png"])
    nome_produto = st.text_input("2. Nome da campanha ou produto", value="Vídeo Cinemático")
    palavras_chave = st.text_input("3. Benefícios esperados (Keywords)", value="iluminação dramática de pôr do sol, guerreiros marchando em campo de batalha antigo")
    review_cliente = st.text_area("4. Avaliação base do cliente", value="A luz e a composição entregam muita ação. Excelente projeto.")
    
    botao_gerar = st.button("🚀 Gerar Campanha de Marketing", use_container_width=True)

with col2:
    st.markdown("### 📤 Resultado Gerado")
    if botao_gerar:
        if imagem_carregada is None:
            st.warning("Por favor, faça o upload de uma imagem primeiro.")
        else:
            imagem = Image.open(imagem_carregada).convert("RGB")
            st.image(imagem, caption="Imagem Analisada", use_column_width=True)
            
            with st.spinner("Analisando imagem e processando redes neurais..."):
                resultado_visao = analisar_imagem(imagem)
                resultado_texto = analisar_review_bert(review_cliente)
                postagem_final = gerar_post_publicitario(nome_produto, resultado_texto, palavras_chave)
            
            st.success("Análise Multimodal Concluída!")
            st.markdown(f"**👁️ Detecção Visual (EfficientNet):** `{resultado_visao}`")
            st.markdown(f"**📊 Análise de Sentimento (BERT):** `{resultado_texto}`")
            
            st.markdown("---")
            st.markdown("📱 **COPYWRITING GERADO (Flan-T5):**")
            st.info(postagem_final)