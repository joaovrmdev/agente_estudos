import streamlit as st
from transformers import pipeline
from PIL import Image
import fitz  # PyMuPDF
import io

# Define o título da página e um ícone
st.set_page_config(page_title="Pergunte aos Seus Documentos", page_icon="❓")

# Função para carregar o modelo de QA de Documentos do Hugging Face.
# O decorator @st.cache_resource garante que o modelo seja carregado apenas uma vez.
@st.cache_resource
def carregar_modelo_qa():
    """
    Carrega e retorna o pipeline de Question Answering para documentos.
    """
    return pipeline("document-question-answering", model="impira/layoutlm-document-qa")

# Função principal que processa os documentos e a pergunta
def processar_documentos(documentos_carregados, pergunta, modelo_qa):
    """
    Itera sobre os documentos carregados, faz a pergunta ao modelo
    e coleta as respostas.
    """
    respostas_encontradas = []

    if not documentos_carregados:
        st.warning("Por favor, carregue um ou mais documentos antes de perguntar.")
        return []

    # Barra de progresso para dar feedback ao usuário
    barra_progresso = st.progress(0, text="Processando documentos...")

    for i, documento in enumerate(documentos_carregados):
        try:
            nome_arquivo = documento.name
            # Atualiza o texto da barra de progresso
            barra_progresso.progress((i + 1) / len(documentos_carregados), text=f"Analisando: {nome_arquivo}")

            # Se o arquivo for um PDF
            if nome_arquivo.lower().endswith(".pdf"):
                # Lê os bytes do PDF
                bytes_pdf = documento.read()
                doc_pdf = fitz.open(stream=bytes_pdf, filetype="pdf")

                # Itera sobre cada página do PDF
                for num_pagina, pagina in enumerate(doc_pdf):
                    # Converte a página em uma imagem (pixmap)
                    pix = pagina.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    imagem = Image.open(io.BytesIO(img_bytes))

                    # Usa o modelo para encontrar a resposta na imagem da página
                    resultado = modelo_qa(image=imagem, question=pergunta)
                    
                    # Filtra resultados com alta confiança e adiciona à lista
                    respostas_filtradas = [r for r in resultado if r['score'] > 0.01] # Ajuste o score se necessário
                    for res in respostas_filtradas:
                        respostas_encontradas.append({
                            "arquivo": nome_arquivo,
                            "pagina": num_pagina + 1,
                            "resposta": res["answer"],
                            "score": res["score"]
                        })

            # Se o arquivo for uma imagem
            elif nome_arquivo.lower().endswith((".png", ".jpg", ".jpeg")):
                imagem = Image.open(documento)
                resultado = modelo_qa(image=imagem, question=pergunta)
                
                respostas_filtradas = [r for r in resultado if r['score'] > 0.01]
                for res in respostas_filtradas:
                    respostas_encontradas.append({
                        "arquivo": nome_arquivo,
                        "pagina": "N/A",
                        "resposta": res["answer"],
                        "score": res["score"]
                    })
        
        except Exception as e:
            st.error(f"Erro ao processar o arquivo {documento.name}: {e}")

    barra_progresso.empty() # Limpa a barra de progresso ao final
    return respostas_encontradas


# --- Interface do Usuário (UI) ---

st.title("❓ Pergunte aos Seus Documentos com LayoutLM")
st.markdown("Faça upload de imagens (PNG, JPG) ou PDFs e faça uma pergunta sobre o conteúdo deles.")

# Carrega o modelo de IA
modelo_qa = carregar_modelo_qa()

# Área de upload de arquivos
documentos_carregados = st.file_uploader(
    "📂 Carregue seus documentos aqui",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True
)

# Campo de texto para a pergunta
pergunta_usuario = st.text_input("📝 Digite sua pergunta aqui:")

# Botão para iniciar o processo
if st.button("Encontrar Respostas"):
    if pergunta_usuario:
        with st.spinner("🧠 O modelo está pensando... Por favor, aguarde."):
            # Processa os documentos e obtém as respostas
            respostas = processar_documentos(documentos_carregados, pergunta_usuario, modelo_qa)

        st.subheader("✅ Resultados:")
        if respostas:
            # Ordena as respostas pela pontuação de confiança (maior primeiro)
            respostas_ordenadas = sorted(respostas, key=lambda x: x['score'], reverse=True)
            
            for res in respostas_ordenadas:
                st.markdown(f"**Resposta:** `{res['resposta']}`")
                st.markdown(f"**Fonte:** `{res['arquivo']}` (Página: `{res['pagina']}`)")
                st.markdown(f"**Confiança:** `{res['score']:.2%}`")
                st.divider()
        else:
            st.info("Nenhuma resposta foi encontrada nos documentos fornecidos para esta pergunta.")
    else:
        st.warning("Por favor, digite uma pergunta.")