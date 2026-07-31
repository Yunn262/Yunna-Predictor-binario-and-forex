import streamlit as st
import pandas as pd
from datetime import datetime
import os

# Configuração da página
st.set_page_config(
    page_title="⚽ Soccer Striker AI",
    page_icon="⚽",
    layout="centered"
)

# Título
st.title("⚽ Soccer Striker AI")
st.caption("Assistente Estatístico de Pênaltis")

# Inicializar estado da sessão
if 'historico' not in st.session_state:
    st.session_state.historico = []
if 'resultado_temp' not in st.session_state:
    st.session_state.resultado_temp = None

# Função para salvar em CSV
def salvar_csv(historico):
    df = pd.DataFrame(historico)
    df.to_csv('historico_penaltis.csv', index=False)

# Função para calcular probabilidades
def calcular_probabilidades(historico, janela=None):
    if not historico:
        return {"⬅️ Esquerda": 0.33, "⬆️ Centro": 0.33, "➡️ Direita": 0.34}
    
    if janela:
        dados = historico[-janela:]
    else:
        dados = historico
    
    direcoes = [h['direcao'] for h in dados]
    total = len(direcoes)
    
    prob_esquerda = direcoes.count("⬅️ Esquerda") / total
    prob_centro = direcoes.count("⬆️ Centro") / total
    prob_direita = direcoes.count("➡️ Direita") / total
    
    return {
        "⬅️ Esquerda": prob_esquerda,
        "⬆️ Centro": prob_centro,
        "➡️ Direita": prob_direita
    }

# Função para análise avançada
def analise_avancada(historico):
    if len(historico) < 3:
        return None
    
    # Padrões de sequência
    ultimas_3 = [h['direcao'] for h in historico[-3:]]
    
    # Contar repetições
    repeticoes = 0
    for i in range(len(historico)-1):
        if historico[i]['direcao'] == historico[i+1]['direcao']:
            repeticoes += 1
    
    # Alternâncias
    alternancias = 0
    for i in range(len(historico)-1):
        if historico[i]['direcao'] != historico[i+1]['direcao']:
            alternancias += 1
    
    return {
        "ultimas_3": ultimas_3,
        "repeticoes": repeticoes,
        "alternancias": alternancias,
        "taxa_repeticao": repeticoes / (len(historico)-1) if len(historico) > 1 else 0
    }

# Sidebar com controles
with st.sidebar:
    st.header("🎮 Controles")
    
    # Botão de reiniciar
    if st.button("🔄 Reiniciar Sessão", type="secondary"):
        st.session_state.historico = []
        st.rerun()
    
    st.divider()
    
    # Configurações de análise
    st.header("⚙️ Configurações")
    janela_analise = st.selectbox(
        "Janela de análise:",
        ["Todas", "Últimas 20", "Últimas 50", "Últimas 100"],
        index=0
    )
    
    # Converter seleção para número
    if janela_analise == "Últimas 20":
        janela = 20
    elif janela_analise == "Últimas 50":
        janela = 50
    elif janela_analise == "Últimas 100":
        janela = 100
    else:
        janela = None
    
    st.divider()
    
    # Estatísticas gerais
    st.header("📊 Estatísticas")
    total_rodadas = len(st.session_state.historico)
    gols = sum(1 for h in st.session_state.historico if h['resultado'] == "✅ Gol")
    defesas = total_rodadas - gols
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rodadas", total_rodadas)
        st.metric("Gols", gols)
    with col2:
        st.metric("Defesas", defesas)
        if total_rodadas > 0:
            st.metric("Aproveitamento", f"{(gols/total_rodadas)*100:.1f}%")
        else:
            st.metric("Aproveitamento", "0%")

# Área principal
col1, col2 = st.columns([2, 1])

with col1:
    # Histórico
    st.header("📝 Histórico")
    if st.session_state.historico:
        for i, rodada in enumerate(reversed(st.session_state.historico), 1):
            st.text(f"{len(st.session_state.historico)-i+1}️⃣ {rodada['direcao']} {rodada['resultado']}")
    else:
        st.info("Nenhuma rodada registrada ainda. Comece a jogar!")

with col2:
    # Input de resultado
    st.header("🎯 Registrar")
    
    # Selecionar direção primeiro
    direcao = st.radio(
        "Direção do chute:",
        ["⬅️ Esquerda", "⬆️ Centro", "➡️ Direita"],
        key="direcao_input"
    )
    
    # Botões de resultado
    col_gol, col_defesa = st.columns(2)
    with col_gol:
        if st.button("✅ Gol", type="primary", use_container_width=True):
            st.session_state.historico.append({
                "resultado": "✅ Gol",
                "direcao": direcao,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            salvar_csv(st.session_state.historico)
            st.rerun()
    
    with col_defesa:
        if st.button("❌ Defesa", type="secondary", use_container_width=True):
            st.session_state.historico.append({
                "resultado": "❌ Defesa",
                "direcao": direcao,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            salvar_csv(st.session_state.historico)
            st.rerun()
    
    # Botão desfazer
    if st.button("↩️ Desfazer última", use_container_width=True):
        if st.session_state.historico:
            st.session_state.historico.pop()
            salvar_csv(st.session_state.historico)
            st.rerun()

# Análise e sugestão
st.divider()
st.header("🤖 Análise da IA")

if st.session_state.historico:
    # Calcular probabilidades
    probs = calcular_probabilidades(st.session_state.historico, janela)
    
    # Encontrar a direção com maior probabilidade de defesa (para evitar)
    direcao_evitar = max(probs, key=probs.get)
    prob_evitar = probs[direcao_evitar]
    
    # Sugerir a direção oposta ou com menor probabilidade
    direcao_sugerir = min(probs, key=probs.get)
    
    # Calcular confiança baseada em vários fatores
    confianca_base = (1 - prob_evitar) * 100
    
    # Ajustar confiança baseado no tamanho da amostra
    fator_amostra = min(len(st.session_state.historico) / 100, 1.0)
    
    # Análise avançada
    analise = analise_avancada(st.session_state.historico)
    if analise:
        # Penalizar se há muitas repetições
        if analise['taxa_repeticoes'] > 0.6:
            confianca_base *= 0.8
    
    confianca_final = confianca_base * (0.5 + 0.5 * fator_amostra)
    confianca_final = min(confianca_final, 95)  # Máximo 95% de confiança
    
    # Exibir probabilidades
    st.subheader("📈 Probabilidade Atual")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⬅️ Esquerda", f"{probs['⬅️ Esquerda']*100:.1f}%")
        st.progress(probs['⬅️ Esquerda'])
    with col2:
        st.metric("⬆️ Centro", f"{probs['⬆️ Centro']*100:.1f}%")
        st.progress(probs['⬆️ Centro'])
    with col3:
        st.metric("➡️ Direita", f"{probs['➡️ Direita']*100:.1f}%")
        st.progress(probs['➡️ Direita'])
    
    # Estrelas de confiança
    estrelas = min(int(confianca_final / 20), 5)
    st.write(f"{'⭐' * estrelas}{'☆' * (5-estrelas)}")
    st.metric("Confiança", f"{confianca_final:.1f}%")
    
    # Sugestão
    st.subheader("🎯 Sugestão")
    if confianca_final > 50:
        st.success(f"**{direcao_sugerir} - CHUTE AQUI!**")
    else:
        st.warning(f"Dados insuficientes para alta confiança. Sugestão: {direcao_sugerir}")
    
    # Insights da análise avançada
    if analise and len(st.session_state.historico) > 10:
        with st.expander("🔍 Análise Detalhada"):
            st.write(f"Últimas 3 direções: {' → '.join(analise['ultimas_3'])}")
            st.write(f"Taxa de repetição: {analise['taxa_repeticoes']*100:.1f}%")
            st.write(f"Total de repetições: {analise['repeticoes']}")
            st.write(f"Total de alternâncias: {analise['alternancias']}")
            
            if analise['taxa_repeticoes'] > 0.7:
                st.info("🔔 Alta tendência de repetição detectada!")
            elif analise['taxa_repeticoes'] < 0.3:
                st.info("🔄 Alta tendência de alternância detectada!")

else:
    st.info("Registre algumas rodadas para receber sugestões da IA!")

# Rodapé
st.divider()
st.caption(f"Total de rodadas registradas: {len(st.session_state.historico)}")
if os.path.exists('historico_penaltis.csv'):
    st.caption("💾 Dados salvos automaticamente em 'historico_penaltis.csv'")
