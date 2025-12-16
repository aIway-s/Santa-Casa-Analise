import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import calendar
import io
import numpy as np
from pysus.ftp.databases.sih import SIH

# ===================== CONFIGURAÇÃO =====================
st.set_page_config(page_title="Gestão Hospitalar - Auditoria Final", layout="wide")
st.title("🏥 Painel de Indicadores - Auditoria (Filtro de Idade Corrigido)")
st.markdown("---")

# ===================== PARÂMETROS =====================
CAPACIDADE_FIXA = {'geral': 89, 'uti_a': 17, 'uti_n': 9, 'uti_p': 1}

# ===================== AUXILIARES =====================
def get_meses_quadrimestre(q):
    if q == "Q1 (Jan-Abr)": return [1, 2, 3, 4]
    if q == "Q2 (Mai-Ago)": return [5, 6, 7, 8]
    if q == "Q3 (Set-Dez)": return [9, 10, 11, 12]
    return []

def encontrar_coluna(df, candidatos):
    cols_upper = [c.upper().strip() for c in df.columns]
    # Busca Exata
    for termo in candidatos:
        for i, col in enumerate(cols_upper):
            if termo == col: return df.columns[i]
    # Busca Parcial
    for termo in candidatos:
        for i, col in enumerate(cols_upper):
            if termo in col: return df.columns[i]
    return None

def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

# --- PONTUAÇÃO ---
def pontuacao_mortalidade(taxa): return 7 if taxa <= 3 else (4 if taxa < 6 else (2 if taxa <= 8 else 0))
def pontuacao_ocupacao(taxa): return 7 if taxa >= 80 else (4 if taxa >= 65 else (2 if taxa >= 55 else 0))
def pontuacao_tmp_medica(dias): return 6 if 0 < dias < 8 else (4 if 8 <= dias < 11 else (2 if 11 <= dias < 14 else 0))
def pontuacao_tmp_cirurgica(dias): return 6 if 0 < dias < 5 else (4 if 5 <= dias < 7 else (2 if 7 <= dias < 9 else 0))
def pontuacao_uti(taxa): return 6 if taxa >= 85 else (4 if taxa >= 70 else (2 if taxa >= 60 else 0))
def pontuacao_infeccao(densidade): return 6 if densidade <= 2.0 else (4 if densidade <= 3.0 else (2 if densidade <= 5.0 else 0))

# ===================== PROCESSAMENTO =====================
@st.cache_data(show_spinner=False)
def processar_mes_unico(ano, month, uf, cnes_filter):
    sih_db = SIH().load()
    year = ano
    dias_mes = get_days_in_month(year, month)
    
    # Capacidades
    caps = {k: v * dias_mes for k, v in CAPACIDADE_FIXA.items()}
    
    # Inicializa
    d = {k: 0 for k in ["saidas_tot", "obitos_tot", "dias_geral", "dias_med", "saidas_med", 
                        "dias_cir", "saidas_cir", "dias_a", "dias_n", "dias_p"]}
    d.update({'sp_columns': [], 'unique_atoprof': [], 'raw_unique_atoprof': [], 'unique_procrea': [], 'raw_unique_procrea': []})
    d["mes"] = month
    
    # ------------------- 1. RD (Mortalidade, Ocupação, TMP) -------------------
    try:
        files_rd = sih_db.get_files(group="RD", uf=uf, year=year, month=month)
        if files_rd:
            df_rd = sih_db.download(files_rd).to_dataframe()
            df_rd.columns = [c.upper().strip() for c in df_rd.columns]
            
            cnes_c = encontrar_coluna(df_rd, ["CNES", "CNES_EXEC"])
            if cnes_c:
                df_rd['CNES_INT'] = pd.to_numeric(df_rd[cnes_c], errors='coerce').fillna(0).astype(int)
                df_rd = df_rd[df_rd['CNES_INT'] == int(cnes_filter)].copy()
                
                if not df_rd.empty:
                    c_morte = encontrar_coluna(df_rd, ["MORTE", "OBITO"])
                    c_dias = encontrar_coluna(df_rd, ["DIAS_PERM", "QT_DIARIAS"])
                    c_clinica = encontrar_coluna(df_rd, ["ESPEC"])
                    
                    print("RD columns:", df_rd.columns.tolist())
                    print("c_clinica (ESPEC):", c_clinica)
                    
                    if c_morte: df_rd[c_morte] = pd.to_numeric(df_rd[c_morte], errors='coerce').fillna(0).astype(int)
                    if c_dias: df_rd[c_dias] = pd.to_numeric(df_rd[c_dias], errors='coerce').fillna(0).astype(int)

                    if c_morte and c_dias:
                        # Mortalidade
                        d["saidas_tot"] = len(df_rd)
                        d["obitos_tot"] = df_rd[df_rd[c_morte] == 1].shape[0]
                        # Ocupação Geral (Soma Total RD)
                        d["dias_geral"] = df_rd[c_dias].sum()

                        # TMP from RD
                        if c_clinica:
                            df_rd['clinica_int'] = pd.to_numeric(df_rd[c_clinica], errors='coerce').fillna(0).astype(int)
                            print("Unique clinica_int:", df_rd['clinica_int'].unique())
                            df_med_rd = df_rd[df_rd['clinica_int'] == 1]  # Clinica Medica
                            d["dias_med"] = df_med_rd[c_dias].sum()
                            d["saidas_med"] = len(df_med_rd)
                            df_cir_rd = df_rd[df_rd['clinica_int'] == 2]  # Cirurgia
                            d["dias_cir"] = df_cir_rd[c_dias].sum()
                            d["saidas_cir"] = len(df_cir_rd)

                    # TMP movido para SP

    except Exception: pass

    # ------------------- 2. SP (UTIs - ATOPROF + Filtro Idade) -------------------
    try:
        files_sp = sih_db.get_files(group="SP", uf=uf, year=year, month=month)
        if files_sp:
            df_sp = sih_db.download(files_sp).to_dataframe()
            df_sp.columns = [c.upper().strip() for c in df_sp.columns]
            
            cnes_s = encontrar_coluna(df_sp, ["CNES", "SP_CNES"])
            if cnes_s:
                df_sp['CNES_INT'] = pd.to_numeric(df_sp[cnes_s], errors='coerce').fillna(0).astype(int)
                df_sp = df_sp[df_sp['CNES_INT'] == int(cnes_filter)].copy()
                
                if not df_sp.empty:
                    c_ato_uti = next((c for c in df_sp.columns if "ATOPROF" in c), "SP_ATOPROF")
                    c_ato_tmp = next((c for c in df_sp.columns if "PROCREA" in c), "SP_PROCREA")
                    c_qtd = next((c for c in df_sp.columns if "QT_" in c), "SP_QTD_ATO")
                    c_val = next((c for c in df_sp.columns if "VAL" in c), "SP_VALATO")
                    c_aih = next((c for c in df_sp.columns if "NAIH" in c), "SP_NAIH")
                    
                    # DEBUG raw
                    d['raw_unique_atoprof'] = sorted(df_sp[c_ato_uti].unique()) if c_ato_uti in df_sp.columns else []
                    d['raw_unique_procrea'] = sorted(df_sp[c_ato_tmp].unique()) if c_ato_tmp in df_sp.columns else []
                    
                    # Limpeza
                    df_sp[c_ato_uti] = df_sp[c_ato_uti].astype(str).str.strip().str.replace(r"[^0-9]", "", regex=True)
                    df_sp[c_ato_tmp] = df_sp[c_ato_tmp].astype(str).str.strip().str.replace(r"[^0-9]", "", regex=True)
                    df_sp[c_qtd] = pd.to_numeric(df_sp[c_qtd], errors='coerce').fillna(0).astype(int)
                    df_sp[c_val] = pd.to_numeric(df_sp[c_val], errors='coerce').fillna(0.0)
                    
                    # Filtro 1: Apenas Linhas Pagas e com QT_ > 0
                    df_ok = df_sp[(df_sp[c_val] > 0) & (df_sp[c_qtd] > 0)].copy()
                    
                    if not df_ok.empty:
                        print("Colunas SP:", df_sp.columns.tolist())
                        print("Valores únicos ATOPROF (UTI):", df_sp[c_ato_uti].unique()[:20])
                        print("Valores únicos PROCREA (TMP):", df_sp[c_ato_tmp].unique()[:20])
                        
                        # Filtros por código ATOPROF específico da UTI
                        mask_a = (df_ok[c_ato_uti] == '0802010083')
                        mask_n = (df_ok[c_ato_uti] == '0802010121')
                        mask_p = (df_ok[c_ato_uti] == '0802010156')

                        # DEBUG SP
                        d['sp_columns'] = list(df_sp.columns)
                        d['unique_atoprof'] = sorted(df_sp[c_ato_uti].unique()) if c_ato_uti in df_sp.columns else []
                        d['unique_procrea'] = sorted(df_sp[c_ato_tmp].unique()) if c_ato_tmp in df_sp.columns else []

                        # Agrupa por AIH e ato, soma QT_ (dias de UTI)
                        df_a = df_ok[mask_a].groupby([c_aih, c_ato_uti])[c_qtd].sum().reset_index()
                        d["dias_a"] = df_a[c_qtd].sum()
                        
                        df_n = df_ok[mask_n].groupby([c_aih, c_ato_uti])[c_qtd].sum().reset_index()
                        d["dias_n"] = df_n[c_qtd].sum()
                        
                        df_p = df_ok[mask_p].groupby([c_aih, c_ato_uti])[c_qtd].sum().reset_index()
                        d["dias_p"] = df_p[c_qtd].sum()

    except Exception: pass

    d.update({"cap_geral": caps['geral'], "cap_a": caps['uti_a'], "cap_n": caps['uti_n'], "cap_p": caps['uti_p']})
    return d

# ===================== PLOTAGEM =====================
def plot_indicador(ax, df, col_y, media, nota, meta, color_ok, title, is_tmp=False, fixed_ylim=None, is_inf=False):
    x = df["periodo"]
    y = df[col_y].fillna(0)
    colors = ['#7209b7' if (val > 100 and not is_tmp and not is_inf) else color_ok for val in y]
    ax.bar(x, y, color=colors, alpha=0.8, width=0.5)
    unit = "‰" if is_inf else ("d" if is_tmp else "%")
    ax.set_title(f"{title}\nMedia: {media:.2f}{unit} | Nota: {nota}", fontweight='bold', fontsize=10)
    if fixed_ylim: ax.set_ylim(fixed_ylim)
    else: ax.set_ylim(0, 10 if y.max() <= 0 else (105 if (not is_tmp and not is_inf and y.max() <= 100) else y.max() * 1.3))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.axhline(media, color='blue', linestyle='--', label=f'Media ({media:.2f})')
    ax.axhline(meta, color='green', linestyle=':', label='Meta')
    ax.legend(loc='lower right', fontsize='x-small')
    for i, val in enumerate(y):
        txt = f"{val:.2f}" + ("" if is_tmp else ("‰" if is_inf else "%"))
        ax.text(i, val + (ax.get_ylim()[1] * 0.02), txt, ha='center', fontweight='bold', fontsize=8)

def gerar_pdf_buffer(df, cnes, t):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        FIG_SIZE = (18, 12)
        # P1
        fig1, axs1 = plt.subplots(2, 2, figsize=FIG_SIZE)
        plt.suptitle(f"Indicadores Gerais - CNES {cnes}", fontsize=16, fontweight='bold')
        plot_indicador(axs1[0,0], df, "tx_mort_m", t['tx_mort'], t['p_mort'], 3, '#2a9d8f', "Mortalidade", fixed_ylim=(0,10))
        plot_indicador(axs1[0,1], df, "tx_ocup_m", t['tx_ocup'], t['p_ocup'], 80, '#2a9d8f', "Ocupacao Geral")
        plot_indicador(axs1[1,0], df, "tmp_med_m", t['tx_med'], t['p_med'], 8, '#2a9d8f', "TMP Clinica Medica", is_tmp=True)
        plot_indicador(axs1[1,1], df, "tmp_cir_m", t['tx_cir'], t['p_cir'], 5, '#2a9d8f', "TMP Clinica Cirurgica", is_tmp=True)
        pdf.savefig(fig1); plt.close()
        # P2
        fig2, axs2 = plt.subplots(2, 2, figsize=FIG_SIZE)
        plt.suptitle(f"Indicadores UTI - CNES {cnes}", fontsize=16, fontweight='bold')
        plot_indicador(axs2[0,0], df, "tx_a_m", t['tx_a'], t['p_a'], 85, '#2a9d8f', "UTI Adulto")
        plot_indicador(axs2[0,1], df, "tx_n_m", t['tx_n'], t['p_n'], 85, '#2a9d8f', "UTI Neo")
        plot_indicador(axs2[1,0], df, "tx_p_m", t['tx_p'], t['p_p'], 85, '#2a9d8f', "UTI Ped")
        plot_indicador(axs2[1,1], df, "dens_inf_m", t['tx_inf'], t['p_inf'], 2.0, '#2a9d8f', "Infeccao CVC", is_inf=True)
        pdf.savefig(fig2); plt.close()
        # P3
        fig3 = plt.figure(figsize=FIG_SIZE)
        plt.axis('off')
        plt.title("RESUMO EXECUTIVO", fontsize=20, fontweight='bold')
        dt = [
            ["INDICADOR", "DADOS (Soma)", "RESULTADO", "NOTA"],
            ["Mortalidade", f"{t['s_obitos']}/{t['s_saidas']}", f"{t['tx_mort']:.2f}%", f"{t['p_mort']}/7"],
            ["Ocup. Geral", f"{t['s_dias_g']}/{t['s_cap_g']}", f"{t['tx_ocup']:.2f}%", f"{t['p_ocup']}/7"],
            ["TMP Medica", f"{t['s_dias_m']}/{t['s_sai_m']}", f"{t['tx_med']:.2f} d", f"{t['p_med']}/6"],
            ["TMP Cirurgica", f"{t['s_dias_c']}/{t['s_sai_c']}", f"{t['tx_cir']:.2f} d", f"{t['p_cir']}/6"],
            ["UTI Adulto", f"{t['s_dias_a']}/{t['s_cap_a']}", f"{t['tx_a']:.2f}%", f"{t['p_a']}/6"],
            ["UTI Neo", f"{t['s_dias_n']}/{t['s_cap_n']}", f"{t['tx_n']:.2f}%", f"{t['p_n']}/6"],
            ["UTI Ped", f"{t['s_dias_p']}/{t['s_cap_p']}", f"{t['tx_p']:.2f}%", f"{t['p_p']}/6"],
            ["Infeccao", f"{t['s_casos']}/{t['s_cvc']}", f"{t['tx_inf']:.2f}‰", f"{t['p_inf']}/6"],
            ["TOTAL", "", "", f"{t['total_pts']:.2f}/50"]
        ]
        tab = plt.table(cellText=dt, colLabels=None, loc='center', bbox=[0.05, 0.2, 0.9, 0.6])
        tab.auto_set_font_size(False); tab.set_fontsize(12); tab.scale(1, 2)
        pdf.savefig(fig3); plt.close()
    buffer.seek(0); return buffer

# ===================== UI =====================
with st.sidebar:
    st.header("Configurações")
    cnes_input = st.text_input("CNES", "2142376")
    uf_input = st.selectbox("Estado", ["MG"], index=0)
    ano_sel = st.selectbox("Ano", [2023, 2024, 2025], index=2)
    quad_sel = st.selectbox("Quadrimestre", ["Q1 (Jan-Abr)", "Q2 (Mai-Ago)", "Q3 (Set-Dez)"], index=1)
    meses_sel = get_meses_quadrimestre(quad_sel)
    
    st.markdown("### Indicador 8 (Manual)")
    manual = []
    with st.expander("Dados CCIH", expanded=False):
        for m in meses_sel:
            c = st.number_input(f"Casos {m:02d}", 0, 100, 0, key=f"c_{m}")
            d = st.number_input(f"Dias CVC {m:02d}", 0, 5000, 0, key=f"d_{m}")
            manual.append((ano_sel, m, c, d))
    if st.button("Limpar Cache"): st.cache_data.clear()

if st.button("Processar Dados", type="primary"):
    bar = st.progress(0); status = st.empty()
    res = []
    for i, m in enumerate(meses_sel):
        status.text(f"Processando {m:02d}/{ano_sel}...")
        res.append(processar_mes_unico(ano_sel, m, uf_input, cnes_input))
        bar.progress((i+1)/len(meses_sel))
    
    status.text("Calculando indicadores..."); bar.progress(100)
    
    df = pd.DataFrame(res)
    df["periodo"] = df["mes"].apply(lambda x: f"{x:02d}")
    
    man = pd.DataFrame(manual, columns=["ano", "mes", "casos", "cvc"])
    df = pd.merge(df, man, on="mes", how="left")
    
    # Indicadores Mensais
    df["tx_mort_m"] = (df["obitos_tot"]/df["saidas_tot"]*100).fillna(0)
    df["tx_ocup_m"] = (df["dias_geral"]/df["cap_geral"]*100).clip(upper=100).fillna(0)
    df["tmp_med_m"] = (df["dias_med"]/df["saidas_med"]).fillna(0)
    df["tmp_cir_m"] = (df["dias_cir"]/df["saidas_cir"]).fillna(0)
    df["tx_a_m"] = (df["dias_a"]/df["cap_a"]*100).fillna(0)
    df["tx_n_m"] = (df["dias_n"]/df["cap_n"]*100).fillna(0)
    df["tx_p_m"] = (df["dias_p"]/df["cap_p"]*100).fillna(0)
    df["dens_inf_m"] = (df["casos"]/df["cvc"]*1000).fillna(0)

    # Totais
    t = {}
    t['s_obitos'] = df['obitos_tot'].sum(); t['s_saidas'] = df['saidas_tot'].sum()
    t['s_dias_g'] = df['dias_geral'].sum(); t['s_cap_g'] = df['cap_geral'].sum()
    t['s_dias_m'] = df['dias_med'].sum(); t['s_sai_m'] = df['saidas_med'].sum()
    t['s_dias_c'] = df['dias_cir'].sum(); t['s_sai_c'] = df['saidas_cir'].sum()
    t['s_dias_a'] = df['dias_a'].sum(); t['s_cap_a'] = df['cap_a'].sum()
    t['s_dias_n'] = df['dias_n'].sum(); t['s_cap_n'] = df['cap_n'].sum()
    t['s_dias_p'] = df['dias_p'].sum(); t['s_cap_p'] = df['cap_p'].sum()
    t['s_casos'] = df['casos'].sum(); t['s_cvc'] = df['cvc'].sum()

    # Taxas Finais
    t['tx_mort'] = (t['s_obitos']/t['s_saidas']*100) if t['s_saidas'] else 0
    t['tx_ocup'] = (t['s_dias_g']/t['s_cap_g']*100) if t['s_cap_g'] else 0
    t['tx_med'] = (t['s_dias_m']/t['s_sai_m']) if t['s_sai_m'] else 0
    t['tx_cir'] = (t['s_dias_c']/t['s_sai_c']) if t['s_sai_c'] else 0
    t['tx_a'] = (t['s_dias_a']/t['s_cap_a']*100) if t['s_cap_a'] else 0
    t['tx_n'] = (t['s_dias_n']/t['s_cap_n']*100) if t['s_cap_n'] else 0
    t['tx_p'] = (t['s_dias_p']/t['s_cap_p']*100) if t['s_cap_p'] else 0
    t['tx_inf'] = (t['s_casos']/t['s_cvc']*1000) if t['s_cvc'] else 0

    # Pontos
    t['p_mort'] = pontuacao_mortalidade(t['tx_mort'])
    t['p_ocup'] = pontuacao_ocupacao(t['tx_ocup'])
    t['p_med'] = pontuacao_tmp_medica(t['tx_med'])
    t['p_cir'] = pontuacao_tmp_cirurgica(t['tx_cir'])
    t['p_a'] = pontuacao_uti(t['tx_a'])
    t['p_n'] = pontuacao_uti(t['tx_n'])
    t['p_p'] = pontuacao_uti(t['tx_p'])
    t['p_inf'] = pontuacao_infeccao(t['tx_inf'])
    t['total_pts'] = t['p_mort'] + t['p_ocup'] + t['p_med'] + t['p_cir'] + t['p_a'] + t['p_n'] + t['p_p'] + t['p_inf']

    status.success("Concluído!")
    
    # LOG VISUAL
    with st.expander("🕵️ LOG DE AUDITORIA", expanded=True):
        st.markdown(f"""
        **1. OCUPAÇÃO GERAL (GLOBAL)**
        - Dias Totais: `{t['s_dias_g']}` (Meta: 9424) | Capacidade: `{t['s_cap_g']}`
        - Resultado: `{t['tx_ocup']:.2f}%` (Meta: 86.08%)
        
        **2. UTIs (SP - ATOPROF Códigos SIGTAP)**
        - Adulto: `{t['s_dias_a']}` dias (Meta: 2013) -> `{t['tx_a']:.2f}%`
        - Neo: `{t['s_dias_n']}` dias (Meta: 943) -> `{t['tx_n']:.2f}%`
        - Ped: `{t['s_dias_p']}` dias (Meta: 114) -> `{t['tx_p']:.2f}%`
        
        **3. TMP CLÍNICAS**
        - Médica: `{t['s_dias_m']}` / `{t['s_sai_m']}` = `{t['tx_med']:.2f}` (Meta: 8.44)
        - Cirúrgica: `{t['s_dias_c']}` / `{t['s_sai_c']}` = `{t['tx_cir']:.2f}` (Meta: 4.20)
        """)
        st.dataframe(df[["periodo", "dias_geral", "dias_a", "dias_n", "dias_p"]])

        st.markdown("**DEBUG SP INSPECTION:**")
        if not df.empty:
            st.write("Colunas SP:", df['sp_columns'].iloc[0])
            all_raw_atoprof = set()
            for u in df['raw_unique_atoprof']:
                all_raw_atoprof.update(u)
            st.write("Raw Unique ATOPROF (UTI, first 50):", sorted(list(all_raw_atoprof))[:50])
            all_raw_procrea = set()
            for u in df['raw_unique_procrea']:
                all_raw_procrea.update(u)
            st.write("Raw Unique PROCREA (TMP, first 50):", sorted(list(all_raw_procrea))[:50])
            all_unique_atoprof = set()
            for u in df['unique_atoprof']:
                all_unique_atoprof.update(u)
            st.write("Unique ATOPROF (UTI, all months):", sorted(all_unique_atoprof))
            all_unique_procrea = set()
            for u in df['unique_procrea']:
                all_unique_procrea.update(u)
            st.write("Unique PROCREA (TMP, all months):", sorted(all_unique_procrea))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pontuação", f"{t['total_pts']} / 50")
    c2.metric("Mortalidade", f"{t['tx_mort']:.2f}%", f"Nota {t['p_mort']}")
    c3.metric("Ocup. Geral", f"{t['tx_ocup']:.2f}%", f"Nota {t['p_ocup']}")
    c4.metric("Infecção", f"{t['tx_inf']:.2f}‰", f"Nota {t['p_inf']}")
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("UTI Adu", f"{t['tx_a']:.2f}%", f"Nota {t['p_a']}")
    c6.metric("UTI Neo", f"{t['tx_n']:.2f}%", f"Nota {t['p_n']}")
    c7.metric("UTI Ped", f"{t['tx_p']:.2f}%", f"Nota {t['p_p']}")
    c8.metric("TMP Med", f"{t['tx_med']:.2f}d", f"Nota {t['p_med']}")

    tab1, tab2, tab3 = st.tabs(["Graficos", "Tabela", "PDF"])
    with tab1:
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tx_mort_m", t['tx_mort'], t['p_mort'], 3, '#2a9d8f', "Mortalidade"); c1.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tx_ocup_m", t['tx_ocup'], t['p_ocup'], 80, '#2a9d8f', "Ocupacao Geral"); c2.pyplot(fig)
        c3, c4 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tmp_med_m", t['tx_med'], t['p_med'], 8, '#2a9d8f', "TMP Med", True); c3.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tmp_cir_m", t['tx_cir'], t['p_cir'], 5, '#2a9d8f', "TMP Cir", True); c4.pyplot(fig)
        st.markdown("### UTIs")
        c5, c6 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tx_a_m", t['tx_a'], t['p_a'], 85, '#2a9d8f', "UTI Adulto"); c5.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tx_n_m", t['tx_n'], t['p_n'], 85, '#2a9d8f', "UTI Neo"); c6.pyplot(fig)
        c7, c8 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tx_p_m", t['tx_p'], t['p_p'], 85, '#2a9d8f', "UTI Ped"); c7.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "dens_inf_m", t['tx_inf'], t['p_inf'], 2.0, '#2a9d8f', "Infeccao", False, None, True); c8.pyplot(fig)
    
    with tab2: st.dataframe(df)
    with tab3:
        st.download_button("Download PDF", gerar_pdf_buffer(df, cnes_input, t), "relatorio.pdf", "application/pdf")