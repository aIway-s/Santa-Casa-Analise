import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import calendar
import io
import traceback

# Importações do PySUS com rastreamento de disponibilidade
sih_api = None
import_error_msg = ""

try:
    from pysus import sih as sih_api
except Exception as e1:
    try:
        from pysus.api import sih as sih_api
    except Exception as e2:
        sih_api = None
        import_error_msg = f"Erro ao importar 'pysus.sih': {e1} | 'pysus.api.sih': {e2}"

sih_download = None
parquets_to_dataframe = None
try:
    from pysus.online_data.SIH import download as sih_download
    from pysus.online_data import parquets_to_dataframe
except Exception as e3:
    pass

# ===================== CONFIGURAÇÃO =====================
st.set_page_config(page_title="Indicadores - Santa Casa", layout="wide")
st.title("🏥 Indicadores - Santa Casa de Formiga")
st.markdown("---")

# ===================== PARÂMETROS =====================
CAPACIDADE_FIXA = {'geral': 89, 'uti_a': 17, 'uti_n': 9, 'uti_p': 1}

MAPA_UTI_ESTRITO = {
    '0802010083': 'A',
    '0802010121': 'N',
    '0802010156': 'P'
}

CODIGOS_ESPEC = {'MEDICA': ['03'], 'CIRURGICA': ['01']}
MOTIVOS_NAO_CONTAR_SAIDA = [26, 21, 22]

# ===================== AUXILIARES =====================
def get_meses_quadrimestre(q):
    if q == "Q1 (Jan-Abr)": return [1, 2, 3, 4]
    if q == "Q2 (Mai-Ago)": return [5, 6, 7, 8]
    if q == "Q3 (Set-Dez)": return [9, 10, 11, 12]
    return []

def encontrar_coluna(df, candidatos):
    if df is None or df.empty: return None
    cols_upper = [str(c).upper().strip() for c in df.columns]
    for termo in candidatos:
        for i, col in enumerate(cols_upper):
            if termo == col: return df.columns[i]
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

# ===================== DOWNLOAD DIAGNÓSTICO PYSUS =====================
def baixar_dados_sih_diagnostico(uf, year, month, group="RD"):
    logs = []
    
    # 1. Tentativa via API Moderna (PySUS 2.x+)
    if sih_api is not None:
        grupos_para_testar = [group, group.lower(), group.upper()]
        for grp in grupos_para_testar:
            try:
                logs.append(f"Tentando PySUS sih(state='{uf}', year={year}, month=[{month}], group='{grp}')...")
                res = sih_api(state=uf, year=int(year), month=[int(month)], group=grp, as_dataframe=True)
                if isinstance(res, pd.DataFrame) and not res.empty:
                    logs.append(f"✅ Sucesso via PySUS 2.x API (`group='{grp}'`)! {len(res)} linhas baixadas.")
                    return res, logs
                else:
                    logs.append(f"⚠️ PySUS retornou objeto vazio para `group='{grp}'`.")
            except Exception as e:
                logs.append(f"❌ Erro na tentativa `group='{grp}'`: {type(e).__name__} - {e}")
            
            try:
                res = sih_api(state=uf, year=int(year), month=int(month), group=grp, as_dataframe=True)
                if isinstance(res, pd.DataFrame) and not res.empty:
                    logs.append(f"✅ Sucesso via PySUS 2.x API (mês inteiro)! {len(res)} linhas baixadas.")
                    return res, logs
            except Exception as e:
                logs.append(f"❌ Erro na tentativa mês int `group='{grp}'`: {type(e).__name__} - {e}")
    else:
        logs.append(f"⚠️ PySUS modern API indisponível. Motivo: {import_error_msg}")

    # 2. Tentativa via API Clássica (PySUS 1.x / online_data)
    if sih_download is not None and parquets_to_dataframe is not None:
        try:
            logs.append(f"Tentando PySUS clássico `SIH.download(state='{uf}', year={year}, month={month}, group='{group}')`...")
            arquivos = sih_download(state=uf, year=int(year), month=int(month), group=group)
            if arquivos:
                logs.append(f"Baixados {len(arquivos)} arquivos parquet: {arquivos}")
                df = parquets_to_dataframe(arquivos)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    logs.append(f"✅ Sucesso via PySUS clássico! {len(df)} linhas lidas.")
                    return df, logs
                else:
                    logs.append("⚠️ parquets_to_dataframe retornou um DataFrame vazio.")
            else:
                logs.append("⚠️ SIH.download retornou lista vazia.")
        except Exception as e:
            logs.append(f"❌ Erro no PySUS clássico: {type(e).__name__} - {e}")
    else:
        logs.append("⚠️ PySUS clássico (`online_data.SIH.download`) não está disponível.")

    return pd.DataFrame(), logs

# ===================== PROCESSAMENTO COM CACHE =====================
@st.cache_data(ttl=3600, show_spinner=False)
def processar_mes_unico(ano, month, uf, cnes_filter):
    year = int(ano)
    dias_mes = get_days_in_month(year, month)
    caps = {k: v * dias_mes for k, v in CAPACIDADE_FIXA.items()}
    
    d = {k: 0 for k in ["saidas_tot", "obitos_tot", "dias_geral", "dias_med", "saidas_med",
                        "dias_cir", "saidas_cir", "dias_a", "dias_n", "dias_p"]}
    d["mes"] = month
    cnes_alvo_str = str(cnes_filter).strip().zfill(7)
    cnes_alvo_int = int(str(cnes_filter).strip()) if str(cnes_filter).strip().isdigit() else 0
    
    diag_info = {"month": month, "logs_rd": [], "logs_sp": [], "total_rd_bruto": 0, "total_rd_cnes": 0, "cnes_encontrados_rd": [], "total_sp_bruto": 0, "total_sp_cnes": 0}

    # ------------------- 1. RD (Hospitalizações Reduzidas) -------------------
    df_rd, logs_rd = baixar_dados_sih_diagnostico(uf=uf, year=year, month=month, group="RD")
    diag_info["logs_rd"] = logs_rd

    if df_rd is not None and not df_rd.empty:
        diag_info["total_rd_bruto"] = len(df_rd)
        df_rd.columns = [str(c).upper().strip() for c in df_rd.columns]
        
        cnes_c = encontrar_coluna(df_rd, ["CNES", "CNES_EXEC", "CODUFMUN"])
        if cnes_c:
            col_cnes_vals = df_rd[cnes_c].astype(str).str.replace(r"\D", "", regex=True)
            diag_info["cnes_encontrados_rd"] = list(col_cnes_vals.unique()[:10])
            
            cnes_col_str = col_cnes_vals.str.zfill(7)
            # Filtro flexível (string zfill ou inteiro)
            df_rd_filtered = df_rd[(cnes_col_str == cnes_alvo_str) | (pd.to_numeric(df_rd[cnes_c], errors='coerce') == cnes_alvo_int)].copy()
            diag_info["total_rd_cnes"] = len(df_rd_filtered)
            
            if not df_rd_filtered.empty:
                df_rd = df_rd_filtered
                c_morte = encontrar_coluna(df_rd, ["MORTE", "OBITO"])
                c_dias = encontrar_coluna(df_rd, ["DIAS_PERM", "QT_DIARIAS"])
                c_espec = encontrar_coluna(df_rd, ["ESPEC", "COD_ESPEC"])
                c_motivo = encontrar_coluna(df_rd, ["COBRANCA", "MOT_SAIDA", "COBRA_SAI"])

                if c_morte: df_rd[c_morte] = pd.to_numeric(df_rd[c_morte], errors='coerce').fillna(0).astype(int)
                if c_dias: df_rd[c_dias] = pd.to_numeric(df_rd[c_dias], errors='coerce').fillna(0).astype(int)

                df_rd = df_rd[df_rd[c_dias] >= 0].copy()

                if c_morte and c_dias:
                    d["saidas_tot"] = len(df_rd)
                    d["obitos_tot"] = int((df_rd[c_morte] == 1).sum())
                    d["dias_geral"] = int(df_rd[c_dias].sum())

                if c_espec and c_motivo:
                    df_rd['ESPEC_STR'] = df_rd[c_espec].astype(str).str.split('.').str[0].str.strip().str.zfill(2)
                    df_rd['MOTIVO_INT'] = pd.to_numeric(df_rd[c_motivo], errors='coerce').fillna(0).astype(int)
                    
                    df_med_dias = df_rd[df_rd['ESPEC_STR'].isin(CODIGOS_ESPEC['MEDICA'])]
                    d["dias_med"] = int(df_med_dias[c_dias].sum())
                    df_med_saidas = df_med_dias[~df_med_dias['MOTIVO_INT'].isin(MOTIVOS_NAO_CONTAR_SAIDA)]
                    d["saidas_med"] = len(df_med_saidas)
                    
                    df_cir_dias = df_rd[df_rd['ESPEC_STR'].isin(CODIGOS_ESPEC['CIRURGICA'])]
                    d["dias_cir"] = int(df_cir_dias[c_dias].sum())
                    df_cir_saidas = df_cir_dias[~df_cir_dias['MOTIVO_INT'].isin(MOTIVOS_NAO_CONTAR_SAIDA)]
                    d["saidas_cir"] = len(df_cir_saidas)

    # ------------------- 2. SP (UTIs) -------------------
    df_sp, logs_sp = baixar_dados_sih_diagnostico(uf=uf, year=year, month=month, group="SP")
    diag_info["logs_sp"] = logs_sp

    if df_sp is not None and not df_sp.empty:
        diag_info["total_sp_bruto"] = len(df_sp)
        df_sp.columns = [str(c).upper().strip() for c in df_sp.columns]
        
        cnes_s = encontrar_coluna(df_sp, ["CNES", "SP_CNES"])
        if cnes_s:
            col_cnes_sp = df_sp[cnes_s].astype(str).str.replace(r"\D", "", regex=True)
            cnes_sp_str = col_cnes_sp.str.zfill(7)
            df_sp_filtered = df_sp[(cnes_sp_str == cnes_alvo_str) | (pd.to_numeric(df_sp[cnes_s], errors='coerce') == cnes_alvo_int)].copy()
            diag_info["total_sp_cnes"] = len(df_sp_filtered)
            
            if not df_sp_filtered.empty:
                df_sp = df_sp_filtered
                c_ato = next((c for c in df_sp.columns if "ATOPROF" in c), "SP_ATOPROF")
                c_qtd = next((c for c in df_sp.columns if "QT_" in c), "SP_QTD_ATO")
                c_val = next((c for c in df_sp.columns if "VAL" in c), "SP_VALATO")
                c_idade = next((c for c in df_sp.columns if "IDADE" in c or "NU_IDADE" in c), None)

                df_sp[c_ato] = df_sp[c_ato].astype(str).str.strip().str.replace(r"[^0-9]", "", regex=True)
                df_sp[c_qtd] = pd.to_numeric(df_sp[c_qtd], errors='coerce').fillna(0).astype(int)
                df_sp[c_val] = pd.to_numeric(df_sp[c_val], errors='coerce').fillna(0.0)
                
                if c_idade: df_sp['IDADE_R'] = pd.to_numeric(df_sp[c_idade], errors='coerce').fillna(-1)
                else: df_sp['IDADE_R'] = -1

                df_ok = df_sp[df_sp[c_val] > 0].copy()
                
                if not df_ok.empty:
                    mask_a = (df_ok[c_ato] == '0802010083') & ((df_ok['IDADE_R'] >= 14) | (df_ok['IDADE_R'] == -1))
                    mask_n = (df_ok[c_ato] == '0802010121') & ((df_ok['IDADE_R'] < 1) | (df_ok['IDADE_R'] == -1))
                    mask_p = (df_ok[c_ato] == '0802010156')

                    d["dias_a"] = int(df_ok.loc[mask_a, c_qtd].sum()) if mask_a.any() else 0
                    d["dias_n"] = int(df_ok.loc[mask_n, c_qtd].sum()) if mask_n.any() else 0
                    d["dias_p"] = int(df_ok.loc[mask_p, c_qtd].sum()) if mask_p.any() else 0

    d.update({"cap_geral": caps['geral'], "cap_a": caps['uti_a'], "cap_n": caps['uti_n'], "cap_p": caps['uti_p']})
    return d, diag_info

# ===================== PLOTAGEM E UI =====================
def plot_indicador(ax, df, col_y, media, title, color_ok):
    x = df["periodo"]
    y = df[col_y].fillna(0)
    ax.bar(x, y, color=color_ok, alpha=0.8)
    ax.set_title(f"{title}\nMédia: {media:.2f}", fontweight='bold', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.axhline(media, color='blue', linestyle='--')
    for i, val in enumerate(y):
        ax.text(i, val, f"{val:.2f}", ha='center', fontsize=8)

def gerar_pdf_buffer(df, cnes, t):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        FIG_SIZE = (18, 12)
        fig1, axs1 = plt.subplots(2, 2, figsize=FIG_SIZE)
        plt.suptitle(f"Indicadores Gerais - CNES {cnes}", fontsize=16, fontweight='bold')
        plot_indicador(axs1[0,0], df, "tx_mort_m", t['tx_mort'], "Mortalidade", "#2a9d8f")
        plot_indicador(axs1[0,1], df, "tx_ocup_m", t['tx_ocup'], "Ocupação Geral", "#2a9d8f")
        plot_indicador(axs1[1,0], df, "tmp_med_m", t['tx_med'], "TMP Médica", "#2a9d8f")
        plot_indicador(axs1[1,1], df, "tmp_cir_m", t['tx_cir'], "TMP Cirúrgica", "#2a9d8f")
        pdf.savefig(fig1); plt.close()
        
        fig2, axs2 = plt.subplots(2, 2, figsize=FIG_SIZE)
        plt.suptitle(f"Indicadores UTI - CNES {cnes}", fontsize=16, fontweight='bold')
        plot_indicador(axs2[0,0], df, "tx_a_m", t['tx_a'], "UTI Adulto", "#2a9d8f")
        plot_indicador(axs2[0,1], df, "tx_n_m", t['tx_n'], "UTI Neo", "#2a9d8f")
        plot_indicador(axs2[1,0], df, "tx_p_m", t['tx_p'], "UTI Ped", "#2a9d8f")
        plot_indicador(axs2[1,1], df, "dens_inf_m", t['tx_inf'], "Infecção CVC", "#2a9d8f")
        pdf.savefig(fig2); plt.close()
        
        fig3 = plt.figure(figsize=FIG_SIZE)
        plt.axis('off')
        plt.title("RESUMO EXECUTIVO", fontsize=20, fontweight='bold')
        dt = [
            ["INDICADOR", "DADOS (Soma)", "RESULTADO", "NOTA"],
            ["Mortalidade", f"{t['s_obitos']}/{t['s_saidas']}", f"{t['tx_mort']:.2f}%", f"{t['p_mort']}/7"],
            ["Ocup. Geral", f"{t['s_dias_g']}/{t['s_cap_g']}", f"{t['tx_ocup']:.2f}%", f"{t['p_ocup']}/7"],
            ["TMP Médica", f"{t['s_dias_m']}/{t['s_sai_m']}", f"{t['tx_med']:.2f} d", f"{t['p_med']}/6"],
            ["TMP Cirúrgica", f"{t['s_dias_c']}/{t['s_sai_c']}", f"{t['tx_cir']:.2f} d", f"{t['p_cir']}/6"],
            ["UTI Adulto", f"{t['s_dias_a']}/{t['s_cap_a']}", f"{t['tx_a']:.2f}%", f"{t['p_a']}/6"],
            ["UTI Neo", f"{t['s_dias_n']}/{t['s_cap_n']}", f"{t['tx_n']:.2f}%", f"{t['p_n']}/6"],
            ["UTI Ped", f"{t['s_dias_p']}/{t['s_cap_p']}", f"{t['tx_p']:.2f}%", f"{t['p_p']}/6"],
            ["Infecção", f"{t['s_casos']}/{t['s_cvc']}", f"{t['tx_inf']:.2f}‰", f"{t['p_inf']}/6"],
            ["TOTAL", "", "", f"{t['total_pts']:.2f}/50"]
        ]
        tab = plt.table(cellText=dt, colLabels=None, loc='center', bbox=[0.05, 0.2, 0.9, 0.6])
        tab.auto_set_font_size(False); tab.set_fontsize(12); tab.scale(1, 2)
        pdf.savefig(fig3); plt.close()
    buffer.seek(0); return buffer

# ===================== INTERFACE (SIDEBAR) =====================
with st.sidebar:
    st.header("Configurações")
    cnes_input = st.text_input("CNES", "2142376")
    uf_input = st.selectbox("Estado", ["MG"], index=0)
    ano_sel = st.selectbox("Ano", [2023, 2024, 2025, 2026], index=3)
    quad_sel = st.selectbox("Quadrimestre", ["Q1 (Jan-Abr)", "Q2 (Mai-Ago)", "Q3 (Set-Dez)"], index=1)
    meses_sel = get_meses_quadrimestre(quad_sel)
    
    st.markdown("### Indicador 8 (Manual CCIH)")
    manual = []
    with st.expander("Dados CCIH", expanded=False):
        for m in meses_sel:
            c = st.number_input(f"Casos {m:02d}", 0, 100, 0, key=f"c_{m}")
            d = st.number_input(f"Dias CVC {m:02d}", 0, 5000, 0, key=f"d_{m}")
            manual.append({"mes": m, "casos": c, "cvc": d})
    
    if st.button("Limpar Cache de Dados"): 
        st.cache_data.clear()
        st.success("Cache limpo com sucesso!")

# ===================== PROCESSAMENTO PRINCIPAL =====================
if st.button("Processar Dados", type="primary"):
    bar = st.progress(0); status = st.empty()
    res = []
    todos_diagnosticos = []
    
    for i, m in enumerate(meses_sel):
        status.text(f"Baixando/Processando Mês {m:02d}/{ano_sel} via PySUS...")
        r, diag = processar_mes_unico(ano_sel, m, uf_input, cnes_input)
        res.append(r)
        todos_diagnosticos.append(diag)
        bar.progress((i+1)/len(meses_sel))
    
    status.text("Calculando indicadores..."); bar.progress(100)
    
    df = pd.DataFrame(res)
    df["periodo"] = df["mes"].apply(lambda x: f"{x:02d}")
    
    man = pd.DataFrame(manual)
    df = pd.merge(df, man, on="mes", how="left")
    
    df["tx_mort_m"] = (df["obitos_tot"]/df["saidas_tot"]*100).fillna(0)
    df["tx_ocup_m"] = (df["dias_geral"]/df["cap_geral"]*100).clip(upper=100).fillna(0)
    df["tmp_med_m"] = (df["dias_med"]/df["saidas_med"]).fillna(0)
    df["tmp_cir_m"] = (df["dias_cir"]/df["saidas_cir"]).fillna(0)
    df["tx_a_m"] = (df["dias_a"]/df["cap_a"]*100).fillna(0)
    df["tx_n_m"] = (df["dias_n"]/df["cap_n"]*100).fillna(0)
    df["tx_p_m"] = (df["dias_p"]/df["cap_p"]*100).fillna(0)
    df["dens_inf_m"] = (df["casos"]/df["cvc"]*1000).fillna(0)

    t = {}
    t['s_obitos'] = df['obitos_tot'].sum(); t['s_saidas'] = df['saidas_tot'].sum()
    t['s_dias_g'] = df['dias_geral'].sum(); t['s_cap_g'] = df['cap_geral'].sum()
    t['s_dias_m'] = df['dias_med'].sum(); t['s_sai_m'] = df['saidas_med'].sum()
    t['s_dias_c'] = df['dias_cir'].sum(); t['s_sai_c'] = df['saidas_cir'].sum()
    t['s_dias_a'] = df['dias_a'].sum(); t['s_cap_a'] = df['cap_a'].sum()
    t['s_dias_n'] = df['dias_n'].sum(); t['s_cap_n'] = df['cap_n'].sum()
    t['s_dias_p'] = df['dias_p'].sum(); t['s_cap_p'] = df['cap_p'].sum()
    t['s_casos'] = df['casos'].sum(); t['s_cvc'] = df['cvc'].sum()

    t['tx_mort'] = (t['s_obitos']/t['s_saidas']*100) if t['s_saidas'] else 0
    t['tx_ocup'] = (t['s_dias_g']/t['s_cap_g']*100) if t['s_cap_g'] else 0
    t['tx_med'] = (t['s_dias_m']/t['s_sai_m']) if t['s_sai_m'] else 0
    t['tx_cir'] = (t['s_dias_c']/t['s_sai_c']) if t['s_sai_c'] else 0
    t['tx_a'] = (t['s_dias_a']/t['s_cap_a']*100) if t['s_cap_a'] else 0
    t['tx_n'] = (t['s_dias_n']/t['s_cap_n']*100) if t['s_cap_n'] else 0
    t['tx_p'] = (t['s_dias_p']/t['s_cap_p']*100) if t['s_cap_p'] else 0
    t['tx_inf'] = (t['s_casos']/t['s_cvc']*1000) if t['s_cvc'] else 0

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

    # Exibição dos diagnósticos em caso de dados zerados
    with st.expander("🛠️ Diagnóstico Detalhado de Download (Clique para ver logs de erro)", expanded=True if t['s_saidas'] == 0 else False):
        for diag in todos_diagnosticos:
            st.write(f"### Mês {diag['month']:02d}")
            st.write(f"- **RD Bruto:** {diag['total_rd_bruto']} linhas | **RD Filtro CNES:** {diag['total_rd_cnes']} linhas")
            if diag['cnes_encontrados_rd']:
                st.write(f"  * CNES encontrados na amostra RD: `{diag['cnes_encontrados_rd']}`")
            st.write("  * **Logs de Download (RD):**")
            for log in diag['logs_rd']:
                st.text(f"    {log}")

            st.write(f"- **SP Bruto:** {diag['total_sp_bruto']} linhas | **SP Filtro CNES:** {diag['total_sp_cnes']} linhas")
            st.write("  * **Logs de Download (SP):**")
            for log in diag['logs_sp']:
                st.text(f"    {log}")
            st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pontuação Total", f"{t['total_pts']} / 50")
    c2.metric("Mortalidade", f"{t['tx_mort']:.2f}%", f"Nota {t['p_mort']}")
    c3.metric("Ocupação Geral", f"{t['tx_ocup']:.2f}%", f"Nota {t['p_ocup']}")
    c4.metric("Infecção", f"{t['tx_inf']:.2f}‰", f"Nota {t['p_inf']}")
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("UTI Adulto", f"{t['tx_a']:.2f}%", f"Nota {t['p_a']}")
    c6.metric("UTI Neo", f"{t['tx_n']:.2f}%", f"Nota {t['p_n']}")
    c7.metric("UTI Ped", f"{t['tx_p']:.2f}%", f"Nota {t['p_p']}")
    c8.metric("TMP Médica", f"{t['tx_med']:.2f}d", f"Nota {t['p_med']}")

    tab1, tab2, tab3 = st.tabs(["Gráficos", "Tabela", "PDF"])
    with tab1:
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tx_mort_m", t['tx_mort'], "Mortalidade", "#2a9d8f"); c1.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tx_ocup_m", t['tx_ocup'], "Ocupação Geral", "#2a9d8f"); c2.pyplot(fig)
        c3, c4 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tmp_med_m", t['tx_med'], "TMP Médica", "#2a9d8f"); c3.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tmp_cir_m", t['tx_cir'], "TMP Cirúrgica", "#2a9d8f"); c4.pyplot(fig)
        st.markdown("### UTIs")
        c5, c6 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tx_a_m", t['tx_a'], "UTI Adulto", "#2a9d8f"); c5.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tx_n_m", t['tx_n'], "UTI Neo", "#2a9d8f"); c6.pyplot(fig)
        c7, c8 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "tx_p_m", t['tx_p'], "UTI Ped", "#2a9d8f"); c7.pyplot(fig)
        fig, ax = plt.subplots(figsize=(6,4)); plot_indicador(ax, df, "dens_inf_m", t['tx_inf'], "Infecção CVC", "#2a9d8f"); c8.pyplot(fig)
    
    with tab2: st.dataframe(df)
    with tab3:
        st.download_button("Download PDF", gerar_pdf_buffer(df, cnes_input, t), "relatorio.pdf", "application/pdf")