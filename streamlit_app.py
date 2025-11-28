import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import calendar
import io
import numpy as np
from loguru import logger
from pysus.ftp.databases.sih import SIH
from pysus.ftp.databases.cnes import CNES

# ===================== CONFIGURAÇÃO DA PÁGINA =====================
st.set_page_config(
    page_title="Santa Casa Analytics",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título Principal
st.title(":hospital: Santa Casa de Formiga - Análise dos Indicadores")
st.markdown("---")

# ===================== CONSTANTES DE LEITOS =====================
CODIGOS_REMOVE_GERAL = ['10', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '85', '86', '95']
CODIGOS_DENOM_ADULTO = ['10', '74', '75', '76'] 
CODIGOS_DENOM_NEO = ['81', '82', '83', '91', '92']
CODIGOS_DENOM_PED = ['77', '78', '79', '95']

# ===================== FUNÇÕES AUXILIARES =====================
def get_meses_quadrimestre(q):
    if q == "Q1 (Jan-Abr)": return [1, 2, 3, 4]
    if q == "Q2 (Mai-Ago)": return [5, 6, 7, 8]
    if q == "Q3 (Set-Dez)": return [9, 10, 11, 12]
    return []

def encontrar_coluna(df, candidatos):
    for col in candidatos:
        if col in df.columns: return col
    return None

def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

# --- PONTUAÇÃO ---
def pontuacao_mortalidade(taxa):
    if taxa <= 3: return 7
    elif 3 < taxa < 6: return 4
    elif 6 <= taxa <= 8: return 2
    else: return 0

def pontuacao_ocupacao(taxa):
    if taxa >= 80: return 7
    elif 65 <= taxa < 80: return 4
    elif 55 <= taxa < 65: return 2
    else: return 0

def pontuacao_tmp_medica(dias):
    if dias == 0: return 0
    if dias < 8: return 6
    elif 8 <= dias < 11: return 4
    elif 11 <= dias < 14: return 2
    else: return 0

def pontuacao_tmp_cirurgica(dias):
    if dias == 0: return 0
    if dias < 5: return 6
    elif 5 <= dias < 7: return 4
    elif 7 <= dias < 9: return 2
    else: return 0

def pontuacao_uti(taxa):
    if taxa >= 85: return 6
    elif 70 <= taxa < 85: return 4
    elif 60 <= taxa < 70: return 2
    else: return 0

def pontuacao_infeccao(densidade):
    if densidade <= 2.0: return 6
    elif 2.0 < densidade <= 3.0: return 4
    elif 3.0 < densidade <= 5.0: return 2
    else: return 0

# ===================== PROCESSAMENTO (COM CACHE) =====================
@st.cache_data(show_spinner=False)
def processar_dados(ano, meses, uf, cnes_filter):
    cnes_db = CNES().load()
    sih = SIH().load()
    
    dict_cap_geral = {}
    dict_cap_uti_a = {}
    dict_cap_uti_n = {}
    dict_cap_uti_p = {}
    
    stats_list = []
    
    # --- FASE 1: CNES ---
    for month in meses:
        year = ano
        dias_mes = get_days_in_month(year, month)
        try:
            files = cnes_db.get_files(group="LT", uf=uf, year=year, month=month)
            if files:
                df = cnes_db.download(files).to_dataframe()
                cols = [c for c in df.columns if "CNES" in c.upper()]
                cnes_col = cols[0] if cols else "CNES"
                df[cnes_col] = df[cnes_col].astype(str).str.strip().str.zfill(7)
                df_hosp = df[df[cnes_col] == str(cnes_filter)].copy()
                
                if "CODLEITO" in df_hosp.columns:
                    df_hosp["CODLEITO"] = df_hosp["CODLEITO"].astype(str).str.strip()
                    col_qt = "QT_EXIST" if "QT_EXIST" in df_hosp.columns else "QT_SUS"
                    df_hosp[col_qt] = pd.to_numeric(df_hosp[col_qt], errors='coerce').fillna(0)

                    # Capacidades
                    l_geral = df_hosp[~df_hosp["CODLEITO"].isin(CODIGOS_REMOVE_GERAL)][col_qt].sum()
                    if l_geral == 0: l_geral = 100
                    dict_cap_geral[f"{month}/{year}"] = l_geral * dias_mes
                    
                    dict_cap_uti_a[f"{month}/{year}"] = df_hosp[df_hosp["CODLEITO"].isin(CODIGOS_DENOM_ADULTO)][col_qt].sum() * dias_mes
                    dict_cap_uti_n[f"{month}/{year}"] = df_hosp[df_hosp["CODLEITO"].isin(CODIGOS_DENOM_NEO)][col_qt].sum() * dias_mes
                    dict_cap_uti_p[f"{month}/{year}"] = df_hosp[df_hosp["CODLEITO"].isin(CODIGOS_DENOM_PED)][col_qt].sum() * dias_mes
                else:
                    dict_cap_geral[f"{month}/{year}"] = 120 * dias_mes; dict_cap_uti_a[f"{month}/{year}"] = 0
                    dict_cap_uti_n[f"{month}/{year}"] = 0; dict_cap_uti_p[f"{month}/{year}"] = 0
        except:
            # Fallback
            dict_cap_geral[f"{month}/{year}"] = 120 * dias_mes; dict_cap_uti_a[f"{month}/{year}"] = 0
            dict_cap_uti_n[f"{month}/{year}"] = 0; dict_cap_uti_p[f"{month}/{year}"] = 0

    # --- FASE 2: SIH ---
    for month in meses:
        year = ano
        # Inicialização Segura das Variáveis
        total_saidas = 0; obitos_inst = 0
        dias_gerais_reais = 0
        dias_med = 0; saidas_med = 0
        dias_cir = 0; saidas_cir = 0
        dias_uti_a = 0; dias_uti_n = 0; dias_uti_p = 0

        try:
            files = sih.get_files(group="RD", uf=uf, year=year, month=month)
            if files:
                df = sih.download(files).to_dataframe()
                df.columns = [c.upper().strip() for c in df.columns]
                cnes_c = encontrar_coluna(df, ["CNES", "CNES_EXEC", "M_CNES"])
                
                if cnes_c:
                    df[cnes_c] = df[cnes_c].astype(str).str.strip().str.zfill(7)
                    df_hosp = df[df[cnes_c] == str(cnes_filter)].copy()
                    
                    if not df_hosp.empty:
                        col_morte = encontrar_coluna(df_hosp, ["MORTE", "OBITO"])
                        col_dias  = encontrar_coluna(df_hosp, ["DIAS_PERM", "QT_DIARIAS", "DIAS"])
                        col_proc  = encontrar_coluna(df_hosp, ["PROC_REA", "PROC_REALIZADO"])
                        col_uti_dias = encontrar_coluna(df_hosp, ["UTI_MES_TO", "QT_DIARIAS_UTI"])
                        col_idade = encontrar_coluna(df_hosp, ["IDADE", "ID_PACIENTE"])

                        if col_morte and col_dias:
                            df_hosp["IS_OBITO"] = pd.to_numeric(df_hosp[col_morte], errors='coerce').fillna(0).astype(int)
                            df_hosp["QTD_DIAS"] = pd.to_numeric(df_hosp[col_dias], errors='coerce').fillna(0).astype(int)
                            
                            if col_uti_dias: df_hosp[col_uti_dias] = pd.to_numeric(df_hosp[col_uti_dias], errors='coerce').fillna(0)
                            else: df_hosp["UTI_TEMP"] = 0; col_uti_dias = "UTI_TEMP"

                            if col_proc:
                                df_hosp["PROC_STR"] = df_hosp[col_proc].astype(str).str.strip()
                                df_hosp["GRUPO_PROC"] = df_hosp["PROC_STR"].str.slice(0, 2)
                            else: df_hosp["PROC_STR"] = ""; df_hosp["GRUPO_PROC"] = "99"

                            # 1. Mortalidade
                            total_saidas = len(df_hosp)
                            obitos_inst = df_hosp[(df_hosp["IS_OBITO"] == 1) & (df_hosp["QTD_DIAS"] >= 1)].shape[0]
                            
                            # 2. Ocupação Geral
                            dias_totais = df_hosp["QTD_DIAS"].sum()
                            dias_uti_total = df_hosp[col_uti_dias].sum()
                            dias_gerais_reais = max(0, dias_totais - dias_uti_total)
                            
                            # 3. TMP
                            df_med = df_hosp[df_hosp["GRUPO_PROC"] == '03']
                            dias_med = df_med["QTD_DIAS"].sum(); saidas_med = len(df_med)
                            
                            df_cir = df_hosp[df_hosp["GRUPO_PROC"] == '04']
                            dias_cir = df_cir["QTD_DIAS"].sum(); saidas_cir = len(df_cir)
                            
                            # UTIs
                            mask_uti = df_hosp[col_uti_dias] > 0
                            if mask_uti.any():
                                for _, row in df_hosp[mask_uti].iterrows():
                                    dias = row[col_uti_dias]
                                    proc = row.get("PROC_STR", "")
                                    try: idade = int(row[col_idade]) if col_idade else 999
                                    except: idade = 999
                                    
                                    if proc.startswith("08020101"): dias_uti_n += dias
                                    elif proc.startswith("080201009"): dias_uti_p += dias
                                    elif proc.startswith("080201008"): dias_uti_a += dias
                                    else:
                                        if idade < 1: dias_uti_n += dias
                                        elif 1 <= idade < 14: dias_uti_p += dias
                                        else: dias_uti_a += dias
        except Exception as e:
            logger.error(f"Erro processamento SIH {month}/{year}: {e}")

        # Consolidar
        cap_g = dict_cap_geral.get(f"{month}/{year}", 1)
        cap_a = dict_cap_uti_a.get(f"{month}/{year}", 0)
        cap_n = dict_cap_uti_n.get(f"{month}/{year}", 0)
        cap_p = dict_cap_uti_p.get(f"{month}/{year}", 0)

        stats_list.append({
            "periodo": f"{str(month).zfill(2)}/{str(year)[-2:]}",
            "saidas_tot": total_saidas, "obitos_tot": obitos_inst,
            "dias_geral": dias_gerais_reais, "cap_geral": cap_g,
            "dias_med": dias_med, "saidas_med": saidas_med,
            "dias_cir": dias_cir, "saidas_cir": saidas_cir,
            "dias_a": dias_uti_a, "cap_a": cap_a,
            "dias_n": dias_uti_n, "cap_n": cap_n,
            "dias_p": dias_uti_p, "cap_p": cap_p
        })

    return pd.DataFrame(stats_list)

# ===================== PLOTAGEM =====================
def plot_indicador(ax, df, col_y, media, nota, meta, color_ok, title, ylabel="Taxa (%)", is_tmp=False, fixed_ylim=None, is_inf=False):
    x = df["periodo"]
    y = df[col_y].fillna(0) # Sanitização
    
    colors = []
    for val in y:
        if val > 100 and not is_tmp and not is_inf: colors.append('#7209b7') # Roxo
        else: colors.append(color_ok)
    
    ax.bar(x, y, color=colors, alpha=0.8, width=0.5)
    
    unit = "‰" if is_inf else ("d" if is_tmp else "%")
    title_fmt = f"{title}\nMédia: {media:.2f}{unit} | Nota: {nota:.2f}"
    ax.set_title(title_fmt, fontweight='bold', fontsize=10)
    
    # === CORREÇÃO CRÍTICA DE LIMITES ===
    # Garante que max_val não seja NaN ou Inf
    max_val = max(y.max() if not y.empty else 0, media)
    if not np.isfinite(max_val) or np.isnan(max_val):
        max_val = 0
    
    if fixed_ylim:
        ax.set_ylim(fixed_ylim)
        limit = fixed_ylim[1]
    else:
        # Lógica de limite dinâmico seguro
        if max_val <= 0:
            limit = 10 
        elif not is_tmp and not is_inf and max_val <= 100:
            limit = 105
        else:
            limit = max_val * 1.35 # Margem generosa
        ax.set_ylim(0, limit)
        
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.axhline(media, color='blue', linestyle='--', label=f'Média ({media:.2f})')
    
    # Meta
    if is_inf: lbl_meta = f"Meta (<={meta}‰)"
    elif is_tmp: lbl_meta = f"Meta (<{meta})"
    else: lbl_meta = f"Meta (>={meta}%)" if meta > 10 else f"Meta (<={meta:.2f}%)"
    
    ax.axhline(meta, color='green', linestyle=':', label=lbl_meta)
    
    if not is_tmp and not is_inf and max_val > 100:
        ax.axhline(100, color='#7209b7', linestyle='--', label='Capac. (100%)')

    ax.legend(loc='lower right', fontsize='x-small')
    
    # Labels
    for i, val in enumerate(y):
        txt = f"{val:.2f}"
        if not is_tmp: txt += "%" if not is_inf else "‰"
        
        y_pos = val + (limit * 0.02)
        ax.text(i, y_pos, txt, ha='center', fontweight='bold', fontsize=9)
        
        # Alerta
        if not is_tmp and not is_inf and val > 100:
            y_pos_risk = val + (limit * 0.10)
            ax.text(i, y_pos_risk, "ALTO RISCO", ha='center', color='#7209b7', fontweight='bold', fontsize=8)

# ===================== FUNÇÃO GERAR PDF =====================
def gerar_pdf_buffer(df_stats, cnes_filter, totais):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        FIG_SIZE = (18, 12)
        
        # PAG 1 - GERAL
        fig1, axs1 = plt.subplots(2, 2, figsize=FIG_SIZE)
        plt.suptitle(f"Painel de Indicadores (Geral) - CNES {cnes_filter}", fontsize=16, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9)
        
        plot_indicador(axs1[0,0], df_stats, "tx_mort_m", totais['tx_mort'], totais['p_mort'], 3, '#2a9d8f', "1. Mortalidade Inst.", fixed_ylim=(0,10))
        plot_indicador(axs1[0,1], df_stats, "tx_ocup_m", totais['tx_ocup'], totais['p_ocup'], 80, '#2a9d8f', "2. Ocupação Geral (Sem UTI)")
        plot_indicador(axs1[1,0], df_stats, "tmp_med_m", totais['tx_med'], totais['p_med'], 8, '#2a9d8f', "3. TMP Clínica Médica", ylabel="Dias", is_tmp=True)
        plot_indicador(axs1[1,1], df_stats, "tmp_cir_m", totais['tx_cir'], totais['p_cir'], 5, '#2a9d8f', "4. TMP Clínica Cirúrgica", ylabel="Dias", is_tmp=True)
        pdf.savefig(fig1); plt.close()

        # PAG 2 - UTI
        fig2, axs2 = plt.subplots(2, 2, figsize=FIG_SIZE)
        plt.suptitle(f"Painel de Indicadores (UTI) - CNES {cnes_filter}", fontsize=16, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9)
        
        plot_indicador(axs2[0,0], df_stats, "tx_a_m", totais['tx_a'], totais['p_a'], 85, '#2a9d8f', "5. UTI Adulto")
        plot_indicador(axs2[0,1], df_stats, "tx_n_m", totais['tx_n'], totais['p_n'], 85, '#2a9d8f', "6. UTI Neonatal")
        plot_indicador(axs2[1,0], df_stats, "tx_p_m", totais['tx_p'], totais['p_p'], 85, '#2a9d8f', "7. UTI Pediátrica")
        plot_indicador(axs2[1,1], df_stats, "dens_inf_m", totais['tx_inf'], totais['p_inf'], 2.0, '#2a9d8f', "8. Infecção CVC Adulto", is_inf=True)
        pdf.savefig(fig2); plt.close()

        # PAG 3 - RESUMO
        fig3 = plt.figure(figsize=FIG_SIZE)
        plt.axis('off')
        plt.title("RESUMO EXECUTIVO - DADOS E PONTUAÇÃO", fontsize=20, fontweight='bold')
        
        t = totais
        data_table = [
            ["INDICADOR", "FÓRMULA (Agregada)", "DADOS BRUTOS (Soma)", "RESULTADO", "NOTA"],
            ["1. Mort. Inst.", "Óbitos / Saídas", f"{t['s_obitos']} / {t['s_saidas']}", f"{t['tx_mort']:.2f}%", f"{t['p_mort']:.2f} / 7"],
            ["2. Ocup. Geral", "Dias / Leitos-Dia", f"{t['s_dias_g']} / {t['s_cap_g']}", f"{t['tx_ocup']:.2f}%", f"{t['p_ocup']:.2f} / 7"],
            ["3. TMP Médica", "Dias / Saídas", f"{t['s_dias_m']} / {t['s_sai_m']}", f"{t['tx_med']:.2f} d", f"{t['p_med']:.2f} / 6"],
            ["4. TMP Cirúrgica", "Dias / Saídas", f"{t['s_dias_c']} / {t['s_sai_c']}", f"{t['tx_cir']:.2f} d", f"{t['p_cir']:.2f} / 6"],
            ["5. UTI Adulto", "Dias / Leitos-Dia", f"{t['s_dias_a']} / {t['s_cap_a']}", f"{t['tx_a']:.2f}%", f"{t['p_a']:.2f} / 6"],
            ["6. UTI Neo", "Dias / Leitos-Dia", f"{t['s_dias_n']} / {t['s_cap_n']}", f"{t['tx_n']:.2f}%", f"{t['p_n']:.2f} / 6"],
            ["7. UTI Ped", "Dias / Leitos-Dia", f"{t['s_dias_p']} / {t['s_cap_p']}", f"{t['tx_p']:.2f}%", f"{t['p_p']:.2f} / 6"],
            ["8. Infecção", "Casos / Dias-CVC * 1000", f"{t['s_casos']} / {t['s_cvc']}", f"{t['tx_inf']:.2f}‰", f"{t['p_inf']:.2f} / 6"]
        ]
        
        total_pts = t['p_mort']+t['p_ocup']+t['p_med']+t['p_cir']+t['p_a']+t['p_n']+t['p_p']+t['p_inf']
        data_table.append(["TOTAL GERAL", "", "", "", f"{total_pts:.2f} / 50"])

        table = plt.table(cellText=data_table, colLabels=None, cellLoc='center', loc='center', bbox=[0.05, 0.2, 0.9, 0.6])
        table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 2)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0: cell.set_facecolor('#4a4e69'); cell.set_text_props(weight='bold', color='white')
            elif row == len(data_table)-1: cell.set_facecolor('#e9c46a'); cell.set_text_props(weight='bold')
            elif row % 2 == 0: cell.set_facecolor('#f2f2f2')

        pdf.savefig(fig3); plt.close()
        
    buffer.seek(0)
    return buffer

# ===================== UI PRINCIPAL =====================
with st.sidebar:
    st.header("Configurações")
    cnes_input = st.text_input("Código CNES", "2142376")
    uf_input = st.selectbox("Estado", ["MG"], index=0)
    
    ano_sel = st.selectbox("Ano", [2023, 2024, 2025], index=2)
    quad_sel = st.selectbox("Quadrimestre", ["Q1 (Jan-Abr)", "Q2 (Mai-Ago)", "Q3 (Set-Dez)"], index=1)
    
    meses_sel = get_meses_quadrimestre(quad_sel)
    
    st.markdown("### Indicador 8 (Manual)")
    manual_input = []
    
    with st.expander(":page_with_curl: Inserir Dados CCIH (IPCSL/CVC)", expanded=False):
        for m in meses_sel:
            st.markdown(f"**{m:02d}/{ano_sel}**")
            c1, c2 = st.columns(2)
            c = c1.number_input(f"Casos ({m})", 0, 100, 0, key=f"c_{m}")
            d = c2.number_input(f"Dias CVC ({m})", 0, 5000, 0, key=f"d_{m}")
            manual_input.append((ano_sel, m, c, d))

if st.button("Processar Dados", type="primary"):
    with st.spinner("Baixando dados do DATASUS e calculando... Isso pode demorar um pouco devido ao tamanho dos arquivos.."):
        df_final = processar_dados(ano_sel, meses_sel, uf_input, cnes_input)
        
        # Merge Manual
        df_manual = pd.DataFrame(manual_input, columns=["ano", "mes", "casos_ipcs", "dias_cvc"])
        df_final["mes_int"] = df_final["periodo"].apply(lambda x: int(x.split('/')[0]))
        df_manual["mes_int"] = df_manual["mes"]
        df_final = pd.merge(df_final, df_manual, on="mes_int", how="left")
        
        # Calcular Colunas Finais
        df_final["tx_mort_m"] = df_final["obitos_tot"] / df_final["saidas_tot"] * 100
        df_final["tx_ocup_m"] = (df_final["dias_geral"] / df_final["cap_geral"] * 100).clip(upper=100)
        df_final["tmp_med_m"] = df_final["dias_med"] / df_final["saidas_med"]
        df_final["tmp_cir_m"] = df_final["dias_cir"] / df_final["saidas_cir"]
        df_final["tx_a_m"] = (df_final["dias_a"] / df_final["cap_a"] * 100).fillna(0)
        df_final["tx_n_m"] = (df_final["dias_n"] / df_final["cap_n"] * 100).fillna(0)
        df_final["tx_p_m"] = (df_final["dias_p"] / df_final["cap_p"] * 100).fillna(0)
        df_final["dens_inf_m"] = (df_final["casos_ipcs"] / df_final["dias_cvc"] * 1000).fillna(0)

        # Totais Agregados
        totais = {}
        totais['s_obitos'] = df_final["obitos_tot"].sum(); totais['s_saidas'] = df_final["saidas_tot"].sum()
        totais['tx_mort'] = (totais['s_obitos'] / totais['s_saidas'] * 100) if totais['s_saidas'] > 0 else 0
        totais['p_mort'] = pontuacao_mortalidade(totais['tx_mort'])

        totais['s_dias_g'] = df_final["dias_geral"].sum(); totais['s_cap_g'] = df_final["cap_geral"].sum()
        totais['tx_ocup'] = (totais['s_dias_g'] / totais['s_cap_g'] * 100) if totais['s_cap_g'] > 0 else 0
        totais['p_ocup'] = pontuacao_ocupacao(totais['tx_ocup'])

        totais['s_dias_m'] = df_final["dias_med"].sum(); totais['s_sai_m'] = df_final["saidas_med"].sum()
        totais['tx_med'] = (totais['s_dias_m'] / totais['s_sai_m']) if totais['s_sai_m'] > 0 else 0
        totais['p_med'] = pontuacao_tmp_medica(totais['tx_med'])

        totais['s_dias_c'] = df_final["dias_cir"].sum(); totais['s_sai_c'] = df_final["saidas_cir"].sum()
        totais['tx_cir'] = (totais['s_dias_c'] / totais['s_sai_c']) if totais['s_sai_c'] > 0 else 0
        totais['p_cir'] = pontuacao_tmp_cirurgica(totais['tx_cir'])

        totais['s_dias_a'] = df_final["dias_a"].sum(); totais['s_cap_a'] = df_final["cap_a"].sum()
        totais['tx_a'] = (totais['s_dias_a'] / totais['s_cap_a'] * 100) if totais['s_cap_a'] > 0 else 0
        totais['p_a'] = pontuacao_uti(totais['tx_a'])

        totais['s_dias_n'] = df_final["dias_n"].sum(); totais['s_cap_n'] = df_final["cap_n"].sum()
        totais['tx_n'] = (totais['s_dias_n'] / totais['s_cap_n'] * 100) if totais['s_cap_n'] > 0 else 0
        totais['p_n'] = pontuacao_uti(totais['tx_n'])

        totais['s_dias_p'] = df_final["dias_p"].sum(); totais['s_cap_p'] = df_final["cap_p"].sum()
        totais['tx_p'] = (totais['s_dias_p'] / totais['s_cap_p'] * 100) if totais['s_cap_p'] > 0 else 0
        totais['p_p'] = pontuacao_uti(totais['tx_p'])

        totais['s_casos'] = df_final["casos_ipcs"].sum(); totais['s_cvc'] = df_final["dias_cvc"].sum()
        totais['tx_inf'] = (totais['s_casos'] / totais['s_cvc'] * 1000) if totais['s_cvc'] > 0 else 0
        totais['p_inf'] = pontuacao_infeccao(totais['tx_inf'])
        
        total_pts = totais['p_mort']+totais['p_ocup']+totais['p_med']+totais['p_cir']+totais['p_a']+totais['p_n']+totais['p_p']+totais['p_inf']

        # --- EXIBIÇÃO WEB (Metric Cards & Tabs) ---
        st.success("Dados processados com sucesso!")
        
        # Resumo Executivo (Cards)
        st.markdown("### Resumo do Desempenho")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pontuação Total", f"{total_pts} / 50", delta_color="normal")
        c2.metric("Mortalidade", f"{totais['tx_mort']:.2f}%", f"Nota {totais['p_mort']}")
        c3.metric("Ocup. Geral", f"{totais['tx_ocup']:.2f}%", f"Nota {totais['p_ocup']}")
        c4.metric("Infecção CVC", f"{totais['tx_inf']:.2f}‰", f"Nota {totais['p_inf']}")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("UTI Adulto", f"{totais['tx_a']:.2f}%", f"Nota {totais['p_a']}")
        c6.metric("UTI Neonatal", f"{totais['tx_n']:.2f}%", f"Nota {totais['p_n']}")
        c7.metric("UTI Pediátrica", f"{totais['tx_p']:.2f}%", f"Nota {totais['p_p']}")
        c8.metric("TMP Médica", f"{totais['tx_med']:.2f}d", f"Nota {totais['p_med']}")

        # Tabs
        tab1, tab2, tab3 = st.tabs([":bar_chart: Gráficos Interativos", ":page_with_curl: Tabela de Dados", ":mailbox_with_mail: Exportar PDF"])

        with tab1:
            col1, col2 = st.columns(2)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tx_mort_m", totais['tx_mort'], totais['p_mort'], 3, '#2a9d8f', "Mortalidade", fixed_ylim=(0,10))
            col1.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tx_ocup_m", totais['tx_ocup'], totais['p_ocup'], 80, '#2a9d8f', "Ocupação Geral")
            col2.pyplot(fig)
            
            col3, col4 = st.columns(2)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tmp_med_m", totais['tx_med'], totais['p_med'], 8, '#2a9d8f', "TMP Médica", is_tmp=True)
            col3.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tmp_cir_m", totais['tx_cir'], totais['p_cir'], 5, '#2a9d8f', "TMP Cirúrgica", is_tmp=True)
            col4.pyplot(fig)
            
            st.markdown("### Terapia Intensiva")
            col5, col6 = st.columns(2)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tx_a_m", totais['tx_a'], totais['p_a'], 85, '#2a9d8f', "UTI Adulto")
            col5.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tx_n_m", totais['tx_n'], totais['p_n'], 85, '#2a9d8f', "UTI Neonatal")
            col6.pyplot(fig)
            
            col7, col8 = st.columns(2)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tx_p_m", totais['tx_p'], totais['p_p'], 85, '#2a9d8f', "UTI Pediátrica")
            col7.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "dens_inf_m", totais['tx_inf'], totais['p_inf'], 2.0, '#2a9d8f', "Infecção CVC", is_inf=True)
            col8.pyplot(fig)

        with tab2:
            st.dataframe(df_final)

        with tab3:
            st.write("Clique abaixo para baixar o relatório completo.")
            pdf_bytes = gerar_pdf_buffer(df_final, cnes_input, totais)
            st.download_button(
                label=" :bookmark_tabs: Baixar Relatório PDF",
                data=pdf_bytes,
                file_name=f"relatorio_indicadores_{quad_sel}_{ano_sel}.pdf",
                mime="application/pdf"
            )