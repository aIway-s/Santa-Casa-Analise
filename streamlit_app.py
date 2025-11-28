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

# ===================== CONFIGURAÇÃO INICIAL =====================
st.set_page_config(
    page_title="Gestao Hospitalar",
    page_icon=":hospital:", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(":hospital: Santa Casa de Formiga - Analise dos Indicadores")
st.markdown("---")

# ===================== CONSTANTES =====================
CODIGOS_REMOVE_GERAL = ['10', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '85', '86', '95']
CODIGOS_DENOM_ADULTO = ['10', '74', '75', '76'] 
CODIGOS_DENOM_NEO = ['81', '82', '83', '91', '92']
CODIGOS_DENOM_PED = ['77', '78', '79', '95']

# Capacidades Padrão (Fallback APENAS se o arquivo não existir mesmo)
CAPACIDADE_PADRAO = {'geral': 100, 'uti_a': 31, 'uti_n': 9, 'uti_p': 1}

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

# ===================== PROCESSAMENTO SEQUENCIAL (SEGURO) =====================
@st.cache_data(show_spinner=False)
def processar_dados(ano, meses, uf, cnes_filter):
    # Instancia clientes uma única vez
    cnes_db = CNES().load()
    sih_db = SIH().load()
    
    dict_cap_geral = {}
    dict_cap_uti_a = {}
    dict_cap_uti_n = {}
    dict_cap_uti_p = {}
    
    stats_list = []
    
    # Loop SEQUENCIAL para evitar travar o FTP
    for month in meses:
        year = ano
        dias_mes = get_days_in_month(year, month)
        
        # --- 1. CNES (CAPACIDADE) ---
        l_geral = CAPACIDADE_PADRAO['geral']
        l_a = CAPACIDADE_PADRAO['uti_a']
        l_n = CAPACIDADE_PADRAO['uti_n']
        l_p = CAPACIDADE_PADRAO['uti_p']
        
        try:
            # Tenta baixar CNES
            files = cnes_db.get_files(group="LT", uf=uf, year=year, month=month)
            if files:
                df = cnes_db.download(files).to_dataframe()
                cols = [c for c in df.columns if "CNES" in c.upper()]
                if cols:
                    cnes_col = cols[0]
                    df[cnes_col] = df[cnes_col].astype(str).str.strip().str.zfill(7)
                    df_hosp = df[df[cnes_col] == str(cnes_filter)].copy()
                    
                    if not df_hosp.empty and "CODLEITO" in df_hosp.columns:
                        df_hosp["CODLEITO"] = df_hosp["CODLEITO"].astype(str).str.strip()
                        col_qt = "QT_EXIST" if "QT_EXIST" in df_hosp.columns else "QT_SUS"
                        df_hosp[col_qt] = pd.to_numeric(df_hosp[col_qt], errors='coerce').fillna(0)

                        # Capacidades Reais
                        val_geral = df_hosp[~df_hosp["CODLEITO"].isin(CODIGOS_REMOVE_GERAL)][col_qt].sum()
                        if val_geral > 0: l_geral = val_geral
                        
                        val_a = df_hosp[df_hosp["CODLEITO"].isin(CODIGOS_DENOM_ADULTO)][col_qt].sum()
                        if val_a > 0: l_a = val_a
                        
                        val_n = df_hosp[df_hosp["CODLEITO"].isin(CODIGOS_DENOM_NEO)][col_qt].sum()
                        if val_n > 0: l_n = val_n
                        
                        val_p = df_hosp[df_hosp["CODLEITO"].isin(CODIGOS_DENOM_PED)][col_qt].sum()
                        if val_p > 0: l_p = val_p

        except Exception as e:
            logger.warning(f"Falha ao baixar CNES {month}/{year}. Usando padrão. Erro: {e}")

        # Salva capacidade em dias
        dict_cap_geral[f"{month}/{year}"] = l_geral * dias_mes
        dict_cap_uti_a[f"{month}/{year}"] = l_a * dias_mes
        dict_cap_uti_n[f"{month}/{year}"] = l_n * dias_mes
        dict_cap_uti_p[f"{month}/{year}"] = l_p * dias_mes

        # --- 2. SIH (PRODUÇÃO) ---
        # Inicializa variáveis com zero
        total_saidas = 0; obitos_inst = 0
        dias_gerais_reais = 0
        dias_med = 0; saidas_med = 0
        dias_cir = 0; saidas_cir = 0
        dias_uti_a = 0; dias_uti_n = 0; dias_uti_p = 0

        try:
            files = sih_db.get_files(group="RD", uf=uf, year=year, month=month)
            if files:
                df = sih_db.download(files).to_dataframe()
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
            logger.error(f"Erro SIH {month}/{year}: {e}")

        # Consolidar dados do mês
        stats_list.append({
            "periodo": f"{str(month).zfill(2)}/{str(year)[-2:]}",
            "mes_int": month, # Importante para o merge
            "saidas_tot": total_saidas, "obitos_tot": obitos_inst,
            "dias_geral": dias_gerais_reais, "cap_geral": dict_cap_geral.get(f"{month}/{year}", 1),
            "dias_med": dias_med, "saidas_med": saidas_med,
            "dias_cir": dias_cir, "saidas_cir": saidas_cir,
            "dias_a": dias_uti_a, "cap_a": dict_cap_uti_a.get(f"{month}/{year}", 1),
            "dias_n": dias_uti_n, "cap_n": dict_cap_uti_n.get(f"{month}/{year}", 1),
            "dias_p": dias_uti_p, "cap_p": dict_cap_uti_p.get(f"{month}/{year}", 1)
        })

    return pd.DataFrame(stats_list)

# ===================== PLOTAGEM =====================
def plot_indicador(ax, df, col_y, media, nota, meta, color_ok, title, ylabel="Taxa (%)", is_tmp=False, fixed_ylim=None, is_inf=False):
    x = df["periodo"]
    y = df[col_y].fillna(0)
    
    colors = []
    for val in y:
        if val > 100 and not is_tmp and not is_inf: colors.append('#7209b7') # Roxo
        else: colors.append(color_ok)
    
    ax.bar(x, y, color=colors, alpha=0.8, width=0.5)
    
    unit = "‰" if is_inf else ("d" if is_tmp else "%")
    title_fmt = f"{title}\nMedia: {media:.2f}{unit} | Nota: {nota:.2f}"
    ax.set_title(title_fmt, fontweight='bold', fontsize=10)
    
    max_val = max(y.max() if not y.empty else 0, media)
    if not np.isfinite(max_val): max_val = 0
    
    if fixed_ylim:
        ax.set_ylim(fixed_ylim)
        limit = fixed_ylim[1]
    else:
        limit = 10 if max_val <= 0 else (105 if (not is_tmp and not is_inf and max_val <= 100) else max_val * 1.35)
        ax.set_ylim(0, limit)
        
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.axhline(media, color='blue', linestyle='--', label=f'Media ({media:.2f})')
    
    if is_inf: lbl_meta = f"Meta (<={meta}‰)"
    elif is_tmp: lbl_meta = f"Meta (<{meta})"
    else: lbl_meta = f"Meta (>={meta}%)" if meta > 10 else f"Meta (<={meta:.2f}%)"
    
    ax.axhline(meta, color='green', linestyle=':', label=lbl_meta)
    
    if not is_tmp and not is_inf and max_val > 100:
        ax.axhline(100, color='#7209b7', linestyle='--', label='Capac. (100%)')

    ax.legend(loc='lower right', fontsize='x-small')
    
    for i, val in enumerate(y):
        txt = f"{val:.2f}"
        if not is_tmp: txt += "%" if not is_inf else "‰"
        
        y_pos = val + (limit * 0.02)
        ax.text(i, y_pos, txt, ha='center', fontweight='bold', fontsize=9)
        
        if not is_tmp and not is_inf and val > 100:
            y_pos_risk = val + (limit * 0.12)
            ax.text(i, y_pos_risk, "LEITOS EXTRAS", ha='center', color='#7209b7', fontweight='bold', fontsize=8)

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
        plot_indicador(axs1[0,1], df_stats, "tx_ocup_m", totais['tx_ocup'], totais['p_ocup'], 80, '#2a9d8f', "2. Ocupacao Geral (Sem UTI)")
        plot_indicador(axs1[1,0], df_stats, "tmp_med_m", totais['tx_med'], totais['p_med'], 8, '#2a9d8f', "3. TMP Clinica Medica", ylabel="Dias", is_tmp=True)
        plot_indicador(axs1[1,1], df_stats, "tmp_cir_m", totais['tx_cir'], totais['p_cir'], 5, '#2a9d8f', "4. TMP Clinica Cirurgica", ylabel="Dias", is_tmp=True)
        pdf.savefig(fig1); plt.close()

        # PAG 2 - UTI
        fig2, axs2 = plt.subplots(2, 2, figsize=FIG_SIZE)
        plt.suptitle(f"Painel de Indicadores (UTI) - CNES {cnes_filter}", fontsize=16, fontweight='bold')
        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9)
        
        plot_indicador(axs2[0,0], df_stats, "tx_a_m", totais['tx_a'], totais['p_a'], 85, '#2a9d8f', "5. UTI Adulto")
        plot_indicador(axs2[0,1], df_stats, "tx_n_m", totais['tx_n'], totais['p_n'], 85, '#2a9d8f', "6. UTI Neonatal")
        plot_indicador(axs2[1,0], df_stats, "tx_p_m", totais['tx_p'], totais['p_p'], 85, '#2a9d8f', "7. UTI Pediatrica")
        plot_indicador(axs2[1,1], df_stats, "dens_inf_m", totais['tx_inf'], totais['p_inf'], 2.0, '#2a9d8f', "8. Infeccao CVC Adulto", is_inf=True)
        pdf.savefig(fig2); plt.close()

        # PAG 3 - RESUMO
        fig3 = plt.figure(figsize=FIG_SIZE)
        plt.axis('off')
        plt.title("RESUMO EXECUTIVO - DADOS E PONTUACAO", fontsize=20, fontweight='bold')
        
        t = totais
        data_table = [
            ["INDICADOR", "FORMULA (Agregada)", "DADOS BRUTOS (Soma)", "RESULTADO", "NOTA (Max)"],
            ["1. Mort. Inst.", "Obitos / Saidas", f"{t['s_obitos']} / {t['s_saidas']}", f"{t['tx_mort']:.2f}%", f"{t['p_mort']:.2f} / 7"],
            ["2. Ocup. Geral", "Dias / Leitos-Dia", f"{t['s_dias_g']} / {t['s_cap_g']}", f"{t['tx_ocup']:.2f}%", f"{t['p_ocup']:.2f} / 7"],
            ["3. TMP Medica", "Dias / Saidas", f"{t['s_dias_m']} / {t['s_sai_m']}", f"{t['tx_med']:.2f} d", f"{t['p_med']:.2f} / 6"],
            ["4. TMP Cirurgica", "Dias / Saidas", f"{t['s_dias_c']} / {t['s_sai_c']}", f"{t['tx_cir']:.2f} d", f"{t['p_cir']:.2f} / 6"],
            ["5. UTI Adulto", "Dias / Leitos-Dia", f"{t['s_dias_a']} / {t['s_cap_a']}", f"{t['tx_a']:.2f}%", f"{t['p_a']:.2f} / 6"],
            ["6. UTI Neo", "Dias / Leitos-Dia", f"{t['s_dias_n']} / {t['s_cap_n']}", f"{t['tx_n']:.2f}%", f"{t['p_n']:.2f} / 6"],
            ["7. UTI Ped", "Dias / Leitos-Dia", f"{t['s_dias_p']} / {t['s_cap_p']}", f"{t['tx_p']:.2f}%", f"{t['p_p']:.2f} / 6"],
            ["8. Infeccao", "Casos / Dias-CVC * 1000", f"{t['s_casos']} / {t['s_cvc']}", f"{t['tx_inf']:.2f}‰", f"{t['p_inf']:.2f} / 6"]
        ]
        
        data_table.append(["TOTAL GERAL", "", "", "", f"{t['total_pts']:.2f} / 50"])

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
    st.header("Configuracoes")
    cnes_input = st.text_input("Codigo CNES", "2142376")
    uf_input = st.selectbox("Estado", ["MG"], index=0)
    
    ano_sel = st.selectbox("Ano", [2023, 2024, 2025], index=2)
    quad_sel = st.selectbox("Quadrimestre", ["Q1 (Jan-Abr)", "Q2 (Mai-Ago)", "Q3 (Set-Dez)"], index=1)
    meses_sel = get_meses_quadrimestre(quad_sel)
    
    st.markdown("### Indicador 8 (Manual)")
    manual_input = []
    with st.expander("Dados CCIH (IPCSL/CVC)", expanded=False):
        for m in meses_sel:
            st.markdown(f"**{m:02d}/{ano_sel}**")
            c1, c2 = st.columns(2)
            c = c1.number_input(f"Casos ({m})", 0, 100, 0, key=f"c_{m}")
            d = c2.number_input(f"Dias CVC ({m})", 0, 5000, 0, key=f"d_{m}")
            manual_input.append((ano_sel, m, c, d))

if st.button("Processar Dados", type="primary"):
    # Alterado o texto para ser sincero com o usuário
    with st.spinner("Baixando dados do DATASUS... (Pode levar 1-2 min)"):
        df_final = processar_dados(ano_sel, meses_sel, uf_input, cnes_input)
        
        # Merge Manual (Garante tipos)
        df_manual = pd.DataFrame(manual_input, columns=["ano", "mes", "casos_ipcs", "dias_cvc"])
        df_final["mes_int"] = df_final["mes_int"].astype(int)
        df_manual["mes"] = df_manual["mes"].astype(int)
        df_manual["mes_int"] = df_manual["mes"]
        
        df_final = pd.merge(df_final, df_manual, on="mes_int", how="left")
        
        # Colunas para plot
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
        
        totais['total_pts'] = totais['p_mort'] + totais['p_ocup'] + totais['p_med'] + totais['p_cir'] + totais['p_a'] + totais['p_n'] + totais['p_p'] + totais['p_inf']

        # --- EXIBIÇÃO ---
        st.success("Calculo concluido com Sucesso!")
        
        # Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pontuacao Total", f"{totais['total_pts']:.2f} / 50")
        col2.metric("Mortalidade", f"{totais['tx_mort']:.2f}%", f"Nota {totais['p_mort']}")
        col3.metric("Ocup. Geral", f"{totais['tx_ocup']:.2f}%", f"Nota {totais['p_ocup']}")
        col4.metric("Infeccao CVC", f"{totais['tx_inf']:.2f}‰", f"Nota {totais['p_inf']}")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("UTI Adulto", f"{totais['tx_a']:.2f}%", f"Nota {totais['p_a']}")
        col6.metric("UTI Neo", f"{totais['tx_n']:.2f}%", f"Nota {totais['p_n']}")
        col7.metric("UTI Ped", f"{totais['tx_p']:.2f}%", f"Nota {totais['p_p']}")
        col8.metric("TMP Medica", f"{totais['tx_med']:.2f}d", f"Nota {totais['p_med']}")

        # Tabs
        tab1, tab2, tab3 = st.tabs([":bar_chart: Graficos", ":clipboard: Tabela", ":page_facing_up: PDF"])

        with tab1:
            c1, c2 = st.columns(2)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tx_mort_m", totais['tx_mort'], totais['p_mort'], 3, '#2a9d8f', "Mortalidade", fixed_ylim=(0,10))
            c1.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tx_ocup_m", totais['tx_ocup'], totais['p_ocup'], 80, '#2a9d8f', "Ocupacao Geral")
            c2.pyplot(fig)
            
            c3, c4 = st.columns(2)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tmp_med_m", totais['tx_med'], totais['p_med'], 8, '#2a9d8f', "TMP Medica", is_tmp=True)
            c3.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tmp_cir_m", totais['tx_cir'], totais['p_cir'], 5, '#2a9d8f', "TMP Cirurgica", is_tmp=True)
            c4.pyplot(fig)
            
            st.markdown("### Terapia Intensiva")
            c5, c6 = st.columns(2)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tx_a_m", totais['tx_a'], totais['p_a'], 85, '#2a9d8f', "UTI Adulto")
            c5.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tx_n_m", totais['tx_n'], totais['p_n'], 85, '#2a9d8f', "UTI Neonatal")
            c6.pyplot(fig)
            
            c7, c8 = st.columns(2)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "tx_p_m", totais['tx_p'], totais['p_p'], 85, '#2a9d8f', "UTI Pediatrica")
            c7.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_indicador(ax, df_final, "dens_inf_m", totais['tx_inf'], totais['p_inf'], 2.0, '#2a9d8f', "Infeccao CVC", is_inf=True)
            c8.pyplot(fig)

        with tab2:
            st.dataframe(df_final)

        with tab3:
            st.write("Clique abaixo para baixar o relatorio.")
            pdf_bytes = gerar_pdf_buffer(df_final, cnes_input, totais)
            st.download_button(
                label="Baixar PDF",
                data=pdf_bytes,
                file_name=f"relatorio_{quad_sel}_{ano_sel}.pdf",
                mime="application/pdf"
            )