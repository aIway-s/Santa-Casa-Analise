import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import calendar
import io
import os
import ftplib
import tempfile
import traceback
import gc

# Importações do PySUS com tratamento de avisos do Pylance / VS Code
sih_api = None
import_error_msg = ""

try:
    from pysus.api import sih as sih_api  # type: ignore
except Exception as e1:
    try:
        from pysus import sih as sih_api  # type: ignore
    except Exception as e2:
        sih_api = None
        import_error_msg = f"Erro ao importar 'pysus.api.sih': {e1} | 'pysus': {e2}"

sih_download = None
sih_class = None
parquets_to_dataframe = None

try:
    from pysus.online_data.SIH import download as sih_download  # type: ignore
    from pysus.online_data import parquets_to_dataframe  # type: ignore
except Exception:
    pass

try:
    from pysus.online_data.SIH import SIH as sih_class  # type: ignore
except Exception:
    pass

read_dbc = None
dbc2dbf = None

# Tentar submódulos do PySUS caso pyreaddbc não esteja disponível
modules_to_check = [
    "pysus.utilities.readdbc",
    "pysus.utilities.dbc",
    "pysus.online_data.SIH",
    "pysus.data.dbc",
    "pysus.preprocessing.dbc",
]

for mod_path in modules_to_check:
    try:
        mod = __import__(mod_path, fromlist=["read_dbc", "dbc2dbf"])
        if read_dbc is None and hasattr(mod, "read_dbc"):
            read_dbc = getattr(mod, "read_dbc")
        if dbc2dbf is None and hasattr(mod, "dbc2dbf"):
            dbc2dbf = getattr(mod, "dbc2dbf")
    except Exception:
        pass

def ler_arquivo_dbc(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return None

    temp_dbf = tempfile.mktemp(suffix=".dbf")
    descompactado = False

    # 1. Tentar via datasus-dbc (Motor em Rust: rápido e sem dependências de compilação C)
    try:
        import datasus_dbc
        datasus_dbc.decompress(filepath, temp_dbf)
        if os.path.exists(temp_dbf) and os.path.getsize(temp_dbf) > 0:
            descompactado = True
    except Exception:
        pass

    # 2. Tentar via dbc-to-dbf (Descompactador PKWARE 100% em Python puro)
    if not descompactado:
        try:
            from dbctodbf import DBCDecompress  # type: ignore
            dbc = DBCDecompress()
            dbc.decompressFile(filepath, temp_dbf)
            if os.path.exists(temp_dbf) and os.path.getsize(temp_dbf) > 0:
                descompactado = True
        except Exception:
            pass

    # 3. Tentar via dbc2dbf do PySUS
    if not descompactado and dbc2dbf is not None:
        try:
            dbc2dbf(filepath, temp_dbf)
            if os.path.exists(temp_dbf) and os.path.getsize(temp_dbf) > 0:
                descompactado = True
        except Exception:
            pass

    # 4. Tentar via read_dbc direto
    if not descompactado and read_dbc is not None:
        try:
            df = read_dbc(filepath)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

    # Se descompactou com sucesso para DBF, converter em DataFrame via dbfread
    if descompactado and os.path.exists(temp_dbf):
        try:
            from dbfread import DBF  # type: ignore
            table = DBF(temp_dbf, encoding='iso-8859-1', ignore_missing_memofile=True)
            df = pd.DataFrame(iter(table))
            try:
                os.remove(temp_dbf)
            except Exception:
                pass
            if df is not None and not df.empty:
                return df
        except Exception:
            if os.path.exists(temp_dbf):
                try:
                    os.remove(temp_dbf)
                except Exception:
                    pass

    # 5. Fallback: verificar se é Parquet disfarçado
    try:
        return pd.read_parquet(filepath)
    except Exception:
        pass

    return None

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
        t_up = termo.upper().strip()
        for i, col in enumerate(cols_upper):
            if t_up == col: return df.columns[i]
    for termo in candidatos:
        t_up = termo.upper().strip()
        for i, col in enumerate(cols_upper):
            if t_up in col: return df.columns[i]
    return None

def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

# --- VERIFICADORES DE ESTRUTURA DE TABELA (RD vs SP) ---
def eh_tabela_rd(df):
    if df is None or df.empty: return False
    cols = [str(c).upper().strip() for c in df.columns]
    tem_rd = any(c in cols for c in ['MORTE', 'DIAS_PERM', 'ESPEC', 'N_AIH', 'UF_ZI', 'DIAG_PRINC', 'COBRANCA'])
    tem_sp = any(c in cols for c in ['SP_ATOPROF', 'SP_PROCDIG', 'SP_QTD_ATO', 'SP_VALATO'])
    return tem_rd and not tem_sp

def eh_tabela_sp(df):
    if df is None or df.empty: return False
    cols = [str(c).upper().strip() for c in df.columns]
    return any(c in cols for c in ['SP_ATOPROF', 'SP_PROCDIG', 'SP_QTD_ATO', 'SP_VALATO', 'SP_ATO', 'ATOPROF']) or ('SP_GESTOR' in cols and 'SP_CNES' in cols)

# --- PONTUAÇÃO ---
def pontuacao_mortalidade(taxa): return 7 if taxa <= 3 else (4 if taxa < 6 else (2 if taxa <= 8 else 0))
def pontuacao_ocupacao(taxa): return 7 if taxa >= 80 else (4 if taxa >= 65 else (2 if taxa >= 55 else 0))
def pontuacao_tmp_medica(dias): return 6 if 0 < dias < 8 else (4 if 8 <= dias < 11 else (2 if 11 <= dias < 14 else 0))
def pontuacao_tmp_cirurgica(dias): return 6 if 0 < dias < 5 else (4 if 5 <= dias < 7 else (2 if 7 <= dias < 9 else 0))
def pontuacao_uti(taxa): return 6 if taxa >= 85 else (4 if taxa >= 70 else (2 if taxa >= 60 else 0))
def pontuacao_infeccao(densidade): return 6 if densidade <= 2.0 else (4 if densidade <= 3.0 else (2 if densidade <= 5.0 else 0))

# ===================== OTIMIZADOR DE MEMÓRIA DE DATAFRAMES =====================
COLUNAS_ESSENCIAIS = {
    'CNES', 'CNES_EXEC', 'CODUFMUN', 'SP_CNES', 'MORTE', 'OBITO', 'DIAG_OBITO',
    'DIAS_PERM', 'QT_DIARIAS', 'DIAS', 'ESPEC', 'COD_ESPEC', 'ESPECIAL',
    'COBRANCA', 'MOT_SAIDA', 'COBRA_SAI', 'MOTIV_SAI', 'SP_GESTOR', 'SP_UF',
    'SP_AA', 'SP_MM', 'SP_NAIH', 'SP_PROCREA', 'SP_DTINTER', 'SP_DTSAIDA',
    'SP_NUM_PR', 'SP_TIPO', 'SP_CPFCGC', 'SP_ATOPROF', 'SP_TP_ATO', 'SP_QTD_ATO',
    'SP_PROCDIG', 'SP_COD_ATO', 'SP_ATO', 'ATOPROF', 'PROCDIG', 'COD_ATO',
    'PROCEDIMENTO', 'SP_QT_ATO', 'SP_QTD', 'QT_ATO', 'QTD_ATO', 'QT_PROCDIG',
    'QUANTIDADE', 'SP_VALATO', 'SP_VAL_ATO', 'SP_VALOR', 'VAL_ATO', 'VALOR_ATO',
    'VAL_PROCDIG', 'VAL_TOT', 'SP_IDADE', 'IDADE', 'NU_IDADE', 'IDADE_PAC',
    'UF_ZI', 'ANO_CMPT', 'MES_CMPT', 'CGC_HOSP', 'N_AIH', 'IDENT', 'CEP', 'MUNIC_RES'
}

def otimizar_dataframe_memoria(df):
    if df is None or df.empty:
        return df
    try:
        cols_upper_map = {str(c).upper().strip(): c for c in df.columns}
        manter_cols = [cols_upper_map[k] for k in cols_upper_map if k in COLUNAS_ESSENCIAIS]
        if manter_cols:
            df = df[manter_cols]
    except Exception:
        pass
    gc.collect()
    return df

# ===================== CONVERSOR UNIVERSAL PYSUS =====================
def converter_pysus_para_dataframe(res):
    if res is None:
        return None
    if isinstance(res, pd.DataFrame):
        return otimizar_dataframe_memoria(res) if not res.empty else None
    
    if isinstance(res, (list, tuple)):
        dfs = []
        for item in res:
            sub_df = converter_pysus_para_dataframe(item)
            if sub_df is not None and not sub_df.empty:
                dfs.append(sub_df)
        if dfs:
            return otimizar_dataframe_memoria(pd.concat(dfs, ignore_index=True))
        return None
    
    if isinstance(res, str):
        if os.path.exists(res):
            if res.upper().endswith('.DBC'):
                df = ler_arquivo_dbc(res)
                if df is not None:
                    return otimizar_dataframe_memoria(df)
            try:
                df = pd.read_parquet(res)
                return otimizar_dataframe_memoria(df)
            except Exception:
                pass
            if parquets_to_dataframe is not None:
                try:
                    df = parquets_to_dataframe([res])
                    return otimizar_dataframe_memoria(df)
                except Exception:
                    pass
    
    if hasattr(res, 'to_dataframe'):
        try:
            return otimizar_dataframe_memoria(res.to_dataframe())
        except Exception:
            pass
            
    if hasattr(res, 'load'):
        try:
            return otimizar_dataframe_memoria(res.load())
        except Exception:
            pass

    return None

# ===================== DOWNLOAD DIRETO VIA FTP DATASUS =====================
def download_direto_ftp_datasus(uf, year, month, group):
    yy = str(year)[-2:]
    mm = f"{int(month):02d}"
    
    variacoes_nome = [
        f"{group.upper()}{uf.upper()}{yy}{mm}.DBC",
        f"{group.upper()}{uf.upper()}{yy}{mm}.dbc",
        f"{group.lower()}{uf.lower()}{yy}{mm}.dbc",
    ]
    
    temp_dir = tempfile.gettempdir()
    
    try:
        ftp = ftplib.FTP('ftp.datasus.gov.br', timeout=20)
        ftp.login()
        ftp.cwd('/dissemin/publicos/SIHSUS/200801_/Dados/')
        
        for fname in variacoes_nome:
            local_path = os.path.join(temp_dir, fname)
            try:
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {fname}", f.write)
                if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
                    df = ler_arquivo_dbc(local_path)
                    if df is not None and not df.empty:
                        ftp.quit()
                        return df, f"Baixou e descompactou {fname} com sucesso ({len(df)} linhas)"
                    else:
                        ftp.quit()
                        return None, f"Baixou {fname} do FTP (tamanho ok), mas não conseguiu descompactar com ler_arquivo_dbc"
            except Exception:
                continue
                
        ftp.quit()
        return None, f"Nenhuma variação de nome ({', '.join(variacoes_nome)}) encontrada no FTP"
    except Exception as e:
        return None, f"Erro na conexão FTP: {e}"

# ===================== DOWNLOAD DIAGNÓSTICO PYSUS =====================
def baixar_dados_sih_diagnostico(uf, year, month, group="RD"):
    logs = []
    target_group = group.upper().strip()
    
    def validar_df(df, metodo_nome):
        if df is None or df.empty:
            return None
        
        if target_group == "RD":
            if eh_tabela_rd(df):
                logs.append(f"✅ Sucesso [{metodo_nome}]! Tabela RD validada com {len(df)} linhas.")
                return df
            elif eh_tabela_sp(df):
                logs.append(f"⚠️ [{metodo_nome}] Baixou tabela SP ao invés de RD. Ignorando...")
                return None
            else:
                logs.append(f"ℹ️ [{metodo_nome}] Baixou {len(df)} linhas, mas colunas RD não confirmadas.")
                return df
        else: # SP
            if eh_tabela_sp(df):
                logs.append(f"✅ Sucesso [{metodo_nome}]! Tabela SP validada com {len(df)} linhas.")
                return df
            elif eh_tabela_rd(df):
                logs.append(f"⚠️ [{metodo_nome}] Baixou tabela RD ao invés de SP. Ignorando...")
                return None
            else:
                logs.append(f"ℹ️ [{metodo_nome}] Baixou {len(df)} linhas, mas colunas SP não confirmadas.")
                return df

    # Lista de tentativas
    tentativas = []

    # 1. API sih com argumentos POSICIONAIS e NOMINAIS
    if sih_api is not None:
        tentativas.extend([
            # Argumentos posicionais (soluciona o erro missing positional arguments)
            (sih_api, (uf, int(year), int(month), target_group), {}, f"sih('{uf}', {year}, {month}, '{target_group}')"),
            (sih_api, (uf, int(year), [int(month)], target_group), {}, f"sih('{uf}', {year}, [{month}], '{target_group}')"),
            (sih_api, (uf, [int(year)], [int(month)], target_group), {}, f"sih('{uf}', [{year}], [{month}], '{target_group}')"),
            (sih_api, (uf, int(year), int(month)), {"group": target_group}, f"sih('{uf}', {year}, {month}, group='{target_group}')"),
            (sih_api, (uf, int(year), int(month)), {"dis_type": target_group}, f"sih('{uf}', {year}, {month}, dis_type='{target_group}')"),
            # Argumentos nominais
            (sih_api, (), {"state": uf, "year": int(year), "month": int(month), "group": target_group}, f"sih(state='{uf}', year={year}, month={month}, group='{target_group}')"),
            (sih_api, (), {"state": uf, "year": int(year), "month": [int(month)], "group": target_group}, f"sih(state='{uf}', year={year}, month=[{month}], group='{target_group}')"),
        ])

    # 2. Classe SIH (pysus.online_data.SIH)
    if sih_class is not None:
        inst = sih_class()
        tentativas.extend([
            (inst.download, (uf, int(year), int(month)), {"group": target_group}, f"SIH().download('{uf}', {year}, {month}, group='{target_group}')"),
            (inst.download, (), {"state": uf, "year": int(year), "month": int(month), "group": target_group}, f"SIH().download(state='{uf}', year={year}, month={month}, group='{target_group}')"),
        ])

    # 3. Download Clássico (sih_download)
    if sih_download is not None:
        tentativas.extend([
            (sih_download, (uf, int(year), int(month)), {"group": target_group}, f"sih_download('{uf}', {year}, {month}, group='{target_group}')"),
        ])

    # Executar tentativas registradas
    for fn, args, kwargs, rotulo in tentativas:
        try:
            logs.append(f"Tentando `{rotulo}`...")
            res = fn(*args, **kwargs)
            df = converter_pysus_para_dataframe(res)
            v_df = validar_df(df, rotulo)
            if v_df is not None:
                return v_df, logs
        except Exception as e:
            logs.append(f"⚠️ Erro em `{rotulo}`: {e}")

    # 4. FALLBACK FINAL DIRETO VIA FTP DO DATASUS (ftp.datasus.gov.br)
    try:
        logs.append(f"Tentando Download Direto FTP DATASUS (`{target_group}{uf}{str(year)[-2:]}{int(month):02d}.DBC`)...")
        df_ftp, msg_ftp = download_direto_ftp_datasus(uf, year, month, target_group)
        logs.append(f"ℹ️ [{target_group}] FTP DATASUS: {msg_ftp}")
        if df_ftp is not None and not df_ftp.empty:
            v_df = validar_df(df_ftp, "FTP Direto DATASUS")
            if v_df is not None:
                return v_df, logs
    except Exception as e:
        logs.append(f"❌ Erro no FTP direto: {e}")

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
    d["tem_rd"] = False
    d["tem_sp"] = False
    
    cnes_alvo_str = str(cnes_filter).strip().zfill(7)
    cnes_alvo_int = int(str(cnes_filter).strip()) if str(cnes_filter).strip().isdigit() else 0
    
    diag_info = {"month": month, "logs_rd": [], "logs_sp": [], "total_rd_bruto": 0, "total_rd_cnes": 0, "cnes_encontrados_rd": [], "total_sp_bruto": 0, "total_sp_cnes": 0, "colunas_rd": [], "colunas_sp": []}

    # ------------------- 1. RD (Hospitalizações Reduzidas) -------------------
    df_rd, logs_rd = baixar_dados_sih_diagnostico(uf=uf, year=year, month=month, group="RD")
    diag_info["logs_rd"] = logs_rd

    if df_rd is not None and not df_rd.empty:
        diag_info["total_rd_bruto"] = len(df_rd)
        df_rd.columns = [str(c).upper().strip() for c in df_rd.columns]
        diag_info["colunas_rd"] = list(df_rd.columns)[:15]
        
        cnes_c = encontrar_coluna(df_rd, ["CNES", "CNES_EXEC", "CODUFMUN", "SP_CNES"])
        if cnes_c and cnes_c in df_rd.columns:
            s_cnes_rd = df_rd[cnes_c].astype(str).str.strip()
            diag_info["cnes_encontrados_rd"] = list(s_cnes_rd.unique()[:10])
            mask_rd = (s_cnes_rd == cnes_alvo_str) | (s_cnes_rd.str.zfill(7) == cnes_alvo_str)
            df_rd_filtered = df_rd[mask_rd].copy()
            diag_info["total_rd_cnes"] = len(df_rd_filtered)
            del df_rd, s_cnes_rd, mask_rd
            gc.collect()
            
            if not df_rd_filtered.empty:
                d["tem_rd"] = True
                df_rd = df_rd_filtered
                c_morte = encontrar_coluna(df_rd, ["MORTE", "OBITO", "DIAG_OBITO"])
                c_dias = encontrar_coluna(df_rd, ["DIAS_PERM", "QT_DIARIAS", "DIAS"])
                c_espec = encontrar_coluna(df_rd, ["ESPEC", "COD_ESPEC", "ESPECIAL"])
                c_motivo = encontrar_coluna(df_rd, ["COBRANCA", "MOT_SAIDA", "COBRA_SAI", "MOTIV_SAI"])

                if c_morte and c_morte in df_rd.columns: 
                    df_rd[c_morte] = pd.to_numeric(df_rd[c_morte], errors='coerce').fillna(0).astype(int)
                if c_dias and c_dias in df_rd.columns: 
                    df_rd[c_dias] = pd.to_numeric(df_rd[c_dias], errors='coerce').fillna(0).astype(int)
                    df_rd = df_rd[df_rd[c_dias] >= 0].copy()

                if c_morte and c_dias and c_morte in df_rd.columns and c_dias in df_rd.columns:
                    d["saidas_tot"] = len(df_rd)
                    d["obitos_tot"] = int((df_rd[c_morte] == 1).sum())
                    d["dias_geral"] = int(df_rd[c_dias].sum())

                if c_espec and c_motivo and c_espec in df_rd.columns and c_motivo in df_rd.columns:
                    df_rd['ESPEC_STR'] = df_rd[c_espec].astype(str).str.split('.').str[0].str.strip().str.zfill(2)
                    df_rd['MOTIVO_INT'] = pd.to_numeric(df_rd[c_motivo], errors='coerce').fillna(0).astype(int)
                    
                    df_med_dias = df_rd[df_rd['ESPEC_STR'].isin(CODIGOS_ESPEC['MEDICA'])]
                    if c_dias and c_dias in df_med_dias.columns:
                        d["dias_med"] = int(df_med_dias[c_dias].sum())
                    df_med_saidas = df_med_dias[~df_med_dias['MOTIVO_INT'].isin(MOTIVOS_NAO_CONTAR_SAIDA)]
                    d["saidas_med"] = len(df_med_saidas)
                    
                    df_cir_dias = df_rd[df_rd['ESPEC_STR'].isin(CODIGOS_ESPEC['CIRURGICA'])]
                    if c_dias and c_dias in df_cir_dias.columns:
                        d["dias_cir"] = int(df_cir_dias[c_dias].sum())
                    df_cir_saidas = df_cir_dias[~df_cir_dias['MOTIVO_INT'].isin(MOTIVOS_NAO_CONTAR_SAIDA)]
                    d["saidas_cir"] = len(df_cir_saidas)

    # ------------------- 2. SP (UTIs) -------------------
    df_sp, logs_sp = baixar_dados_sih_diagnostico(uf=uf, year=year, month=month, group="SP")
    diag_info["logs_sp"] = logs_sp

    if df_sp is not None and not df_sp.empty:
        diag_info["total_sp_bruto"] = len(df_sp)
        df_sp.columns = [str(c).upper().strip() for c in df_sp.columns]
        diag_info["colunas_sp"] = list(df_sp.columns)[:15]
        
        cnes_s = encontrar_coluna(df_sp, ["CNES", "SP_CNES", "CNES_EXEC", "CODUFMUN"])
        if cnes_s and cnes_s in df_sp.columns:
            s_cnes_sp = df_sp[cnes_s].astype(str).str.strip()
            mask_sp = (s_cnes_sp == cnes_alvo_str) | (s_cnes_sp.str.zfill(7) == cnes_alvo_str)
            df_sp_filtered = df_sp[mask_sp].copy()
            diag_info["total_sp_cnes"] = len(df_sp_filtered)
            del df_sp, s_cnes_sp, mask_sp
            gc.collect()
            
            if not df_sp_filtered.empty:
                d["tem_sp"] = True
                df_sp = df_sp_filtered
                
                c_ato = encontrar_coluna(df_sp, ["SP_ATOPROF", "SP_PROCDIG", "SP_COD_ATO", "SP_ATO", "ATOPROF", "PROCDIG", "COD_ATO", "PROCEDIMENTO"])
                c_qtd = encontrar_coluna(df_sp, ["SP_QTD_ATO", "SP_QT_ATO", "SP_QTD", "QT_ATO", "QTD_ATO", "QT_PROCDIG", "QUANTIDADE"])
                c_val = encontrar_coluna(df_sp, ["SP_VALATO", "SP_VAL_ATO", "SP_VALOR", "VAL_ATO", "VALOR_ATO", "VAL_PROCDIG", "VAL_TOT"])
                c_idade = encontrar_coluna(df_sp, ["SP_IDADE", "IDADE", "NU_IDADE", "IDADE_PAC", "IDADE_PACIENTE"])

                if c_ato and c_qtd and c_val and c_ato in df_sp.columns and c_qtd in df_sp.columns and c_val in df_sp.columns:
                    df_sp[c_ato] = df_sp[c_ato].astype(str).str.strip().str.replace(r"[^0-9]", "", regex=True)
                    df_sp[c_qtd] = pd.to_numeric(df_sp[c_qtd], errors='coerce').fillna(0).astype(int)
                    df_sp[c_val] = pd.to_numeric(df_sp[c_val], errors='coerce').fillna(0.0)
                    
                    if c_idade and c_idade in df_sp.columns: 
                        df_sp['IDADE_R'] = pd.to_numeric(df_sp[c_idade], errors='coerce').fillna(-1)
                    else: 
                        df_sp['IDADE_R'] = -1

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
    for i, row in df.iterrows():
        val = row[col_y]
        lbl = f"{val:.2f}" if (pd.notnull(val) and val > 0) else "S/ Dados"
        ax.text(i, max(val, 0.5), lbl, ha='center', fontsize=8)

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
    ano_sel = st.selectbox("Ano", [2023, 2024, 2025, 2026], index=2) # 2025
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
        status.text(f"Baixando/Processando Mês {m:02d}/{ano_sel} via PySUS & DATASUS FTP...")
        r, diag = processar_mes_unico(ano_sel, m, uf_input, cnes_input)
        res.append(r)
        todos_diagnosticos.append(diag)
        bar.progress((i+1)/len(meses_sel))
    
    status.text("Calculando indicadores..."); bar.progress(100)
    
    df = pd.DataFrame(res)
    df["periodo"] = df["mes"].apply(lambda x: f"{x:02d}")
    
    man = pd.DataFrame(manual)
    df = pd.merge(df, man, on="mes", how="left")
    
    # Cálculo mensal individual (limitado a 100% max no display de UTI)
    df["tx_mort_m"] = (df["obitos_tot"]/df["saidas_tot"]*100).where(df["tem_rd"], 0)
    df["tx_ocup_m"] = (df["dias_geral"]/df["cap_geral"]*100).where(df["tem_rd"], 0).clip(upper=100)
    df["tmp_med_m"] = (df["dias_med"]/df["saidas_med"]).where(df["tem_rd"], 0)
    df["tmp_cir_m"] = (df["dias_cir"]/df["saidas_cir"]).where(df["tem_rd"], 0)
    df["tx_a_m"] = (df["dias_a"]/df["cap_a"]*100).where(df["tem_sp"], 0).clip(upper=100)
    df["tx_n_m"] = (df["dias_n"]/df["cap_n"]*100).where(df["tem_sp"], 0).clip(upper=100)
    df["tx_p_m"] = (df["dias_p"]/df["cap_p"]*100).where(df["tem_sp"], 0).clip(upper=100)
    df["dens_inf_m"] = (df["casos"]/df["cvc"]*1000).fillna(0)

    # Filtrar apenas meses que possuem dados publicados para compor a média consolidada
    df_rd_val = df[df["tem_rd"]]
    df_sp_val = df[df["tem_sp"]]

    t = {}
    # Totais RD (Apenas meses com RD disponível)
    t['s_obitos'] = df_rd_val['obitos_tot'].sum() if not df_rd_val.empty else 0
    t['s_saidas'] = df_rd_val['saidas_tot'].sum() if not df_rd_val.empty else 0
    t['s_dias_g'] = df_rd_val['dias_geral'].sum() if not df_rd_val.empty else 0
    t['s_cap_g'] = df_rd_val['cap_geral'].sum() if not df_rd_val.empty else 0
    t['s_dias_m'] = df_rd_val['dias_med'].sum() if not df_rd_val.empty else 0
    t['s_sai_m'] = df_rd_val['saidas_med'].sum() if not df_rd_val.empty else 0
    t['s_dias_c'] = df_rd_val['dias_cir'].sum() if not df_rd_val.empty else 0
    t['s_sai_c'] = df_rd_val['saidas_cir'].sum() if not df_rd_val.empty else 0
    
    # Totais SP (Apenas meses com SP disponível)
    t['s_dias_a'] = df_sp_val['dias_a'].sum() if not df_sp_val.empty else 0
    t['s_cap_a'] = df_sp_val['cap_a'].sum() if not df_sp_val.empty else 0
    t['s_dias_n'] = df_sp_val['dias_n'].sum() if not df_sp_val.empty else 0
    t['s_cap_n'] = df_sp_val['cap_n'].sum() if not df_sp_val.empty else 0
    t['s_dias_p'] = df_sp_val['dias_p'].sum() if not df_sp_val.empty else 0
    t['s_cap_p'] = df_sp_val['cap_p'].sum() if not df_sp_val.empty else 0
    
    # CCIH Manual
    t['s_casos'] = df['casos'].sum(); t['s_cvc'] = df['cvc'].sum()

    # Taxas Consolidadas
    t['tx_mort'] = (t['s_obitos']/t['s_saidas']*100) if t['s_saidas'] else 0
    t['tx_ocup'] = min((t['s_dias_g']/t['s_cap_g']*100), 100.0) if t['s_cap_g'] else 0
    t['tx_med'] = (t['s_dias_m']/t['s_sai_m']) if t['s_sai_m'] else 0
    t['tx_cir'] = (t['s_dias_c']/t['s_sai_c']) if t['s_sai_c'] else 0
    t['tx_a'] = min((t['s_dias_a']/t['s_cap_a']*100), 100.0) if t['s_cap_a'] else 0
    t['tx_n'] = min((t['s_dias_n']/t['s_cap_n']*100), 100.0) if t['s_cap_n'] else 0
    t['tx_p'] = min((t['s_dias_p']/t['s_cap_p']*100), 100.0) if t['s_cap_p'] else 0
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

    # Exibição dos diagnósticos
    with st.expander("🛠️ Diagnóstico Detalhado de Download", expanded=False):
        for diag in todos_diagnosticos:
            st.write(f"### Mês {diag['month']:02d}")
            st.write(f"- **RD Bruto:** {diag['total_rd_bruto']} linhas | **RD Filtro CNES:** {diag['total_rd_cnes']} linhas")
            if diag['colunas_rd']:
                st.write(f"  * Primeiras colunas RD: `{diag['colunas_rd']}`")
            if diag['cnes_encontrados_rd']:
                st.write(f"  * CNES encontrados na amostra RD: `{diag['cnes_encontrados_rd']}`")
            st.write("  * **Logs de Download (RD):**")
            for log in diag['logs_rd']:
                st.text(f"    {log}")

            st.write(f"- **SP Bruto:** {diag['total_sp_bruto']} linhas | **SP Filtro CNES:** {diag['total_sp_cnes']} linhas")
            if diag['colunas_sp']:
                st.write(f"  * Primeiras colunas SP: `{diag['colunas_sp']}`")
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