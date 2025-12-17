import streamlit as st
import pandas as pd
from pysus.ftp.databases.sih import SIH

st.title("🕵️ Auditoria de TMP - Caça aos Números")

# Configurações
UF = "MG"
CNES_ALVO = "2142376" # Santa Casa Formiga
ANO = 2025
MES = 5 # Maio (Exemplo de um mês do Q2)

if st.button("RASTREAR DADOS BRUTOS (MAIO/25)"):
    sih = SIH().load()
    files = sih.get_files(group="RD", uf=UF, year=ANO, month=MES)
    df = sih.download(files).to_dataframe()
    
    # Limpeza básica
    df.columns = [c.upper().strip() for c in df.columns]
    
    # Filtro CNES
    cnes_col = next((c for c in df.columns if "CNES" in c), "CNES")
    df['CNES_STR'] = df[cnes_col].astype(str).str.strip().str.lstrip('0')
    df = df[df['CNES_STR'] == str(CNES_ALVO).lstrip('0')]
    
    # Colunas de Interesse
    c_dias = next((c for c in df.columns if "DIAS" in c), "DIAS_PERM")
    c_morte = next((c for c in df.columns if "MORTE" in c), "MORTE")
    
    df[c_dias] = pd.to_numeric(df[c_dias], errors='coerce').fillna(0)
    
    st.write(f"### Dados Brutos de {MES}/{ANO} (Total Saídas: {len(df)})")
    
    # 1. AGRUPAMENTO POR COLUNA 'CLINICA' (A clássica do SUS)
    # 1=Cirurgica, 2=Obstetrica, 3=Medica, 4=Cronicos, 5=Pediatria
    if "CLINICA" in df.columns:
        st.subheader("1. Agrupado por Coluna 'CLINICA'")
        res_clin = df.groupby("CLINICA")[c_dias].agg(['count', 'sum']).reset_index()
        res_clin.columns = ['COD_CLINICA', 'SAIDAS (Denom)', 'DIAS (Num)']
        st.dataframe(res_clin)
        st.info("👆 Verifique se o numero 601 (ou proporcional ao mês) aparece aqui na linha 3")

    # 2. AGRUPAMENTO POR 'ESPEC' (Especialidade do Leito)
    # 33=Clinica Geral, 03=Cirurgia Geral, etc.
    c_espec = next((c for c in df.columns if "ESPEC" in c), None)
    if c_espec:
        st.subheader(f"2. Agrupado por Coluna '{c_espec}'")
        res_espec = df.groupby(c_espec)[c_dias].agg(['count', 'sum']).reset_index()
        res_espec.columns = ['COD_ESPEC', 'SAIDAS (Denom)', 'DIAS (Num)']
        st.dataframe(res_espec)
        st.info("👆 Verifique se a soma de alguma dessas linhas bate com seus dados.")

    # 3. AGRUPAMENTO POR GRUPO DE PROCEDIMENTO
    # 03=Clinico, 04=Cirurgico
    c_proc = next((c for c in df.columns if "PROC_REA" in c), None)
    if c_proc:
        df['GRUPO_PROC'] = df[c_proc].astype(str).str.slice(0, 2)
        st.subheader("3. Agrupado por Grupo de Procedimento")
        res_proc = df.groupby("GRUPO_PROC")[c_dias].agg(['count', 'sum']).reset_index()
        res_proc.columns = ['GRUPO', 'SAIDAS (Denom)', 'DIAS (Num)']
        st.dataframe(res_proc)