import pandas as pd
from pysus.ftp.databases.sih import SIH

# Configurações
ano = 2023
mes = 5  # Maio
uf = "MG"
cnes_filter = 2142376

sih_db = SIH().load()

# Baixar SP
files_sp = sih_db.get_files(group="SP", uf=uf, year=ano, month=mes)
if files_sp:
    df_sp = sih_db.download(files_sp).to_dataframe()
    df_sp.columns = [c.upper().strip() for c in df_sp.columns]
    
    # Filtrar CNES
    cnes_s = None
    for col in df_sp.columns:
        if "CNES" in col.upper():
            cnes_s = col
            break
    if cnes_s:
        df_sp['CNES_INT'] = pd.to_numeric(df_sp[cnes_s], errors='coerce').fillna(0).astype(int)
        df_sp = df_sp[df_sp['CNES_INT'] == int(cnes_filter)].copy()
        
        print("Colunas SP:", df_sp.columns.tolist())
        
        # Encontrar coluna ATOPROF
        c_ato = None
        for col in df_sp.columns:
            if "ATOPROF" in col.upper():
                c_ato = col
                break
        if c_ato:
            print(f"Coluna ATOPROF: {c_ato}")
            print("Valores únicos ATOPROF (primeiros 20):", df_sp[c_ato].unique()[:20])
            
            # Verificar códigos específicos
            codigos_interesse = ['060100001', '060200001', '0802010083', '0802010121', '0802010156']
            for cod in codigos_interesse:
                count = (df_sp[c_ato] == cod).sum()
                print(f"Código {cod}: {count} ocorrências")
        else:
            print("Coluna ATOPROF não encontrada")
    else:
        print("Coluna CNES não encontrada")
else:
    print("Nenhum arquivo SP encontrado")