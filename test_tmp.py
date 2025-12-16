import sys
sys.path.append('/workspaces/Santa-Casa-Analise')

from streamlit_app import processar_mes_unico

# Testar para um mês
result = processar_mes_unico(2023, 5, "MG", "2142376")
print("Resultado:", result)