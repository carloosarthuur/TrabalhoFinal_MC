import pandas as pd
import json
import pulp
import time

def resolver_nra_pli(caminho_json, caminho_nurses, caminho_rooms, nome_instancia='i01', time_limit=300, verbose=False):
    # CARREGAMENTO DOS DADOS E PESOS
    with open(caminho_json, 'r') as f:
        instance_info = json.load(f)
    
    # Extração dos pesos das penalidades do JSON
    peso_habilidade = instance_info['weights']['S2_room_nurse_skill']
    peso_trabalho = instance_info['weights']['S4_nurse_excessive_workload']
    
    # Leitura dos arquivos CSV com dados de enfermeiros e quartos
    df_nurses = pd.read_csv(caminho_nurses)
    df_rooms = pd.read_csv(caminho_rooms)
    
    # Dicionários para acesso rápido a habilidades, capacidades e requisitos
    enfermeiros = df_nurses['nurse_id'].unique()
    turnos = df_nurses['global_shift'].unique()
    
    habilidade_enf = df_nurses.drop_duplicates('nurse_id').set_index('nurse_id')['skill_level'].to_dict()
    cap_maxima_enf = df_nurses.set_index(['nurse_id', 'global_shift'])['max_load'].to_dict()
    disponibilidade = set(zip(df_nurses['nurse_id'], df_nurses['global_shift']))
    quartos_ocupados = list(zip(df_rooms['room_id'], df_rooms['global_shift']))
    req_habilidade = df_rooms.set_index(['room_id', 'global_shift'])['max_skill_required'].to_dict()
    carga_trabalho = df_rooms.set_index(['room_id', 'global_shift'])['total_room_workload'].to_dict()

    # CRIAÇÃO DO MODELO
    model = pulp.LpProblem(f"NRA_Otimizacao_{nome_instancia}", pulp.LpMinimize)

    # VARIÁVEIS DE DECISÃO
    x = pulp.LpVariable.dicts("x", 
                              ((n, r, s) for n in enfermeiros for (r, s) in quartos_ocupados),
                              cat='Binary')
                              
    d = pulp.LpVariable.dicts("d", 
                              ((n, r, s) for n in enfermeiros for (r, s) in quartos_ocupados),
                              lowBound=0, cat='Integer')

    e = pulp.LpVariable.dicts("e", 
                              ((n, s) for n in enfermeiros for s in turnos),
                              lowBound=0, cat='Integer')

    # FUNÇÃO OBJETIVO
    model += (
        pulp.lpSum(peso_habilidade * d[n, r, s] for n in enfermeiros for (r, s) in quartos_ocupados) +
        pulp.lpSum(peso_trabalho * e[n, s] for n in enfermeiros for s in turnos)
    ), "Minimizar_Penalidades"

    # RESTRIÇÕES
    
    # Cobertura
    for (r, s) in quartos_ocupados:
        model += pulp.lpSum(x[n, r, s] for n in enfermeiros) == 1, f"Cobertura_{r}_{s}"

    # Disponibilidade
    for n in enfermeiros:
        for (r, s) in quartos_ocupados:
            if (n, s) not in disponibilidade:
                model += x[n, r, s] == 0, f"Disponibilidade_{n}_{r}_{s}"

    # Déficit de Habilidade
    for n in enfermeiros:
        for (r, s) in quartos_ocupados:
            if (n, s) in disponibilidade:
                hab_faltante = req_habilidade[(r, s)] - habilidade_enf[n]
                if hab_faltante > 0:
                    model += d[n, r, s] >= hab_faltante * x[n, r, s], f"Deficit_Hab_{n}_{r}_{s}"
            else:
                model += d[n, r, s] == 0

    # Excesso de Trabalho
    for n in enfermeiros:
        for s in turnos:
            if (n, s) in disponibilidade:
                quartos_neste_turno = [r_i for (r_i, s_i) in quartos_ocupados if s_i == s]
                trabalho_alocado = pulp.lpSum(carga_trabalho[(r_i, s)] * x[n, r_i, s] for r_i in quartos_neste_turno)
                capacidade = cap_maxima_enf[(n, s)]
                model += e[n, s] >= trabalho_alocado - capacidade, f"Excesso_Trab_{n}_{s}"
            else:
                model += e[n, s] == 0, f"Zero_Excesso_{n}_{s}"

# RESOLUÇÃO DO MODELO
    tempo_inicio = time.time()
    model.solve(pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit,options=['randomSeed', '42'])) # tambemgarante reprodutibilidade
    tempo_fim = time.time()
    
    tempo_execucao = tempo_fim - tempo_inicio
    status = pulp.LpStatus[model.status]
    
    custo_total = None

    # Se achou uma solução (Optimal ou pelo menos uma Feasible antes do tempo acabar)
    if pulp.value(model.objective) is not None:
        custo_total = pulp.value(model.objective)
        
        if tempo_execucao >= time_limit * 0.95:
            status = "Timeout (Feasible)"
        
        if verbose:
            print(f"\nStatus da Solução: {status}")
            print(f"Tempo de Execução do Solver: {tempo_execucao:.4f} segundos")
            print(f"Custo Total (Função Objetivo): {custo_total}")
            
            # Extração das alocações feitas para exibição
            alocacoes = []
            for n in enfermeiros:
                for (r, s) in quartos_ocupados:
                    if pulp.value(x[n, r, s]) == 1:
                        alocacoes.append({'Turno': s, 'Quarto': r, 'Enfermeiro': n})
            
            # Converte para DataFrame para visualizar melhor as 10 primeiras
            df_alocacoes = pd.DataFrame(alocacoes).sort_values(by=['Turno', 'Quarto'])
            print("\nPrimeiras alocações (Top 10):")
            print(df_alocacoes.head(10).to_string(index=False))
            
    else:
        # Caso o tempo acabe sem nenhuma solução encontrada
        if tempo_execucao >= time_limit * 0.95:
            status = "Timeout (No Solution)"
            
        if verbose:
            print(f"\nStatus da Solução: {status}")
            print("Não foi possível encontrar uma solução factível no tempo limite.")

    return status, tempo_execucao, custo_total

def testar_todas_instancias_pli(lista_instancias, tempo_limite):
    resultados = []
    print("\nIniciando execução em lote do PLI para TODAS as instâncias...")
    
    for instancia in lista_instancias:
        print(f"Processando Instância: {instancia.upper()}... ", end="", flush=True)
        
        # Define os caminhos dos arquivos para a instância atual
        caminho_json = f'ihtc2024-nra/{instancia}/instance_info.json'
        caminho_nurses = f'ihtc2024-nra/{instancia}/nurse_shifts.csv'
        caminho_rooms = f'ihtc2024-nra/{instancia}/occupied_room_shifts.csv'
        
        # Chama a função de resolução
        status, t_exec, c_tot = resolver_nra_pli(
            caminho_json, caminho_nurses, caminho_rooms, 
            nome_instancia=instancia, time_limit=tempo_limite, verbose=False
        )
        
        # Guarda os resultados da instância
        resultados.append([
            instancia, status, round(t_exec, 2), c_tot
        ])
        print(f"Concluído! Status: {status} | Custo: {c_tot} | Tempo: {t_exec:.2f}s")
        

    # Salva a tabela final de resultados
    df = pd.DataFrame(resultados, columns=['Instancia', 'Status', 'Tempo_s', 'Custo_Total'])
    nome_arquivo = 'tabela_resultados_pli.csv'
    df.to_csv(nome_arquivo, index=False)
    
    print(f"\nTestes em lote concluídos! Resultados salvos em '{nome_arquivo}'.\n")
    print(df.to_string(index=False))

opcoes = ['i01', 'i02', 'i03', 'i04', 'i05', 'i06', 'i18']

print("TESTE DE INSTÂNCIAS - PROJETO NRA (PLI)")

print("\nModos de Execução:")
print(" [1] Execução Padrão Manual (Testar apenas 1 instância)")
print(" [2] Grade de Testes (Testar TODAS as instâncias e gerar CSV)")

modo = input("\nEscolha o modo de execução (1 ou 2): ").strip()

limit_input = input("Tempo limite por instância em segundos [300]: ").strip()
time_limit_val = int(limit_input) if limit_input else 300

# Lógica para execução em lote
if modo == '2':
    testar_todas_instancias_pli(opcoes, time_limit_val)

# Lógica para execução de uma única instância escolhida
elif modo == '1':
    print("\nInstâncias disponíveis:")
    for i, opcao in enumerate(opcoes):
        print(f" [{i+1}] - {opcao}")  
        
    escolha = input("\nEscolha a instância (1 a 7) ou o nome (ex: i01): ").strip().lower()

    try:
        instancia_escolhida = opcoes[int(escolha)-1]
    except (ValueError, IndexError):
        instancia_escolhida = escolha if escolha in opcoes else exit("Opção inválida.")

    print(f"\nPreparando ambiente para a instância: {instancia_escolhida.upper()} <<<")

    # Caminhos dos arquivos para a instância individual
    caminho_json = f'ihtc2024-nra/{instancia_escolhida}/instance_info.json'
    caminho_nurses = f'ihtc2024-nra/{instancia_escolhida}/nurse_shifts.csv'
    caminho_rooms = f'ihtc2024-nra/{instancia_escolhida}/occupied_room_shifts.csv'

    # Resolve e exibe detalhes (verbose=True)
    status, t_exec, c_tot = resolver_nra_pli(
        caminho_json, caminho_nurses, caminho_rooms, 
        nome_instancia=instancia_escolhida, 
        time_limit=time_limit_val, 
        verbose=True
    )

    # Salva o resultado individual em CSV
    resultado = [[instancia_escolhida, status, round(t_exec, 2), c_tot]]
    df_res = pd.DataFrame(resultado, columns=['Instancia', 'Status', 'Tempo_s', 'Custo_Total'])
    nome_csv = f'resultado_individual_pli_{instancia_escolhida}.csv'
    df_res.to_csv(nome_csv, index=False)
    
    print(f"\nExecução concluída! Resultado salvo em '{nome_csv}'")
    
else:
    print("Modo inválido. Encerrando o programa.")