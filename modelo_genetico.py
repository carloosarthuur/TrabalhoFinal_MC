import pandas as pd
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import time

# garantindo reprodutibilidade dos resultados aleatórios
random.seed(42)  
np.random.seed(42)

def resolver_nra_ga(caminho_json, caminho_nurses, caminho_rooms, nome_instancia='i01', populacao_tamanho=100, geracoes=500, R=5, taxa_mutacao=0.05, plotar=True):
    # CARREGAMENTO DE CONFIGURAÇÕES E PESOS
    with open(caminho_json, 'r') as f:
        instance_info = json.load(f)
    
    # Extração dos pesos das penalidades do JSON
    peso_habilidade = instance_info['weights']['S2_room_nurse_skill']
    peso_trabalho = instance_info['weights']['S4_nurse_excessive_workload']
    
    # Leitura dos arquivos CSV com dados de enfermeiros e quartos
    df_nurses = pd.read_csv(caminho_nurses)
    df_rooms = pd.read_csv(caminho_rooms)
    

    # Dicionários para acesso rápido a habilidades, capacidades e requisitos
    habilidade_enf = df_nurses.drop_duplicates('nurse_id').set_index('nurse_id')['skill_level'].to_dict()
    cap_maxima_enf = df_nurses.set_index(['nurse_id', 'global_shift'])['max_load'].to_dict()
    
    tarefas = list(zip(df_rooms['room_id'], df_rooms['global_shift']))
    req_habilidade = df_rooms.set_index(['room_id', 'global_shift'])['max_skill_required'].to_dict()
    carga_trabalho = df_rooms.set_index(['room_id', 'global_shift'])['total_room_workload'].to_dict()

    # Mapeia quais enfermeiros estão disponíveis em cada turno
    disp_por_turno = {}
    for shift in df_rooms['global_shift'].unique():
        enf_disponiveis = df_nurses[df_nurses['global_shift'] == shift]['nurse_id'].tolist()
        disp_por_turno[shift] = enf_disponiveis

    # Gera uma solução aleatória válida por turno
    def gerar_individuo():
        individuo = []
        for quarto, turno in tarefas:
            enfermeiro_escolhido = random.choice(disp_por_turno[turno])
            individuo.append(enfermeiro_escolhido)
        return individuo

    # Calcula as penalidades da solução
    def calcular_fitness(individuo):
        penalidade_hab = 0
        trabalho_por_enf_turno = {} 

        for i, (quarto, turno) in enumerate(tarefas):
            enf = individuo[i]
            
            # Penalidade de habilidade insuficiente
            hab_faltante = req_habilidade[(quarto, turno)] - habilidade_enf[enf]
            if hab_faltante > 0:
                penalidade_hab += hab_faltante
                
            # Acumula carga de trabalho por enfermeiro/turno
            chave_trab = (enf, turno)
            trabalho_por_enf_turno[chave_trab] = trabalho_por_enf_turno.get(chave_trab, 0) + carga_trabalho[(quarto, turno)]
            
        # Penalidade de excesso de carga de trabalho
        penalidade_trab = 0
        for (enf, turno), carga_total in trabalho_por_enf_turno.items():
            excesso = carga_total - cap_maxima_enf[(enf, turno)]
            if excesso > 0:
                penalidade_trab += excesso
                
        return (peso_habilidade * penalidade_hab) + (peso_trabalho * penalidade_trab)

    # Combina genes de dois pais (Uniforme)
    def cruzamento(pai1, pai2):
        filho = []
        for g1, g2 in zip(pai1, pai2):
            filho.append(g1 if random.random() < 0.5 else g2)
        return filho

    # Altera aleatoriamente enfermeiros na escala
    def mutacao(individuo, taxa_mutacao_alvo):
        for i, (quarto, turno) in enumerate(tarefas):
            if random.random() < taxa_mutacao_alvo:
                individuo[i] = random.choice(disp_por_turno[turno])
        return individuo


    # Loop de execuções (R iterações para análise estatística)
    historico_r_execucoes = []
    melhores_solucoes_finais = []
    tempos_execucao = []
    
    print(f"Iniciando {R} execuções do Algoritmo Genético")
    
    for r in range(R):
        tempo_inicio_r = time.time()
        
        populacao = [gerar_individuo() for _ in range(populacao_tamanho)]
        historico_melhor_geracao = []
        
        # Evolução por gerações
        for geracao in range(geracoes):

            # Ordena por melhor menor custo
            populacao = sorted(populacao, key=calcular_fitness)
            historico_melhor_geracao.append(calcular_fitness(populacao[0]))
            
            # Preserva os 10% melhores
            nova_populacao = populacao[:int(0.1 * populacao_tamanho)]
            
            # Reprodução até completar a nova população
            while len(nova_populacao) < populacao_tamanho:
                torneio = random.sample(populacao[:int(0.5 * populacao_tamanho)], 2)
                pai1, pai2 = torneio[0], torneio[1]

                # Gera filho com crossover e mutação
                filho = mutacao(cruzamento(pai1, pai2), taxa_mutacao_alvo=taxa_mutacao)
                nova_populacao.append(filho)
                
            populacao = nova_populacao
            
        # RESOLUÇÃO DO MODELO
        tempo_fim_r = time.time()
        tempo_iteracao = tempo_fim_r - tempo_inicio_r
        tempos_execucao.append(tempo_iteracao)
        

        historico_r_execucoes.append(historico_melhor_geracao)
        melhor_fitness = calcular_fitness(populacao[0])
        melhores_solucoes_finais.append(melhor_fitness)
        
        print(f"Execução {r+1}/{R} concluída. Melhor fitness: {melhor_fitness} | Tempo da iteração: {tempo_iteracao:.2f} s")

    tempo_medio = np.mean(tempos_execucao)
    melhor_global = min(melhores_solucoes_finais)
    
    print(f"\nTempo médio por execução: {tempo_medio:.2f} segundos")
    print(f"Média do melhor fitness: {np.mean(melhores_solucoes_finais):.2f}")
    print(f"Melhor fitness global (nas {R} execuções): {melhor_global}")

    # PLOTAGEM 
    if plotar:
        # Gera gráfico de convergência
        historico_matriz = np.array(historico_r_execucoes)
        media_geracoes = np.mean(historico_matriz, axis=0)
        
        plt.figure(figsize=(10, 6))
        
        for i, hist in enumerate(historico_r_execucoes):
            plt.plot(hist, alpha=0.3, color='blue', label=f'Exec {i+1}' if i==0 else "")
            
        plt.plot(media_geracoes, color='red', linewidth=2, label='Média das Execuções')
        
        plt.title(f'Convergência do Algoritmo Genético - Instância {nome_instancia}')
        plt.xlabel('Gerações')
        plt.ylabel('Custo Total (Função Objetivo)')
        plt.legend()
        plt.grid(True)
        
        nome_grafico = f'grafico_convergencia_ga_{nome_instancia}.png'
        plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
        print(f"\nGráfico salvo como '{nome_grafico}'.")
        plt.show()

    return melhor_global, np.mean(melhores_solucoes_finais), tempo_medio

def testar_multiplos_parametros(caminho_json, caminho_nurses, caminho_rooms, nome_instancia):
    configs = [
        # Definição das combinações de parâmetros (População, Gerações, Taxa de Mutação)
        (100, 100, 0.05),   # MODIFIQUE AQUI PARA TROCAR PARAMENTROS DA OPÇÃO 2
        (100, 250, 0.05),   # MODIFIQUE AQUI PARA TROCAR PARAMENTROS DA OPÇÃO 2
        (100, 500, 0.05)    # MODIFIQUE AQUI PARA TROCAR PARAMENTROS DA OPÇÃO 2
    ]
    resultados = []
    
    for pop, gen, mut in configs:
        print(f"\nTestando: Pop={pop}, Gen={gen}, Mut={mut}")
        # Executa o AG para a configuração atual
        best_fit, avg_fit, avg_time = resolver_nra_ga(
            caminho_json, caminho_nurses, caminho_rooms, nome_instancia,
            populacao_tamanho=pop, geracoes=gen, R=5, taxa_mutacao=mut, plotar=False # PODE MUDAR O R AQUI
        )
        
        # Armazena os resultados formatados
        resultados.append([nome_instancia, pop, gen, mut, best_fit, round(avg_fit, 2), round(avg_time, 2)])
   
    # Cria DataFrame e exporta os resultados da grade de testes para CSV
    df = pd.DataFrame(resultados, columns=['Instancia', 'Populacao', 'Geracoes', 'Mutacao', 'Melhor_Fitness', 'Fitness_Medio', 'Tempo_Medio_s'])
    df.to_csv(f'tabela_parametros_ga_{nome_instancia}.csv', index=False)
    print(f"\nTestes concluídos! Resultados:\n{df.to_string(index=False)}")


opcoes = ['i01', 'i02', 'i03', 'i04', 'i05', 'i06', 'i18']
print("\n" + "="*40 + "\n   SISTEMA DE TESTES - NRA\n" + "="*40)

# Exibe o menu de instâncias disponíveis
for i, opcao in enumerate(opcoes):
    print(f" [{i+1}] - {opcao}") 
escolha = input("Escolha a instância (1 a 7, onde 1=i01 e 7=i18) ou o nome (ex: i01): ").strip().lower()


# Validação da escolha do usuário
try:
    instancia_escolhida = opcoes[int(escolha)-1]
except (ValueError, IndexError):
    instancia_escolhida = escolha if escolha in opcoes else exit("Opção inválida.")

print(f"\nPreparando ambiente para: {instancia_escolhida.upper()}")

# Define os caminhos dos arquivos de dados da instância selecionada
caminho_json = f'ihtc2024-nra/{instancia_escolhida}/instance_info.json'
caminho_nurses = f'ihtc2024-nra/{instancia_escolhida}/nurse_shifts.csv'
caminho_rooms = f'ihtc2024-nra/{instancia_escolhida}/occupied_room_shifts.csv'

# Seleção do modo de operação
modo = input("\n[1] Execução Padrão Manual  [2] Grade de Testes (CSV)\nModo: ").strip()

if modo == '2':
    # Executa a bateria de testes automáticos definida em testar_multiplos_parametros
    testar_multiplos_parametros(caminho_json, caminho_nurses, caminho_rooms, instancia_escolhida)
else:
    # Captura parâmetros manuais do usuário ou usa valores padrão
    pop = int(input("Tamanho da População [100]: ") or 100)
    gen = int(input("Número de Gerações [500]: ") or 500)
    mut = float(input("Taxa de Mutação [0.05]: ") or 0.05)
    
    # Execução com os parâmetros escolhidos
    best_fit, avg_fit, avg_time = resolver_nra_ga(
        caminho_json, caminho_nurses, caminho_rooms, 
        nome_instancia=instancia_escolhida,
        populacao_tamanho=pop,
        geracoes=gen,
        taxa_mutacao=mut
    )

    # Organiza e salva o resultado da execução manual em um CSV individual
    resultado = [[instancia_escolhida, pop, gen, mut, best_fit, round(avg_fit, 2), round(avg_time, 2)]]
    df_res = pd.DataFrame(resultado, columns=['Instancia', 'Populacao', 'Geracoes', 'Mutacao', 'Melhor_Fitness', 'Fitness_Medio', 'Tempo_Medio_s'])
    nome_csv = f'resultado_individual_ga_{instancia_escolhida}.csv'
    df_res.to_csv(nome_csv, index=False)
    print(f"\n>>> Execução concluída! Resultado salvo em '{nome_csv}'")