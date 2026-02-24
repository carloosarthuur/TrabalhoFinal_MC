# Otimização da Alocação Enfermeiro-Quarto (NRA)

Este repositório contém os códigos-fonte para a resolução de uma versão simplificada do problema de **Alocação de Enfermeiros aos Quartos (Nurse-to-Room Assignment - NRA)**, baseado no *Integrated Healthcare Timetabling Problem* (IHTC-2024). 

O projeto explora e compara duas abordagens distintas de otimização combinatória para minimizar as penalidades por déficit de habilidade técnica e excesso de carga de trabalho dos enfermeiros.

---

## Arquivos do Projeto

O repositório é composto por dois scripts principais:

1. **`modelo_pli.py` (Programação Linear Inteira):** Utiliza a biblioteca PuLP para modelar o problema de forma exata matematicamente. Ele busca a solução ótima global respeitando as restrições rígidas e flexíveis.
   
2. **`modelo_genetico.py` (Algoritmo Genético):** Uma meta-heurística estocástica projetada para encontrar soluções de alta qualidade em um tempo viável. Utiliza operadores de Seleção por Torneio, Crossover Uniforme e Mutação focada na disponibilidade dos turnos.

Alem dos Scripts Temos Algumas Pastas com conteudos Importantes

3. **`ihtc2024-nra`:** arquivo contendo as instancias do problema utilizadas
4.  **`PLI_vs_AG`:** arquivo contendo os dois csvs utilizados nas tabeelas do artigo alem de uma pasta com os resultados individuais gerados pelo algoritimo genetico um a um (usados para compor `tabela_resultados_ag`) e uma pasta com os graficos gerados para cada instancia
5. **`Sensibilidade_AG`:** é onde foi armazenado todos os csvs gerados utilizando o `modelo_genetico.py`, foram utilizados para criar `tabela_parametros_ga_sensibilidade`
---

## Pré-requisitos e Instalação

Para rodar os códigos, você precisará do **Python 3.8+** instalado em sua máquina.

1. Clone este repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
   cd SEU_REPOSITORIO
   ```

2. Instale as bibliotecas necessárias. Você pode usar o comando abaixo:
   ```bash
   pip install requirements.txt
   ```
   *(Nota: Recomendo usar um ambiente virtual; O PuLP geralmente já inclui o solver CBC padrão necessário para o modelo PLI).*

---

## Estrutura de Dados Esperada

Para que os scripts funcionem corretamente, os arquivos das instâncias do IHTC-2024 devem estar organizados em uma pasta chamada `ihtc2024-nra` no mesmo diretório dos scripts, seguindo esta estrutura hierárquica:

```text
.
├── PLI_vs_AG/
├── Sensibilidade_AG/
├── modelo_genetico.py
├── modelo_pli.py
└── ihtc2024-nra/
    ├── i01/
    │   ├── instance_info.json
    │   ├── nurse_shifts.csv
    │   └── occupied_room_shifts.csv
    ├── i02/
    └── ...
```

---

## Como Executar

Ambos os programas foram desenvolvidos com menus interativos no terminal para facilitar a execução de testes e a coleta de dados.

### 1. Modelo Exato (PLI)
Execute o comando no terminal:
```bash
python modelo_pli.py
```
Você verá um menu com as seguintes opções:
* **[1] Execução Padrão Manual:** Permite escolher uma única instância (ex: `i02`) e visualizar os detalhes de alocação (Top 10) diretamente no terminal. `resultado_individual_pli_[instancia]*.csv`
* **[2] Grade de Testes:** Executa automaticamente **todas** as instâncias (`i01` a `i18`) respeitando o tempo limite configurado e gera um relatório consolidado no arquivo `tabela_resultados_pli.csv`.
* **Configuração de Tempo:** O programa perguntará qual o tempo limite do solver em segundos (o padrão é 300s).

### 2. Algoritmo Genético (AG)
Execute o comando no terminal:
```bash
python modelo_genetico.py
```
O menu interativo oferecerá:
* **[1] Execução Padrão Manual:** Você escolhe uma instância e define manualmente o Tamanho da População, Número de Gerações e Taxa de Mutação. Ao final, o programa plota o gráfico de convergência e salva os resultados no arquivo `resultado_individual_ga_[instancia].csv`.
* **[2] Grade de Testes (CSV):** Executa uma análise de sensibilidade sendo possivel modificar pupulação, n° de gerações e taxa de mutação, para a instancia escolhida e salva em  `tabela_parametros_ga_[instancia].csv`.
---

## Arquivos de Saída (Outputs)

Ao rodar os testes, os seguintes arquivos serão gerados na raiz do seu projeto para análise:

* **`grafico_convergencia_ga_*.png`**: Gráficos exibindo a evolução da função de *fitness* (aptidão) ao longo das gerações do Algoritmo Genético (somente para a opção manual).
* **`tabela_resultados_pli.csv`**: Contém a consolidação do PLI, incluindo Instância, Status do Solver (ex: Optimal, Timeout), Tempo(s) de processamento e Custo Total (Função Objetivo).
* **`tabela_parametros_ga_*.csv`** / **`resultado_individual_ga_*.csv`**: Contêm as métricas da meta-heurística, incluindo Melhor Fitness, Fitness Médio das repetições e Tempo Médio de Execução.

---

## Autores / Contexto

Carlos Arthur Lima da Silva
Projeto feito inteiramente por uma pessoa
