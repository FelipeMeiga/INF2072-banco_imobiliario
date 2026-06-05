# Banco Imobiliario IA

Projeto de simulacao de um jogo estilo Monopoly/Banco Imobiliario com agentes
de inteligencia artificial treinados para jogar partidas completas. O projeto
nao copia nomes, arte ou marca oficial; o tabuleiro usa nomes genericos, mas
modela as principais mecanicas classicas do jogo.

O objetivo do trabalho e investigar se agentes treinados por aprendizado de
reforco conseguem aprender estrategias relevantes em um ambiente com compra de
propriedades, leiloes, construcao, hipotecas, falencia e negociacao entre
jogadores. O foco atual e o agente PPO, depois de tentativas com DQN e NEAT.

## Instalar

```powershell
pip install -r requirements.txt
```

Dependencias principais:

- `pygame`: visualizacao da partida.
- `numpy`: vetorizacao de estados e acoes.
- `torch`: redes neurais e treino PPO/DQN.
- `neat-python`: experimentos NEAT.

## Rodar

Assistir uma partida:

```powershell
py main.py
```

Treinar PPO:

```powershell
py train_ppo.py --episodes 2000 --entropy-coef 0.05 --tournament-every 100 --tournament-games 100
```

Continuar treino PPO a partir de checkpoint:

```powershell
py train_ppo.py --resume models\ppo_raw_checkpoints\ppo_raw_ep_000700.pt --episodes 2000 --tournament-every 100 --tournament-games 100
```

`--episodes` indica o episodio final alvo. Se o checkpoint esta no episodio
700 e o comando usa `--episodes 2000`, o treino continua do episodio 701 ate o
2000.

Rodar campeonato manual:

```powershell
py tournament.py --checkpoints models/best_ppo_raw_agent.pt models/ppo_raw_checkpoints --games 1000 --latest 8
```

Abrir o ultimo replay salvo:

```powershell
$env:BANCO_REPLAY_PATH = "replays\latest_replay.json"
py main.py
```

Voltar a jogar/gravar partidas novas:

```powershell
Remove-Item Env:BANCO_REPLAY_PATH
py main.py
```

## Organizacao do projeto

```text
t1/
|-- main.py
|-- main_github.py
|-- train_ppo.py
|-- train_dqn.py
|-- train_neat.py
|-- train_neat_github.py
|-- tournament.py
|-- game/
|   |-- env.py
|   |-- board.py
|   |-- actions.py
|   |-- encoders.py
|   |-- models.py
|   |-- github_monopoly.py
|-- agents/
|   |-- ppo_agent.py
|   |-- neural_agent.py
|   |-- neat_agent.py
|   |-- random_agent.py
|   |-- replay_buffer.py
|-- ui/
|   |-- pygame_view.py
|-- models/
|-- replays/
```

## Principais classes e responsabilidades

### `game.env.BancoImobiliarioEnv`

Motor principal do jogo. Mantem o estado completo da partida, valida acoes,
aplica regras, calcula recompensas, controla fases, leiloes, trocas, falencia e
fim de jogo.

API principal:

```python
state = env.get_state()
actions = env.get_valid_actions(player_index)
next_state, reward, done, info = env.step(action)
```

### `game.models.Space` e `game.models.Player`

Representam casas do tabuleiro e jogadores. `Space` guarda tipo da casa, preco,
aluguel, dono, hipoteca, grupo de cor e construcoes. `Player` guarda dinheiro,
posicao, falencia, cadeia e cartas de saida da cadeia.

### `game.board`

Define o tabuleiro com 40 casas, grupos de cores, ferrovias, companhias,
impostos, cartas, cadeia e constantes do jogo.

### `game.actions`

Define os tipos de acoes aceitas pelo ambiente, como `roll_dice`,
`buy_property`, `auction_bid`, `build_house`, `start_trade`, `submit_trade`,
`accept_trade` e `decline_trade`.

### `game.encoders`

Transforma estados e acoes estruturados em vetores numericos para redes
neurais. O encoder atual tem dois modos:

```text
raw:  state_size=318 | action_size=151
rich: state_size=494 | action_size=163
```

O encoder `raw` e o padrao atual do PPO. Ele ainda inclui algumas features
compactas para tornar trocas e complementaridade de grupos observaveis pela
rede.

### `agents.ppo_agent.PPOActorCritic`

Rede Actor-Critic usada pelo PPO. O actor recebe o par `(estado, acao)` e gera
um logit para aquela acao valida. O critic recebe apenas o estado e estima
`V(s)`.

### `agents.ppo_agent.PPOAgent`

Empacota a rede PPO e escolhe uma acao entre as acoes validas. Para cada estado,
o ambiente fornece apenas as acoes legais; o agente pontua todas e amostra uma
acao pela distribuicao categorial.

### `agents.neural_agent.QNetwork` e `agents.neural_agent.NeuralAgent`

Implementacao DQN antiga. Tambem usa a ideia de avaliar pares `(estado, acao)`,
mas aprende Q-values com replay buffer e target network.

### `agents.neat_agent.NeatAgent`

Adaptador para redes evoluidas por NEAT no ambiente principal.

### `agents.random_agent.RandomAgent`

Baseline heuristico. Apesar do nome, nao e puramente aleatorio: tende a comprar,
construir, participar de leiloes e montar/avaliar trocas por valor situacional.
No campeonato aparece como `heuristic_baseline`. O campeonato tambem inclui
`pure_random_baseline`.

### `ui.pygame_view.PygameView`

Interface Pygame. Mostra tabuleiro, jogadores, informacoes da casa atual,
trocas, historico, controles e timeline de replay.

## Modelagem do ambiente

O jogo e modelado como um ambiente multiagente por turnos. Em cada momento
existe um jogador que deve agir. Isso pode ser o jogador do turno, o jogador
atual de um leilao ou o jogador que precisa responder uma troca.

### Fases

O ambiente usa fases para limitar as acoes validas:

- `ready_to_roll`: jogador pode rolar dados ou fazer acoes financeiras.
- `awaiting_buy`: jogador caiu em propriedade sem dono e decide comprar ou
  passar para leilao.
- `auction`: jogadores participam de leilao.
- `building_trade`: jogador monta uma proposta de troca por etapas.
- `pending_trade_response`: outro jogador aceita ou recusa a proposta.
- `game_over`: partida terminada.

### Regras implementadas

O ambiente principal implementa:

- passagem pelo Go com bonus;
- compra de propriedades;
- leilao quando uma propriedade nao e comprada;
- aluguel comum, aluguel dobrado em grupo completo e aluguel com casas/hotel;
- ferrovias com aluguel escalado;
- companhias com aluguel baseado nos dados;
- cartas de Chance e Community Chest;
- cadeia, pagamento de fianca, carta de saida e regra das tres duplas;
- hipoteca e quitar hipoteca;
- juros ao receber propriedade hipotecada;
- venda de construcoes pela metade do custo;
- construcao uniforme dentro do grupo;
- limite de 32 casas e 12 hoteis;
- trocas entre jogadores;
- falencia;
- fim por ultimo jogador ativo;
- fim por limite de turnos com vencedor por patrimonio.

### Trocas

As trocas nao sao mais propostas prontas geradas pelo ambiente. O agente monta a
troca em etapas:

1. `start_trade`
2. `select_trade_target`
3. `add_trade_offer_property`
4. `finish_trade_offer`
5. `set_trade_offer_money`
6. `add_trade_request_property`
7. `finish_trade_request`
8. `set_trade_request_money`
9. `submit_trade`
10. `accept_trade` ou `decline_trade`

Regras importantes:

- propriedades de grupo com casas/hotel nao podem ser negociadas;
- doacoes unilaterais sao bloqueadas;
- trocas apenas dinheiro-por-dinheiro sao bloqueadas;
- uma proposta precisa ter contrapartida dos dois lados;
- propriedades hipotecadas podem ser negociadas;
- ao receber propriedade hipotecada, o novo dono paga juros de transferencia
  quando aplicavel;
- `cancel_trade` so aparece como fallback tecnico quando nao existe outra acao
  valida na etapa atual.

O encoder inclui sinais de complementaridade de troca. Exemplo:

```text
Jogador A tem 2/3 Orange.
Jogador B tem o Orange faltante.
Jogador B tem 2/3 Pink.
Jogador A tem o Pink faltante.
```

Nesse caso o estado indica que existe potencial de troca mutuamente benefica.
O ambiente nao monta a proposta; ele apenas torna esse padrao observavel.

### Replays

A visualizacao salva partidas em `replays/`. Cada replay JSON guarda:

- seed da partida;
- modelo usado;
- lista exata de acoes;
- cursor atual;
- se a partida terminou.

Como as acoes sao gravadas, a partida pode ser reassistida exatamente mesmo que
o modelo atual tenha mudado.

Controles:

- `S`: salva replay atual.
- `L`: carrega `replays/latest_replay.json`.
- slider inferior: avanca ou volta para qualquer acao gravada.
- `U` ou `Backspace`: volta uma acao pela timeline.

## Recompensas e punicoes

O reward principal vem do ambiente em `env.step(action)`.

### Patrimonio

A base do reward e a mudanca de patrimonio liquido:

```text
reward_base = (patrimonio_depois - patrimonio_antes) / 100
```

O patrimonio inclui dinheiro, propriedades, hipotecas e construcoes. Falencia e
perda de patrimonio geram rewards negativos por consequencia direta.

### Rewards estrategicos

O ambiente tambem aplica shaping leve para acelerar o aprendizado:

- `buy_property` e `auction_bid`: `+1.0`;
- `build_house` bem-sucedido: `+4.0`;
- completar grupo de cor: `+12.0` por grupo novo;
- vencer a partida: `+100.0`;
- perder ao fim da partida: `-100.0`.

### Compra e leilao

Existe shaping especifico para compra/leilao:

- comprar ou dar lance em propriedade que completa grupo recebe bonus maior;
- comprar ou dar lance em propriedade que bloqueia monopolio de outro jogador
  tambem recebe bonus;
- passar compra ou sair de leilao quando ainda havia valor estrategico pode
  gerar penalidade;
- lances tem teto racional para evitar que agentes gastem todo o dinheiro em
  uma propriedade sem sentido.

### Trocas

As trocas usam recompensa por consequencia real:

- se uma troca aumenta o patrimonio de um jogador, ele recebe reward positivo;
- se uma troca prejudica o jogador, recebe reward negativo;
- o ambiente retorna `rewards_by_player`, ou seja, recompensa separada para cada
  jogador afetado;
- no PPO, essa recompensa e atribuida a ultima decisao do jogador correto.

Isso corrige um problema observado durante o desenvolvimento: antes, um jogador
podia propor uma troca ruim, outro aceitava, e o prejuizo nao voltava
corretamente para quem tinha proposto.

### Clip no treino

No PPO, antes de entrar no rollout, o reward e limitado:

```text
REWARD_CLIP_MIN = -10.0
REWARD_CLIP_MAX = 10.0
```

Isso reduz explosoes na funcao de valor.

## Tentativas de solucao

### DQN

Primeira abordagem neural usada no projeto.

Implementacao:

- arquivo principal: `train_dqn.py`;
- agente: `agents/neural_agent.py`;
- replay buffer: `agents/replay_buffer.py`;
- rede aprende `Q(estado, acao)`;
- target network;
- epsilon-greedy;
- replay buffer;
- ajuste adversarial no alvo quando a proxima acao e de outro jogador.

Problemas observados:

- o treino era instavel;
- reward chegou a ficar extremamente negativo antes dos ajustes de patrimonio;
- agentes evitavam comprar ou demoravam a desenvolver o tabuleiro;
- o espaco de acoes variavel tornou o problema mais dificil para DQN;
- trocas eram especialmente ruins, porque dependem de sequencias longas e
  credito temporal dificil;
- loss isolada nao indicava qualidade real de jogo.

Resultado pratico:

- DQN ajudou a estruturar o ambiente e os encoders;
- nao foi a abordagem mais adequada para o estado atual do problema;
- foi mantido como comparacao historica.

Limitacoes:

- pouco adequado para acoes longas e estruturadas de negociacao;
- sensivel ao replay buffer em ambiente multiagente nao estacionario;
- Q-values podem ficar instaveis com recompensas de longo prazo.

### PPO

Abordagem principal atual.

Motivos da troca para PPO:

- lida melhor com acoes amostradas por politica;
- permite trabalhar diretamente com distribuicao sobre acoes validas;
- combina melhor com self-play;
- facilita medir entropia, policy loss e value loss;
- evita alguns problemas de Q-learning em ambiente multiagente.

Resultado atual:

Em um campeonato manual com 1000 partidas totais:

```text
rank | competidor       | jogos | vitorias | win% | pos_med | patrimonio | score
0001 | ppo_raw_ep0450   |   418 |      183 | 43.8 |    2.11 |   16008.70 | 5767.86
0002 | ppo_raw_ep0400   |   781 |      295 | 37.8 |    2.23 |   13348.28 | 4889.37
0008 | heuristic_baseli |   388 |       50 | 12.9 |    2.90 |    4154.20 | 1414.13
0009 | pure_random_base |   404 |       35 |  8.7 |    3.06 |    1985.09 |  758.90
```

Interpretacao:

- PPO superou claramente `heuristic_baseline` e `pure_random_baseline`;
- `ppo_raw_ep0450` foi o melhor checkpoint observado no campeonato maior;
- a diferenca para os baselines foi grande o suficiente para nao parecer apenas
  sorte;
- campeonatos pequenos com 100 jogos totais ainda apresentaram muita variancia;
- 1000 jogos totais produziram uma leitura mais confiavel.

Limitacoes:

- ainda ha instabilidade entre checkpoints;
- partidas de Monopoly tem variancia alta por dados, cartas e ordem de turno;
- 100 jogos totais por campeonato nao eliminam a sorte;
- trocas ainda sao a parte mais dificil do aprendizado;
- PPO atual e feedforward, sem memoria explicita de negociacoes recusadas;
- a politica e compartilhada entre jogadores, o que simplifica o treino, mas
  limita especializacao;
- nao existe modelagem explicita de intencao dos adversarios.

### NEAT no ambiente principal

Foi implementada uma versao NEAT que usa o mesmo par `(estado_raw, acao_raw)` do
ambiente principal.

Caracteristicas:

- arquivo: `train_neat.py`;
- agente: `agents/neat_agent.py`;
- self-play;
- hall of fame;
- champion gate para evitar substituir campeoes por candidatos piores.

Motivo do experimento:

- o video/referencia usava NEAT;
- NEAT pode encontrar estruturas de rede sem gradiente;
- poderia mostrar comportamento emergente com outra abordagem.

Limitacoes observadas:

- muito caro computacionalmente;
- precisa de muitos jogos por genoma para reduzir sorte;
- poucas geracoes pequenas nao garantem cobertura suficiente;
- nao ha garantia de superar PPO;
- avaliar genomas em Monopoly e lento por natureza.

### NEAT GitHub-style

Foi criada tambem uma implementacao separada mais parecida com o repositorio
`MonopolyNEAT`.

Arquivos:

- `game/github_monopoly.py`;
- `train_neat_github.py`;
- `main_github.py`;
- `neat_github_config.ini`.

Caracteristicas:

- vetor fixo de 126 entradas;
- 9 saidas da rede;
- ambiente separado do principal;
- torneio em chaves;
- fitness baseado em progresso no bracket e taxa de pontuacao.

Limitacoes:

- ambiente nao e identico ao principal;
- a rede nao recebe lista de acoes validas como no PPO;
- trocas no estilo GitHub sao menos controladas pelo agente;
- para usar a escala original, como `--pop-size 256 --games-per-bracket 2000`,
  o custo fica muito alto;
- resultados de pequenas geracoes sao muito ruidosos.

### Baselines

O campeonato usa dois baselines:

- `pure_random_baseline`: escolhe acoes validas aleatoriamente.
- `heuristic_baseline`: usa regras simples para comprar, construir, leiloar,
  sair da cadeia e negociar. Nas trocas, procura propriedades que fecham grupos
  para si, oferece propriedades que fecham grupos para o outro jogador, calcula
  uma compensacao em dinheiro e aceita ou recusa pela diferenca de valor
  situacional. Ferrovias e companhias tambem entram com peso menor quando
  melhoram colecoes ja existentes.

Esses baselines sao importantes porque loss/reward isolados nao dizem se o
agente joga melhor.

## PPO atual em detalhes

### Rede

Classe: `PPOActorCritic`.

Arquitetura padrao:

- hidden size: `512`;
- actor com duas camadas escondidas;
- critic com duas camadas escondidas;
- ativacao ReLU;
- actor recebe `estado + acao`;
- critic recebe apenas `estado`.

O actor nao tem uma saida fixa para cada acao. Em vez disso, para cada estado:

1. o ambiente gera a lista de acoes validas;
2. cada acao e codificada;
3. a rede calcula um logit para cada par `(estado, acao)`;
4. a distribuicao categorial e formada apenas sobre acoes validas;
5. uma acao e amostrada durante treino.

Isso funciona melhor para acoes estruturadas, como:

- lance de leilao com valor;
- construir em uma propriedade especifica;
- hipotecar propriedade;
- montar troca com propriedades e dinheiro.

### Critic

O critic estima `V(s)`, nao Q-values. Ele tenta prever o retorno esperado a
partir do estado, independentemente da acao especifica.

### PPO loss

O treino usa:

- clipped surrogate objective;
- value loss;
- bonus de entropia;
- GAE para vantagens.

Parametros atuais principais:

```text
GAMMA = 0.98
GAE_LAMBDA = 0.95
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
PPO_EPOCHS = 2
CLIP_EPSILON = 0.20
VALUE_COEF = 0.50
ENTROPY_COEF = 0.05
MAX_GRAD_NORM = 0.50
UPDATE_EVERY_EPISODES = 4
TRADE_CURRICULUM_RATIO = 0.50
```

`--entropy-coef` pode ser ajustado por linha de comando. Ele foi aumentado para
manter mais exploracao em trocas, porque o agente chegou a reduzir demais a
frequencia de propostas.

### Rollout por jogador

Como todos os jogadores usam a mesma politica, o PPO poderia misturar recompensas
de jogadores diferentes. Para reduzir esse problema, o treino separa as
trajetorias por jogador.

Quando uma acao de um jogador afeta outro, `env.step` retorna
`rewards_by_player`. O treino acumula a recompensa de cada jogador na ultima
decisao daquele jogador.

Isso e especialmente importante em trocas:

```text
Jogador 1 propoe troca ruim.
Jogador 4 aceita.
Jogador 1 perde patrimonio.
Reward negativo volta para a ultima decisao do Jogador 1.
```

### Curriculum de trocas

Parte dos episodios comeca em estados favoraveis a negociacao. Exemplo:

```text
Jogador A tem 2/3 Orange e 1/3 Pink.
Jogador B tem 1/3 Orange e 2/3 Pink.
```

O ambiente nao monta a troca. Ele apenas cria o estado. O agente ainda precisa:

- iniciar troca;
- escolher alvo;
- oferecer a propriedade certa;
- pedir a propriedade certa;
- definir dinheiro;
- enviar proposta;
- aceitar ou recusar como outro jogador.

### Checkpoints e campeonatos

O treino salva:

```text
models/ppo_raw_agent.pt
models/ppo_raw_checkpoints/
models/best_ppo_raw_agent.pt
```

O melhor checkpoint e escolhido por campeonato, nao apenas por reward de treino.
Isso e necessario porque o reward por episodio e muito ruidoso.

## Resultados observados

Durante o desenvolvimento, foram observados estes comportamentos:

- no inicio, agentes frequentemente nao compravam propriedades suficientes;
- DQN gerou rewards muito negativos antes de ajustes;
- leiloes inicialmente permitiam gastos absurdos;
- agentes faziam loops de construir/vender e hipotecar/quitar;
- trocas antigas permitiam propostas sem sentido, inclusive doar propriedades
  ou trocar dinheiro por dinheiro;
- PPO inicialmente aceitava quase todas as trocas porque o estado indicava o
  jogador errado durante resposta;
- depois da correcao de reward por jogador, as trocas passaram a ser punidas ou
  recompensadas de forma mais coerente;
- remover `cancel_trade` como acao normal reduziu o atalho de iniciar e cancelar
  trocas;
- aumentar entropia ajudou a manter exploracao de negociacao;
- campeonatos pequenos tinham alta variancia;
- campeonato maior com 1000 jogos totais indicou superioridade clara do PPO
  sobre os baselines.

Resultado mais relevante conhecido:

```text
ppo_raw_ep0450:
  jogos: 418
  vitorias: 183
  win%: 43.8
  pos_med: 2.11
  patrimonio: 16008.70
  score: 5767.86

heuristic_baseline:
  jogos: 388
  vitorias: 50
  win%: 12.9
  pos_med: 2.90
  patrimonio: 4154.20
  score: 1414.13

pure_random_baseline:
  jogos: 404
  vitorias: 35
  win%: 8.7
  pos_med: 3.06
  patrimonio: 1985.09
  score: 758.90
```

Conclusao atual: o PPO ja joga melhor que os baselines disponiveis, mas ainda
nao deve ser considerado uma solucao final.

## Limitacoes atuais

### Ambiente

- Ainda ha shaping manual em compra, leilao, construcao e grupo completo.
- As regras sao uma aproximacao das principais mecanicas, nao uma replica legal
  perfeita de todas as variantes.
- O fim por limite de turnos ainda pode influenciar estrategias.
- A ordem dos jogadores e randomizada no reset, mas jogos de tabuleiro continuam
  tendo forte variancia por dados/cartas.

### Trocas

- A negociacao e sequencial e longa, o que dificulta credito temporal.
- O PPO nao tem memoria recorrente de propostas recusadas.
- Nao ha sistema de contraoferta.
- Nao ha modelagem explicita de "quanto o adversario precisa dessa propriedade".
- O agente pode aprender a fazer muitas propostas se isso parecer barato no
  curto prazo.

### PPO

- A politica compartilhada simplifica o treino, mas todos os jogadores usam a
  mesma rede.
- O critic estima apenas `V(s)`, nao avalia explicitamente cada oponente.
- PPO e sensivel a hiperparametros e pode piorar temporariamente entre
  checkpoints.
- Campeonatos pequenos podem escolher campeoes por sorte.

### DQN

- Mais instavel neste ambiente.
- Sofre com acoes variaveis e estrategias longas.
- Replay buffer mistura experiencias de uma politica que muda em self-play.

### NEAT

- Muito caro para avaliar de forma confiavel.
- Precisa de muitas partidas por genoma.
- Pequenas geracoes sao altamente ruidosas.

## Melhorias possiveis

Sem considerar apenas "treinar por mais tempo", os principais caminhos de
melhoria sao:

### 1. Politica com memoria

Usar uma rede recorrente, como LSTM/GRU, para que o agente lembre de propostas
recentes, recusas, padroes de adversarios e contexto de negociacao.

### 2. Attention ou arquitetura por entidades

Em vez de vetor plano, representar jogadores, propriedades e grupos como
entidades. Uma arquitetura com attention poderia comparar diretamente:

```text
minhas propriedades
propriedades do alvo
grupos quase completos
dinheiro disponivel
risco de aluguel
```

### 3. Melhor abstracao de trocas

Manter a decisao emergente, mas reduzir a sequencia de acoes. Exemplo:

- escolher grupo desejado;
- escolher propriedade alvo;
- escolher compensacao;
- escolher dinheiro em buckets.

Isso preserva aprendizado, mas reduz combinatoria.

### 4. Contraofertas

Adicionar uma fase de negociacao onde uma troca recusada pode gerar uma
contraoferta ou nova proposta informada pelo historico.

### 5. Avaliacao estatistica melhor

Usar campeonatos maiores, round-robin balanceado, seeds fixas por comparacao e
intervalos de confianca. Outra alternativa e usar Elo, Glicko ou TrueSkill para
avaliar checkpoints.

### 6. Self-play populacional

Em vez de todos os jogadores usarem sempre a mesma politica mais recente,
treinar contra uma populacao de checkpoints antigos. Isso reduz esquecimento e
estrategias que exploram apenas a versao atual de si mesmo.

### 7. MAPPO ou politicas separadas

Usar uma abordagem multiagente mais explicita, como MAPPO, ou politicas por
assento/jogador. Isso pode melhorar credito em interacoes competitivas.

### 8. Rewards auxiliares melhor controlados

Criar metricas auxiliares sem entregar a resposta pronta, como:

- liquidez minima;
- risco de falencia;
- valor potencial de grupo completo;
- taxa de aceite de propostas;
- custo de spam de propostas.

Esses sinais precisam ser calibrados para nao substituir a estrategia emergente.

### 9. Melhor modelagem de leiloes

O teto racional atual evita lances absurdos, mas ainda pode ser refinado com
valor contextual, bloqueio de adversario e liquidez pos-compra.

### 10. Paralelizacao de ambientes

Rodar multiplos ambientes em paralelo para coletar rollouts maiores por update,
reduzindo variancia sem depender apenas de mais tempo sequencial.

## Observacoes finais

O projeto evoluiu de uma implementacao DQN instavel para uma abordagem PPO mais
adequada ao ambiente. A IA atual ja supera os baselines em campeonatos maiores,
mas o problema ainda e aberto: Monopoly/Banco Imobiliario combina sorte,
planejamento de longo prazo, negociacao, liquidez e interacao adversarial.

O melhor resultado ate agora indica que a abordagem esta funcionando, mas ainda
ha espaco relevante para melhorar arquitetura, representacao do estado,
negociacao e avaliacao estatistica.
