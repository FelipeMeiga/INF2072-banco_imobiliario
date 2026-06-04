# Banco Imobiliario IA com PPO, DQN e NEAT

Projeto de simulacao de um jogo estilo Monopoly/Banco Imobiliario, com motor de
regras separado da interface e agentes treinados principalmente por PPO. O DQN
antigo continua no repositorio para comparacao.

O projeto nao copia nomes, arte ou marca oficial. O tabuleiro usa nomes
genericos, mas implementa as principais mecanicas classicas: compra, aluguel,
leilao, cadeia, cartas, hipoteca, casas, hoteis, trocas e falencia.

## Instalacao

```bash
pip install -r requirements.txt
```

Dependencias principais:

- `pygame`: visualizacao da partida.
- `numpy`: vetorizacao de estado/acoes.
- `torch`: rede neural, PPO, DQN e treino.

## Como rodar

Assistir uma partida com agentes neurais:

```bash
python main.py
```

Assistir uma partida no ambiente GitHub-style/NEAT:

```bash
py main_github.py
```

Esse visualizador carrega `models/best_neat_github_agent.pkl` quando existir.
Para apontar para outro checkpoint:

```bash
$env:BANCO_GITHUB_MODEL_PATH="models/outro_neat_github.pkl"
py main_github.py
```

Treinar o PPO atual com encoder `raw`, voltado a comportamento mais emergente:

```bash
py train_ppo.py --episodes 2000 --tournament-every 100 --tournament-games 100
```

Para manter mais exploracao, especialmente nas trocas, ajuste a entropia:

```bash
py train_ppo.py --episodes 2000 --entropy-coef 0.05 --tournament-every 100 --tournament-games 100
```

Esse comando salva modelos em:

```text
models/ppo_raw_agent.pt
models/ppo_raw_checkpoints/
models/best_ppo_raw_agent.pt
```

Retomar o PPO a partir de um checkpoint:

```bash
py train_ppo.py --resume models/ppo_raw_checkpoints/ppo_raw_ep_000700.pt --episodes 2000 --tournament-every 100 --tournament-games 100
```

Nesse caso, `--episodes 2000` significa episodio final alvo. Se o checkpoint
esta no episodio 700, o treino continua do episodio 701 ate o 2000. Checkpoints
novos salvam estado do modelo, otimizador, RNGs e rollout parcial do PPO.

Observacao: o encoder `raw` atual tem `state_size=318` e `action_size=151`.
O encoder `rich` atual tem `state_size=494` e `action_size=163`. Checkpoints
PPO antigos com tamanhos diferentes sao ignorados ou precisam ser retreinados.

Treinar o PPO antigo com encoder `rich`, que entrega mais features estrategicas
prontas para a rede:

```bash
python train_ppo.py --encoder rich --episodes 2000
```

Treinar uma abordagem NEAT usando o mesmo encoder `raw`:

```bash
python train_neat.py --generations 100 --games-per-genome 2
```

Esse treino evolui redes que recebem o par `(estado_raw, acao_raw)` e retornam
um score para cada acao valida. O treino usa self-play, hall of fame e um
`champion gate`: um novo genoma so substitui o campeao salvo se vencer uma
bateria curta contra o campeao atual. O melhor genoma protegido e salvo em:

```text
models/best_neat_raw_agent.pkl
```

Campeoes historicos ficam em:

```text
models/neat_hall_of_fame/
```

Quando esse arquivo existir, `python main.py` carrega o NEAT antes do PPO.

Parametros uteis:

```bash
python train_neat.py --generations 200 --games-per-genome 4 --hall-of-fame-games 1 --champion-games 24
```

- `games-per-genome`: jogos de self-play por genoma.
- `hall-of-fame-games`: rodadas extras contra campeoes antigos.
- `hall-of-fame-size`: quantos campeoes antigos entram como adversarios.
- `champion-games`: jogos usados para decidir se um candidato vira campeao.
- `champion-margin`: margem minima para trocar o campeao salvo.

Treinar uma abordagem NEAT no estilo do repositorio GitHub `MonopolyNEAT`:

```bash
py train_neat_github.py --generations 100 --games-per-bracket 40
```

Esse modo usa um ambiente separado em `game/github_monopoly.py`, com 126
entradas e 9 saidas fixas da rede, como no repositorio original. A rede nao
recebe uma lista de acoes validas; o ambiente pergunta decisoes especificas
quando elas aparecem na partida.

Para usar a escala original do repositorio, o comando equivalente seria muito
mais caro:

```bash
py train_neat_github.py --pop-size 256 --games-per-bracket 2000
```

Na pratica, comece com valores menores e aumente quando quiser cobertura mais
confiavel.

Para rodar por tempo limitado e continuar depois:

```bash
py train_neat_github.py --pop-size 256 --games-per-bracket 2000 --max-hours 8 --checkpoint-every 1
```

O treino para somente no fim de uma geracao. Ao atingir o tempo, ele salva um
checkpoint em `models/neat_github_checkpoints/` e imprime o comando de
continuidade:

```bash
py train_neat_github.py --resume models/neat_github_checkpoints/neat-github-N
```

Se voce interromper manualmente com `Ctrl+C`, continue a partir do ultimo
checkpoint salvo. Usar `--checkpoint-every 1` reduz o trabalho perdido para no
maximo uma geracao.

Treinar o DQN antigo:

```bash
python train_dqn.py --episodes 1000
```

Treinar salvando checkpoints:

```bash
python train_dqn.py --episodes 1000 --checkpoint-every 100
```

Treinar com campeonato periodico:

```bash
python train_dqn.py --episodes 1000 --checkpoint-every 100 --tournament-every 200 --tournament-games 40
```

Rodar campeonato manual:

```bash
python tournament.py --checkpoints models/ppo_raw_checkpoints --games 100 --latest 8
```

Comparar NEAT contra baselines:

```bash
python tournament.py --checkpoints models/best_neat_raw_agent.pkl --games 100
```

## Estrutura do projeto

```text
t1/
|-- main.py
|-- main_github.py
|-- train_dqn.py
|-- train_neat.py
|-- train_neat_github.py
|-- train_ppo.py
|-- tournament.py
|-- neat_github_config.ini
|-- neat_raw_config.ini
|-- requirements.txt
|-- game/
|   |-- actions.py
|   |-- board.py
|   |-- encoders.py
|   |-- env.py
|   |-- github_monopoly.py
|   |-- models.py
|   `-- property.py
|-- agents/
|   |-- neural_agent.py
|   |-- random_agent.py
|   |-- replay_buffer.py
|   `-- __init__.py
|-- ui/
|   |-- pygame_view.py
|   `-- __init__.py
`-- models/
    |-- dqn_agent.pt
    |-- best_dqn_agent.pt
    `-- checkpoints/
```

## Ambiente do jogo

O ambiente principal fica em `game/env.py`, na classe `BancoImobiliarioEnv`.

Ele e responsavel por:

- guardar o estado completo da partida;
- validar acoes possiveis;
- aplicar regras do jogo;
- calcular recompensas;
- controlar turnos, fases e fim de jogo;
- gerar snapshots para voltar uma jogada na visualizacao.

O ambiente nao depende de Pygame. A interface, o treino e os agentes conversam
com ele usando:

```python
state = env.get_state()
actions = env.get_valid_actions(player_index)
next_state, reward, done, info = env.step(action)
```

## Fases do ambiente

O jogo usa fases para controlar quais acoes sao validas.

Principais fases:

- `ready_to_roll`: jogador pode rolar dados ou fazer acoes financeiras.
- `awaiting_buy`: jogador caiu em propriedade sem dono e pode comprar ou passar.
- `building_trade`: jogador esta montando uma proposta de troca por etapas.
- `auction`: propriedade esta em leilao.
- `pending_trade_response`: outro jogador precisa aceitar ou recusar uma troca.
- `game_over`: partida finalizada.

## Tabuleiro

O tabuleiro fica em `game/board.py`.

Ele possui 40 casas:

- `go`;
- propriedades coloridas;
- ferrovias;
- companhias;
- impostos;
- cartas de chance;
- cartas de community chest;
- cadeia;
- free parking;
- go to jail.

Os grupos de cor ficam em `GROUPS`. O jogador precisa possuir todas as
propriedades de um grupo para construir casas/hotel.

## Regras implementadas

O motor atual implementa:

- passagem pelo Go pagando bonus;
- compra de propriedades;
- leilao quando uma propriedade nao e comprada;
- aluguel de propriedades comuns;
- aluguel dobrado em grupo completo sem casas;
- aluguel escalonado com casas e hotel;
- ferrovias com aluguel escalado pela quantidade possuida;
- companhias com aluguel baseado nos dados;
- impostos;
- cartas de Chance e Community Chest;
- cadeia;
- saida da cadeia por dupla, pagamento ou carta;
- tres duplas seguidas enviam para a cadeia;
- hipoteca e quitar hipoteca;
- juros de hipoteca;
- venda de construcoes pela metade do custo;
- construcao uniforme dentro do grupo;
- limite de 32 casas e 12 hoteis;
- trocas entre jogadores;
- falencia;
- fim por ultimo jogador ativo;
- fim por limite de turnos com vencedor por patrimonio.

## Acoes

As acoes ficam em `game/actions.py`.

Acoes principais:

- `roll_dice`;
- `buy_property`;
- `pass_buy`;
- `auction_bid`;
- `auction_pass`;
- `build_house`;
- `sell_house`;
- `mortgage_property`;
- `unmortgage_property`;
- `pay_jail_fine`;
- `use_jail_card`;
- `start_trade`;
- `select_trade_target`;
- `add_trade_offer_property`;
- `finish_trade_offer`;
- `set_trade_offer_money`;
- `add_trade_request_property`;
- `finish_trade_request`;
- `set_trade_request_money`;
- `submit_trade`;
- `cancel_trade`;
- `accept_trade`;
- `decline_trade`.

As acoes sao dicionarios estruturados. Exemplos:

```python
{"type": "roll_dice"}
```

```python
{"type": "auction_bid", "amount": 120}
```

```python
{"type": "build_house", "property_index": 21}
```

```python
{
    "type": "select_trade_target",
    "target_player": 1,
}
```

```python
{
    "type": "add_trade_offer_property",
    "property_index": 3,
}
```

```python
{
    "type": "set_trade_offer_money",
    "amount": 100,
}
```

```python
{
    "type": "submit_trade",
    "target_player": 1,
    "offer_properties": [3],
    "offer_money": 100,
    "request_properties": [12],
    "request_money": 0,
}
```

## Modo GitHub-style

O arquivo `game/github_monopoly.py` e uma porta do ambiente usado no repositorio
`MonopolyNEAT`.

Principais diferencas em relacao ao ambiente principal:

- o jogo termina em empate apos 300 turnos, como no repositorio;
- a rede recebe um vetor fixo de estado, nao `(estado, acao)`;
- o vetor padrao tem 126 entradas, preservando a configuracao original;
- existe uma posicao 127 no adapter para contexto de dinheiro de troca, mas ela
  fica desligada por padrao porque o repositorio declarou `INPUTS = 126`;
- a rede tem 9 saidas continuas;
- o leilao pede um unico lance de cada jogador;
- as trocas sao geradas aleatoriamente pelo ambiente e a rede decide se oferece
  e se aceita;
- o fitness do treino GitHub-style usa vitorias em torneio eliminatorio, nao
  reward denso.

As 9 saidas da rede sao:

```text
0: comprar propriedade ou mandar para leilao
1: decisao de cadeia: carta, rolar ou pagar
2: hipotecar propriedade selecionada
3: quitar hipoteca da propriedade selecionada
4: valor do lance no leilao, convertido para dinheiro ate 4000
5: quantidade de casas a construir no grupo selecionado
6: quantidade de casas a vender no grupo selecionado
7: propor troca gerada pelo ambiente
8: aceitar troca recebida
```

O treino desse modo fica em `train_neat_github.py`.

Fluxo do torneio GitHub-style:

1. A populacao e embaralhada em grupos de 4.
2. Cada grupo joga `games-per-bracket` partidas.
3. Vitoria soma `1.0`; empate soma `0.25` para cada rede.
4. A melhor rede do grupo avanca uma chave.
5. Ao final, o fitness prioriza o bracket alcancado e usa a taxa de pontos
   nas partidas como desempate seletivo.

A formula atual e:

```text
fitness = bracket * 100 + score_rate * 10
```

Onde `score_rate` e `wins_score / jogos_disputados_pelo_genoma`. Isso evita o
problema em que todos os campeoes de geracao ficavam com fitness igual a `20`.
O score nao vem de reward por jogada, mas de sobreviver e vencer em partidas
completas contra outras redes.

## Regras de troca

As trocas no ambiente principal agora sao montadas por decisoes sequenciais do
agente. O ambiente nao entrega uma proposta estrategica pronta. Ele apenas
oferece as escolhas legais da etapa atual:

1. iniciar troca;
2. escolher jogador alvo;
3. adicionar propriedades oferecidas;
4. definir dinheiro oferecido;
5. adicionar propriedades pedidas;
6. definir dinheiro pedido;
7. enviar ou cancelar a proposta.

Depois do envio, a troca vira `pending_trade_response` e o outro jogador decide
entre `accept_trade` e `decline_trade`.

Regras importantes:

- propriedades de um grupo com casas ou hotel nao podem ser negociadas;
- uma troca precisa ter contrapartida dos dois lados;
- doacoes unilaterais de dinheiro/propriedade sao bloqueadas;
- trocas puramente dinheiro-por-dinheiro sao bloqueadas;
- propriedades hipotecadas podem ser negociadas;
- ao receber propriedade hipotecada, o novo dono paga os juros de transferencia
  quando aplicavel.

Essas regras evitam casos como um jogador doar terreno sem motivo ou transferir
um grupo construido mantendo casas no terreno.

## PPO e trocas

O PPO usa uma politica compartilhada entre os quatro jogadores. Para evitar que
uma recompensa de um jogador contamine a decisao de outro, o treino separa as
trajetorias por jogador dentro de cada partida.

Em cada `env.step`, o ambiente retorna `rewards_by_player` com a mudanca real de
patrimonio de cada jogador. O treino acumula essa recompensa na ultima decisao
do respectivo jogador. Com isso, se um jogador monta uma proposta ruim e outro
aceita depois, o prejuizo volta para a decisao de propor a troca.

O estado tambem codifica o jogador que esta agindo de fato. Em resposta de
troca, `current_player` no estado passa a ser quem decide aceitar ou recusar,
nao apenas o jogador dono do turno original.

Para facilitar trocas emergentes sem entregar propostas prontas, o encoder inclui
features compactas de complementaridade:

- se o jogador atual consegue completar um grupo pegando propriedade de cada
  adversario;
- se o adversario consegue completar um grupo pegando propriedade do jogador
  atual;
- se existe complementaridade mutua, como `2/3 Orange` contra `2/3 Pink`.

O curriculum de trocas cria algumas partidas com estados complementares, mas o
agente ainda precisa decidir iniciar, montar, enviar, aceitar ou recusar a troca.

Durante o treino, o campo `trades` no log usa o formato:

```text
trades=inicios/envios/cancelamentos/aceites/recusas
```

Exemplo: `trades=8/2/5/1/1` significa que o agente iniciou 8 montagens de troca,
enviou 2 propostas, cancelou 5, teve 1 aceite e 1 recusa.

## Anti-loops financeiros

Para evitar loops artificiais como:

```text
construir -> vender -> construir -> vender
hipotecar -> quitar -> hipotecar -> quitar
```

o ambiente restringe algumas acoes:

- vender construcoes e hipotecar so aparecem como acoes livres quando o jogador
  esta com dinheiro negativo;
- quitar hipoteca exige dinheiro suficiente e uma reserva de caixa;
- uma propriedade que vendeu construcao no turno nao pode construir de novo no
  mesmo turno;
- uma construcao feita no turno nao pode ser vendida no mesmo turno, exceto em
  liquidacao automatica por divida.

## Leiloes

Quando o jogador cai em uma propriedade sem dono e escolhe `pass_buy`, o jogo
inicia um leilao.

O leilao tem teto de lance racional. O ambiente nao oferece lances que gastam
todo o dinheiro do jogador sem motivo.

O teto considera:

- preco da propriedade;
- reserva minima de caixa;
- se a propriedade completa um grupo;
- se a propriedade bloqueia monopolio de outro jogador;
- quantidade de ferrovias/companhias ja possuidas.

Isso evita que a exploracao inicial do DQN destrua o jogador em lances absurdos.

## Replay e voltar jogada

Na visualizacao, o ambiente e criado com `enable_undo=True`.

Antes de cada `env.step(action)`, o ambiente salva um snapshot completo:

- tabuleiro;
- jogadores;
- dinheiro;
- propriedades;
- casas/hoteis;
- hipotecas;
- fase;
- leilao;
- troca pendente;
- estado interno do random.

Com isso, a tecla `U` ou `Backspace` volta uma jogada real, nao apenas o texto
do historico.

## Visualizacao

A interface fica em `ui/pygame_view.py`.

Controles:

- `N` ou seta para direita: executa uma acao da IA;
- `SPACE`: pausa ou retoma modo automatico;
- `+`: aumenta a velocidade;
- `-`: diminui a velocidade;
- `U` ou `Backspace`: volta uma jogada;
- `R`: reinicia;
- `ESC`: fecha.

A lateral mostra:

- jogador atual;
- fase;
- dados;
- dinheiro dos jogadores;
- quantidade de propriedades;
- construcoes;
- casa atual;
- ultima mensagem;
- historico recente.

## Agente neural

O agente neural fica em `agents/neural_agent.py`.

Ele usa DQN, mas com uma diferenca importante: a rede nao tem uma saida fixa
para cada acao. Em vez disso, ela aprende:

```text
Q(estado, acao)
```

Ou seja, a rede recebe o vetor do estado concatenado com o vetor de uma acao
candidata e devolve um unico valor Q.

Isso permite avaliar acoes estruturadas, como:

- proposta de troca com propriedades e dinheiro;
- lance de leilao com valor;
- construcao em uma propriedade especifica;
- hipoteca de uma propriedade especifica.

Durante a escolha:

1. O ambiente gera as acoes validas.
2. O agente codifica o estado.
3. O agente codifica cada acao valida.
4. A rede calcula Q para cada par `(estado, acao)`.
5. O agente escolhe a acao com maior Q.

Durante treino, tambem existe exploracao epsilon-greedy.

## Agente aleatorio/heuristico

O arquivo `agents/random_agent.py` contem um agente simples baseado em regras.

Ele nao e puramente aleatorio. Ele tende a:

- comprar propriedades;
- construir quando possivel;
- pagar/usar carta para sair da cadeia em algumas situacoes;
- participar de leiloes;
- aceitar algumas trocas;
- usar hipoteca/venda apenas defensivamente.

Por isso, no campeonato ele e chamado de `heuristic_baseline`.

O campeonato tambem tem `pure_random_baseline`, que escolhe qualquer acao valida
ao acaso.

## Encoder do DQN

O arquivo `game/encoders.py` transforma estado e acao em vetores numericos.

O estado inclui:

- jogador atual;
- fase;
- dados;
- duplas consecutivas;
- dinheiro;
- posicao;
- falencia;
- cadeia;
- turnos na cadeia;
- cartas de sair da cadeia;
- donos das propriedades;
- casas/hoteis;
- hipotecas;
- troca em montagem ou troca pendente;
- casas/hoteis restantes no banco;
- turno atual.

A acao inclui:

- tipo da acao;
- jogador alvo;
- propriedades oferecidas em troca;
- propriedades pedidas em troca;
- propriedade alvo da acao;
- dinheiro oferecido/pedido;
- valor de lance.

Se `STATE_SIZE` ou `ACTION_SIZE` mudar, modelos antigos ficam incompativeis e
precisam ser treinados novamente.

## Treino DQN

O treino fica em `train_dqn.py`.

Fluxo de cada episodio:

1. Cria um ambiente novo.
2. Calcula o epsilon do episodio.
3. Enquanto a partida nao termina:
   - pega o jogador que deve agir;
   - busca acoes validas;
   - escolhe acao com epsilon-greedy;
   - aplica `env.step(action)`;
   - salva transicao no replay buffer;
   - periodicamente otimiza a rede.
4. Atualiza a target network periodicamente.
5. Salva modelo/checkpoint periodicamente.
6. Opcionalmente roda campeonato.

## Replay buffer

O replay buffer fica em `agents/replay_buffer.py`.

Cada transicao salva:

- vetor `(estado, acao)`;
- reward;
- proximo estado;
- proximas acoes validas;
- se a partida acabou.

Durante otimizacao, o treino amostra batches aleatorios do buffer. Isso reduz a
correlacao entre jogadas consecutivas e deixa o DQN mais estavel.

## Loss function

A loss fica em `train_dqn.py`.

Ela usa Smooth L1 Loss, tambem chamada de Huber loss:

```python
loss_fn = nn.SmoothL1Loss()
loss = loss_fn(current_q, expected_q)
```

O valor atual:

```text
current_q = Q(estado, acao)
```

O alvo:

```text
expected_q = reward + gamma * max Q(proximo_estado, proxima_acao)
```

Em jogo multiagente, existe um ajuste adversarial:

- se o proximo jogador a agir for o mesmo jogador, o melhor Q futuro entra como
  oportunidade;
- se o proximo jogador for um adversario, o melhor Q dele entra como ameaca e
  e subtraido parcialmente do alvo.

Na pratica:

```text
mesmo jogador:
expected_q = reward + gamma * max Q(proximo_estado, minha_proxima_acao)

adversario:
expected_q = reward - gamma * OPPONENT_Q_WEIGHT * max Q(proximo_estado, acao_do_adversario)
```

Isso faz o DQN levar em conta que uma acao aparentemente boa pode entregar uma
resposta forte para outro jogador.

Se a partida acabou:

```text
expected_q = reward
```

Entao a loss mede o erro entre o valor Q previsto e o alvo calculado.

## Reward

O reward combina:

- mudanca de patrimonio liquido;
- bonus por comprar propriedade;
- bonus por construir;
- bonus por completar grupo;
- bonus/penalidade final por vencer ou perder.

O patrimonio usado no reward nao usa sentinelas artificiais de falencia. Isso
evita rewards gigantes negativos como `-30000`.

## Epsilon

O treino usa epsilon-greedy:

- com probabilidade epsilon, escolhe acao aleatoria;
- caso contrario, escolhe maior Q.

Atualmente:

- `epsilon_start = 1.0`;
- `epsilon_end = 0.05`;
- decai ate o minimo em 40% dos episodios.

Isso reduz mais cedo a fase em que os agentes tomam decisoes quase totalmente
aleatorias.

## Hiperparametros atuais

Para reduzir instabilidade do DQN, o treino usa:

- `LEARNING_RATE = 3e-5`;
- `BATCH_SIZE = 128`;
- `OPTIMIZE_EVERY_STEPS = 20`;
- `TARGET_UPDATE_EVERY = 50`;
- `GAMMA = 0.98`;
- `OPPONENT_Q_WEIGHT = 0.75`;
- reward clipado entre `-10` e `+10` antes de entrar no replay buffer.

O reward bruto ainda vem do ambiente, mas o valor usado para treinar e somar no
log do episodio e limitado. Isso reduz saltos grandes nos Q-values.

## Checkpoints

O modelo principal e salvo em:

```bash
models/dqn_agent.pt
```

Checkpoints periodicos sao salvos em:

```bash
models/checkpoints/
```

Exemplo:

```bash
models/checkpoints/dqn_ep_000500.pt
```

O melhor checkpoint encontrado pelo campeonato e salvo em:

```bash
models/best_dqn_agent.pt
```

## Campeonato hibrido

O campeonato fica em `tournament.py`.

Ele nao substitui a loss do DQN. Ele serve para avaliar desempenho real em
partidas completas.

Competidores:

- checkpoints DQN;
- `pure_random_baseline`;
- `heuristic_baseline`.

Metricas:

- `jogos`: partidas jogadas;
- `vitorias`: partidas vencidas;
- `win%`: taxa de vitoria;
- `pos_med`: posicao media final;
- `patrimonio`: patrimonio medio final;
- `score`: pontuacao normalizada do campeonato.

O score prioriza taxa de vitoria, usa posicao media como desempate e patrimonio
medio como sinal auxiliar.

Isso permite escolher modelos pelo desempenho real, nao apenas pela loss.

## Observacoes importantes

- Modelos antigos podem ficar incompativeis depois de mudancas no estado ou nas
  acoes.
- A loss so mede erro de Q-value; ela nao garante que o agente joga melhor.
- Para avaliar aprendizado em jogos longos, use tambem campeonato, win rate,
  patrimonio medio e comportamento observado.
- O agente ainda nao faz minimax, expectimax ou modelagem explicita dos outros
  jogadores. Esse comportamento pode emergir por self-play, mas nao e planejado
  diretamente.
