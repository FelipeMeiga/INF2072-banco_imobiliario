# Banco Imobiliario IA com DQN

Projeto de simulacao de um jogo estilo Monopoly/Banco Imobiliario, com motor de
regras separado da interface e agentes treinados por DQN.

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
- `torch`: rede neural, DQN e treino.

## Como rodar

Assistir uma partida com agentes neurais:

```bash
python main.py
```

Treinar o DQN:

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
python tournament.py --checkpoints models/checkpoints --games 100 --latest 8
```

## Estrutura do projeto

```text
t1/
|-- main.py
|-- train_dqn.py
|-- tournament.py
|-- requirements.txt
|-- game/
|   |-- actions.py
|   |-- board.py
|   |-- encoders.py
|   |-- env.py
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
- `propose_trade`;
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
    "type": "propose_trade",
    "target_player": 1,
    "offer_properties": [3, 6],
    "offer_money": 100,
    "request_properties": [12],
    "request_money": 0,
}
```

## Regras de troca

As trocas sao validadas pelo ambiente antes de virarem troca pendente.

Regras importantes:

- propriedades de um grupo com casas ou hotel nao podem ser negociadas;
- uma troca precisa ter contrapartida dos dois lados;
- doacoes unilaterais de dinheiro/propriedade sao bloqueadas;
- propriedades hipotecadas podem ser negociadas;
- ao receber propriedade hipotecada, o novo dono paga os juros de transferencia
  quando aplicavel.

Essas regras evitam casos como um jogador doar terreno sem motivo ou transferir
um grupo construido mantendo casas no terreno.

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
