# Banco Imobiliário IA

Projeto base para um Banco Imobiliário onde agentes de IA jogam entre si.

Nesta versão:

- o motor do jogo fica separado da interface;
- a interface Pygame apenas visualiza a partida;
- os jogadores são agentes aleatórios;
- o jogo aceita ações estruturadas via `env.step(action)`;
- já existe suporte básico a compra, aluguel, imposto, sorte/revés, falência e trocas.

## Instalação

```bash
pip install -r requirements.txt
```

## Rodar

```bash
python main.py
```

## Estrutura

```text
banco_imobiliario_ai/
├── main.py
├── requirements.txt
├── game/
│   ├── actions.py
│   ├── board.py
│   ├── env.py
│   └── models.py
├── agents/
│   └── random_agent.py
└── ui/
    └── pygame_view.py
```

## Exemplo de ação de troca

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

## Próximos passos naturais

1. Criar um agente baseado em regras.
2. Transformar `get_state()` em vetor numérico para rede neural.
3. Definir uma função de recompensa melhor.
4. Criar um loop de treino sem Pygame para rodar milhares de partidas rápido.

## Agente neural com DQN

Esta versão também inclui um agente neural simples baseado em DQN.

A rede não escolhe apenas um número de ação. Ela aprende uma função Q(estado, ação),
o que permite avaliar ações estruturadas, inclusive propostas de troca com propriedades e dinheiro.

### Treinar o modelo

```bash
python train_dqn.py --episodes 300
```

## Campeonato de checkpoints DQN

O treino pode salvar checkpoints e comparar modelos em um campeonato de self-play.
A loss continua treinando o DQN, mas o campeonato mede desempenho real em partidas:
vitorias, colocacao media e patrimonio medio.

Treinar salvando checkpoints:

```bash
python train_dqn.py --episodes 1000 --checkpoint-every 100
```

Treinar e rodar campeonato periodico entre os checkpoints mais recentes:

```bash
python train_dqn.py --episodes 1000 --checkpoint-every 100 --tournament-every 200 --tournament-games 40
```

Rodar campeonato manual depois do treino:

```bash
python tournament.py --checkpoints models/checkpoints --games 100 --latest 8
```

O campeonato inclui dois baselines:

- `pure_random_baseline`: escolhe qualquer acao valida ao acaso.
- `heuristic_baseline`: usa o agente simples baseado em regras.

Quando o campeonato roda durante o treino, o melhor checkpoint DQN encontrado e
salvo automaticamente em:

```bash
models/best_dqn_agent.pt
```

O epsilon agora decai ate o minimo em 40% dos episodios, para reduzir a fase em
que os agentes tomam decisoes quase totalmente aleatorias.

O modelo será salvo em:

```bash
models/dqn_agent.pt
```

### Assistir a partida com agente neural

```bash
python main_neural.py
```

Se o modelo treinado ainda não existir, o programa roda uma rede não treinada com exploração aleatória.


## Controles da visualização

A visualização agora inicia pausada para permitir acompanhar cada decisão dos agentes com calma.

- `N` ou seta para direita: executa uma ação da IA
- `SPACE`: alterna entre pausado e automático
- `+`: aumenta a velocidade no modo automático
- `-`: diminui a velocidade no modo automático
- `R`: reinicia a partida
- `ESC`: fecha a janela

A lateral da tela mostra o último acontecimento e um histórico recente das últimas ações.

## Mecânica de regiões, casas e hotel

As propriedades comuns agora pertencem a uma região/país, como Brasil, Argentina,
Chile, Uruguai, Portugal, Espanha, França e Itália.

Quando um jogador possui todas as propriedades de uma mesma região, ele libera a
ação:

```python
{"type": "build_house", "property_index": 21}
```

Cada propriedade pode receber até 5 níveis de construção:

- 0 = sem casa
- 1 a 4 = casas
- 5 = hotel

Construir custa dinheiro e aumenta muito o aluguel daquela propriedade. Além disso,
quando um jogador possui a região completa, o aluguel sem casas também dobra.

Essa mecânica foi adicionada para aumentar a pressão econômica e reduzir partidas
que só terminam pelo limite de turnos.

Depois dessa mudança, modelos treinados antes podem ficar incompatíveis com a nova
entrada da rede neural. Refaça o treinamento com:

```bash
python train_dqn.py --episodes 300
```
