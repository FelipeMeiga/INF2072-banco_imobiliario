<<<<<<< HEAD
# INF2072-banco_imobiliario
=======
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
>>>>>>> 910e8bd (init commit)
