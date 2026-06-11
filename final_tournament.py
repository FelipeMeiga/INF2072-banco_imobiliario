import os
import json
import time
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tournament import load_checkpoint_competitor, run_tournament, Competitor, CompetitorStats

def championship_score(stats):
    games = stats.get("games", 1)
    wins = stats.get("wins", 0)
    total_rank = stats.get("total_rank", 0)
    total_net = stats.get("total_net_worth", 0.0)
    win_rate = wins / games if games else 0.0
    avg_rank = total_rank / games if games else 0.0
    avg_net = total_net / games if games else 0.0
    return (win_rate * 10000.0) - (avg_rank * 100.0) + (avg_net / 10.0)

def main():
    parser = argparse.ArgumentParser(description="Rodar o torneio final entre os melhores modelos.")
    parser.add_argument("--games", type=int, default=1000, help="Número de jogos no torneio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade.")
    parser.add_argument("--output", type=str, default="results/final_tournament_results.json", help="Caminho do JSON de saída.")
    parser.add_argument("--out-dir", type=str, default="reports/final_tournament", help="Diretório de saída dos gráficos.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Definir caminhos para os modelos de forma robusta
    ppo_path = "models/ppo_raw_ep_002550.pt"
    
    # Buscar o checkpoint NEAT
    neat_path = None
    for path in ["models/neat_hof_gen_000029_fit_3231.pkl"]:
        if os.path.exists(path):
            neat_path = path
            break
    if not neat_path:
        raise FileNotFoundError("Nenhum arquivo de checkpoint NEAT encontrado em models/")

    dqn_path = None
    for path in ["models/dqn_ep_000150.pt"]:
        if os.path.exists(path):
            dqn_path = path
            break
    if not dqn_path:
        raise FileNotFoundError("Nenhum arquivo de checkpoint DQN encontrado em models/")

    print(f"Carregando PPO de: {ppo_path}")
    print(f"Carregando NEAT de: {neat_path}")
    print(f"Carregando DQN de: {dqn_path}")

    # Carregar competidores
    competitors = []
    
    comp_ppo = load_checkpoint_competitor(ppo_path, device=device, verbose=True)
    if comp_ppo is None:
        raise ValueError("Falha ao carregar o competitor PPO")
    competitors.append(comp_ppo)
    
    comp_neat = load_checkpoint_competitor(neat_path, device=device, verbose=True)
    if comp_neat is None:
        raise ValueError("Falha ao carregar o competitor NEAT")
    competitors.append(comp_neat)
    
    comp_dqn = load_checkpoint_competitor(dqn_path, device=device, verbose=True)
    if comp_dqn is None:
        raise ValueError("Falha ao carregar o competitor DQN")
    competitors.append(comp_dqn)

    # Adicionar o competidor Heurístico
    comp_heur = Competitor(name="heuristic", kind="heuristic")
    competitors.append(comp_heur)

    print(f"Iniciando torneio final com {args.games} jogos entre:")
    for c in competitors:
        print(f" - {c.name} ({c.kind})")

    # Rodar torneio
    # include_baselines=False para garantir que apenas os 4 competidores joguem entre si em todas as partidas
    stats = run_tournament(
        competitors=competitors,
        games=args.games,
        seed=args.seed,
        device=device,
        include_baselines=False,
        progress_every=50,
    )

    # Converter stats em dicionário serializável
    serialized_stats = {}
    for name, s in stats.items():
        serialized_stats[name] = {
            "games": s.games,
            "wins": s.wins,
            "total_rank": s.total_rank,
            "total_net_worth": s.total_net_worth,
            "total_turns": s.total_turns,
            "action_counts": s.action_counts,
        }

    # Salvar em JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({
            "metadata": {
                "games": args.games,
                "seed": args.seed,
                "device": device,
                "timestamp": time.time()
            },
            "stats": serialized_stats
        }, fh, indent=2)
    print(f"Resultados do torneio salvos em: {output_path}")

    # LER O ARQUIVO JSON SALVO PARA GERAR OS GRÁFICOS
    print("Lendo o arquivo JSON para gerar os gráficos...")
    with open(output_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    
    stats_data = data.get("stats", {})
    
    # Calcular scores e preparar dados para plotagem
    scored_competitors = []
    for name, s in stats_data.items():
        score = championship_score(s)
        win_rate = (s.get("wins", 0) / s.get("games", 1)) * 100.0
        scored_competitors.append((name, score, win_rate, s))
        
    # Ordenar por score de campeonato (decrescente)
    scored_competitors.sort(key=lambda x: x[1], reverse=True)
    
    # Criar pasta de gráficos
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    names = [item[0] for item in scored_competitors]
    win_rates = [item[2] for item in scored_competitors]
    scores = [item[1] for item in scored_competitors]
    
    # 1. Gráfico de Win Rate
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, win_rates, color=['#4CAF50', '#2196F3', '#FFC107', '#9C27B0'][:len(names)])
    plt.ylabel('Taxa de Vitória (Win Rate %)')
    plt.title(f'Comparação de Win Rate - Torneio Final ({args.games} Jogos)')
    plt.ylim(0, max(win_rates) * 1.15 if win_rates else 100)
    
    # Adicionar rótulos de valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 pontos de deslocamento vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    plt.tight_layout()
    winrate_path = out_dir / "final_winrate_comparison.png"
    plt.savefig(winrate_path)
    plt.close()
    
    # 2. Gráfico de Championship Score
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, scores, color=['#4CAF50', '#2196F3', '#FFC107', '#9C27B0'][:len(names)])
    plt.ylabel('Score de Campeonato')
    plt.title('Comparação de Championship Score')
    plt.ylim(min(0, min(scores) * 1.15) if scores else 0, max(scores) * 1.15 if scores else 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    plt.tight_layout()
    score_path = out_dir / "final_score_comparison.png"
    plt.savefig(score_path)
    plt.close()
    
    print("Tabela de Resultados:")
    print("rank | competitor       | jogos | vitorias | win%   | pos_med | patrimonio | score")
    for rank, (name, score, win_rate, s) in enumerate(scored_competitors, start=1):
        games = s.get("games", 1)
        avg_rank = s.get("total_rank", 0) / games
        avg_net = s.get("total_net_worth", 0.0) / games
        print(
            f"{rank:04d} | "
            f"{name[:16]:16s} | "
            f"{games:5d} | "
            f"{s.get('wins', 0):8d} | "
            f"{win_rate:5.1f} | "
            f"{avg_rank:7.2f} | "
            f"{avg_net:10.2f} | "
            f"{score:8.2f}"
        )
        
    print(f"Gráficos salvos em: {out_dir}")

if __name__ == "__main__":
    main()
