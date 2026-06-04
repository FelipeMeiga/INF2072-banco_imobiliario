import os

import pygame

from game.board import NUM_SPACES

WIDTH = 1320
HEIGHT = 800
FPS = 60

BOARD_SIZE = 680
BOARD_X = 40
BOARD_Y = 60

SIDE_PANEL_X = 760
SIDE_PANEL_Y = 60
SIDE_PANEL_WIDTH = 520
SIDE_PANEL_HEIGHT = 680

WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (190, 190, 190)
DARK_GRAY = (80, 80, 80)
LIGHT_GRAY = (225, 225, 225)
GREEN = (50, 170, 80)
RED = (210, 60, 60)
YELLOW = (230, 190, 60)
ORANGE = (240, 140, 40)
BROWN = (120, 70, 30)
PANEL_BG = (255, 252, 238)


class PygameView:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Banco Imobiliário IA - Simulação")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 26, bold=True)
        self.small_font = pygame.font.SysFont("arial", 20)
        self.tiny_font = pygame.font.SysFont("arial", 14)
        self.timeline_rect = pygame.Rect(70, HEIGHT - 34, WIDTH - 140, 8)
        self.timeline_hit_rect = self.timeline_rect.inflate(24, 28)
        self.timeline_total = 0
        self.dragging_timeline = False

    def handle_events(self) -> dict:
        commands = {
            "running": True,
            "toggle_pause": False,
            "step_once": False,
            "speed_up": False,
            "speed_down": False,
            "reset": False,
            "undo": False,
            "save_replay": False,
            "load_latest": False,
            "seek_index": None,
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                commands["running"] = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    commands["running"] = False
                elif event.key == pygame.K_SPACE:
                    commands["toggle_pause"] = True
                elif event.key in (pygame.K_n, pygame.K_RIGHT):
                    commands["step_once"] = True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    commands["speed_up"] = True
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    commands["speed_down"] = True
                elif event.key == pygame.K_r:
                    commands["reset"] = True
                elif event.key in (pygame.K_u, pygame.K_BACKSPACE):
                    commands["undo"] = True
                elif event.key == pygame.K_s:
                    commands["save_replay"] = True
                elif event.key == pygame.K_l:
                    commands["load_latest"] = True

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.timeline_total > 0 and self.timeline_hit_rect.collidepoint(event.pos):
                    self.dragging_timeline = True
                    commands["seek_index"] = self._timeline_index_from_x(event.pos[0])

            if event.type == pygame.MOUSEMOTION:
                if self.dragging_timeline and self.timeline_total > 0:
                    commands["seek_index"] = self._timeline_index_from_x(event.pos[0])

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self.dragging_timeline and self.timeline_total > 0:
                    commands["seek_index"] = self._timeline_index_from_x(event.pos[0])
                self.dragging_timeline = False

        return commands

    def _timeline_index_from_x(self, x):
        if self.timeline_total <= 0 or self.timeline_rect.width <= 0:
            return 0
        ratio = (x - self.timeline_rect.x) / float(self.timeline_rect.width)
        ratio = max(0.0, min(1.0, ratio))
        return int(round(ratio * self.timeline_total))

    def draw(
        self,
        env,
        paused=False,
        step_delay=None,
        replay_cursor=0,
        replay_total=0,
        replay_path=None,
        replay_locked=False,
    ):
        self.timeline_total = max(0, int(replay_total))
        self.screen.fill((35, 120, 70))
        self._draw_board(env)
        self._draw_center_trade(env)
        self._draw_side_panel(env, paused=paused, step_delay=step_delay)
        self._draw_timeline(
            replay_cursor=replay_cursor,
            replay_total=replay_total,
            replay_path=replay_path,
            replay_locked=replay_locked,
        )
        pygame.display.flip()
        self.clock.tick(FPS)

    def quit(self):
        pygame.quit()

    def _draw_board(self, env):
        pygame.draw.rect(self.screen, WHITE, (BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE))
        pygame.draw.rect(self.screen, BLACK, (BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE), 3)

        for i, space in enumerate(env.board):
            rect = get_space_rect(i)

            if space.type == "go":
                color = YELLOW
            elif space.type in ("property", "railroad", "utility"):
                color = WHITE
            elif space.type == "tax":
                color = ORANGE
            elif space.type in ("chance", "community_chest"):
                color = GREEN
            elif space.type == "jail":
                color = GRAY
            else:
                color = LIGHT_GRAY

            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 1)

            if space.owner is not None:
                owner_color = env.players[space.owner].color
                owner_bar = pygame.Rect(rect.x, rect.y, rect.width, 8)
                pygame.draw.rect(self.screen, owner_color, owner_bar)

            index_text = self.tiny_font.render(str(i), True, BLACK)
            self.screen.blit(index_text, (rect.x + 3, rect.y + 3))

            name_text = self.tiny_font.render(space.name[:8], True, BLACK)
            self.screen.blit(name_text, (rect.x + 3, rect.y + 18))

            if space.type in ("property", "railroad", "utility"):
                price_text = self.tiny_font.render(f"${space.price}", True, DARK_GRAY)
                self.screen.blit(price_text, (rect.x + 3, rect.y + 34))

                if getattr(space, "houses", 0) > 0:
                    self._draw_construction_indicator(rect, space)

        self._draw_players(env)

    def _draw_construction_indicator(self, rect, space):
        houses = getattr(space, "houses", 0)

        if houses >= 5:
            label = "HOTEL"
            color = RED
        else:
            label = f"{houses}C"
            color = BROWN

        indicator = pygame.Rect(rect.x + 3, rect.bottom - 16, rect.width - 6, 13)
        pygame.draw.rect(self.screen, color, indicator, border_radius=3)
        text = self.tiny_font.render(label, True, WHITE)
        text_rect = text.get_rect(center=indicator.center)
        self.screen.blit(text, text_rect)

    def _draw_players(self, env):
        offsets = [
            (16, 16),
            (38, 16),
            (16, 38),
            (38, 38),
        ]

        for i, player in enumerate(env.players):
            if player.bankrupt:
                continue

            rect = get_space_rect(player.position)
            ox, oy = offsets[i % len(offsets)]

            pygame.draw.circle(
                self.screen,
                player.color,
                (rect.x + ox, rect.y + oy),
                9,
            )
            pygame.draw.circle(
                self.screen,
                BLACK,
                (rect.x + ox, rect.y + oy),
                9,
                2,
            )

    def _draw_center_trade(self, env):
        cell = BOARD_SIZE // 11
        inner_rect = pygame.Rect(
            BOARD_X + cell + 18,
            BOARD_Y + cell + 18,
            BOARD_SIZE - (cell + 18) * 2,
            BOARD_SIZE - (cell + 18) * 2,
        )
        panel_width = min(470, inner_rect.width)
        panel_height = min(390, inner_rect.height)
        panel = pygame.Rect(0, 0, panel_width, panel_height)
        panel.center = inner_rect.center

        shadow = panel.move(6, 7)
        pygame.draw.rect(self.screen, (25, 55, 35), shadow, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BG, panel, border_radius=8)
        pygame.draw.rect(self.screen, BLACK, panel, 2, border_radius=8)

        pending_trade = getattr(env, "pending_trade", None)
        trade_draft = getattr(env, "trade_draft", None)
        last_trade_result = getattr(env, "last_trade_result", None)
        trade = pending_trade or trade_draft or last_trade_result

        y = panel.y + 16
        title = self.font.render("Trocas", True, BLACK)
        title_rect = title.get_rect(center=(panel.centerx, y + 16))
        self.screen.blit(title, title_rect)
        y += 42

        if not trade:
            return

        proposer = env.players[trade["from"]]
        target_index = trade.get("to")
        target = env.players[target_index] if target_index is not None else None

        if pending_trade:
            subtitle_text = f"{proposer.name} -> {target.name}"
            subtitle_color = DARK_GRAY
        elif trade_draft:
            target_name = target.name if target is not None else "alvo..."
            subtitle_text = f"Montando: {proposer.name} -> {target_name}"
            subtitle_color = DARK_GRAY
        else:
            status = trade.get("status")
            status_text = "aceita" if status == "accepted" else "recusada"
            subtitle_text = f"Ultima troca: {status_text}"
            subtitle_color = GREEN if status == "accepted" else RED

        subtitle = self.small_font.render(subtitle_text, True, subtitle_color)
        subtitle_rect = subtitle.get_rect(center=(panel.centerx, y + 12))
        self.screen.blit(subtitle, subtitle_rect)
        y += 34

        if not pending_trade:
            matchup = self.tiny_font.render(
                f"{proposer.name} -> {target.name if target is not None else 'alvo...'}",
                True,
                DARK_GRAY,
            )
            matchup_rect = matchup.get_rect(center=(panel.centerx, y - 5))
            self.screen.blit(matchup, matchup_rect)

        column_gap = 18
        column_width = (panel.width - 44 - column_gap) // 2
        left = pygame.Rect(panel.x + 18, y, column_width, panel.bottom - y - 18)
        right = pygame.Rect(left.right + column_gap, y, column_width, left.height)

        self._draw_trade_column(
            title="Oferece",
            player=proposer,
            properties=trade["offer_properties"],
            money=trade["offer_money"],
            env=env,
            rect=left,
        )
        self._draw_trade_column(
            title="Pede",
            player=target,
            properties=trade["request_properties"],
            money=trade["request_money"],
            env=env,
            rect=right,
        )

    def _draw_trade_column(self, title, player, properties, money, env, rect):
        player_color = player.color if player is not None else DARK_GRAY

        pygame.draw.rect(self.screen, WHITE, rect, border_radius=6)
        pygame.draw.rect(self.screen, player_color, rect, 2, border_radius=6)

        y = rect.y + 10
        title_text = self.small_font.render(title, True, player_color)
        self.screen.blit(title_text, (rect.x + 10, y))
        y += 28

        money_text = self.small_font.render(f"Dinheiro: ${money}", True, BLACK)
        self.screen.blit(money_text, (rect.x + 10, y))
        y += 30

        props_title = self.tiny_font.render("Itens:", True, DARK_GRAY)
        self.screen.blit(props_title, (rect.x + 10, y))
        y += 20

        if properties:
            for prop_index in properties[:7]:
                space = env.board[prop_index]
                prefix = f"{prop_index:02d}"
                line = f"{prefix} - {space.name}"
                y = draw_wrapped_text(
                    self.screen,
                    line,
                    rect.x + 10,
                    y,
                    rect.width - 20,
                    self.tiny_font,
                    BLACK,
                    16,
                    max_lines=2,
                )
                y += 3
                if y > rect.bottom - 22:
                    more = self.tiny_font.render("...", True, DARK_GRAY)
                    self.screen.blit(more, (rect.x + 10, rect.bottom - 20))
                    break
        else:
            none_text = self.tiny_font.render("Nenhuma propriedade", True, DARK_GRAY)
            self.screen.blit(none_text, (rect.x + 10, y))

    def _draw_side_panel(self, env, paused=False, step_delay=None):
        panel = pygame.Rect(SIDE_PANEL_X, SIDE_PANEL_Y, SIDE_PANEL_WIDTH, SIDE_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, panel, border_radius=12)
        pygame.draw.rect(self.screen, BLACK, panel, 2, border_radius=12)

        y = SIDE_PANEL_Y + 20
        title = self.font.render("Banco Imobiliário IA", True, BLACK)
        self.screen.blit(title, (SIDE_PANEL_X + 20, y))
        y += 45

        current = env.current_player
        turn_text = self.small_font.render(
            f"Turno: {current.name} | Fase: {env.phase}",
            True,
            current.color,
        )
        self.screen.blit(turn_text, (SIDE_PANEL_X + 20, y))
        y += 30

        dice_text = self.small_font.render(f"Dados: {env.dice[0]} + {env.dice[1]}", True, BLACK)
        self.screen.blit(dice_text, (SIDE_PANEL_X + 20, y))
        y += 35

        status = "PAUSADO" if paused else "AUTO"
        speed_text = "" if step_delay is None else f" | delay: {step_delay:.2f}s"
        turn_count = self.small_font.render(
            f"Turnos: {env.turn_count} | Ações: {getattr(env, 'action_count', 0)}",
            True,
            BLACK,
        )
        self.screen.blit(turn_count, (SIDE_PANEL_X + 20, y))
        y += 28

        mode_text = self.tiny_font.render(
            f"Modo: {status}{speed_text}",
            True,
            DARK_GRAY,
        )
        self.screen.blit(mode_text, (SIDE_PANEL_X + 20, y))
        y += 24

        controls = self.tiny_font.render(
            "SPACE pausa | N/→ avança | +/- velocidade | R reinicia",
            True,
            DARK_GRAY,
        )
        controls = self.tiny_font.render(
            "SPACE pausa | N/-> avanca | U volta | S salva | L replay | +/- vel | R reinicia",
            True,
            DARK_GRAY,
        )
        self.screen.blit(controls, (SIDE_PANEL_X + 20, y))
        y += 28

        players_title = self.small_font.render("Jogadores:", True, BLACK)
        self.screen.blit(players_title, (SIDE_PANEL_X + 20, y))
        y += 28

        for i, player in enumerate(env.players):
            props = len(env.get_owned_properties(i))
            houses = sum(getattr(space, "houses", 0) for space in env.board if getattr(space, "owner", None) == i)
            status = "FALIDO" if player.bankrupt else f"${player.money} | props: {props} | constr: {houses}"
            text = self.small_font.render(f"{player.name}: {status}", True, player.color)
            self.screen.blit(text, (SIDE_PANEL_X + 20, y))
            y += 26

        y += 15

        if getattr(env, "pending_trade", None):
            y = self._draw_pending_trade(env, y)
        elif getattr(env, "trade_draft", None):
            y = self._draw_trade_draft(env, y)
        else:
            y = self._draw_current_space(env, y)

        y += 15
        message_title = self.small_font.render("Último acontecimento:", True, BLACK)
        self.screen.blit(message_title, (SIDE_PANEL_X + 20, y))
        y += 25

        y = draw_wrapped_text(
            self.screen,
            env.last_message,
            SIDE_PANEL_X + 20,
            y,
            SIDE_PANEL_WIDTH - 40,
            self.tiny_font,
            BLACK,
            18,
            max_lines=3,
        )

        y += 12
        history_title = self.small_font.render("Histórico recente:", True, BLACK)
        self.screen.blit(history_title, (SIDE_PANEL_X + 20, y))
        y += 24

        for event in getattr(env, "event_history", [])[-6:]:
            line = f"#{event['number']} T{event['turn']} {event['player']}: {event['message']}"
            y = draw_wrapped_text(
                self.screen,
                line,
                SIDE_PANEL_X + 20,
                y,
                SIDE_PANEL_WIDTH - 40,
                self.tiny_font,
                DARK_GRAY,
                16,
                max_lines=2,
            )
            y += 4

        if env.done:
            winner_text = self.font.render("JOGO FINALIZADO", True, RED)
            self.screen.blit(winner_text, (SIDE_PANEL_X + 20, SIDE_PANEL_Y + SIDE_PANEL_HEIGHT - 55))

    def _draw_current_space(self, env, y):
        current = env.current_player
        space = env.board[current.position]

        title = self.small_font.render("Casa atual:", True, BLACK)
        self.screen.blit(title, (SIDE_PANEL_X + 20, y))
        y += 25

        lines = [
            f"{space.name}",
            f"Tipo: {space.type}",
        ]

        if space.type in ("property", "railroad", "utility"):
            owner_name = "Nenhum"
            if space.owner is not None:
                owner_name = env.players[space.owner].name
            construction = "hotel" if getattr(space, "houses", 0) >= 5 else f"{getattr(space, 'houses', 0)} casas"
            current_rent = env.get_rent_for_space(space) if hasattr(env, "get_rent_for_space") else (space.current_rent() if hasattr(space, "current_rent") else space.rent)
            lines.extend([
                f"Região: {space.group or '-'}",
                f"Preço: ${space.price}",
                f"Aluguel atual: ${current_rent}",
                f"Construção: {construction}",
                f"Custo construção: ${getattr(space, 'build_cost', 0)}",
                f"Hipotecada: {'sim' if getattr(space, 'mortgaged', False) else 'nao'}",
                f"Dono: {owner_name}",
            ])

        for line in lines:
            text = self.tiny_font.render(line, True, BLACK)
            self.screen.blit(text, (SIDE_PANEL_X + 20, y))
            y += 20

        return y

    def _draw_pending_trade(self, env, y):
        lines = env.describe_pending_trade()

        for line in lines[:16]:
            if line == "":
                y += 8
                continue
            text = self.tiny_font.render(line, True, BLACK)
            self.screen.blit(text, (SIDE_PANEL_X + 20, y))
            y += 18

        return y

    def _draw_trade_draft(self, env, y):
        lines = env.describe_trade_draft()

        for line in lines[:16]:
            if line == "":
                y += 8
                continue
            text = self.tiny_font.render(line, True, BLACK)
            self.screen.blit(text, (SIDE_PANEL_X + 20, y))
            y += 18

        return y

    def _draw_timeline(self, replay_cursor, replay_total, replay_path, replay_locked):
        cursor = max(0, min(int(replay_cursor), int(replay_total)))
        total = max(0, int(replay_total))

        panel = pygame.Rect(40, HEIGHT - 54, WIDTH - 80, 42)
        pygame.draw.rect(self.screen, WHITE, panel, border_radius=8)
        pygame.draw.rect(self.screen, BLACK, panel, 1, border_radius=8)

        title = "Replay carregado" if replay_locked else "Gravando replay"
        if replay_path:
            title = f"{title}: {os.path.basename(str(replay_path))}"

        left_text = self.tiny_font.render(title, True, BLACK)
        self.screen.blit(left_text, (panel.x + 14, panel.y + 5))

        right_text = self.tiny_font.render(f"acao {cursor}/{total}", True, DARK_GRAY)
        right_rect = right_text.get_rect(topright=(panel.right - 14, panel.y + 5))
        self.screen.blit(right_text, right_rect)

        pygame.draw.rect(self.screen, LIGHT_GRAY, self.timeline_rect, border_radius=4)

        if total > 0:
            progress = cursor / float(total)
            filled = pygame.Rect(
                self.timeline_rect.x,
                self.timeline_rect.y,
                int(self.timeline_rect.width * progress),
                self.timeline_rect.height,
            )
            pygame.draw.rect(self.screen, GREEN, filled, border_radius=4)
            knob_x = self.timeline_rect.x + int(self.timeline_rect.width * progress)
        else:
            knob_x = self.timeline_rect.x

        knob_color = RED if replay_locked else GREEN
        pygame.draw.circle(self.screen, knob_color, (knob_x, self.timeline_rect.centery), 9)
        pygame.draw.circle(self.screen, BLACK, (knob_x, self.timeline_rect.centery), 9, 2)


def get_space_rect(index):
    cell = BOARD_SIZE // 11

    if 0 <= index <= 10:
        x = BOARD_X + BOARD_SIZE - cell * (index + 1)
        y = BOARD_Y + BOARD_SIZE - cell
        return pygame.Rect(x, y, cell, cell)

    if 11 <= index <= 20:
        offset = index - 10
        x = BOARD_X
        y = BOARD_Y + BOARD_SIZE - cell * (offset + 1)
        return pygame.Rect(x, y, cell, cell)

    if 21 <= index <= 30:
        offset = index - 20
        x = BOARD_X + cell * offset
        y = BOARD_Y
        return pygame.Rect(x, y, cell, cell)

    offset = index - 30
    x = BOARD_X + BOARD_SIZE - cell
    y = BOARD_Y + cell * offset
    return pygame.Rect(x, y, cell, cell)


def draw_wrapped_text(screen, text, x, y, max_width, font, color, line_height=22, max_lines=None):
    words = str(text).split(" ")
    line = ""
    lines_drawn = 0

    for word in words:
        test_line = line + word + " "
        width, _ = font.size(test_line)

        if width <= max_width:
            line = test_line
        else:
            if max_lines is not None and lines_drawn >= max_lines:
                return y

            rendered = font.render(line, True, color)
            screen.blit(rendered, (x, y))
            y += line_height
            lines_drawn += 1
            line = word + " "

    if line and (max_lines is None or lines_drawn < max_lines):
        rendered = font.render(line, True, color)
        screen.blit(rendered, (x, y))
        y += line_height

    return y
