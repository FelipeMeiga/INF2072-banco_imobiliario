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


class PygameView:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Banco Imobiliário IA - Simulação")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 26, bold=True)
        self.small_font = pygame.font.SysFont("arial", 20)
        self.tiny_font = pygame.font.SysFont("arial", 14)

    def handle_events(self) -> dict:
        commands = {
            "running": True,
            "toggle_pause": False,
            "step_once": False,
            "speed_up": False,
            "speed_down": False,
            "reset": False,
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

        return commands

    def draw(self, env, paused=False, step_delay=None):
        self.screen.fill((35, 120, 70))
        self._draw_board(env)
        self._draw_side_panel(env, paused=paused, step_delay=step_delay)
        pygame.display.flip()
        self.clock.tick(FPS)

    def quit(self):
        pygame.quit()

    def _draw_board(self, env):
        pygame.draw.rect(self.screen, WHITE, (BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE))
        pygame.draw.rect(self.screen, BLACK, (BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE), 3)

        for i, space in enumerate(env.board):
            rect = get_space_rect(i)

            if space.type == "start":
                color = YELLOW
            elif space.type == "property":
                color = WHITE
            elif space.type == "tax":
                color = ORANGE
            elif space.type == "chance":
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

            if space.type == "property":
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

        if env.pending_trade:
            y = self._draw_pending_trade(env, y)
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

        if space.type == "property":
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
