"""An NxN grid of robots, each one a conversation with the model, each turn one batched generation.

    python -m slm.arena --checkpoint runs/m10_rl_336m/checkpoints/final.pt --task key_door --agents 4 --turns 12

Every robot is a chat conversation: a system turn (who it is, the task, the rules), then per turn a user turn with
its observation (position, what it can see within `sight`, messages heard within `comm_range`) and an assistant turn
in which the model acts by calling its tools inside the think span -- `move("north")`, `say("the code is 4719")`,
and the task's unique tools -- and then writes a one-line status. The tools are declared functions (slm.tools.functions):
one block per function at the top of the conversation, the impls registered in that robot's PySession, so a call
runs through the normal <|python_call|> path and its result comes back as the sandbox's string. Actions take
effect when the call runs; all robots generate in one batch, so a turn is simultaneous.

Why this shape: the project's thesis for the swarm (docs/roadmap.md goal 5) is parallel inference as a *system*.
Here the system is a world with partial observability and asymmetric tools, so the only way to finish a task is
to talk. Nothing is trained: this measures what the M10 model already does with a protocol it has only seen in
pieces (declared functions, multi-turn sessions, chat). Tasks are verifiable, so an episode has a score, and a
sweep over seeds is an eval (`scripts/arena_eval.py`).
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field

DIRS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def _code_in(x) -> str:
    """The 4-digit code inside whatever the model passed ("7311", "the code is 7311", 7311). The model copies the
    sentence it was given more often than it extracts the number; the tool accepts both (a scaffold, logged as one)."""
    m = re.search(r"\d{4}", str(x))
    return m.group(0) if m else str(x).strip()


@dataclass
class Agent:
    id: int
    name: str
    x: int
    y: int
    tools: list[str]                       # names of the task tools this robot has (besides move/say/look)
    inbox: list[str] = field(default_factory=list)      # messages heard this turn (delivered next observation)
    history: list[dict] = field(default_factory=list)   # prior (user, assistant) turns, truncated
    inventory: list[str] = field(default_factory=list)
    memory: dict = field(default_factory=dict)          # task-private state (e.g. the key this robot read)
    log: list[dict] = field(default_factory=list)       # per-turn record for the transcript / portal

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)


def dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _direction(frm: tuple[int, int], to: tuple[int, int]) -> str:
    dx, dy = to[0] - frm[0], to[1] - frm[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


# ------------------------------------------------------------------------------------------------ tasks
class Task:
    """A task defines the world's objects, which robot gets which unique tool, the goal text, and the checker.
    `done_at` and the events use the 1-based turn (the turn the records carry; `World.turn` is 0-based while a
    turn is running)."""

    name = "base"
    comm_range = 3
    sight = 2

    def __init__(self, n: int, n_agents: int, rng: random.Random) -> None:
        self.n, self.n_agents, self.rng = n, n_agents, rng
        self.objects: dict[tuple[int, int], str] = {}
        self.done_at: int | None = None
        self.events: list[str] = []

    def goal(self, agent: Agent) -> str:
        raise NotImplementedError

    def tools(self, world: "World", agent: Agent) -> list:
        return []

    def place_agents(self, world: "World") -> None:
        for a in world.agents:
            a.x, a.y = self.rng.randrange(self.n), self.rng.randrange(self.n)

    def score(self, world: "World") -> dict:
        raise NotImplementedError

    def extra_observation(self, world: "World", agent: Agent) -> str:
        return ""

    def after_turn(self, world: "World", agent: Agent, record: dict) -> None:
        return None

    def question(self, world: "World", agent: Agent) -> tuple[str, list[str]]:
        """The sub-goal for this robot this turn, as a question it answers by calling ONE tool, and which tool to
        declare for it. Measured (2026-10-02): the 336M model calls a declared function correctly when exactly one is
        declared and the question names it ("use read_key() to find out"); with four declared it writes unrelated
        code, and it cannot carry its own previous result across turns, so the question re-states the fact it must
        use (what it heard, what its last call returned). Planning is the task's; execution and copying the right
        value into the right call are the model's. The scaffold is the honest first rung."""
        return "What do you do? Call a tool.", [d.name for d in world.decls[agent.id]]


class KeyDoor(Task):
    """Pairs: robot K holds a key code only `read_key()` reveals; robot D has `open_door(code)` which only works
    standing on the door cell. K must tell D the code (within comm range), D must walk to the door and open it.
    With more than two robots there are several doors, each with its own key holder and door opener."""

    name = "key_door"

    def __init__(self, n, n_agents, rng):
        super().__init__(n, n_agents, rng)
        self.pairs = max(1, n_agents // 2)
        self.codes = [str(rng.randint(1000, 9999)) for _ in range(self.pairs)]
        cells = rng.sample([(x, y) for x in range(n) for y in range(n)], self.pairs)
        self.doors = {c: i for i, c in enumerate(cells)}
        self.opened: set[int] = set()
        for c, i in self.doors.items():
            self.objects[c] = f"door {i + 1}"

    def place_agents(self, world):
        super().place_agents(world)
        # key holder and door opener of a pair start within comm range of each other, so turn 1 can already talk
        for i in range(self.pairs):
            k, d = world.agents[2 * i], world.agents[2 * i + 1] if 2 * i + 1 < len(world.agents) else None
            if d is not None:
                d.x = max(0, min(self.n - 1, k.x + self.rng.randint(-1, 1))); d.y = max(0, min(self.n - 1, k.y + self.rng.randint(-1, 1)))

    def goal(self, agent):
        i = agent.id // 2
        if agent.id % 2 == 0:
            return (f"Door {i + 1} is at {self._door_cell(i)}. Only you can read its key code, with read_key(). Only robot "
                    f"{self._partner(agent).name} can open the door, and it needs the code. Tell it the code with say(...) when it is near you.")
        return (f"Door {i + 1} is at {self._door_cell(i)}. Robot {self._partner(agent).name} knows its key code; it will say it when near "
                f"you. Walk to the door cell with move(...) and call open_door(code) there with the code you were told.")

    def _door_cell(self, i):
        return next(c for c, j in self.doors.items() if j == i)

    def _partner(self, agent):
        return self.world.agents[agent.id ^ 1]

    def tools(self, world, agent):
        from slm.tools.functions import FunctionDecl

        self.world = world
        i = agent.id // 2
        if agent.id % 2 == 0:
            def read_key(_i=i):
                return f"The key code for door {_i + 1} is {self.codes[_i]}."
            return [FunctionDecl("read_key", "def read_key() -> str", f"Returns the secret key code for door {i + 1}. Only you can read it; your partner needs it.", read_key)]

        def open_door(code: str, _i=i, _a=agent):
            if _a.pos != self._door_cell(_i):
                return f"You are at {_a.pos}; door {_i + 1} is at {self._door_cell(_i)}. Move there first."
            if _code_in(code) == self.codes[_i]:
                if _i not in self.opened:
                    self.opened.add(_i); self.events.append(f"turn {world.turn + 1}: {_a.name} opened door {_i + 1}")
                    if len(self.opened) == self.pairs and self.done_at is None:
                        self.done_at = world.turn + 1
                return f"Door {_i + 1} is open. Task complete."
            return f"Wrong code {code!r}. Ask {self._partner(_a).name} for the code."
        return [FunctionDecl("open_door", "def open_door(code: str) -> str", f"Opens door {i + 1} when you stand on its cell and give the right 4-digit code. Only you have this tool.", open_door)]

    def score(self, world):
        return {"opened": len(self.opened), "of": self.pairs, "success": len(self.opened) == self.pairs, "done_turn": self.done_at,
                "progress": round(len(self.opened) / self.pairs, 2)}

    def question(self, world, agent):
        i = agent.id // 2
        partner = self._partner(agent)
        heard = [m for m in agent.inbox if re.search(r"\d{4}", m) and m.startswith(partner.name + " ")]
        if heard:
            agent.memory["code_msg"] = heard[-1]
        if agent.id % 2 == 0:
            if not agent.memory.get("code"):
                return f"What is the key code for door {i + 1}? Use read_key() to find out.", ["read_key"]
            code = agent.memory["code"]
            if dist(agent.pos, partner.pos) <= self.comm_range:
                return f"The key code is {code}. Tell robot {partner.name}: call say(\"the code is {code}\"). What does say() return?", ["say"]
            d = _direction(agent.pos, partner.pos)
            return f"Robot {partner.name} is too far to hear you. Call move(\"{d}\") to step one cell {d}. What does move() return?", ["move"]
        door = self._door_cell(i)
        if agent.pos != door:
            d = _direction(agent.pos, door)
            return f"Door {i + 1} is at {door} and you are at {agent.pos}. Call move(\"{d}\") to step one cell {d}. What does move() return?", ["move"]
        msg = agent.memory.get("code_msg")
        if msg:
            code = re.search(r"\d{4}", msg).group(0)
            return f"You are on door {i + 1}'s cell. Earlier {msg}. Call open_door(\"{code}\") to open the door. What does open_door() return?", ["open_door"]
        return f"You are on door {i + 1}'s cell but have not heard the code yet. Use look() to see who is near. What does look() return?", ["look"]

    def after_turn(self, world, agent, record):
        """Remember what this robot's own calls returned (the model cannot carry it over by itself)."""
        for code_txt, result in record.get("calls", []):
            if "read_key" in code_txt and re.search(r"\d{4}", str(result)):
                agent.memory["code"] = re.search(r"\d{4}", str(result)).group(0)


class Relay(Task):
    """One robot (the source) knows a code; the robot with `submit(code)` (the sink) starts out of comm range,
    with the others strung between them. The code must hop robot to robot. Robots cannot move: it is a pure
    communication task (a chain of `say`s)."""

    name = "relay"
    comm_range = 2
    sight = 2

    def __init__(self, n, n_agents, rng):
        super().__init__(n, n_agents, rng)
        self.code = str(rng.randint(1000, 9999))
        self.submitted: str | None = None

    def place_agents(self, world):
        y = self.n // 2
        for i, a in enumerate(world.agents):  # a line, two cells apart: each robot hears only its neighbours
            a.x, a.y = min(self.n - 1, 2 * i), y

    def goal(self, agent):
        last = self.world.agents[-1]
        if agent.id == 0:
            return (f"You know a secret code: read_code() tells you. Robot {last.name} must submit it, but it is too far away to hear you. "
                    f"Tell the code to the robot next to you with say(...), and ask it to pass it on.")
        if agent.id == len(self.world.agents) - 1:
            return "A secret code is being passed along the line of robots towards you. When you hear a 4-digit code, call submit(code) with it."
        return "A secret code is being passed along the line of robots. When you hear a 4-digit code, repeat it with say(...) so the next robot hears it. Nobody can move."

    def tools(self, world, agent):
        from slm.tools.functions import FunctionDecl

        self.world = world
        if agent.id == 0:
            return [FunctionDecl("read_code", "def read_code() -> str", "Returns the secret code. Only you can read it.", lambda: f"The secret code is {self.code}.")]
        if agent.id == len(world.agents) - 1:
            def submit(code: str, _a=agent):
                self.submitted = _code_in(code)
                if self.submitted == self.code and self.done_at is None:
                    self.done_at = world.turn + 1; self.events.append(f"turn {world.turn + 1}: {_a.name} submitted the right code")
                return "Correct! Task complete." if self.submitted == self.code else f"{code!r} is not the code. Keep listening."
            return [FunctionDecl("submit", "def submit(code: str) -> str", "Submits the 4-digit code you were told. Only you have this tool.", submit)]
        return []

    def score(self, world):
        reached = max([a.id for a in world.agents if a.memory.get("code_msg") or a.memory.get("code")] or [0])
        return {"submitted": self.submitted, "code": self.code, "success": self.submitted == self.code, "done_turn": self.done_at,
                "hops": len(world.agents) - 1, "hops_reached": reached, "progress": round(reached / max(1, len(world.agents) - 1), 2)}

    def question(self, world, agent):
        last = len(world.agents) - 1
        heard = [m for m in agent.inbox if re.search(r"\d{4}", m)]
        if heard:
            agent.memory["code_msg"] = heard[-1]
        msg = agent.memory.get("code_msg")
        if agent.id == 0:
            if not agent.memory.get("code"):
                return "What is the secret code? Use read_code() to find out.", ["read_code"]
            code = agent.memory["code"]
            return f"The secret code is {code}. Tell the next robot: call say(\"the code is {code}\"). What does say() return?", ["say"]
        if msg:
            code = re.search(r"\d{4}", msg).group(0)
            if agent.id == last:
                return f"Earlier {msg}. Call submit(\"{code}\") to submit it. What does submit() return?", ["submit"]
            return f"Earlier {msg}. Pass it on: call say(\"the code is {code}\"). What does say() return?", ["say"]
        return "You have not heard the code yet. Use look() to see who is near. What does look() return?", ["look"]

    def after_turn(self, world, agent, record):
        for code_txt, result in record.get("calls", []):
            if "read_code" in code_txt and re.search(r"\d{4}", str(result)):
                agent.memory["code"] = re.search(r"\d{4}", str(result)).group(0)


class Triangulate(Task):
    """A buried target: every robot's `distance_to_target()` returns its own distance to it; one robot has `dig()`, which works
    only on the target cell. Robots must share distances so the digger can work out where to go. Hard."""

    name = "triangulate"
    comm_range = 4
    sight = 1

    def __init__(self, n, n_agents, rng):
        super().__init__(n, n_agents, rng)
        self.target = (rng.randrange(n), rng.randrange(n))
        self.dug_at: tuple[int, int] | None = None

    def goal(self, agent):
        if agent.id == 0:
            return ("Something is buried at a secret cell. Every robot's distance_to_target() gives its own distance to it. Ask the others for their "
                    "distances with say(...), work out the cell, move there and call dig() on it. Only you can dig.")
        return ("Something is buried at a secret cell. distance_to_target() gives your distance to it (north/south/east/west steps). Tell robot "
                f"{self.world.agents[0].name} your position and distance with say(...) so it can find the cell.")

    def tools(self, world, agent):
        from slm.tools.functions import FunctionDecl

        self.world = world
        out = [FunctionDecl("distance_to_target", "def distance_to_target() -> int", "Returns your distance in steps to the buried target.", lambda _a=agent: dist(_a.pos, self.target))]
        if agent.id == 0:
            def dig(_a=agent):
                self.dug_at = _a.pos
                if _a.pos == self.target:
                    if self.done_at is None:
                        self.done_at = world.turn + 1; self.events.append(f"turn {world.turn + 1}: {_a.name} dug up the target")
                    return "You found it! Task complete."
                return f"Nothing here at {_a.pos}. The target is {dist(_a.pos, self.target)} steps away."
            out.append(FunctionDecl("dig", "def dig() -> str", "Digs at your current cell. Only you have this tool.", dig))
        return out

    def score(self, world):
        start = world.agents[0].memory.get("start_distance")
        cur = dist(world.agents[0].pos, self.target)
        return {"target": self.target, "dug_at": self.dug_at, "success": self.dug_at == self.target, "done_turn": self.done_at,
                "digger_distance": cur, "progress": round(1 - cur / start, 2) if start else 0.0}

    def question(self, world, agent):
        digger = world.agents[0]
        if agent.id != 0:
            dsn = dist(agent.pos, self.target)
            if not agent.memory.get("sensed"):
                return "What does distance_to_target() return? Use distance_to_target() to find out.", ["distance_to_target"]
            if dist(agent.pos, digger.pos) <= self.comm_range:
                return (f"Tell robot {digger.name} your distance: call say(\"I am at {agent.pos}, distance {dsn}\"). "
                        f"What does say() return?"), ["say"]
            d = _direction(agent.pos, digger.pos)
            return f"Robot {digger.name} is too far to hear you. Call move(\"{d}\") to step one cell {d}. What does move() return?", ["move"]
        cur = dist(agent.pos, self.target)
        if cur == 0:
            return "You are standing right on the buried target. Call dig() to dig it up. What does dig() return?", ["dig"]
        reports = [m for m in agent.inbox if "distance" in m]
        if reports:
            agent.memory.setdefault("reports", []).extend(reports)
        last = agent.memory.get("last_sense")
        agent.memory.setdefault("start_distance", cur)
        agent.memory["last_sense"] = cur
        if last is None or cur < last:
            d = agent.memory.get("dir") or self.rng.choice(list(DIRS))
        else:
            d = self.rng.choice([x for x in DIRS if x != agent.memory.get("dir")])
        agent.memory["dir"] = d
        return f"The target is {cur} steps away. Call move(\"{d}\") to step one cell {d}. What does move() return?", ["move"]

    def after_turn(self, world, agent, record):
        for code_txt, result in record.get("calls", []):
            if "distance_to_target" in code_txt:
                agent.memory["sensed"] = True


TASKS = {"key_door": KeyDoor, "relay": Relay, "triangulate": Triangulate}


# ------------------------------------------------------------------------------------------------ the world
class World:
    def __init__(self, task: str, n: int = 8, n_agents: int = 4, seed: int = 0, max_history: int = 1) -> None:
        self.rng = random.Random(seed)
        self.n, self.seed, self.max_history = n, seed, max_history
        if task == "key_door" and (n_agents < 2 or n_agents % 2):
            raise ValueError("key_door needs an even number of robots (key holder + door opener per door)")
        if n_agents < 2:
            raise ValueError(f"{task} needs at least 2 robots")
        if n_agents > 32:
            raise ValueError("at most 32 robots")
        self.agents = [Agent(i, f"R{i + 1}", 0, 0, []) for i in range(n_agents)]
        self.task: Task = TASKS[task](n, n_agents, self.rng)
        self.task.place_agents(self)
        self.turn = 0
        self.decls: dict[int, list] = {}
        self.pending: list[tuple[Agent, str]] = []  # (speaker, text) said this turn, delivered at the next observation
        for a in self.agents:
            self.decls[a.id] = self._common_tools(a) + self.task.tools(self, a)
            a.tools = [d.name for d in self.decls[a.id]]

    # --- tools every robot has
    def _common_tools(self, agent: Agent) -> list:
        from slm.tools.functions import FunctionDecl

        def move(direction: str, _a=agent):
            d = DIRS.get(str(direction).strip().lower())
            if d is None:
                return f"Unknown direction {direction!r}. Use north, south, east or west."
            nx, ny = _a.x + d[0], _a.y + d[1]
            if not (0 <= nx < self.n and 0 <= ny < self.n):
                return f"Blocked: the grid edge. You are still at {_a.pos}."
            _a.x, _a.y = nx, ny
            return f"You moved {direction} to {_a.pos}."

        def say(message: str, _a=agent):
            text = " ".join(str(message).split())[:200]
            self.pending.append((_a, text))
            near = [b.name for b in self.agents if b is not _a and dist(_a.pos, b.pos) <= self.task.comm_range]
            return f"You said: {text!r}. Heard by: {', '.join(near) if near else 'nobody (no robot within range)'}."

        def look(_a=agent):
            return self._visible(_a) or "Nothing within sight."

        return [FunctionDecl("move", "def move(direction: str) -> str", "Move one cell north, south, east or west.", move),
                FunctionDecl("say", "def say(message: str) -> str", f"Say something; robots within {self.task.comm_range} cells hear it next turn.", say),
                FunctionDecl("look", "def look() -> str", "What is within sight: nearby robots and objects with their positions.", look)]

    def _visible(self, agent: Agent) -> str:
        parts = [f"{b.name} at {b.pos}" for b in self.agents if b is not agent and dist(agent.pos, b.pos) <= self.task.sight]
        parts += [f"{what} at {c}" for c, what in self.task.objects.items() if dist(agent.pos, c) <= self.task.sight]
        return "; ".join(parts)

    # --- conversation
    def system_prompt(self, agent: Agent) -> str:
        return (f"You are robot {agent.name} on a {self.n}x{self.n} grid (x east, y south, both from 0). Each turn you get an observation; "
                f"act by calling your tools (move, say, look, and your special tool) and finish with one short line of status. "
                f"You can only hear robots within {self.task.comm_range} cells. Task: {self.task.goal(agent)}")

    SUFFIX = "\nThink step by step, then give the final answer on its own line as '#### <answer>'."

    def observation(self, agent: Agent) -> str:
        """The user turn IS the question. Measured (2026-10-02): the same question with a "Turn 1. You are at ...
        In sight ... You heard ..." preamble, or the briefing in front of it, turns the model from calling the tool
        into describing it; the facts a turn needs are inside the question already. The briefing and the full
        observation are kept for the transcript and the portal (`status(agent)`), not shown to the model."""
        q, _ = self.turn_question(agent)
        return f"{q}{self.SUFFIX}"

    def turn_question(self, agent: Agent) -> tuple[str, list[str]]:
        """`Task.question` once per robot per turn: it updates the robot's memory and may draw from the task RNG, so
        asking twice would change the question between the prompt and the record (seen on triangulate)."""
        cached = agent.memory.get("_turn")
        if cached is None or cached[0] != self.turn:
            cached = (self.turn, self.task.question(self, agent))
            agent.memory["_turn"] = cached
        return cached[1]

    def status(self, agent: Agent) -> str:
        heard = "; ".join(agent.inbox) if agent.inbox else "nothing"
        vis = self._visible(agent) or "nothing"
        return f"Turn {self.turn + 1}. {agent.name} at {agent.pos}. In sight: {vis}. Heard: {heard}."

    def turn_tools(self, agent: Agent) -> list:
        """The declared tools for this turn: only what the question needs (one, usually)."""
        _, names = self.turn_question(agent)
        return [d for d in self.decls[agent.id] if d.name in names]

    def messages(self, agent: Agent) -> list[dict]:
        history = agent.history[-2 * self.max_history:]
        return history + [{"role": "user", "content": self.observation(agent)}]

    # --- one turn
    def deliver(self) -> None:
        for a in self.agents:
            a.inbox = []
        for speaker, text in self.pending:
            for b in self.agents:
                if b is not speaker and dist(speaker.pos, b.pos) <= self.task.comm_range:
                    b.inbox.append(f"{speaker.name} said: {text!r}")
        self.pending = []

    def state(self) -> dict:
        return {"turn": self.turn, "n": self.n, "task": self.task.name, "agents": [{"id": a.id, "name": a.name, "x": a.x, "y": a.y, "tools": a.tools} for a in self.agents],
                "objects": [{"x": c[0], "y": c[1], "what": w} for c, w in self.task.objects.items()], "score": self.task.score(self), "events": list(self.task.events)}


# ------------------------------------------------------------------------------------------------ the runner
class Runner:
    """Runs a world against a model: every turn, one batched generation over all robots (their conversations are
    different lengths; `sample_with_tools` takes a list of prompts). `generate` can be injected for tests."""

    def __init__(self, world: World, model=None, tok=None, max_new_tokens: int = 128, max_calls: int = 4, generate=None) -> None:
        from slm.tools.pysandbox import PySession

        self.world, self.model, self.tok = world, model, tok
        self.max_new_tokens, self.max_calls = max_new_tokens, max_calls
        self.sessions = {a.id: PySession() for a in world.agents}
        self.generate = generate or self._generate

    def _generate(self, prompts: list[list[int]], sessions: list) -> list:
        import torch

        from slm.tools.loop import sample_with_tools
        from slm.utils.sdpa import sdpa_context

        gen = torch.Generator(device=next(self.model.parameters()).device); gen.manual_seed(self.world.seed * 1000 + self.world.turn)
        with torch.no_grad(), sdpa_context("decode"):
            return sample_with_tools(self.model, self.tok, prompts, self.max_new_tokens, 1.0, 1.0, 1, gen, max_calls=self.max_calls, sessions=sessions)

    def step(self) -> dict:
        from slm.data.chat import format_chat, parse_assistant
        from slm.tools.functions import functions_env
        from slm.tools.pysandbox import PySession

        w = self.world
        w.deliver()
        prompts, sessions, obs, declared = [], [], [], []
        for a in w.agents:
            msgs = w.messages(a)
            tools = w.turn_tools(a)
            self.sessions[a.id] = PySession()  # fresh each turn, with only this turn's tool: a tool declared earlier stays callable otherwise
            self.sessions[a.id].register(functions_env(tools))
            prompts.append(format_chat(self.tok, msgs, add_generation_prompt=True, think_required=True, functions=tools).ids if self.tok else msgs)
            sessions.append(self.sessions[a.id]); obs.append(msgs[-1]["content"]); declared.append([d.name for d in tools])
        t0 = time.time()
        tcs = self.generate(prompts, sessions)
        records = []
        for a, tc, o, tools in zip(w.agents, tcs, obs, declared):
            p = parse_assistant(self.tok, tc.ids) if self.tok else tc
            think, answer = (p["think"], p["answer"]) if isinstance(p, dict) else (tc.get("think"), tc.get("answer"))
            calls = [list(c) for c in getattr(tc, "calls", [])] if not isinstance(tc, dict) else tc.get("calls", [])
            a.history += [{"role": "user", "content": o}, {"role": "assistant", "content": (answer or "").strip()[:300], "ids": list(tc.ids) if hasattr(tc, "ids") else None}]
            if a.history[-1]["ids"] is None:
                a.history[-1].pop("ids")
            rec = {"turn": w.turn + 1, "agent": a.name, "pos": a.pos, "observation": o, "status": w.status(a), "think": think, "calls": calls, "answer": answer,
                   "n_calls": len(calls), "tools": tools}
            w.task.after_turn(w, a, rec)
            a.log.append(rec); records.append(rec)
        w.turn += 1
        return {"turn": w.turn, "seconds": round(time.time() - t0, 2), "records": records, "state": w.state()}

    def run(self, turns: int, stop_when_done: bool = True, on_turn=None) -> dict:
        t0 = time.time()
        for _ in range(turns):
            out = self.step()
            if on_turn:
                on_turn(out)
            if stop_when_done and self.world.task.score(self.world).get("success"):
                break
        return {"task": self.world.task.name, "n": self.world.n, "agents": len(self.world.agents), "seed": self.world.seed, "turns": self.world.turn,
                "score": self.world.task.score(self.world), "events": self.world.task.events, "seconds": round(time.time() - t0, 1),
                "transcript": [r for a in self.world.agents for r in a.log]}


def main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=r"C:\slm-data\tokenizer\v1")
    ap.add_argument("--task", default="key_door", choices=sorted(TASKS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from pathlib import Path as _P

    from slm.data.tokenizer import SlmTokenizer
    from slm.eval.quality import load_model

    model, _ = load_model(_P(a.checkpoint), "cuda")
    tok = SlmTokenizer.load(a.tokenizer)
    world = World(a.task, a.n, a.agents, a.seed)
    runner = Runner(world, model, tok, a.max_new)

    def show(out):
        print(f"--- turn {out['turn']} ({out['seconds']}s)")
        for r in out["records"]:
            print(f"  {r['agent']} @{r['pos']}  calls={[c[0][:60] for c in r['calls']]}  -> {(r['answer'] or '').strip()[:80]!r}")
    res = runner.run(a.turns, on_turn=show)
    print(json.dumps({k: v for k, v in res.items() if k != "transcript"}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
