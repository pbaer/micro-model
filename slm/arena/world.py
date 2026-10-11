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
    history: list[dict] = field(default_factory=list)   # prior (user, assistant) turns (World keeps max_history)
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

    @classmethod
    def check(cls, n: int, n_agents: int) -> None:
        """Raise ValueError when the task cannot be built for this grid and robot count."""
        if n_agents < 2:
            raise ValueError(f"{cls.name} needs at least 2 robots")

    def agent_info(self, agent: Agent) -> dict:
        """Extra per-robot fields for the state and the portal (a team, a status); none by default."""
        return {}

    # Which player ("A" or "B") a robot belongs to when two models play against each other (Runner(players=...)).
    # Cooperative tasks have one side; the competitive ones split their robots (sides() says how).
    def side_of(self, agent: Agent) -> str:
        return "A"

    def sides(self) -> dict:
        """What each player plays, for the record and the portal: {"A": "team red", "B": "team blue"}."""
        return {"A": "every robot"}

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

    @classmethod
    def check(cls, n, n_agents):
        if n_agents < 2 or n_agents % 2:
            raise ValueError("key_door needs an even number of robots (key holder + door opener per door)")

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
            def read_key(*_, _i=i):
                return f"The key code for door {_i + 1} is {self.codes[_i]}."
            return [FunctionDecl("read_key", "def read_key() -> str", f"Returns the secret key code for door {i + 1}. Only you can read it; your partner needs it.", read_key)]

        def open_door(code: str, *_, _i=i, _a=agent):
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
            return [FunctionDecl("read_code", "def read_code() -> str", "Returns the secret code. Only you can read it.", lambda *_: f"The secret code is {self.code}.")]
        if agent.id == len(world.agents) - 1:
            def submit(code: str, *_, _a=agent):
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
        out = [FunctionDecl("distance_to_target", "def distance_to_target() -> int", "Returns your distance in steps to the buried target.", lambda *_, _a=agent: dist(_a.pos, self.target))]
        if agent.id == 0:
            def dig(*_, _a=agent):
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


# ------------------------------------------------------------------------------------------------ competitive tasks
# The scaffold rule above holds (one named tool per turn, the facts restated in the question), but each of these
# leaves one real decision to the model and scores it against chance: which of two heard codes is its team's
# (team_relay), which step brings it closer to or further from another robot given two positions (tag), which step
# heads for the nearest of three listed coins (gold_rush). The decision is offered as complete calls to pick from
# ('call one of move("west"), move("north"), ...'), in a random order. Measured on the first smoke runs (2026-10-10):
# offered `move(...)` and a list of words, the model never chose -- it wrote move("binder") / move("marker") or copied
# say("the code is ...") verbatim, every one of 48 moves invalid. Given the calls themselves it copies one, and which
# one is the decision. Each score carries the chance rate of its decision (the share of offered calls that are right).

def _move_calls(rng: random.Random) -> tuple[str, list[str]]:
    """The four moves as complete calls in a random order, and that order."""
    d = list(DIRS)
    rng.shuffle(d)
    calls = [f'move("{x}")' for x in d]
    return ", ".join(calls[:-1]) + " or " + calls[-1], d


def _step(pos: tuple[int, int], d: str, n: int) -> tuple[int, int]:
    nx, ny = pos[0] + DIRS[d][0], pos[1] + DIRS[d][1]
    return (nx, ny) if 0 <= nx < n and 0 <= ny < n else pos


def _moved(record: dict) -> str | None:
    """The direction of this turn's move call ('east'), 'invalid' for a move with no valid direction, None if no move."""
    for code_txt, result in record.get("calls", []):
        if re.match(r"\s*move\s*\(", code_txt or ""):
            m = re.search(r"\b(north|south|east|west)\b", code_txt.lower())
            return m.group(1) if m and not str(result).startswith("Unknown direction") else "invalid"
    return None


def _name_in(x, names: list[str]) -> str | None:
    """The robot name inside whatever the model passed ("R5", "robot R5", 'R5')."""
    m = re.search(r"\bR(\d+)\b", str(x))
    return f"R{m.group(1)}" if m and f"R{m.group(1)}" in names else None


class TeamRelay(Task):
    """Two teams race to relay their own secret code down their own line. Team red stands on one row, team blue on the
    row below, both two cells apart. With a comm range of 3 a robot hears the teammate behind it AND the rival robot
    diagonally behind it, and those two speak in the same turn (they are at the same depth of their chains), so every
    first pass faces both codes at once. With range 2 the rival's code arrived a turn after the robot's own, and a
    sink that never mixed them up could not be told from one that did (caught by the scripted test). Chains cannot
    skip a robot: the teammate two places back is 4 cells away. The question lists what it heard, labelled by team, in a random order, and
    asks it to pass on its own team's code without saying which number that is. The first team whose last robot
    submits its own code wins (both in the same turn: a tie). Scored: wins, and how often a robot passed on or
    submitted the rival's code instead of its own."""

    name = "team_relay"
    comm_range = 3
    sight = 2
    TEAMS = ("red", "blue")

    @classmethod
    def check(cls, n, n_agents):
        if n_agents < 4 or n_agents % 2:
            raise ValueError("team_relay needs an even number of robots, at least 4 (two teams of a source, relays and a sink)")
        per = n_agents // 2
        if 2 * (per - 1) > n - 1:
            raise ValueError(f"team_relay with {per} robots per team needs a grid of at least {2 * per - 1} cells across (got {n})")

    def __init__(self, n, n_agents, rng):
        super().__init__(n, n_agents, rng)
        a, b = rng.sample(range(1000, 10000), 2)
        self.codes = {"red": str(a), "blue": str(b)}
        self.finished: dict[str, int] = {}  # team -> 1-based turn its sink submitted the right code
        self.right = {"red": 0, "blue": 0}  # relays / sinks that passed on or submitted their own team's code
        self.wrong = {"red": 0, "blue": 0}  # ... the rival team's code

    def team(self, agent):
        return self.TEAMS[agent.id % 2]

    def members(self, team):
        return [a for a in self.world.agents if self.team(a) == team]

    def rival(self, team):
        return "blue" if team == "red" else "red"

    def agent_info(self, agent):
        return {"team": self.team(agent)}

    def side_of(self, agent):
        return "A" if self.team(agent) == "red" else "B"

    def sides(self):
        return {"A": "team red", "B": "team blue"}

    def place_agents(self, world):
        self.world = world
        y = max(0, self.n // 2 - 1)
        for a in world.agents:  # red on row y, blue on row y+1, robot i of a team at x = 2i
            a.x, a.y = 2 * (a.id // 2), y + (a.id % 2)

    def _role(self, agent):
        team = self.members(self.team(agent))
        i = team.index(agent)
        return "source" if i == 0 else "sink" if i == len(team) - 1 else "relay"

    def goal(self, agent):
        t, role = self.team(agent), self._role(agent)
        sink = self.members(t)[-1].name
        rival = f"Team {self.rival(t)} is racing you with its own code on the row beside yours, and you will hear its robots too."
        if role == "source":
            return f"You are on team {t}. read_code() tells you your team's secret code. Pass it to your teammate next to you with say(...). {rival}"
        if role == "sink":
            return f"You are on team {t}, at the end of your team's line. When you hear your team's code, call submit(code) with it: the first team to submit wins. {rival}"
        return f"You are on team {t}. Pass your team's code along your team's line with say(...); robot {sink} at the end submits it. {rival}"

    def tools(self, world, agent):
        from slm.tools.functions import FunctionDecl

        self.world = world
        t, role = self.team(agent), self._role(agent)
        if role == "source":
            return [FunctionDecl("read_code", "def read_code() -> str", f"Returns team {t}'s secret code. Only you can read it.",
                                 lambda *_, _t=t: f"Team {_t}'s secret code is {self.codes[_t]}.")]
        if role == "sink":
            def submit(code: str, *_, _a=agent, _t=t):
                c = _code_in(code)
                if c == self.codes[_t]:
                    if _t not in self.finished:
                        self.finished[_t] = world.turn + 1
                        self.events.append(f"turn {world.turn + 1}: team {_t} submitted its code ({_a.name})")
                        if self.done_at is None:
                            self.done_at = world.turn + 1
                    return f"Correct! Team {_t} finished."
                if c == self.codes[self.rival(_t)]:
                    return f"{c} is team {self.rival(_t)}'s code, not yours. Keep listening for your own team's code."
                return f"{code!r} is not your team's code. Keep listening."
            return [FunctionDecl("submit", "def submit(code: str) -> str", f"Submits team {t}'s 4-digit code. Only you have this tool on your team.", submit)]
        return []

    def winner(self):
        if not self.finished:
            return None
        first = min(self.finished.values())
        best = [t for t, turn in self.finished.items() if turn == first]
        return best[0] if len(best) == 1 else "tie"

    def score(self, world):
        def reach(t):
            team = self.members(t)
            got = [i for i, a in enumerate(team) if a.memory.get("code") or a.memory.get("own_msg")]
            return round(max(got or [0]) / max(1, len(team) - 1), 2)
        w = self.winner()
        verdict = None if w is None else "tie: both teams finished in the same turn" if w == "tie" else f"team {w} wins"
        passes = sum(self.right.values()) + sum(self.wrong.values())
        return {"winner": w, "winner_side": {"red": "A", "blue": "B", "tie": "tie"}.get(w), "verdict": verdict,
                "red_finished": self.finished.get("red"), "blue_finished": self.finished.get("blue"),
                "red_reach": reach("red"), "blue_reach": reach("blue"),
                "own_code_passes": sum(self.right.values()), "rival_code_passes": sum(self.wrong.values()),
                "rival_code_rate": round(sum(self.wrong.values()) / passes, 2) if passes else None,
                "success": w is not None, "done_turn": self.done_at, "progress": max(reach("red"), reach("blue"))}

    def question(self, world, agent):
        t, role = self.team(agent), self._role(agent)
        names = {a.name: a for a in world.agents}
        for m in agent.inbox:
            if not re.search(r"\d{4}", m):
                continue
            speaker = names.get(m.split(" ", 1)[0])
            if speaker is None:
                continue
            key = "own_msg" if self.team(speaker) == t else "rival_msg"
            agent.memory[key] = f"{speaker.name} (team {self.team(speaker)}) said {m.split(' said: ', 1)[-1]}"
        if role == "source":
            if not agent.memory.get("code"):  # the relay task's exact words: with "team red's" in it the model answered nothing
                return "What is the secret code? Use read_code() to find out.", ["read_code"]  # (red) or read_code(team_brown) (blue)
            code = agent.memory["code"]  # the relay task's exact words (the heard lines carry the speaker's team anyway); with
            # "Your team's secret code ... say(\"team red: ...\")" both sources kept calling read_code(secret_code) instead
            return f"The secret code is {code}. Tell the next robot: call say(\"the code is {code}\"). What does say() return?", ["say"]
        own, rival = agent.memory.get("own_msg"), agent.memory.get("rival_msg")
        if not own:  # nothing of its own team yet: wait (the one decision this task scores needs two candidates)
            return "You have not heard your team's code yet. Use look() to see who is near. What does look() return?", ["look"]
        heard = [own] + ([rival] if rival else [])
        self.rng.shuffle(heard)
        lines = " ".join(f"Earlier {h}." for h in heard)
        codes = list(dict.fromkeys(re.search(r"\d{4}", h).group(0) for h in heard))  # both codes (one when they agree)
        agent.memory["q_codes"] = len(codes) > 1  # a decision only when two different codes are on offer (chance 0.5)
        tool = "submit" if role == "sink" else "say"
        calls = [f'submit("{c}")' if tool == "submit" else f'say("the code is {c}")' for c in codes]
        # "Call X(...)" opening its own sentence, as in the relay task: "Submit your own team's code: call submit(...)"
        # came back as print(submit("call submit("7890")) on every turn of the first smoke run
        if len(calls) == 1:
            return f"You are on team {t}. {lines} Call {calls[0]} to pass on your team's code. What does {tool}() return?", [tool]
        return f"You are on team {t}. {lines} Call {' or '.join(calls)}, whichever is your own team's code. What does {tool}() return?", [tool]

    def after_turn(self, world, agent, record):
        t = self.team(agent)
        for code_txt, result in record.get("calls", []):
            if "read_code" in code_txt and re.search(r"\d{4}", str(result)):
                agent.memory["code"] = re.search(r"\d{4}", str(result)).group(0)
            elif ("say" in code_txt or "submit" in code_txt) and agent.memory.pop("q_codes", None):
                c = _code_in(code_txt)
                if c == self.codes[t]:
                    self.right[t] += 1
                elif c == self.codes[self.rival(t)]:
                    self.wrong[t] += 1


class Tag(Task):
    """Chasers against runners. The first third of the robots (at least one) chase; the rest run. A chaser's question
    gives its position and the nearest live runner's and asks which way is closer; a runner's gives the nearest
    chaser's and asks which way is further. Neither names the direction. A chaser next to a runner (one step away or
    on its cell) calls tag(name); a tagged runner is out. The chasers win when every runner is tagged; the runners win
    by surviving the turn budget. Scored per side: the share of moves that actually closed or opened the distance."""

    name = "tag"
    comm_range = 3
    sight = 3

    def __init__(self, n, n_agents, rng):
        super().__init__(n, n_agents, rng)
        self.n_chasers = max(1, n_agents // 3)
        self.tagged: dict[str, int] = {}  # runner name -> 1-based turn
        self.moves = {side: {"good": 0, "made": 0, "invalid": 0, "chance": 0.0} for side in ("chaser", "runner")}

    def side(self, agent):
        return "chaser" if agent.id < self.n_chasers else "runner"

    def agent_info(self, agent):
        return {"team": self.side(agent), "tagged": agent.name in self.tagged}

    def side_of(self, agent):
        return "A" if self.side(agent) == "chaser" else "B"

    def sides(self):
        return {"A": "the chasers", "B": "the runners"}

    def runners(self, live=True):
        return [a for a in self.world.agents if self.side(a) == "runner" and not (live and a.name in self.tagged)]

    def place_agents(self, world):
        self.world = world
        cells = [(x, y) for x in range(self.n) for y in range(self.n)]
        chasers = [a for a in world.agents if self.side(a) == "chaser"]
        for a in chasers:
            a.x, a.y = self.rng.choice(cells)
        for a in world.agents:
            if self.side(a) == "runner":  # start at least 3 steps from every chaser when the grid allows it
                far = [c for c in cells if all(dist(c, ch.pos) >= 3 for ch in chasers)] or cells
                a.x, a.y = self.rng.choice(far)

    def goal(self, agent):
        if self.side(agent) == "chaser":
            return "You are a chaser. Catch the runners: step towards the nearest one, and when it is next to you call tag(name) with its name."
        return "You are a runner. Stay away from the chasers: each turn step away from the nearest one. A chaser next to you can tag you, and then you are out."

    def tools(self, world, agent):
        from slm.tools.functions import FunctionDecl

        self.world = world
        if self.side(agent) != "chaser":
            return []

        def tag(name: str, *_, _a=agent):
            who = _name_in(name, [r.name for r in self.runners(live=False)])
            if who is None:
                return f"{name!r} is not a runner."
            if who in self.tagged:
                return f"{who} is already tagged."
            r = next(x for x in world.agents if x.name == who)
            if dist(_a.pos, r.pos) > 1:
                return f"{who} is at {r.pos}, {dist(_a.pos, r.pos)} steps away: too far to tag. Get next to it first."
            self.tagged[who] = world.turn + 1
            self.events.append(f"turn {world.turn + 1}: {_a.name} tagged {who}")
            if not self.runners() and self.done_at is None:
                self.done_at = world.turn + 1
            return f"You tagged {who}! It is out."
        return [FunctionDecl("tag", "def tag(name: str) -> str", "Tags a runner that is next to you (one step away or on your cell). Only chasers have this tool.", tag)]

    def score(self, world):
        total = len(self.runners(live=False))
        out = len(self.tagged)
        m = self.moves
        rate = lambda side, k: round(m[side][k] / m[side]["made"], 2) if m[side]["made"] else None  # noqa: E731
        verdict = "the chasers win: every runner tagged" if out == total else None
        return {"tagged": out, "runners": total, "chasers": self.n_chasers, "verdict": verdict, "winner_side": "A" if out == total else None,
                "chaser_good_moves": rate("chaser", "good"), "chaser_chance": rate("chaser", "chance"),
                "runner_good_moves": rate("runner", "good"), "runner_chance": rate("runner", "chance"),
                "chaser_moves": m["chaser"]["made"], "runner_moves": m["runner"]["made"],
                "invalid_moves": m["chaser"]["invalid"] + m["runner"]["invalid"],
                "success": out == total, "done_turn": self.done_at, "progress": round(out / total, 2) if total else 0.0}

    def question(self, world, agent):
        opts, _ = _move_calls(self.rng)
        if self.side(agent) == "chaser":
            live = self.runners()
            if not live:
                return "Every runner has been tagged. Use look() to see the grid. What does look() return?", ["look"]
            r = min(live, key=lambda x: (dist(agent.pos, x.pos), x.id))
            if dist(agent.pos, r.pos) <= 1:
                return f"Runner {r.name} is right next to you at {r.pos}. Call tag(\"{r.name}\") to tag it. What does tag() return?", ["tag"]
            agent.memory["q_move"] = (agent.pos, r.pos)
            return (f"You are at {agent.pos}. The nearest runner, {r.name}, is at {r.pos}. Which step gets you closer to it? "
                    f"Call one of {opts}. What does move() return?"), ["move"]
        if agent.name in self.tagged:
            return "You have been tagged and are out of the game. Use look() to watch. What does look() return?", ["look"]
        chasers = [a for a in world.agents if self.side(a) == "chaser"]
        c = min(chasers, key=lambda x: (dist(agent.pos, x.pos), x.id))
        agent.memory["q_move"] = (agent.pos, c.pos)
        return (f"You are at {agent.pos}. The nearest chaser, {c.name}, is at {c.pos}. Which step gets you further away from it? "
                f"Call one of {opts}. What does move() return?"), ["move"]

    def after_turn(self, world, agent, record):
        q = agent.memory.pop("q_move", None)
        d = _moved(record)
        if q is None or d is None:
            return
        start, other = q
        side = self.side(agent)
        helps = (lambda p: dist(p, other) < dist(start, other)) if side == "chaser" else (lambda p: dist(p, other) > dist(start, other))
        st = self.moves[side]
        st["made"] += 1
        st["chance"] += sum(helps(_step(start, x, self.n)) for x in DIRS) / 4
        if d == "invalid":
            st["invalid"] += 1
        else:
            st["good"] += int(helps(_step(start, d, self.n)))


class GoldRush(Task):
    """Free for all: coins lie on the grid and every robot has grab(), which takes the coin on its cell; the first
    robot to grab a coin gets it. Each turn a robot is shown its three nearest coins in a random order and asked which
    way to step towards the nearest one, so it has to compare the distances itself. Scored: coins per robot (the most
    wins), and the share of moves that closed in on a truly nearest coin."""

    name = "gold_rush"
    comm_range = 3
    sight = 3

    def __init__(self, n, n_agents, rng):
        super().__init__(n, n_agents, rng)
        k = min(max(6, 2 * n_agents), (n * n) // 3)
        for c in rng.sample([(x, y) for x in range(n) for y in range(n)], k):
            self.objects[c] = "coin"
        self.total = k
        self.got: dict[str, int] = {}
        self.moves = {"nearest": 0, "any": 0, "made": 0, "invalid": 0, "chance": 0.0}

    def coins(self):
        return [c for c, w in self.objects.items() if w == "coin"]

    def agent_info(self, agent):
        return {"coins": self.got.get(agent.name, 0)}

    def side_of(self, agent):  # alternate robots, so neither player gets the better starting cells by construction
        return "A" if agent.id % 2 == 0 else "B"

    def sides(self):
        return {"A": "R1, R3, ... (odd robots)", "B": "R2, R4, ... (even robots)"}

    def goal(self, agent):
        return "Coins lie on the grid. Walk to them and pick them up with grab() on their cell: the first robot to grab a coin gets it, and the robot with the most coins wins."

    def tools(self, world, agent):
        from slm.tools.functions import FunctionDecl

        self.world = world

        def grab(*_, _a=agent):
            if self.objects.get(_a.pos) != "coin":
                return f"There is no coin at {_a.pos} (someone may have taken it)."
            del self.objects[_a.pos]
            self.got[_a.name] = self.got.get(_a.name, 0) + 1
            self.events.append(f"turn {world.turn + 1}: {_a.name} grabbed the coin at {_a.pos}")
            if not self.coins() and self.done_at is None:
                self.done_at = world.turn + 1
            return f"You picked up the coin at {_a.pos}! You have {self.got[_a.name]} coin{'s' if self.got[_a.name] != 1 else ''}."
        return [FunctionDecl("grab", "def grab() -> str", "Picks up the coin on your cell, if one is still there.", grab)]

    def score(self, world):
        collected = sum(self.got.values())
        ranked = sorted(self.got.items(), key=lambda kv: (-kv[1], kv[0]))
        top = ranked[0][1] if ranked else 0
        leaders = [n for n, c in ranked if c == top] if top else []
        lead = None if not leaders else leaders[0] if len(leaders) == 1 else "tie: " + ", ".join(leaders)
        done = collected == self.total
        verdict = None if not done else f"{lead} wins with {top} coins" if lead and not lead.startswith("tie") else f"{lead} with {top} coins each"
        by_side = {"A": 0, "B": 0}
        for a in world.agents:
            by_side[self.side_of(a)] += self.got.get(a.name, 0)
        winner_side = None if not done else "A" if by_side["A"] > by_side["B"] else "B" if by_side["B"] > by_side["A"] else "tie"
        return {"collected": collected, "coins": self.total, "leader": lead, "leader_coins": top, "verdict": verdict,
                "coins_A": by_side["A"], "coins_B": by_side["B"], "winner_side": winner_side,
                "standings": ", ".join(f"{n} {c}" for n, c in ranked[:6]) or "-",
                "nearest_coin_moves": self._rate("nearest"), "nearest_chance": self._rate("chance"), "any_coin_moves": self._rate("any"),
                "moves": self.moves["made"], "invalid_moves": self.moves["invalid"],
                "success": done, "done_turn": self.done_at, "progress": round(collected / self.total, 2)}

    def _rate(self, k):
        return round(self.moves[k] / self.moves["made"], 2) if self.moves["made"] else None

    def question(self, world, agent):
        coins = self.coins()
        if agent.pos in coins:  # no coordinates here: with them the model wrote get(3, 2) instead of grab() (smoke run)
            return "You are standing on a coin. Call grab() to pick it up. What does grab() return?", ["grab"]
        if not coins:
            return "Every coin has been picked up. Use look() to see the grid. What does look() return?", ["look"]
        near = sorted(coins, key=lambda c: (dist(agent.pos, c), c))[:3]
        best = dist(agent.pos, near[0])
        shown = near[:]
        self.rng.shuffle(shown)
        agent.memory["q_move"] = (agent.pos, [c for c in near if dist(agent.pos, c) == best], near)
        listed = ", ".join(str(c) for c in shown[:-1]) + (" and " if len(shown) > 1 else "") + str(shown[-1])
        return (f"You are at {agent.pos}. Coins lie at {listed}. Which coin is nearest to you? Step towards it: call one of "
                f"{_move_calls(self.rng)[0]}. What does move() return?"), ["move"]

    def after_turn(self, world, agent, record):
        q = agent.memory.pop("q_move", None)
        d = _moved(record)
        if q is None or d is None:
            return
        start, nearest, listed = q
        closer = lambda p, cs: any(dist(p, c) < dist(start, c) for c in cs)  # noqa: E731
        m = self.moves
        m["made"] += 1
        m["chance"] += sum(closer(_step(start, x, self.n), nearest) for x in DIRS) / 4
        if d == "invalid":
            m["invalid"] += 1
            return
        p = _step(start, d, self.n)
        m["nearest"] += int(closer(p, nearest))
        m["any"] += int(closer(p, listed))


TASKS = {"key_door": KeyDoor, "relay": Relay, "triangulate": Triangulate, "team_relay": TeamRelay, "tag": Tag, "gold_rush": GoldRush}


# ------------------------------------------------------------------------------------------------ the world
class World:
    def __init__(self, task: str, n: int = 8, n_agents: int = 4, seed: int = 0, max_history: int = 1) -> None:
        self.rng = random.Random(seed)
        self.n, self.seed, self.max_history = n, seed, max_history
        if task not in TASKS:
            raise ValueError(f"unknown arena task {task!r}; known: {', '.join(sorted(TASKS))}")
        if n_agents > 32:
            raise ValueError("at most 32 robots")
        TASKS[task].check(n, n_agents)
        self.agents = [Agent(i, f"R{i + 1}", 0, 0, []) for i in range(n_agents)]
        self.task: Task = TASKS[task](n, n_agents, self.rng)
        self.task.place_agents(self)
        self.turn = 0
        self.decls: dict[int, list] = {}
        self.pending: list[tuple[Agent, str]] = []  # (speaker, text) said this turn, delivered at the next observation
        self.player_of: dict[int, str] = {}  # robot id -> player key ("A"/"B"), set by the Runner
        for a in self.agents:
            self.decls[a.id] = self._common_tools(a) + self.task.tools(self, a)
            a.tools = [d.name for d in self.decls[a.id]]

    # --- tools every robot has
    # Every tool binds its robot (and its door / team) as a keyword-only default after `*_`: extra positional arguments
    # land in `*_` and are ignored. With the robot as a plain default (`def grab(_a=agent)`) the model's grab("pick") bound
    # _a to the string "pick" and the tool crashed (AttributeError), seen in a gold rush on 2026-10-10.
    def _common_tools(self, agent: Agent) -> list:
        from slm.tools.functions import FunctionDecl

        def move(direction: str, *_, _a=agent):
            d = DIRS.get(str(direction).strip().lower())
            if d is None:
                return f"Unknown direction {direction!r}. Use north, south, east or west."
            nx, ny = _a.x + d[0], _a.y + d[1]
            if not (0 <= nx < self.n and 0 <= ny < self.n):
                return f"Blocked: the grid edge. You are still at {_a.pos}."
            _a.x, _a.y = nx, ny
            return f"You moved {direction} to {_a.pos}."

        def say(message: str, *_, _a=agent):
            text = " ".join(str(message).split())[:200]
            self.pending.append((_a, text))
            near = [b.name for b in self.agents if b is not _a and dist(_a.pos, b.pos) <= self.task.comm_range]
            return f"You said: {text!r}. Heard by: {', '.join(near) if near else 'nobody (no robot within range)'}."

        def look(*_, _a=agent):
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
        return {"turn": self.turn, "n": self.n, "task": self.task.name,
                "agents": [{"id": a.id, "name": a.name, "x": a.x, "y": a.y, "tools": a.tools, "player": self.player_of.get(a.id, "A"),
                            **self.task.agent_info(a)} for a in self.agents],
                "objects": [{"x": c[0], "y": c[1], "what": w} for c, w in self.task.objects.items()], "score": self.task.score(self), "events": list(self.task.events)}


# ------------------------------------------------------------------------------------------------ players
_ARG = r"""\(\s*(?:"([^"]*)"|'([^']*)'|([^()"']*?))\s*\)"""


def text_action(text: str, tool_names: list[str]) -> tuple[str, list[str]] | None:
    """The call a plain-text player wrote, as `(name, args)`, or None. The questions offer complete calls ('call one of
    move("west"), ...'), so a model that has never seen our tool protocol can still act by writing one.

    Forms: `move("west")` (any quotes, or none) and the prose form `Move "west"` (the name, case-insensitive, then a
    quoted value: SmolLM2 answers "The final answer is: 4. Move "south""). Where: the last call on its final answer line
    (a line starting with '####' or containing "final answer"); with no such line, the call it names, if it names exactly
    one. An answer that names several different calls and settles on none is restating the options, not choosing
    (taking the last one would credit whichever option the list happened to end on): no action."""
    if not tool_names or not text:
        return None
    names = "|".join(map(re.escape, tool_names))
    rx = re.compile(r"(?i)\b(" + names + r")\s*" + _ARG + r"|(?i:\b(" + names + r"))\s+(?:\"([^\"]+)\"|'([^']+)')")

    def calls(chunk):
        out = []
        for m in rx.finditer(chunk):
            if m.group(1):
                arg = next((g for g in m.groups()[1:4] if g is not None), "").strip()
                out.append((m.group(1).lower(), [arg] if arg else []))
            else:
                out.append((m.group(5).lower(), [(m.group(6) or m.group(7)).strip()]))
        return [(n, a) for n, a in out if n in tool_names]

    finals = [ln for ln in text.splitlines() if ln.strip().startswith("####") or "final answer" in ln.lower()]
    if finals:
        found = calls(finals[-1])
        if found:
            return found[-1]
    found = calls(text)
    distinct = {(n, tuple(a)) for n, a in found}
    return found[-1] if len(distinct) == 1 else None


@dataclass
class Player:
    """One model playing some of the robots. `native`: our checkpoint, acting through declared functions and
    <|python_call|> inside the think span (`sample_with_tools`); otherwise a chat model that has never seen that protocol
    (an external open-weight model, `slm.eval.external.HfChatModel`), which answers in plain text and whose written call
    is executed (`text_action`). `generate` overrides the model call (tests, the portal's device-aware sampler)."""

    label: str
    model: object = None
    tok: object = None
    native: bool = True
    generate: object = None


class Runner:
    """Runs a world: every turn, each player generates for its robots in batches of at most `batch_size` (their
    conversations are different lengths; `sample_with_tools` takes a list of prompts). With one player every robot is
    that model; with `players={"A": ..., "B": ...}` a competitive task's sides (`Task.side_of`) are played by different
    models -- ours against ours, or against an external chat model. `batch_size` bounds the rows of one generation call:
    smaller costs time, saves peak VRAM (two resident models and their KV caches can spill past the 16 GiB card on WDDM,
    which shows as a 60x slowdown, not an error). `generate` (one player) can be injected for tests."""

    def __init__(self, world: World, model=None, tok=None, max_new_tokens: int = 128, max_calls: int = 4, generate=None,
                 players: dict | None = None, batch_size: int | None = None, failed_history: str = "keep",
                 retry_temperature: float | None = 0.8) -> None:
        from slm.tools.pysandbox import PySession

        self.world, self.model, self.tok = world, model, tok
        self.max_new_tokens, self.max_calls = max_new_tokens, max_calls
        self.sessions = {a.id: PySession() for a in world.agents}
        self.players = players or {"A": Player("model", model, tok, native=True, generate=generate)}
        self.batch_size = batch_size if batch_size and batch_size > 0 else None
        two = "B" in self.players
        for a in world.agents:
            world.player_of[a.id] = world.task.side_of(a) if two else "A"
        self.vram_peak_gib = 0.0
        # Decoding is greedy, so an identical prompt gives an identical answer. A robot whose turn failed gets the same
        # question again: `failed_history` says whether the failed exchange stays in its context ("keep": the model may
        # copy it, as a gold-rush robot copied get("pick") six times) or not ("drop": the prompt is then identical, and a
        # deterministic failure repeats forever, as a team-relay source repeated says("...")); `retry_temperature`, when
        # set, samples that robot's next turn instead (top-p 0.95), so a failure need not repeat. A turn with no call at all
        # counts as failed for the retry: under greedy decoding an unchanged question repeats the silence too. Measured
        # (results §28): keep + retry at 0.8 decided 5 of 5 team relays with the fewest failed calls; "drop" doubled them.
        if failed_history not in ("keep", "drop"):
            raise ValueError("failed_history must be 'keep' or 'drop'")
        self.failed_history, self.retry_temperature = failed_history, retry_temperature
        self.current_temperature = 0.0  # the temperature of the generation call in flight (the portal's generate reads it)

    @property
    def generate(self):  # the single player's generator (the portal and older callers set it)
        return self.players["A"].generate

    @generate.setter
    def generate(self, fn):
        self.players["A"].generate = fn

    def _native(self, player: Player, prompts: list, sessions: list) -> list:
        if player.generate is not None:
            return player.generate(prompts, sessions)
        import torch

        from slm.tools.loop import sample_with_tools
        from slm.utils.sdpa import sdpa_context

        gen = torch.Generator(device=next(player.model.parameters()).device); gen.manual_seed(self.world.seed * 1000 + self.world.turn)
        t = self.current_temperature
        with torch.no_grad(), sdpa_context("decode"):
            if t and t > 0:  # a retry after a failed turn: sampled, so a deterministic failure need not repeat
                return sample_with_tools(player.model, player.tok, prompts, self.max_new_tokens, t, 0.95, 0, gen, max_calls=self.max_calls, sessions=sessions)
            return sample_with_tools(player.model, player.tok, prompts, self.max_new_tokens, 1.0, 1.0, 1, gen, max_calls=self.max_calls, sessions=sessions)

    def _text(self, player: Player, prompts: list, sessions: list, tool_names: list, no_args: list | None = None) -> list:
        """A plain-text player: generate (greedy, like ours), then execute the call each answer wrote."""
        if player.generate is not None:
            texts = player.generate(prompts, sessions)
        else:
            t = self.current_temperature or 0.0
            gens = player.model.batch_generate_chat(prompts, self.max_new_tokens, temperature=t, top_p=0.95 if t else 1.0,
                                                    seed=self.world.seed * 1000 + self.world.turn, batch_size=max(1, len(prompts)))
            texts = [g.text for g in gens]
        out = []
        for i, (text, sess, names) in enumerate(zip(texts, sessions, tool_names)):
            act = text_action(text, names)
            calls = []
            if act:
                name, args = act
                if no_args and name in no_args[i]:  # read_code('secret_code'): the tool takes nothing; the argument is noise
                    args = []
                code = f"{name}({', '.join(repr(x) for x in args)})"
                try:
                    calls.append([code, str(sess.functions[name](*args))])
                except Exception as e:  # noqa: BLE001 - a wrong argument count is the player's error, recorded as one
                    calls.append([code, f"error: {type(e).__name__}: {e}"])
            out.append({"think": None, "answer": text, "calls": calls})
        return out

    def step(self) -> dict:
        from slm.data.chat import format_chat, parse_assistant
        from slm.tools.functions import functions_env
        from slm.tools.pysandbox import PySession

        w = self.world
        w.deliver()
        prompts, sessions, obs, declared, no_args, history_seen = {}, {}, {}, {}, {}, {}
        for a in w.agents:
            player = self.players[w.player_of.get(a.id, "A")]
            msgs = w.messages(a)
            tools = w.turn_tools(a)
            self.sessions[a.id] = PySession()  # fresh each turn, with only this turn's tool: a tool declared earlier stays callable otherwise
            self.sessions[a.id].register(functions_env(tools))
            if player.native and player.tok is not None:
                prompts[a.id] = format_chat(player.tok, msgs, add_generation_prompt=True, think_required=True, functions=tools).ids
            else:
                prompts[a.id] = [{k: v for k, v in m.items() if k != "ids"} for m in msgs] if not player.native else msgs
            sessions[a.id] = self.sessions[a.id]; obs[a.id] = msgs[-1]["content"]; declared[a.id] = [d.name for d in tools]
            history_seen[a.id] = [{"role": m["role"], "content": m.get("content", "")} for m in msgs[:-1]]  # what precedes the question
            no_args[a.id] = {d.name for d in tools if re.search(r"\(\s*\)", d.signature)}
        t0 = time.time()
        tcs: dict[int, object] = {}
        cuda = _cuda_ready()
        if cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()
        for key, player in self.players.items():
            mine = [a for a in w.agents if w.player_of.get(a.id, "A") == key]
            groups = [(0.0, [a.id for a in mine if not (self.retry_temperature and a.memory.get("_failed"))])]
            if self.retry_temperature:
                groups.append((self.retry_temperature, [a.id for a in mine if a.memory.get("_failed")]))
            chunks = []
            for temp, ids in groups:
                step = self.batch_size or max(1, len(ids))
                chunks += [(temp, ids[i:i + step]) for i in range(0, len(ids), step)]
            for temp, chunk in chunks:
                self.current_temperature = temp
                if player.native:
                    outs = self._native(player, [prompts[j] for j in chunk], [sessions[j] for j in chunk])
                else:
                    outs = self._text(player, [prompts[j] for j in chunk], [sessions[j] for j in chunk], [declared[j] for j in chunk],
                                      [no_args[j] for j in chunk])
                tcs.update(zip(chunk, outs))
        vram = None
        if cuda:
            import torch

            vram = {"peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2)}
            self.vram_peak_gib = max(self.vram_peak_gib, vram["peak_gib"])
        records = []
        for a in w.agents:
            key = w.player_of.get(a.id, "A")
            player, tc, o, tools = self.players[key], tcs[a.id], obs[a.id], declared[a.id]
            native_ids = player.native and player.tok is not None and not isinstance(tc, dict)
            if native_ids:  # our checkpoint's generated turn: parse the think span and the answer out of the ids
                p = parse_assistant(player.tok, tc.ids)
                think, answer = p["think"], p["answer"]
            elif isinstance(tc, dict):  # a plain-text player, or a scripted one in tests
                think, answer = tc.get("think"), tc.get("answer")
            else:
                think, answer = getattr(tc, "think", None), getattr(tc, "answer", None)
            calls = [list(c) for c in getattr(tc, "calls", [])] if not isinstance(tc, dict) else tc.get("calls", [])
            # A turn whose every call failed is not kept in the robot's history: the question restates what the robot needs,
            # so the history only adds an example to copy, and a failed one is copied (a gold-rush robot on a coin called
            # get("pick") six turns running, 2026-10-10). Success or a plain answer is kept as before.
            failed = bool(calls) and all(str(c[1]).lstrip().startswith("error") for c in calls)
            a.memory["_failed"] = failed or not calls  # the next turn is sampled when retry_temperature is set
            if not failed or self.failed_history == "keep":
                # the answer is kept whole: a plain-text player sees this content as its earlier reply, and a clipped one taught
                # SmolLM2 to stop mid-word at the same place every turn (2026-10-11); our checkpoints replay `ids` instead
                a.history += [{"role": "user", "content": o}, {"role": "assistant", "content": (answer or "").strip(), "ids": list(tc.ids) if native_ids else None}]
                if a.history[-1]["ids"] is None:
                    a.history[-1].pop("ids")
            # the robot's context window this turn, for the portal's follow panel: what came before the question, how many
            # tokens the prompt and the reply took (our checkpoints; an external model's tokenizer is its own), the window
            ctx = getattr(getattr(player.model, "cfg", None), "max_seq_len", None) or getattr(player.model, "max_positions", None)
            rec = {"turn": w.turn + 1, "agent": a.name, "pos": a.pos, "observation": o, "status": w.status(a), "think": think, "calls": calls, "answer": answer,
                   "n_calls": len(calls), "tools": tools, "player": key, "history": history_seen[a.id],
                   "prompt_tokens": len(prompts[a.id]) if native_ids else None, "gen_tokens": len(tc.ids) if native_ids else None, "context": ctx}
            w.task.after_turn(w, a, rec)
            a.log.append(rec); records.append(rec)
        w.turn += 1
        return {"turn": w.turn, "seconds": round(time.time() - t0, 2), "records": records, "state": w.state(), "vram": vram}

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


def _cuda_ready() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.is_initialized()
    except Exception:  # noqa: BLE001
        return False


def load_player(spec: str, tokenizer: str, device: str = "cuda") -> Player:
    """A player from a checkpoint path (ours: native tool protocol) or `external:<name>` (a local open-weight chat
    model from slm.eval.external: plain-text calls)."""
    from pathlib import Path as _P

    if spec.startswith("external:"):
        from slm.eval import external as E

        name = spec.split(":", 1)[1]
        return Player(name, E.load_external(name, device=device), None, native=False)
    from slm.data.tokenizer import SlmTokenizer
    from slm.eval.quality import load_model

    model, _ = load_model(_P(spec), device)
    return Player(_P(spec).parent.parent.name + "/" + _P(spec).name, model, SlmTokenizer.load(tokenizer), native=True)


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
    ap.add_argument("--opponent", default=None, help="side B of a competitive task: a checkpoint path or external:<name>")
    ap.add_argument("--batch-size", type=int, default=None, help="max robots per generation call (lower = less peak VRAM)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    players = {"A": load_player(a.checkpoint, a.tokenizer)}
    if a.opponent:
        players["B"] = load_player(a.opponent, a.tokenizer)
    world = World(a.task, a.n, a.agents, a.seed)
    runner = Runner(world, max_new_tokens=a.max_new, players=players, batch_size=a.batch_size)

    def show(out):
        print(f"--- turn {out['turn']} ({out['seconds']}s)")
        for r in out["records"]:
            print(f"  {r['agent']} @{r['pos']}  calls={[c[0][:60] for c in r['calls']]}  -> {(r['answer'] or '').strip()[:80]!r}")
    res = runner.run(a.turns, on_turn=show)
    res["players"] = {k: {"label": p.label, "native": p.native, "plays": world.task.sides().get(k)} for k, p in players.items()}
    res["vram_peak_gib"] = runner.vram_peak_gib
    print(json.dumps({k: v for k, v in res.items() if k != "transcript"}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
