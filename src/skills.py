"""Scripted macros: path, face, interact. One goal consumes many env steps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.crafter_env import (
    BLOCKING_OBJECTS,
    DELTA_MOVE,
    PLACE_WHERE,
    WALKABLE,
    CrafterTextEnv,
    GameState,
    facing_from_delta,
)

MAX_MACRO_STEPS = 80
MAX_SLEEP_STEPS = 120
MAX_WANDER_STEPS = 24


@dataclass
class WorldMap:
    """Tiles seen so far (grows as the agent walks)."""

    tiles: dict[tuple[int, int], str] = field(default_factory=dict)
    camp: tuple[int, int] | None = None

    def update(self, state: GameState) -> None:
        for tile in state.tiles:
            self.tiles[tile.pos] = tile.name
            if tile.name == "table" and self.camp is None:
                self.camp = tile.pos

    def named(self, name: str) -> list[tuple[int, int]]:
        return [pos for pos, n in self.tiles.items() if n == name]

    def get(self, pos: tuple[int, int]) -> str | None:
        return self.tiles.get(pos)


@dataclass
class MacroResult:
    ok: bool
    goal: str
    reason: str
    actions: list[str] = field(default_factory=list)
    reward: float = 0.0
    steps: int = 0
    state: GameState | None = None
    done: bool = False


def execute_goal(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
    goal: str,
    pair_stations: bool = True,
) -> MacroResult:
    """Dispatch a planner goal to a scripted macro."""
    if goal == "flee":
        return _flee(env, state, world)
    if goal == "attack":
        return _face_and_do(env, state, world, ("zombie", "skeleton"), goal)
    if goal == "kite":
        return _kite(env, state, world)
    if goal == "sleep_safe":
        return _sleep_safe(env, state, world)
    if goal == "drink":
        return _face_and_do(env, state, world, ("water",), goal)
    if goal == "eat":
        return _face_and_do(env, state, world, ("cow", "plant"), goal)
    if goal == "collect_wood":
        return _face_and_do(env, state, world, ("tree",), goal)
    if goal == "collect_sapling":
        return _face_and_do(env, state, world, ("grass",), goal)
    if goal == "collect_stone":
        return _face_and_do(env, state, world, ("stone",), goal)
    if goal == "collect_coal":
        return _face_and_do(env, state, world, ("coal",), goal)
    if goal == "collect_iron":
        return _face_and_do(env, state, world, ("iron",), goal)
    if goal == "collect_diamond":
        return _face_and_do(env, state, world, ("diamond",), goal)
    if goal == "place_table":
        return _place_item(env, state, world, "table")
    if goal == "place_furnace":
        if pair_stations and world.named("table"):
            return _relocate_furnace(env, state, world)
        return _place_item(env, state, world, "furnace")
    if goal == "place_plant":
        return _place_item(env, state, world, "plant")
    if goal == "place_stone":
        return _place_item(env, state, world, "stone")
    if goal.startswith("make_"):
        return _craft(env, state, world, goal[len("make_"):])
    if goal == "relocate_furnace":
        return _relocate_furnace(env, state, world)
    if goal == "craft_iron":
        return _craft_iron(env, state, world)
    if goal == "retreat_to_camp":
        return _retreat_to_camp(env, state, world)
    if goal == "farm_plants":
        return _farm_plants(env, state, world)
    if goal == "explore":
        return _explore(env, state, world)
    return MacroResult(False, goal, f"unknown goal {goal}", state=state)


def _step(
    env: CrafterTextEnv,
    world: WorldMap,
    action: str,
    result: MacroResult,
) -> tuple[GameState, bool]:
    state, reward, done, _ = env.step(action)
    world.update(state)
    result.actions.append(action)
    result.reward += reward
    result.steps += 1
    result.state = state
    result.done = done
    return state, done


def _in_bounds(pos: tuple[int, int], size: int = 64) -> bool:
    return 0 <= pos[0] < size and 0 <= pos[1] < size


def _is_walkable(world: WorldMap, pos: tuple[int, int], unknown_ok: bool = False) -> bool:
    if not _in_bounds(pos):
        return False
    name = world.get(pos)
    if name is None:
        return unknown_ok
    if name in BLOCKING_OBJECTS or name in {"tree", "water", "stone", "lava", "coal", "iron", "diamond", "table", "furnace"}:
        return False
    return name in WALKABLE


def _neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = pos
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def _bfs(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    world: WorldMap,
    max_nodes: int = 2000,
) -> list[tuple[int, int]] | None:
    if start in goals:
        return []
    queue = deque([start])
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue and len(came) < max_nodes:
        cur = queue.popleft()
        for nxt in _neighbors(cur):
            if nxt in came:
                continue
            if nxt not in goals and not _is_walkable(world, nxt, unknown_ok=True):
                continue
            came[nxt] = cur
            if nxt in goals:
                path = [nxt]
                while came[path[-1]] is not None:
                    path.append(came[path[-1]])  # type: ignore[arg-type]
                path.reverse()
                return path[1:]
            queue.append(nxt)
    return None


def _stand_spots(world: WorldMap, targets: list[tuple[int, int]]) -> set[tuple[int, int]]:
    spots: set[tuple[int, int]] = set()
    for t in targets:
        for n in _neighbors(t):
            name = world.get(n)
            if name is None or _is_walkable(world, n):
                spots.add(n)
    return spots


def _follow_path(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
    path: list[tuple[int, int]],
    result: MacroResult,
    cap: int,
) -> tuple[GameState, bool]:
    for dest in path:
        if result.steps >= cap or result.done:
            break
        dx = dest[0] - state.x
        dy = dest[1] - state.y
        move = DELTA_MOVE.get((dx, dy))
        if not move:
            result.reason = f"non-adjacent path step {state.pos} -> {dest}"
            return state, True
        prev = state.pos
        state, done = _step(env, world, move, result)
        if done:
            return state, True
        if state.pos == prev:
            result.reason = f"blocked moving {move} from {prev}"
            return state, True
    return state, False


def _face(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
    target: tuple[int, int],
    result: MacroResult,
) -> tuple[GameState, bool]:
    dx = target[0] - state.x
    dy = target[1] - state.y
    if abs(dx) + abs(dy) != 1:
        result.reason = f"cannot face non-adjacent {target} from {state.pos}"
        return state, True
    move = DELTA_MOVE[(dx, dy)]
    want = facing_from_delta(dx, dy)
    if state.facing == want:
        return state, False
    state, done = _step(env, world, move, result)
    return state, done


def _face_and_do(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
    names: tuple[str, ...],
    goal: str,
    cap: int = MAX_MACRO_STEPS,
    positions: list[tuple[int, int]] | None = None,
) -> MacroResult:
    result = MacroResult(False, goal, "", state=state)
    world.update(state)

    targets: list[tuple[int, int]] = list(positions or [])
    for name in names:
        targets.extend(world.named(name))
    if not targets:
        wander = _explore(env, state, world, cap=MAX_WANDER_STEPS)
        result.actions.extend(wander.actions)
        result.reward += wander.reward
        result.steps += wander.steps
        result.state = wander.state
        result.done = wander.done
        result.reason = f"no {','.join(names)} in map; wandered"
        return result

    start = state.pos
    if start in targets:
        # Standing on grass while collecting saplings: step to a neighbor first.
        for n in _neighbors(start):
            if _is_walkable(world, n):
                state, done = _follow_path(env, state, world, [n], result, cap)
                if done:
                    result.reason = result.reason or "died while stepping off target"
                    return result
                start = state.pos
                break

    spots = _stand_spots(world, targets)
    if start in spots:
        path: list[tuple[int, int]] = []
    else:
        path = _bfs(start, spots, world) or []
        if not path:
            wander = _explore(env, state, world, cap=MAX_WANDER_STEPS)
            result.actions.extend(wander.actions)
            result.reward += wander.reward
            result.steps += wander.steps
            result.state = wander.state
            result.done = wander.done
            result.reason = f"no path to {','.join(names)}"
            return result

    state, stopped = _follow_path(env, state, world, path, result, cap)
    if stopped:
        result.reason = result.reason or "stopped on the way"
        return result

    target = min(targets, key=lambda p: abs(p[0] - state.x) + abs(p[1] - state.y))
    if abs(target[0] - state.x) + abs(target[1] - state.y) != 1:
        result.reason = f"ended {state.pos} not adjacent to {target}"
        return result

    state, done = _face(env, state, world, target, result)
    if done:
        result.reason = "died while facing target"
        return result

    before = dict(state.inventory)
    before_achs = dict(state.achievements)
    state, done = _step(env, world, "do", result)
    if done and state.vitals.get("health", 0) <= 0:
        result.reason = "died while doing"
        return result

    changed = state.inventory != before or state.achievements != before_achs
    result.ok = changed or goal in {"attack", "collect_sapling"}
    result.reason = "did" if result.ok else f"do did nothing (front={state.tile_in_front})"
    result.state = state
    return result


def _place_item(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
    kind: str,
    cap: int = MAX_MACRO_STEPS,
) -> MacroResult:
    goal = f"place_{kind}"
    result = MacroResult(False, goal, "", state=state)
    world.update(state)
    allowed = PLACE_WHERE.get(kind, WALKABLE)

    candidates = [
        pos for pos, name in world.tiles.items()
        if name in allowed
    ]
    # Prefer a tile adjacent to the player, then nearest walkable-adjacent grounds.
    spots_to_ground: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for ground in candidates:
        for stand in _neighbors(ground):
            if stand == ground:
                continue
            if _is_walkable(world, stand) or stand == state.pos:
                spots_to_ground.append((stand, ground))
    if not spots_to_ground:
        result.reason = f"no placeable {kind} tile in map"
        return result

    spots_to_ground.sort(
        key=lambda sg: abs(sg[0][0] - state.x) + abs(sg[0][1] - state.y)
    )
    dest, ground = spots_to_ground[0]
    if dest != state.pos:
        path = _bfs(state.pos, {dest}, world)
        if path is None:
            result.reason = f"no path to place {kind}"
            return result
        state, stopped = _follow_path(env, state, world, path, result, cap)
        if stopped:
            result.reason = result.reason or "stopped walking to place"
            return result

    state, done = _face(env, state, world, ground, result)
    if done:
        result.reason = "died while facing place tile"
        return result

    state, done = _step(env, world, f"place_{kind}", result)
    result.state = state
    if state.achievements.get(f"place_{kind}"):
        result.ok = True
        result.reason = f"placed {kind}"
    else:
        result.reason = f"place_{kind} failed (front={state.tile_in_front})"
    return result


def _craft(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
    kind: str,
    cap: int = MAX_MACRO_STEPS,
) -> MacroResult:
    goal = f"make_{kind}"
    result = MacroResult(False, goal, "", state=state)
    world.update(state)

    tables = world.named("table")
    if not tables:
        result.reason = "no table in map"
        return result

    furnaces = world.named("furnace")
    need_furnace = kind.startswith("iron")
    if need_furnace and not furnaces:
        result.reason = "no furnace in map"
        return result

    def ok_stand(pos: tuple[int, int]) -> bool:
        near_table = any(max(abs(pos[0] - t[0]), abs(pos[1] - t[1])) <= 1 for t in tables)
        if not near_table:
            return False
        if need_furnace:
            return any(max(abs(pos[0] - f[0]), abs(pos[1] - f[1])) <= 1 for f in furnaces)
        return True

    stands: set[tuple[int, int]] = set()
    for table in tables:
        for n in _neighbors(table):
            if (n == state.pos or _is_walkable(world, n)) and ok_stand(n):
                stands.add(n)
    if state.pos not in stands:
        path = _bfs(state.pos, stands, world)
        if path is None:
            result.reason = "no path to table" + ("+furnace" if need_furnace else "")
            return result
        state, stopped = _follow_path(env, state, world, path, result, cap)
        if stopped:
            result.reason = result.reason or "stopped walking to craft"
            return result

    before = state.inventory.get(kind, 0)
    state, done = _step(env, world, f"make_{kind}", result)
    result.state = state
    if state.inventory.get(kind, 0) > before or state.achievements.get(f"make_{kind}"):
        result.ok = True
        result.reason = f"crafted {kind}"
    else:
        result.reason = f"make_{kind} failed (inv wood={state.inventory.get('wood', 0)})"
    return result


def _relocate_furnace(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
    cap: int = MAX_MACRO_STEPS,
) -> MacroResult:
    """Place a furnace on a tile adjacent to an existing table."""
    result = MacroResult(False, "relocate_furnace", "", state=state)
    world.update(state)
    if state.inventory.get("stone", 0) < 4:
        result.reason = "need 4 stone to relocate furnace"
        return result
    tables = world.named("table")
    if not tables:
        result.reason = "no table to relocate next to"
        return result

    allowed = PLACE_WHERE["furnace"]
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for table in tables:
        for ground in _neighbors(table):
            name = world.get(ground)
            if name in allowed or name is None:
                for stand in _neighbors(ground):
                    if stand == state.pos or _is_walkable(world, stand, unknown_ok=True):
                        candidates.append((stand, ground))
    if not candidates:
        result.reason = "no tile next to table for a furnace"
        return result
    candidates.sort(key=lambda sg: abs(sg[0][0] - state.x) + abs(sg[0][1] - state.y))
    dest, ground = candidates[0]
    if dest != state.pos:
        path = _bfs(state.pos, {dest}, world)
        if path is None:
            result.reason = "no path to table-adjacent furnace tile"
            return result
        state, stopped = _follow_path(env, state, world, path, result, cap)
        if stopped:
            result.reason = result.reason or "stopped walking to relocate"
            return result

    state, done = _face(env, state, world, ground, result)
    if done:
        result.reason = "died facing relocate tile"
        return result
    before = len(world.named("furnace"))
    state, done = _step(env, world, "place_furnace", result)
    result.state = state
    if state.achievements.get("place_furnace") or len(world.named("furnace")) > before:
        result.ok = True
        result.reason = "placed furnace next to table"
    else:
        result.reason = f"relocate furnace failed (front={state.tile_in_front})"
    return result


def _has_escape(state: GameState, world: WorldMap) -> bool:
    return any(_is_walkable(world, n) for n in _neighbors(state.pos))


def _stations_adjacent(world: WorldMap) -> bool:
    tables = world.named("table")
    furnaces = world.named("furnace")
    for table in tables:
        for furnace in furnaces:
            if max(abs(table[0] - furnace[0]), abs(table[1] - furnace[1])) <= 2:
                return True
    return False


def _absorb(into: MacroResult, other: MacroResult) -> MacroResult:
    into.actions.extend(other.actions)
    into.reward += other.reward
    into.steps += other.steps
    into.state = other.state or into.state
    into.done = other.done
    into.ok = other.ok
    into.reason = other.reason
    return into


def _craft_iron(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
) -> MacroResult:
    """Stone-pick ore run: coal → iron → camp → iron pickaxe."""
    result = MacroResult(False, "craft_iron", "", state=state)

    def abort_if_critical() -> bool:
        return state.vitals.get("health", 9) <= 2

    gathers = (
        ("coal", ("coal",), "collect_coal"),
        ("iron", ("iron",), "collect_iron"),
        ("wood", ("tree",), "collect_wood"),
    )
    for item, names, label in gathers:
        if state.inventory.get(item, 0) >= 1:
            continue
        if abort_if_critical():
            result.reason = "aborted ore run; health critical"
            result.state = state
            return result
        got = _face_and_do(env, state, world, names, label)
        _absorb(result, got)
        result.goal = "craft_iron"
        if got.done or got.state is None:
            return result
        state = got.state
        if state.inventory.get(item, 0) < 1:
            result.reason = f"ore run: no {item} yet"
            return result

    if abort_if_critical():
        result.reason = "aborted ore run; health critical"
        result.state = state
        return result

    if world.camp or world.named("table"):
        home = _retreat_to_camp(env, state, world)
        _absorb(result, home)
        result.goal = "craft_iron"
        if home.done or home.state is None:
            return result
        state = home.state

    if not world.named("furnace") or not _stations_adjacent(world):
        relocated = _relocate_furnace(env, state, world)
        _absorb(result, relocated)
        result.goal = "craft_iron"
        if relocated.done or relocated.state is None:
            return result
        state = relocated.state
        if not _stations_adjacent(world):
            result.reason = relocated.reason or "could not pair stations"
            return result

    pick = _craft(env, state, world, "iron_pickaxe")
    _absorb(result, pick)
    result.goal = "craft_iron"
    if result.done or result.state is None:
        return result
    state = result.state
    inv = state.inventory
    if result.ok and inv.get("wood", 0) >= 1 and inv.get("coal", 0) >= 1 and inv.get("iron", 0) >= 1:
        sword = _craft(env, state, world, "iron_sword")
        result.actions.extend(sword.actions)
        result.reward += sword.reward
        result.steps += sword.steps
        result.state = sword.state or state
        result.done = sword.done
        if sword.ok:
            result.reason = "crafted iron pickaxe and sword"
    elif result.ok:
        result.reason = "crafted iron pickaxe"
    return result


def _retreat_to_camp(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
    cap: int = MAX_MACRO_STEPS,
) -> MacroResult:
    result = MacroResult(False, "retreat_to_camp", "", state=state)
    camp = world.camp
    if camp is None:
        tables = world.named("table")
        if tables:
            camp = tables[0]
            world.camp = camp
    if camp is None:
        result.reason = "no camp"
        return result
    if max(abs(state.x - camp[0]), abs(state.y - camp[1])) <= 1:
        result.ok = True
        result.reason = "at camp"
        return result
    spots = _stand_spots(world, [camp])
    if not spots:
        result.reason = "no stand spots at camp"
        return result
    path = _bfs(state.pos, spots, world)
    if path is None:
        fled = _flee(env, state, world, cap=4)
        _absorb(result, fled)
        result.goal = "retreat_to_camp"
        result.reason = "no path to camp; fled"
        return result
    hops = 0
    while path and result.steps < cap and not result.done and hops < 40:
        hops += 1
        dest = path[0]
        path = path[1:]
        state, stopped = _follow_path(env, state, world, [dest], result, cap)
        if stopped:
            result.reason = result.reason or "stopped walking home"
            return result
        if _adjacent_monster(state):
            fled = _flee(env, state, world, cap=3)
            result.actions.extend(fled.actions)
            result.reward += fled.reward
            result.steps += fled.steps
            result.state = fled.state or state
            result.done = fled.done
            if fled.done or fled.state is None:
                result.reason = "died retreating"
                return result
            state = fled.state
            path = _bfs(state.pos, spots, world) or []
            if not path:
                result.reason = "cut off from camp"
                return result
    result.state = state
    if max(abs(state.x - camp[0]), abs(state.y - camp[1])) <= 1:
        result.ok = True
        result.reason = "retreated to camp"
    else:
        result.reason = "still en route to camp"
    return result


def _farm_plants(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
) -> MacroResult:
    plants = [
        p for p in world.named("plant")
        if abs(p[0] - state.x) + abs(p[1] - state.y) <= 8
    ]
    if plants:
        return _face_and_do(
            env, state, world, (), "farm_plants", positions=plants
        )
    if state.inventory.get("sapling", 0) > 0:
        planted = _place_item(env, state, world, "plant")
        planted.goal = "farm_plants"
        return planted
    grass = [
        p for p in world.named("grass")
        if abs(p[0] - state.x) + abs(p[1] - state.y) <= 4
    ]
    if not grass:
        wander = _explore(env, state, world, cap=6)
        wander.goal = "farm_plants"
        wander.reason = "no nearby grass; wandered"
        return wander
    sapling = _face_and_do(
        env, state, world, (), "collect_sapling", positions=grass
    )
    sapling.goal = "farm_plants"
    return sapling


def _kite(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
) -> MacroResult:
    """Hit once, then step back. Off unless sword-health-escape already passed."""
    if state.vitals.get("health", 9) < 6 or not _has_escape(state, world):
        fled = _flee(env, state, world, cap=4)
        fled.goal = "kite"
        fled.reason = f"{fled.reason}; kite gated so fled"
        return fled
    result = _face_and_do(env, state, world, ("zombie", "skeleton"), "kite")
    if result.done or result.state is None:
        return result
    state = result.state
    if _adjacent_monster(state) and state.vitals.get("health", 9) <= 7:
        retreat = _flee(env, state, world, cap=4)
        result.actions.extend(retreat.actions)
        result.reward += retreat.reward
        result.steps += retreat.steps
        result.state = retreat.state or state
        result.done = retreat.done
        if retreat.ok:
            result.ok = True
            result.reason = f"{result.reason}; kited"
    return result


def _adjacent_monster(state: GameState) -> bool:
    return any(t.name in {"zombie", "skeleton"} and t.chebyshev == 1 for t in state.tiles)


def _sleep_safe(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
) -> MacroResult:
    result = MacroResult(False, "sleep_safe", "", state=state)
    world.update(state)
    if _adjacent_monster(state):
        result.reason = "monster adjacent"
        return result

    start_energy = state.vitals.get("energy", 0)
    state, done = _step(env, world, "sleep", result)
    if done:
        result.reason = "died after sleep"
        return result

    while state.sleeping and result.steps < MAX_SLEEP_STEPS and not result.done:
        if _adjacent_monster(state):
            result.reason = "monster arrived while sleeping"
            result.state = state
            return result
        state, done = _step(env, world, "sleep", result)
        if done:
            break

    result.state = state
    result.ok = state.vitals.get("energy", 0) > start_energy or state.achievements.get("wake_up", False)
    result.reason = "slept" if result.ok else "sleep did not restore energy"
    return result


def _flee(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
    cap: int = 12,
) -> MacroResult:
    result = MacroResult(False, "flee", "", state=state)
    world.update(state)
    threats = [t for t in state.tiles if t.name in {"zombie", "skeleton"}]
    if not threats:
        result.ok = True
        result.reason = "no threat"
        return result

    def threat_dist(pos: tuple[int, int]) -> int:
        return min(abs(pos[0] - t.x) + abs(pos[1] - t.y) for t in threats)

    for _ in range(cap):
        if result.done:
            break
        here = threat_dist(state.pos)
        options: list[tuple[int, str]] = []
        for delta, move in DELTA_MOVE.items():
            nxt = (state.x + delta[0], state.y + delta[1])
            if _is_walkable(world, nxt):
                options.append((threat_dist(nxt), move))
        if not options:
            result.reason = "boxed in"
            result.state = state
            return result
        options.sort(reverse=True)
        best_dist, move = options[0]
        state, done = _step(env, world, move, result)
        if best_dist > here:
            result.ok = True
        if not _adjacent_monster(state) and best_dist >= 2:
            result.ok = True
            result.reason = "gained distance"
            result.state = state
            return result
        if done:
            result.reason = "died fleeing"
            return result

    result.state = state
    result.reason = "fled" if result.ok else "still close to threat"
    return result


def _explore(
    env: CrafterTextEnv,
    state: GameState,
    world: WorldMap,
    cap: int = MAX_WANDER_STEPS,
) -> MacroResult:
    result = MacroResult(False, "explore", "", state=state)
    world.update(state)

    frontiers: set[tuple[int, int]] = set()
    for pos, name in world.tiles.items():
        if name not in WALKABLE:
            continue
        for n in _neighbors(pos):
            if n not in world.tiles:
                frontiers.add(pos)
                break

    path = None
    if frontiers:
        path = _bfs(state.pos, frontiers, world)
    if path:
        state, _ = _follow_path(env, state, world, path[:cap], result, cap)
        result.ok = True
        result.reason = "walked to frontier"
        result.state = state
        return result

    # Deterministic scan if the frontier is empty or unreachable.
    order = ("move_right", "move_down", "move_left", "move_up")
    idx = (state.x + state.y) % 4
    for i in range(cap):
        if result.done:
            break
        move = order[(idx + i) % 4]
        prev = state.pos
        state, done = _step(env, world, move, result)
        if state.pos != prev:
            result.ok = True
            result.reason = "wandered"
            result.state = state
            return result
        if done:
            break
    result.state = state
    result.reason = "wander blocked"
    return result
