#!/usr/bin/env python3
"""
식인종-선교사 게임

Python 3.12.13 호환.
외부 패키지 없이 실행 가능.

실행:
    python missionaries_cannibals.py
    python missionaries_cannibals.py --auto
    python missionaries_cannibals.py --manual
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import argparse


TOTAL_MISSIONARIES = 3
TOTAL_CANNIBALS = 3


class Side(str, Enum):
    LEFT = "L"
    RIGHT = "R"

    @property
    def other(self) -> "Side":
        return Side.RIGHT if self is Side.LEFT else Side.LEFT

    @property
    def ko(self) -> str:
        return "왼쪽" if self is Side.LEFT else "오른쪽"


class Result(str, Enum):
    CONTINUE = "continue"
    GAME_OVER = "game_over"
    WIN = "win"


@dataclass(frozen=True, slots=True)
class Command:
    """
    배에 태울 사람 수.
    m: 선교사 수
    c: 식인종 수
    """
    name: str
    m: int
    c: int


COMMANDS: tuple[Command, ...] = (
    Command("1M", 1, 0),
    Command("2M", 2, 0),
    Command("1C", 0, 1),
    Command("2C", 0, 2),
    Command("1M1C", 1, 1),
)

COMMAND_BY_NAME: dict[str, Command] = {
    command.name.upper(): command for command in COMMANDS
}
COMMAND_BY_NAME.update({
    "M": COMMAND_BY_NAME["1M"],
    "MM": COMMAND_BY_NAME["2M"],
    "C": COMMAND_BY_NAME["1C"],
    "CC": COMMAND_BY_NAME["2C"],
    "MC": COMMAND_BY_NAME["1M1C"],
    "CM": COMMAND_BY_NAME["1M1C"],
    "1C1M": COMMAND_BY_NAME["1M1C"],
})


@dataclass(frozen=True, slots=True)
class State:
    """
    상태 정의:
    - 왼쪽 선교사 수
    - 왼쪽 식인종 수
    - 배 위치

    오른쪽 사람 수는 전체 인원에서 왼쪽 사람 수를 뺀 값으로 계산한다.
    """
    left_m: int
    left_c: int
    boat: Side

    @property
    def right_m(self) -> int:
        return TOTAL_MISSIONARIES - self.left_m

    @property
    def right_c(self) -> int:
        return TOTAL_CANNIBALS - self.left_c

    def counts_on(self, side: Side) -> tuple[int, int]:
        if side is Side.LEFT:
            return self.left_m, self.left_c
        return self.right_m, self.right_c

    def is_count_range_valid(self) -> bool:
        return (
            0 <= self.left_m <= TOTAL_MISSIONARIES
            and 0 <= self.left_c <= TOTAL_CANNIBALS
            and 0 <= self.right_m <= TOTAL_MISSIONARIES
            and 0 <= self.right_c <= TOTAL_CANNIBALS
        )

    def is_safe_bank(self, missionaries: int, cannibals: int) -> bool:
        """
        선교사가 있는 강둑에서는 선교사 수가 식인종 수보다 작으면 안 된다.
        선교사가 0명이면 잡아먹힐 선교사가 없으므로 안전한 상태로 본다.
        """
        return missionaries == 0 or missionaries >= cannibals

    def is_safe(self) -> bool:
        return (
            self.is_count_range_valid()
            and self.is_safe_bank(self.left_m, self.left_c)
            and self.is_safe_bank(self.right_m, self.right_c)
        )

    def is_win(self) -> bool:
        return self.right_m == TOTAL_MISSIONARIES and self.right_c == TOTAL_CANNIBALS

    def display(self) -> str:
        boat_left = "🚤" if self.boat is Side.LEFT else "  "
        boat_right = "🚤" if self.boat is Side.RIGHT else "  "
        return (
            f"[왼쪽] M={self.left_m}, C={self.left_c} {boat_left}  |강|  "
            f"{boat_right} [오른쪽] M={self.right_m}, C={self.right_c}"
        )


class InvalidMove(ValueError):
    pass


def initial_state() -> State:
    return State(left_m=3, left_c=3, boat=Side.LEFT)


def parse_command(raw: str) -> Command:
    text = raw.strip().upper().replace(" ", "")
    if text not in COMMAND_BY_NAME:
        raise InvalidMove(
            f"알 수 없는 Command입니다: {raw!r}. "
            f"사용 가능: {', '.join(command.name for command in COMMANDS)}"
        )
    return COMMAND_BY_NAME[text]


def can_board(state: State, command: Command) -> bool:
    side_m, side_c = state.counts_on(state.boat)
    return side_m >= command.m and side_c >= command.c


def apply_command(state: State, command: Command) -> State:
    """
    현재 상태와 Command를 받아 다음 상태를 만든다.

    1. 배 위치에 따라 선교사/식인종 수를 수정한다.
    2. 배 위치를 반대로 변경한다.
    3. 다음 상태를 반환한다.
    """
    if command.m + command.c == 0:
        raise InvalidMove("배에는 최소 1명 이상 타야 합니다.")
    if command.m + command.c > 2:
        raise InvalidMove("배에는 최대 2명까지만 탈 수 있습니다.")
    if not can_board(state, command):
        raise InvalidMove(
            f"{state.boat.ko}에 {command.name}을 수행할 인원이 부족합니다."
        )

    if state.boat is Side.LEFT:
        next_state = State(
            left_m=state.left_m - command.m,
            left_c=state.left_c - command.c,
            boat=Side.RIGHT,
        )
    else:
        next_state = State(
            left_m=state.left_m + command.m,
            left_c=state.left_c + command.c,
            boat=Side.LEFT,
        )

    if not next_state.is_count_range_valid():
        raise InvalidMove("사람 수가 가능한 범위를 벗어났습니다.")
    return next_state


def judge(state: State) -> Result:
    """
    다음 상태에 대해 계속 진행, 게임 오버, 게임 승리를 판정한다.
    """
    if not state.is_safe():
        return Result.GAME_OVER
    if state.is_win():
        return Result.WIN
    return Result.CONTINUE


def available_commands(state: State, *, safe_only: bool = True) -> list[tuple[Command, State, Result]]:
    """
    현재 상태에서 수행 가능한 Command 목록을 반환한다.

    safe_only=True이면 게임오버가 되는 Command는 제외한다.
    """
    result: list[tuple[Command, State, Result]] = []

    for command in COMMANDS:
        try:
            next_state = apply_command(state, command)
        except InvalidMove:
            continue

        next_result = judge(next_state)
        if safe_only and next_result is Result.GAME_OVER:
            continue

        result.append((command, next_state, next_result))

    return result


def print_state_and_commands(state: State) -> None:
    print("\n현재 상태")
    print(" ", state.display())

    commands = available_commands(state, safe_only=False)
    if commands:
        print("\n사용 가능한 Command")
        for command, next_state, next_result in commands:
            suffix = ""
            if next_result is Result.GAME_OVER:
                suffix = " -> 게임오버"
            elif next_result is Result.WIN:
                suffix = " -> 승리"
            print(f"  - {command.name:5s}: {next_state.display()}{suffix}")
    else:
        print("\n사용 가능한 Command가 없습니다.")


def run_manual() -> None:
    """
    수동 입력 모드.
    """
    state = initial_state()

    print("식인종-선교사 게임: 수동 모드")
    print("Command: 1M, 2M, 1C, 2C, 1M1C")
    print("종료: q, quit, exit")

    while True:
        print_state_and_commands(state)
        raw = input("\nCommand 입력> ")

        if raw.strip().lower() in {"q", "quit", "exit"}:
            print("게임을 종료합니다.")
            break

        try:
            command = parse_command(raw)
            next_state = apply_command(state, command)
        except InvalidMove as exc:
            print(f"잘못된 입력: {exc}")
            continue

        result = judge(next_state)
        print(f"\n실행 Command: {command.name}")
        print("다음 상태")
        print(" ", next_state.display())

        if result is Result.GAME_OVER:
            print("판정: 게임오버")
            break
        if result is Result.WIN:
            print("판정: 승리")
            break

        print("판정: 계속 진행")
        state = next_state


def solve_auto() -> list[tuple[Command, State]]:
    """
    자동화 모드.

    무한 반복 방지:
    1. 직전 Command와 같은 Command는 바로 반복하지 않는다.
    2. 과거에 등장한 상태는 다시 방문하지 않는다.
    """
    start = initial_state()
    visited: set[State] = {start}

    queue: deque[tuple[State, list[tuple[Command, State]], Command | None]] = deque()
    queue.append((start, [], None))

    while queue:
        state, path, previous_command = queue.popleft()

        for command, next_state, next_result in available_commands(state, safe_only=True):
            if previous_command is not None and command.name == previous_command.name:
                continue
            if next_state in visited:
                continue

            next_path = path + [(command, next_state)]

            if next_result is Result.WIN:
                return next_path

            visited.add(next_state)
            queue.append((next_state, next_path, command))

    raise RuntimeError("승리 경로를 찾지 못했습니다.")


def run_auto() -> None:
    """
    자동 풀이 실행.
    """
    print("식인종-선교사 게임: 자동 모드")
    state = initial_state()
    print(f"START: {state.display()}")

    solution = solve_auto()

    for step, (command, next_state) in enumerate(solution, start=1):
        print(f"{step:02d}. {command.name:5s} -> {next_state.display()}")
        state = next_state

    print(f"\n총 이동 횟수: {len(solution)}")
    print("판정: 승리")


def main() -> None:
    parser = argparse.ArgumentParser(description="식인종-선교사 게임")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manual", action="store_true", help="수동 입력 모드 실행")
    mode.add_argument("--auto", action="store_true", help="자동 풀이 모드 실행")
    args = parser.parse_args()

    if args.auto:
        run_auto()
        return
    if args.manual:
        run_manual()
        return

    print("실행 모드를 선택하세요.")
    print("1. 수동 모드")
    print("2. 자동 모드")
    selected = input("> ").strip()

    if selected == "1":
        run_manual()
    elif selected == "2":
        run_auto()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
