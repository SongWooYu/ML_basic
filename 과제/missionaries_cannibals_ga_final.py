# Python 3.12.13
# 식인종-선교사 게임 + 유전자 알고리즘 자동 Command 입력
# 목적:
#   - 유전자 1개 = Command 1개
#   - 염색체 1개 = Command 20개
#   - 염색체 평가 시, Command를 앞에서부터 1개씩 꺼내 최대 20번 실행
#   - 성공 / 식인종에게 먹힘 / 이동 불가 / 반복 상태 발생 시 해당 염색체 평가는 중단
#   - 여러 염색체를 만들고, 적합도 평가 -> 선택 -> 교차 -> 돌연변이 반복

import random
from dataclasses import dataclass, field


# =========================================================
# 1. 게임 기본 설정
# =========================================================

TOTAL_M = 3
TOTAL_C = 3

# 유전자 딕셔너리
# key = 유전자 값, value = Command 정보
# 모든 가능한 Command를 포함한다.
GENE_COMMANDS = {
    0: {"cmd": "1M", "m": 1, "c": 0},
    1: {"cmd": "2M", "m": 2, "c": 0},
    2: {"cmd": "1C", "m": 0, "c": 1},
    3: {"cmd": "2C", "m": 0, "c": 2},
    4: {"cmd": "1M1C", "m": 1, "c": 1},
}

GENE_VALUES = list(GENE_COMMANDS.keys())


# =========================================================
# 2. 유전자 알고리즘 설정
# =========================================================

CHROMOSOME_SIZE = 20      # 염색체의 유전자 수 = 최대 Command 입력 횟수
POPULATION_SIZE = 100     # 한 세대의 염색체 수
MAX_GENERATION = 1000     # 최대 세대 수
MUTATION_RATE = 0.08      # 유전자 1개가 돌연변이될 확률
CROSSOVER_RATE = 0.90     # 교차 확률
ELITE_COUNT = 4           # 상위 염색체 보존 개수
RANDOM_SEED = 42          # 실행 결과 재현용


# =========================================================
# 3. 게임 함수
# =========================================================

def make_status(left_m, left_c, right_m, right_c, boat):
    """상태 정의"""
    return {
        "left_m": left_m,
        "left_c": left_c,
        "right_m": right_m,
        "right_c": right_c,
        "boat": boat,  # "left" 또는 "right"
    }


def make_start_status():
    """최초 상태: 모두 오른쪽에 있고 배도 오른쪽에 있음"""
    return make_status(0, 0, 3, 3, "right")


def state_key(status):
    """반복 상태 확인을 위한 key"""
    return (
        status["left_m"],
        status["left_c"],
        status["right_m"],
        status["right_c"],
        status["boat"],
    )


def is_safe(missionary, cannibal):
    """선교사가 0명이면 안전, 아니면 선교사 수가 식인종 수 이상이어야 안전"""
    return missionary == 0 or missionary >= cannibal


def judge(status):
    """다음 상태 판정: 계속 / 식인종에게 먹힘 / 성공"""
    values = [
        status["left_m"], status["left_c"],
        status["right_m"], status["right_c"],
    ]
    if any(value < 0 for value in values):
        return "invalid", "사람 수가 음수가 되었습니다."

    if status["left_m"] + status["right_m"] != TOTAL_M:
        return "invalid", "선교사 전체 수가 맞지 않습니다."

    if status["left_c"] + status["right_c"] != TOTAL_C:
        return "invalid", "식인종 전체 수가 맞지 않습니다."

    if not is_safe(status["left_m"], status["left_c"]):
        return "eaten", "왼쪽 선교사가 식인종에게 먹혔습니다."

    if not is_safe(status["right_m"], status["right_c"]):
        return "eaten", "오른쪽 선교사가 식인종에게 먹혔습니다."

    if status["left_m"] == TOTAL_M and status["left_c"] == TOTAL_C:
        return "win", "성공했습니다: 모두 왼쪽으로 이동했습니다."

    return "continue", "계속 진행합니다."


def apply_command(status, gene):
    """현재 상태와 유전자 값을 받아 다음 상태를 계산한다."""
    command = GENE_COMMANDS[gene]
    move_m = command["m"]
    move_c = command["c"]
    cmd = command["cmd"]

    next_status = status.copy()

    if status["boat"] == "right":
        # 오른쪽 -> 왼쪽
        if status["right_m"] < move_m or status["right_c"] < move_c:
            return status, "invalid", f"이동 불가: 오른쪽에 {cmd} 이동 인원이 부족합니다."

        next_status["right_m"] -= move_m
        next_status["right_c"] -= move_c
        next_status["left_m"] += move_m
        next_status["left_c"] += move_c
        next_status["boat"] = "left"

    else:
        # 왼쪽 -> 오른쪽
        if status["left_m"] < move_m or status["left_c"] < move_c:
            return status, "invalid", f"이동 불가: 왼쪽에 {cmd} 이동 인원이 부족합니다."

        next_status["left_m"] -= move_m
        next_status["left_c"] -= move_c
        next_status["right_m"] += move_m
        next_status["right_c"] += move_c
        next_status["boat"] = "right"

    result, message = judge(next_status)
    return next_status, result, message


# =========================================================
# 4. 염색체 정의
# =========================================================

@dataclass
class Chromosome:
    genes: list[int] = field(default_factory=list)
    fitness: int = 1
    result: str = "not_evaluated"
    message: str = ""
    used_turn: int = 0
    used_genes: list[int] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.genes:
            self.genes = [random.choice(GENE_VALUES) for _ in range(CHROMOSOME_SIZE)]

    def command_names(self):
        return [GENE_COMMANDS[gene]["cmd"] for gene in self.genes]

    def used_command_names(self):
        return [GENE_COMMANDS[gene]["cmd"] for gene in self.used_genes]

    def evaluate(self):
        """
        염색체 평가.
        핵심 요구사항:
        - 유전자 수는 20개다.
        - 평가 시작 시 유전자 20개를 복사한다.
        - 한 번의 Command 실행마다 유전자 1개를 앞에서 꺼낸다.
        - 따라서 최대 20번의 Command 입력까지만 진행한다.
        """
        status = make_start_status()
        visited = {state_key(status)}
        remaining_genes = self.genes.copy()

        self.fitness = 1
        self.result = "fail"
        self.message = "20개 유전자를 모두 사용했지만 성공하지 못했습니다."
        self.used_turn = 0
        self.used_genes = []
        self.history = []

        max_left_people = 0
        final_left_people = 0
        valid_move_count = 0
        invalid_count = 0
        eaten_count = 0
        repeat_count = 0
        same_command_count = 0
        previous_gene = None

        # 턴 진행: 유전자 20개에서 1개씩 빼면서 최대 20번 Command 실행
        while len(remaining_genes) > 0:
            gene = remaining_genes.pop(0)
            command_name = GENE_COMMANDS[gene]["cmd"]
            self.used_turn += 1
            self.used_genes.append(gene)

            if previous_gene == gene:
                same_command_count += 1
            previous_gene = gene

            next_status, result, message = apply_command(status, gene)

            # 이동 불가인 경우: 상태 변화 없이 중단
            if result == "invalid":
                invalid_count += 1
                self.result = "invalid"
                self.message = message
                self.history.append({
                    "turn": self.used_turn,
                    "gene": gene,
                    "cmd": command_name,
                    "remaining": len(remaining_genes),
                    "status": status.copy(),
                    "result": result,
                    "message": message,
                })
                break

            status = next_status
            valid_move_count += 1

            left_people = status["left_m"] + status["left_c"]
            max_left_people = max(max_left_people, left_people)
            final_left_people = left_people

            current_key = state_key(status)
            if current_key in visited and result == "continue":
                repeat_count += 1
                self.result = "repeat"
                self.message = "과거 상태가 반복되어 평가를 중단했습니다."
                self.history.append({
                    "turn": self.used_turn,
                    "gene": gene,
                    "cmd": command_name,
                    "remaining": len(remaining_genes),
                    "status": status.copy(),
                    "result": "repeat",
                    "message": self.message,
                })
                break
            visited.add(current_key)

            self.history.append({
                "turn": self.used_turn,
                "gene": gene,
                "cmd": command_name,
                "remaining": len(remaining_genes),
                "status": status.copy(),
                "result": result,
                "message": message,
            })

            if result == "win":
                self.result = "win"
                self.message = message
                break

            if result == "eaten":
                eaten_count += 1
                self.result = "eaten"
                self.message = message
                break

        # 적합도 계산
        # 성공한 염색체는 큰 보상.
        # 단, 같은 성공이면 사용 턴이 적은 쪽이 더 좋은 적합도를 갖는다.
        fitness = 1
        fitness += max_left_people * 200
        fitness += final_left_people * 50
        fitness += valid_move_count * 30

        fitness -= invalid_count * 500
        fitness -= eaten_count * 700
        fitness -= repeat_count * 300
        fitness -= same_command_count * 30
        fitness -= self.used_turn * 2

        if self.result == "win":
            fitness += 10000
            fitness -= self.used_turn * 100

        self.fitness = max(1, fitness)
        return self.fitness


# =========================================================
# 5. 유전자 알고리즘 함수
# =========================================================

def create_population():
    population = []
    while len(population) < POPULATION_SIZE:
        population.append(Chromosome())
    return population


def evaluate_population(population):
    i = 0
    while i < len(population):
        population[i].evaluate()
        i += 1
    population.sort(key=lambda chromo: chromo.fitness, reverse=True)


def select_parent(population):
    """룰렛 휠 선택: 적합도가 높을수록 선택 확률이 높다."""
    total_fitness = sum(chromo.fitness for chromo in population)
    if total_fitness <= 0:
        return random.choice(population)

    pick = random.uniform(0, total_fitness)
    current = 0

    i = 0
    while i < len(population):
        current += population[i].fitness
        if current >= pick:
            return population[i]
        i += 1

    return population[-1]


def crossover(parent1, parent2):
    """단일 지점 교차"""
    if random.random() > CROSSOVER_RATE:
        return Chromosome(parent1.genes.copy())

    cross_point = random.randint(1, CHROMOSOME_SIZE - 1)
    child_genes = parent1.genes[:cross_point] + parent2.genes[cross_point:]
    return Chromosome(child_genes)


def mutate(chromosome):
    """각 유전자에 대하여 일정 확률로 Command를 변경"""
    i = 0
    while i < CHROMOSOME_SIZE:
        if random.random() < MUTATION_RATE:
            chromosome.genes[i] = random.choice(GENE_VALUES)
        i += 1


def make_next_population(population):
    """선택, 교차, 돌연변이를 통해 다음 세대를 만든다."""
    new_population = []

    # 엘리트 보존
    i = 0
    while i < ELITE_COUNT:
        new_population.append(Chromosome(population[i].genes.copy()))
        i += 1

    # 나머지는 선택, 교차, 돌연변이로 생성
    while len(new_population) < POPULATION_SIZE:
        father = select_parent(population)
        mother = select_parent(population)
        child = crossover(father, mother)
        mutate(child)
        new_population.append(child)

    return new_population


# =========================================================
# 6. 출력 함수
# =========================================================

def print_command_dictionary():
    print("유전자 딕셔너리")
    for gene, info in GENE_COMMANDS.items():
        print(f"  {gene} -> {info['cmd']}  (M={info['m']}, C={info['c']})")


def print_best_summary(generation, best):
    print(
        f"세대 {generation:03d} | "
        f"최고 적합도 {best.fitness:5d} | "
        f"결과 {best.result:7s} | "
        f"사용 Command {best.used_turn:2d}/20 | "
        f"명령 {best.used_command_names()}"
    )


def print_final_result(best):
    print("\n==============================")
    print("최종 결과")
    print("==============================")
    print(f"염색체 전체 유전자: {best.genes}")
    print(f"염색체 전체 Command: {best.command_names()}")
    print(f"실제 사용 유전자: {best.used_genes}")
    print(f"실제 사용 Command: {best.used_command_names()}")
    print(f"사용 Command 수: {best.used_turn}/20")
    print(f"적합도: {best.fitness}")
    print(f"결과: {best.message}")

    print("\n[턴별 진행]")
    for row in best.history:
        s = row["status"]
        print(
            f"{row['turn']:02d}. "
            f"gene={row['gene']} cmd={row['cmd']:<5s} | "
            f"남은 유전자={row['remaining']:2d} | "
            f"L(M={s['left_m']}, C={s['left_c']}), "
            f"R(M={s['right_m']}, C={s['right_c']}), "
            f"boat={s['boat']:<5s} | "
            f"{row['message']}"
        )


# =========================================================
# 7. 메인 실행부
# =========================================================

def main():
    random.seed(RANDOM_SEED)

    print("식인종-선교사 게임 자동 입력: 유전자 알고리즘")
    print(f"염색체 유전자 수: {CHROMOSOME_SIZE}")
    print(f"한 염색체 평가 시 최대 Command 입력 수: {CHROMOSOME_SIZE}")
    print(f"개체 수: {POPULATION_SIZE}")
    print(f"최대 세대 수: {MAX_GENERATION}")
    print_command_dictionary()
    print()

    population = create_population()
    best_overall = None
    generation = 0

    # 세대 반복 while
    while generation < MAX_GENERATION:
        evaluate_population(population)
        best = population[0]

        if best_overall is None or best.fitness > best_overall.fitness:
            best_overall = Chromosome(best.genes.copy())
            best_overall.evaluate()

        if generation == 0 or generation % 10 == 0 or best.result == "win":
            print_best_summary(generation, best)

        if best.result == "win":
            best_overall = best
            break

        population = make_next_population(population)
        generation += 1

    if best_overall is None:
        print("평가된 염색체가 없습니다.")
        return

    print_final_result(best_overall)


if __name__ == "__main__":
    main()
