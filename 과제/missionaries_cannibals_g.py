# Python 3.12.13
# 식인종-선교사 게임: 유전자 알고리즘 자동 Command 탐색
# 평가함수 개선 버전
# 핵심 목적:
#   1) 상태 -> Command -> 다음 상태 -> 판정 구조 유지
#   2) 유전자 1개 = Command 1개
#   3) 염색체 = Command 순서열
#   4) 성공/실패 여부만 보지 않고, 실패 직전까지의 유효한 경로도 부분 평가
#   5) 단, 성공 경로가 항상 실패 경로보다 높은 적합도를 갖도록 계층형 평가 적용

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

TOTAL_M = 3
TOTAL_C = 3

RANDOM_SEED = 7
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

# ============================================================
# 1. 유전자 공간
# ============================================================
# 유전자 1개 = Command 1개
# 가능한 Command는 배에 태울 수 있는 모든 경우의 수이다.
GENE_COMMANDS: Dict[int, Dict[str, int | str]] = {
    0: {"name": "1M", "m": 1, "c": 0},
    1: {"name": "2M", "m": 2, "c": 0},
    2: {"name": "1C", "m": 0, "c": 1},
    3: {"name": "2C", "m": 0, "c": 2},
    4: {"name": "1M1C", "m": 1, "c": 1},
}

# ============================================================
# 2. 유전자 알고리즘 설정
# ============================================================
POPULATION_SIZE = 100
MAX_GENE_COUNT = 20
MAX_GENERATION = 500
MUTATION_RATE = 0.08
ELITE_COUNT = 6

# ============================================================
# 3. 평가함수 정책
# ============================================================
# 결과 계층을 먼저 나눈다.
# win은 실패 경로보다 항상 높아야 한다.
# fail은 유전자 끝까지 안전하게 갔지만 성공하지 못한 경우다.
# repeat/eaten/invalid는 중간에 문제가 생긴 경로다.
RESULT_BASE_SCORE = {
    "win": 100_000,
    "fail": 20_000,
    "repeat": 10_000,
    "eaten": 3_000,
    "invalid": 1_000,
}

# 진행도 보상: 실패했더라도 좋은 prefix는 교차/돌연변이로 재활용될 수 있으므로 점수를 일부 준다.
WEIGHT_MAX_LEFT_PEOPLE = 300
WEIGHT_FINAL_LEFT_PEOPLE = 100
WEIGHT_VALID_MOVE = 50

# 벌점: 나쁜 종료 원인과 비효율을 반영한다.
PENALTY_INVALID = 800
PENALTY_EATEN = 1_000
PENALTY_REPEAT = 600
PENALTY_SAME_COMMAND = 30
PENALTY_USED_STEP = 5
PENALTY_WIN_USED_STEP = 500  # 성공 경로끼리는 짧을수록 유리하게 함


def make_status(left_m: int, left_c: int, right_m: int, right_c: int, boat: str) -> Dict[str, int | str]:
    return {
        "left_m": left_m,
        "left_c": left_c,
        "right_m": right_m,
        "right_c": right_c,
        "boat": boat,
    }


def initial_status() -> Dict[str, int | str]:
    return make_status(left_m=0, left_c=0, right_m=3, right_c=3, boat="right")


def state_key(status: Dict[str, int | str]) -> Tuple[int, int, int, int, str]:
    return (
        int(status["left_m"]),
        int(status["left_c"]),
        int(status["right_m"]),
        int(status["right_c"]),
        str(status["boat"]),
    )


def is_safe(missionary: int, cannibal: int) -> bool:
    return missionary == 0 or missionary >= cannibal


def judge(status: Dict[str, int | str]) -> Tuple[str, str]:
    left_m = int(status["left_m"])
    left_c = int(status["left_c"])
    right_m = int(status["right_m"])
    right_c = int(status["right_c"])

    if any(v < 0 for v in [left_m, left_c, right_m, right_c]):
        return "invalid", "잘못된 상태: 사람 수가 음수입니다."

    if left_m + right_m != TOTAL_M:
        return "invalid", "잘못된 상태: 선교사 전체 수가 맞지 않습니다."

    if left_c + right_c != TOTAL_C:
        return "invalid", "잘못된 상태: 식인종 전체 수가 맞지 않습니다."

    if not is_safe(left_m, left_c):
        return "eaten", "게임오버: 왼쪽 선교사가 식인종에게 먹혔습니다."

    if not is_safe(right_m, right_c):
        return "eaten", "게임오버: 오른쪽 선교사가 식인종에게 먹혔습니다."

    if left_m == TOTAL_M and left_c == TOTAL_C:
        return "win", "성공했습니다: 모두 왼쪽으로 이동했습니다."

    return "continue", "계속 진행"


def apply_gene(status: Dict[str, int | str], gene: int) -> Tuple[Dict[str, int | str], str, str]:
    command = GENE_COMMANDS[gene]
    move_m = int(command["m"])
    move_c = int(command["c"])
    next_status = status.copy()

    if status["boat"] == "right":
        if int(status["right_m"]) < move_m or int(status["right_c"]) < move_c:
            return status, "invalid", "이동 불가: 오른쪽에 해당 사람 수가 부족합니다."

        next_status["right_m"] = int(next_status["right_m"]) - move_m
        next_status["right_c"] = int(next_status["right_c"]) - move_c
        next_status["left_m"] = int(next_status["left_m"]) + move_m
        next_status["left_c"] = int(next_status["left_c"]) + move_c
        next_status["boat"] = "left"
    else:
        if int(status["left_m"]) < move_m or int(status["left_c"]) < move_c:
            return status, "invalid", "이동 불가: 왼쪽에 해당 사람 수가 부족합니다."

        next_status["left_m"] = int(next_status["left_m"]) - move_m
        next_status["left_c"] = int(next_status["left_c"]) - move_c
        next_status["right_m"] = int(next_status["right_m"]) + move_m
        next_status["right_c"] = int(next_status["right_c"]) + move_c
        next_status["boat"] = "right"

    return (*judge(next_status),) and (next_status, *judge(next_status))


@dataclass
class EvaluationBreakdown:
    base_score: int = 0
    progress_score: int = 0
    penalty_score: int = 0
    final_fitness: int = 1
    max_left_people: int = 0
    final_left_people: int = 0
    valid_move_count: int = 0
    invalid_count: int = 0
    eaten_count: int = 0
    repeated_state_count: int = 0
    same_command_count: int = 0
    used_count: int = 0


@dataclass
class Chromosome:
    size: int
    genes: Optional[List[int]] = None
    fitness: int = 1
    result: str = "none"
    message: str = ""
    used_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    breakdown: EvaluationBreakdown = field(default_factory=EvaluationBreakdown)

    def __post_init__(self) -> None:
        if self.genes is None:
            self.genes = [random.choice(list(GENE_COMMANDS.keys())) for _ in range(self.size)]
        else:
            self.genes = self.genes.copy()

    def command_names(self) -> List[str]:
        assert self.genes is not None
        return [str(GENE_COMMANDS[g]["name"]) for g in self.genes]

    def used_command_names(self) -> List[str]:
        assert self.genes is not None
        return [str(GENE_COMMANDS[g]["name"]) for g in self.genes[:self.used_count]]

    def calculate_fitness(self) -> int:
        status = initial_status()
        visited = {state_key(status)}
        prev_gene: Optional[int] = None

        max_left_people = 0
        final_left_people = 0
        valid_move_count = 0
        invalid_count = 0
        eaten_count = 0
        repeated_state_count = 0
        same_command_count = 0

        self.history = []
        self.result = "none"
        self.message = ""
        self.used_count = 0

        index = 0
        while index < self.size:
            assert self.genes is not None
            gene = self.genes[index]
            command_name = str(GENE_COMMANDS[gene]["name"])

            if prev_gene is not None and gene == prev_gene:
                same_command_count += 1

            next_status, result, message = apply_gene(status, gene)
            self.used_count = index + 1

            self.history.append({
                "step": index + 1,
                "gene": gene,
                "command": command_name,
                "status": next_status.copy(),
                "result": result,
                "message": message,
            })

            # 이동 불가는 그 유전자 위치에서 평가를 중단한다.
            # 단, 이전 prefix는 평가에 반영된다.
            if result == "invalid":
                invalid_count += 1
                self.result = "invalid"
                self.message = message
                break

            status = next_status
            valid_move_count += 1

            left_people = int(status["left_m"]) + int(status["left_c"])
            final_left_people = left_people
            max_left_people = max(max_left_people, left_people)

            if result == "eaten":
                eaten_count += 1
                self.result = "eaten"
                self.message = message
                break

            if result == "win":
                self.result = "win"
                self.message = message
                break

            key = state_key(status)
            if key in visited:
                repeated_state_count += 1
                self.result = "repeat"
                self.message = "반복 상태: 과거 상태와 같아져서 중단했습니다."
                break
            visited.add(key)

            prev_gene = gene
            index += 1

        if self.result == "none":
            self.result = "fail"
            self.message = "실패: 제한된 유전자 수 안에 성공하지 못했습니다."

        base_score = RESULT_BASE_SCORE[self.result]
        progress_score = (
            max_left_people * WEIGHT_MAX_LEFT_PEOPLE
            + final_left_people * WEIGHT_FINAL_LEFT_PEOPLE
            + valid_move_count * WEIGHT_VALID_MOVE
        )
        penalty_score = (
            invalid_count * PENALTY_INVALID
            + eaten_count * PENALTY_EATEN
            + repeated_state_count * PENALTY_REPEAT
            + same_command_count * PENALTY_SAME_COMMAND
            + self.used_count * PENALTY_USED_STEP
        )

        if self.result == "win":
            penalty_score += self.used_count * PENALTY_WIN_USED_STEP

        final_fitness = max(1, int(base_score + progress_score - penalty_score))

        self.fitness = final_fitness
        self.breakdown = EvaluationBreakdown(
            base_score=base_score,
            progress_score=progress_score,
            penalty_score=penalty_score,
            final_fitness=final_fitness,
            max_left_people=max_left_people,
            final_left_people=final_left_people,
            valid_move_count=valid_move_count,
            invalid_count=invalid_count,
            eaten_count=eaten_count,
            repeated_state_count=repeated_state_count,
            same_command_count=same_command_count,
            used_count=self.used_count,
        )
        return self.fitness

    def clone(self) -> "Chromosome":
        assert self.genes is not None
        copied = Chromosome(self.size, self.genes)
        copied.fitness = self.fitness
        copied.result = self.result
        copied.message = self.message
        copied.used_count = self.used_count
        copied.history = [h.copy() for h in self.history]
        copied.breakdown = self.breakdown
        return copied


def select_by_roulette(population: List[Chromosome]) -> Chromosome:
    total_fitness = sum(c.fitness for c in population)
    pick = random.uniform(0, total_fitness)
    current = 0.0
    for chromosome in population:
        current += chromosome.fitness
        if current >= pick:
            return chromosome
    return population[-1]


def crossover(father: Chromosome, mother: Chromosome) -> Tuple[Chromosome, Chromosome]:
    if father.size <= 1:
        return father.clone(), mother.clone()
    assert father.genes is not None
    assert mother.genes is not None
    index = random.randint(1, father.size - 1)
    child1_genes = father.genes[:index] + mother.genes[index:]
    child2_genes = mother.genes[:index] + father.genes[index:]
    return Chromosome(father.size, child1_genes), Chromosome(father.size, child2_genes)


def mutate(chromosome: Chromosome) -> None:
    assert chromosome.genes is not None
    index = 0
    while index < chromosome.size:
        if random.random() < MUTATION_RATE:
            old_gene = chromosome.genes[index]
            new_gene = random.choice(list(GENE_COMMANDS.keys()))
            while new_gene == old_gene:
                new_gene = random.choice(list(GENE_COMMANDS.keys()))
            chromosome.genes[index] = new_gene
        index += 1


def create_population(population_size: int, gene_count: int) -> List[Chromosome]:
    return [Chromosome(gene_count) for _ in range(population_size)]


def evaluate_population(population: List[Chromosome]) -> None:
    for chromosome in population:
        chromosome.calculate_fitness()
    population.sort(key=lambda c: c.fitness, reverse=True)


def make_next_population(population: List[Chromosome]) -> List[Chromosome]:
    next_population: List[Chromosome] = []
    while len(next_population) < ELITE_COUNT and len(next_population) < len(population):
        next_population.append(population[len(next_population)].clone())

    while len(next_population) < POPULATION_SIZE:
        father = select_by_roulette(population)
        mother = select_by_roulette(population)
        child1, child2 = crossover(father, mother)
        mutate(child1)
        mutate(child2)
        next_population.append(child1)
        if len(next_population) < POPULATION_SIZE:
            next_population.append(child2)
    return next_population


def print_header() -> None:
    print("식인종-선교사 게임 자동 입력: 유전자 알고리즘")
    print(f"최대 유전자 수: {MAX_GENE_COUNT}")
    print(f"개체 수: {POPULATION_SIZE}")
    print(f"각 유전자 수별 최대 세대 수: {MAX_GENERATION}")
    print("유전자 공간")
    for gene, command in GENE_COMMANDS.items():
        print(f"  {gene} -> {command['name']} (M={command['m']}, C={command['c']})")
    print("\n평가 정책")
    print("  성공/실패 종류를 먼저 구분한 뒤, 진행도 보상과 실패 벌점을 더해 적합도를 계산한다.")
    print("  중간에 실패한 염색체도 좋은 prefix가 있으면 일부 점수를 받지만, 성공 염색체보다 높아질 수 없다.")


def print_chromosome_summary(chromosome: Chromosome) -> None:
    print("염색체 유전자 수:", chromosome.size)
    print("염색체 전체 유전자:", chromosome.genes)
    print("염색체 전체 Command:", chromosome.command_names())
    print("실제 사용 Command:", chromosome.used_command_names())
    print("사용 Command 수:", chromosome.used_count)
    print("결과:", chromosome.result, "/", chromosome.message)
    print("적합도:", chromosome.fitness)
    print("평가 상세:")
    print(f"  base_score     = {chromosome.breakdown.base_score}")
    print(f"  progress_score = {chromosome.breakdown.progress_score}")
    print(f"  penalty_score  = {chromosome.breakdown.penalty_score}")
    print(f"  final_fitness  = {chromosome.breakdown.final_fitness}")
    print(f"  max_left_people={chromosome.breakdown.max_left_people}, final_left_people={chromosome.breakdown.final_left_people}, valid_move_count={chromosome.breakdown.valid_move_count}")
    print(f"  invalid={chromosome.breakdown.invalid_count}, eaten={chromosome.breakdown.eaten_count}, repeat={chromosome.breakdown.repeated_state_count}, same_cmd={chromosome.breakdown.same_command_count}")


def print_solution_path(chromosome: Chromosome) -> None:
    print("\n[성공 경로]")
    for item in chromosome.history:
        status = item["status"]
        print(
            f"{int(item['step']):02d}. "
            f"gene={int(item['gene'])} "
            f"cmd={str(item['command']):5s} -> "
            f"L(M={status['left_m']}, C={status['left_c']}), "
            f"R(M={status['right_m']}, C={status['right_c']}), "
            f"boat={status['boat']} / {item['message']}"
        )


def print_result_table(results: List[Dict[str, object]]) -> None:
    print("\n[유전자 수별 탐색 요약]")
    print("gene_count | found | generation | fitness | used | result  | command")
    print("-----------|-------|------------|---------|------|---------|----------------")
    for row in results:
        command_text = " ".join(row["commands"]) if row["commands"] else "-"
        print(
            f"{row['gene_count']:10d} | "
            f"{str(row['found']):5s} | "
            f"{row['generation']:10s} | "
            f"{row['fitness']:7s} | "
            f"{row['used']:4s} | "
            f"{row['result']:7s} | "
            f"{command_text}"
        )


def run_genetic_algorithm() -> None:
    print_header()
    shortest_solution: Optional[Chromosome] = None
    results: List[Dict[str, object]] = []

    gene_count = MAX_GENE_COUNT
    while gene_count >= 1:
        print("\n" + "=" * 50)
        print(f"유전자 수 {gene_count}개로 탐색 시작")
        print("=" * 50)

        population = create_population(POPULATION_SIZE, gene_count)
        generation = 0
        found_solution: Optional[Chromosome] = None

        while generation < MAX_GENERATION:
            evaluate_population(population)
            best = population[0]
            if generation % 50 == 0 or best.result == "win":
                print(
                    f"세대 {generation:03d} | "
                    f"최고 적합도 {best.fitness:6d} | "
                    f"결과 {best.result:7s} | "
                    f"사용 Command {best.used_count:2d} | "
                    f"명령 {best.used_command_names()}"
                )
            if best.result == "win":
                found_solution = best.clone()
                shortest_solution = found_solution
                print("성공 염색체 발견")
                print_chromosome_summary(found_solution)
                break
            population = make_next_population(population)
            generation += 1

        if found_solution is None:
            results.append({
                "gene_count": gene_count,
                "found": False,
                "generation": "-",
                "fitness": "-",
                "used": "-",
                "result": "-",
                "commands": [],
            })
        else:
            results.append({
                "gene_count": gene_count,
                "found": True,
                "generation": str(generation),
                "fitness": str(found_solution.fitness),
                "used": str(found_solution.used_count),
                "result": found_solution.result,
                "commands": found_solution.used_command_names(),
            })
        gene_count -= 1

    print("\n" + "=" * 50)
    print("최종 결과")
    print("=" * 50)
    print_result_table(results)

    if shortest_solution is None:
        print("\n성공 경로를 찾지 못했습니다.")
        return

    print("\n가장 짧게 성공한 염색체")
    print_chromosome_summary(shortest_solution)
    print_solution_path(shortest_solution)


if __name__ == "__main__":
    run_genetic_algorithm()
