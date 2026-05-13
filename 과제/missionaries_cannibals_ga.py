import random

TOTAL_M = 3
TOTAL_C = 3

# 유전자 공간
# 유전자는 0, 1, 2, 3 중 하나이고, 각 번호는 하나의 Command를 의미한다.
# 사용자가 말한 딕셔너리 구조: 1c, 1c1m, 2c, 2m
GENE_COMMANDS = {
    0: {"name": "1c", "m": 0, "c": 1},
    1: {"name": "1c1m", "m": 1, "c": 1},
    2: {"name": "2c", "m": 0, "c": 2},
    3: {"name": "2m", "m": 2, "c": 0},
    4: {"name": "1m", "m": 1, "c": 0},
}

# 유전자 알고리즘 설정
POPULATION_SIZE = 80       # 한 세대의 개체 수
MAX_GENE_COUNT = 20        # 처음 염색체의 유전자 수
MAX_GENERATION = 500       # 각 유전자 수에서 반복할 최대 세대 수
MUTATION_RATE = 0.08       # 돌연변이 확률
ELITE_COUNT = 4            # 다음 세대로 그대로 넘길 우수 개체 수

# 재현 가능한 실행을 위해 고정한다.
# 매번 다른 결과를 원하면 아래 줄을 주석 처리한다.
random.seed(7)


def make_status(left_m, left_c, right_m, right_c, boat):
    """게임 상태 정의"""
    return {
        "left_m": left_m,
        "left_c": left_c,
        "right_m": right_m,
        "right_c": right_c,
        "boat": boat,  # "left" 또는 "right"
    }


def initial_status():
    """최초 상태: 모두 오른쪽에 있고, 배도 오른쪽에 있다."""
    return make_status(0, 0, 3, 3, "right")


def state_key(status):
    """방문 상태 비교용 키"""
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
    """상태 판정: 계속 / 성공 / 식인종에게 먹힘 / 잘못된 상태"""
    values = [
        status["left_m"], status["left_c"],
        status["right_m"], status["right_c"],
    ]

    if any(v < 0 for v in values):
        return "invalid", "잘못된 상태: 사람 수가 음수입니다."

    if status["left_m"] + status["right_m"] != TOTAL_M:
        return "invalid", "잘못된 상태: 선교사 전체 수가 맞지 않습니다."

    if status["left_c"] + status["right_c"] != TOTAL_C:
        return "invalid", "잘못된 상태: 식인종 전체 수가 맞지 않습니다."

    if not is_safe(status["left_m"], status["left_c"]):
        return "eaten", "게임오버: 왼쪽 선교사가 식인종에게 먹혔습니다."

    if not is_safe(status["right_m"], status["right_c"]):
        return "eaten", "게임오버: 오른쪽 선교사가 식인종에게 먹혔습니다."

    if status["left_m"] == TOTAL_M and status["left_c"] == TOTAL_C:
        return "win", "성공했습니다: 모두 왼쪽으로 이동했습니다."

    return "continue", "계속 진행"


def apply_gene(status, gene):
    """현재 상태와 유전자 하나를 받아 다음 상태를 만든다."""
    command = GENE_COMMANDS[gene]
    move_m = command["m"]
    move_c = command["c"]

    next_status = status.copy()

    if status["boat"] == "right":
        # 오른쪽 -> 왼쪽
        if status["right_m"] < move_m or status["right_c"] < move_c:
            return status, "invalid", "이동 불가: 오른쪽에 해당 사람 수가 부족합니다."

        next_status["right_m"] -= move_m
        next_status["right_c"] -= move_c
        next_status["left_m"] += move_m
        next_status["left_c"] += move_c
        next_status["boat"] = "left"

    else:
        # 왼쪽 -> 오른쪽
        if status["left_m"] < move_m or status["left_c"] < move_c:
            return status, "invalid", "이동 불가: 왼쪽에 해당 사람 수가 부족합니다."

        next_status["left_m"] -= move_m
        next_status["left_c"] -= move_c
        next_status["right_m"] += move_m
        next_status["right_c"] += move_c
        next_status["boat"] = "right"

    result, message = judge(next_status)
    return next_status, result, message


class Chromosome:
    """염색체: 유전자(Command 번호)의 리스트"""

    def __init__(self, size, genes=None):
        self.size = size
        if genes is None:
            self.genes = [random.choice(list(GENE_COMMANDS.keys())) for _ in range(size)]
        else:
            self.genes = genes.copy()

        self.fitness = 1
        self.result = "none"
        self.message = ""
        self.used_turn = 0
        self.history = []

    def command_names(self):
        return [GENE_COMMANDS[g]["name"] for g in self.genes]

    def used_command_names(self):
        return [GENE_COMMANDS[g]["name"] for g in self.genes[:self.used_turn]]

    def calculate_fitness(self):
        """염색체를 게임에 적용하여 적합도를 계산한다."""
        status = initial_status()
        visited = {state_key(status)}

        max_left_people = 0
        final_left_people = 0
        valid_move_count = 0
        invalid_count = 0
        eaten_count = 0
        repeated_state_count = 0
        same_command_count = 0

        prev_gene = None
        self.history = []
        self.result = "none"
        self.message = ""
        self.used_turn = 0

        turn = 0
        while turn < self.size:
            gene = self.genes[turn]
            command_name = GENE_COMMANDS[gene]["name"]

            if prev_gene is not None and gene == prev_gene:
                same_command_count += 1

            next_status, result, message = apply_gene(status, gene)
            self.used_turn = turn + 1

            self.history.append({
                "turn": turn + 1,
                "command": command_name,
                "status": next_status.copy(),
                "result": result,
                "message": message,
            })

            if result == "invalid":
                invalid_count += 1
                self.result = result
                self.message = message
                break

            status = next_status
            valid_move_count += 1

            left_people = status["left_m"] + status["left_c"]
            final_left_people = left_people
            if left_people > max_left_people:
                max_left_people = left_people

            if result == "eaten":
                eaten_count += 1
                self.result = result
                self.message = message
                break

            if result == "win":
                self.result = result
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
            turn += 1

        if self.result == "none":
            self.result = "fail"
            self.message = "실패: 제한된 턴 안에 성공하지 못했습니다."

        # 적합도 평가
        # 기본 방향: 왼쪽으로 많이 이동할수록 좋고, 성공하면 매우 크게 보상한다.
        fitness = 1
        fitness += max_left_people * 120
        fitness += final_left_people * 30
        fitness += valid_move_count * 10

        # 무의미하거나 위험한 상황은 벌점
        fitness -= invalid_count * 500
        fitness -= eaten_count * 800
        fitness -= repeated_state_count * 300
        fitness -= same_command_count * 30
        fitness -= self.used_turn * 2

        if self.result == "win":
            fitness += 10000
            # 같은 성공이면 턴 수가 짧을수록 높은 점수
            fitness -= self.used_turn * 100

        # 룰렛 선택을 위해 최소 1 이상으로 유지
        self.fitness = max(1, fitness)
        return self.fitness

    def copy(self):
        copied = Chromosome(self.size, self.genes)
        copied.fitness = self.fitness
        copied.result = self.result
        copied.message = self.message
        copied.used_turn = self.used_turn
        copied.history = [h.copy() for h in self.history]
        return copied


# 선택 연산: 룰렛 휠 선택

def select(population):
    total_fitness = sum(c.fitness for c in population)
    pick = random.uniform(0, total_fitness)
    current = 0

    for chromosome in population:
        current += chromosome.fitness
        if current >= pick:
            return chromosome

    return population[-1]


# 교차 연산: 한 지점을 기준으로 부모 유전자를 교환

def crossover(father, mother):
    if father.size <= 1:
        return father.copy(), mother.copy()

    index = random.randint(1, father.size - 1)
    child1_genes = father.genes[:index] + mother.genes[index:]
    child2_genes = mother.genes[:index] + father.genes[index:]

    return Chromosome(father.size, child1_genes), Chromosome(father.size, child2_genes)


# 돌연변이 연산: 랜덤 위치의 유전자 값을 다른 Command로 변경

def mutate(chromosome):
    i = 0
    while i < chromosome.size:
        if random.random() < MUTATION_RATE:
            old_gene = chromosome.genes[i]
            new_gene = random.choice(list(GENE_COMMANDS.keys()))
            while new_gene == old_gene:
                new_gene = random.choice(list(GENE_COMMANDS.keys()))
            chromosome.genes[i] = new_gene
        i += 1


def create_population(size, gene_count):
    population = []
    i = 0
    while i < size:
        population.append(Chromosome(gene_count))
        i += 1
    return population


def evaluate_population(population):
    for chromosome in population:
        chromosome.calculate_fitness()
    population.sort(key=lambda x: x.fitness, reverse=True)


def make_next_population(population, gene_count):
    new_population = []

    # 우수 개체 보존
    elite_index = 0
    while elite_index < ELITE_COUNT and elite_index < len(population):
        new_population.append(population[elite_index].copy())
        elite_index += 1

    # 선택 -> 교차 -> 돌연변이
    while len(new_population) < POPULATION_SIZE:
        father = select(population)
        mother = select(population)
        child1, child2 = crossover(father, mother)

        mutate(child1)
        mutate(child2)

        new_population.append(child1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)

    return new_population


def print_chromosome_summary(chromosome):
    print("염색체 전체 Command:", chromosome.command_names())
    print("실제 사용 Command:", chromosome.used_command_names())
    print("사용 턴 수:", chromosome.used_turn)
    print("적합도:", chromosome.fitness)
    print("결과:", chromosome.message)


def print_solution_path(chromosome):
    print("\n[성공 경로]")
    for h in chromosome.history:
        status = h["status"]
        print(
            f"{h['turn']:02d}. {h['command']:5s} -> "
            f"L(M={status['left_m']}, C={status['left_c']}), "
            f"R(M={status['right_m']}, C={status['right_c']}), "
            f"boat={status['boat']} / {h['message']}"
        )


def run_genetic_algorithm():
    """20개 유전자부터 시작해서 1개씩 줄여가며 성공 경로를 찾는다."""
    best_solution = None

    gene_count = MAX_GENE_COUNT
    while gene_count >= 1:
        print(f"\n==============================")
        print(f"유전자 수 {gene_count}개로 탐색 시작")
        print(f"==============================")

        population = create_population(POPULATION_SIZE, gene_count)
        generation = 0
        found_in_this_gene_count = None

        while generation < MAX_GENERATION:
            evaluate_population(population)
            best = population[0]

            if generation % 50 == 0 or best.result == "win":
                print(
                    f"세대 {generation:03d} | "
                    f"최고 적합도 {best.fitness:5d} | "
                    f"결과 {best.result:7s} | "
                    f"사용 턴 {best.used_turn:2d} | "
                    f"명령 {best.used_command_names()}"
                )

            if best.result == "win":
                found_in_this_gene_count = best.copy()
                print("성공 염색체 발견")
                print_chromosome_summary(found_in_this_gene_count)
                break

            population = make_next_population(population, gene_count)
            generation += 1

        if found_in_this_gene_count is not None:
            best_solution = found_in_this_gene_count

        # 1개씩 줄여가면서 더 짧은 염색체를 시도한다.
        gene_count -= 1

    print("\n==============================")
    print("최종 결과")
    print("==============================")

    if best_solution is None:
        print("20턴 이내 성공 경로를 찾지 못했습니다.")
    else:
        print("20턴 이내 성공 경로를 찾았습니다.")
        print_chromosome_summary(best_solution)
        print_solution_path(best_solution)


if __name__ == "__main__":
    run_genetic_algorithm()
