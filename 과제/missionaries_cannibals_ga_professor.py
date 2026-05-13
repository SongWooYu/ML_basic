import random
import copy

# 1. 기존 게임 설정 유지
TOTAL_M = 3
TOTAL_C = 3
# 유전자 후보군 (Command 리스트) [cite: 30, 74]
COMMAND_LIST = ["1M", "2M", "1C", "2C", "1M1C"]

# 2. 유전자 알고리즘 파라미터 [cite: 315-317]
POPULATION_SIZE = 50   # 한 세대당 염색체 수
GENE_LENGTH = 20       # 유전자 개수 (최대 20턴 제한)
MUTATION_RATE = 0.1    # 돌연변이 확률 [cite: 316]

class Chromosome:
    def __init__(self, genes=None):
        # 20개의 무작위 명령어로 구성된 염색체 생성 [cite: 112, 140]
        if genes is None:
            self.genes = [random.choice(COMMAND_LIST) for _ in range(GENE_LENGTH)]
        else:
            self.genes = genes
        self.fitness = 0

    def run_game(self):
        """기존 로직을 사용하여 20턴 동안 게임 시뮬레이션 [cite: 35-43]"""
        status = {"left_m": 0, "left_c": 0, "right_m": 3, "right_c": 3, "boat": "right"}
        score = 0
        
        # 20번의 유전자(명령어) 실행 [cite: 335]
        for cmd in self.genes:
            # 기존 제공해주신 game 로직 수행
            next_status, end, message = self.simulate_step(status, cmd)
            
            # 판정 결과에 따른 적합도 부여 [cite: 133]
            if end:
                if "성공" in message:
                    score += 200  # 성공 시 대폭 가산
                    status = next_status
                    break
                else: # 게임오버 (강에 잡아먹힘 등)
                    score -= 10   # 감점
                    break
            
            status = next_status
            score += 5  # 한 턴을 안전하게 생존할 때마다 가산
            
        # 최종 적합도: (왼쪽으로 옮긴 인원수)를 중요 지표로 설정 [cite: 43]
        self.fitness = score + (status["left_m"] + status["left_c"]) * 20
        return self.fitness

    def simulate_step(self, status, command):
        """제공해주신 game() 및 judge() 함수 로직을 그대로 사용"""
        move_m, move_c = self.get_move_counts(command)
        next_status = status.copy()

        if status["boat"] == "right":
            if status["right_m"] < move_m or status["right_c"] < move_c:
                return status, True, "이동 불가"
            next_status["right_m"] -= move_m
            next_status["right_c"] -= move_c
            next_status["left_m"] += move_m
            next_status["left_c"] += move_c
            next_status["boat"] = "left"
        else:
            if status["left_m"] < move_m or status["left_c"] < move_c:
                return status, True, "이동 불가"
            next_status["left_m"] -= move_m
            next_status["left_c"] -= move_c
            next_status["right_m"] += move_m
            next_status["right_c"] += move_c
            next_status["boat"] = "right"

        # judge 로직
        end, message = self.judge_logic(next_status)
        return next_status, end, message

    def get_move_counts(self, cmd):
        mapping = {"1M": (1, 0), "2M": (2, 0), "1C": (0, 1), "2C": (0, 2), "1M1C": (1, 1)}
        return mapping[cmd]

    def judge_logic(self, s):
        def is_safe(m, c): return m == 0 or m >= c
        if not is_safe(s["left_m"], s["left_c"]) or not is_safe(s["right_m"], s["right_c"]):
            return True, "게임오버"
        if s["left_m"] == 3 and s["left_c"] == 3:
            return True, "성공"
        return False, "계속"

# 3. 메인 GA 루프 (Outer While) [cite: 154, 211-224]
population = [Chromosome() for _ in range(POPULATION_SIZE)]
generation = 0

while generation < 500:
    # 적합도 계산 및 정렬 [cite: 141-142, 392]
    population.sort(key=lambda x: x.run_game(), reverse=True)
    
    current_best = population[0]
    print(f"[{generation}세대] 최고 적합도: {current_best.fitness}")

    # 성공 조건 달성 시 종료 [cite: 43, 224]
    if current_best.fitness >= 300: 
        print("\n최적의 해답 유전자를 찾았습니다!")
        print(f"순서: {current_best.genes}")
        break

    # 다음 세대 생성 (선택, 교차, 돌연변이) [cite: 147-153]
    new_pop = population[:2] # 엘리트 보존 (상위 2개)
    
    while len(new_pop) < POPULATION_SIZE:
        # 룰렛 휠 선택 [cite: 162, 354-360]
        p1 = random.choice(population[:10]) 
        p2 = random.choice(population[:10])
        
        # 교차 (Crossover) [cite: 194-197]
        cp = random.randint(1, GENE_LENGTH - 1)
        child_genes = p1.genes[:cp] + p2.genes[cp:]
        child = Chromosome(child_genes)
        
        # 돌연변이 (Mutation) [cite: 204-206]
        if random.random() < MUTATION_RATE:
            child.genes[random.randint(0, GENE_LENGTH-1)] = random.choice(COMMAND_LIST)
            
        new_pop.append(child)
    
    population = new_pop
    generation += 1