from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
import random
from typing import Iterable
import webbrowser

import pandas as pd


# Parametros do algoritmo genetico.
POPULATION_SIZE = 120
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.18
GENERATIONS = 250
EARLY_STOPPING_PATIENCE = 100
RANDOM_SEED = 42
MULTI_RUN_ATTEMPTS = 12
RESULT_FILE = "populacaofinal.xlsx"
DASHBOARD_FILE = "dashboard.html"
TEMPLATE_FILE = "modelo_apresentacoes.csv"
TOP_SOLUTIONS = 30

@dataclass(frozen=True, order=True)
class Slot:
    # Representa um horario unico de apresentacao.
    dia: str
    horario: str


@dataclass(frozen=True)
class Presentation:
    # Guarda os dados da apresentacao a ser alocada.
    aluno: str
    orientador: str | None = None


@dataclass
class Assignment:
    # Representa uma banca completa vinculada a um aluno em um slot.
    apresentacao: Presentation
    slot: Slot
    professor1: str
    professor2: str
    suplente: str


@dataclass
class RankedSolution:
    # Guarda uma solucao completa do algoritmo para exibicao no ranking.
    posicao: int
    fitness: int
    penalty: int
    assignments: list[Assignment]


@dataclass(frozen=True)
class RunSummary:
    # Resume o resultado de uma tentativa completa do algoritmo com uma semente especifica.
    execution: int
    seed: int
    best_fitness: int
    best_penalty: int


@dataclass(frozen=True)
class PenaltyWeights:
    # Bloco central de pesos.
    hard_professor_em_duas_bancas_no_mesmo_horario: int = 1000
    hard_professor_indisponivel_titular: int = 1000
    hard_banca_incompleta: int = 1000
    soft_orientador_na_banca: int = 100
    soft_suplente_fora_do_horario: int = 100
    soft_prioridade_noite: int = 0.3
    soft_professor_sem_intervalo: int = 0.4
    soft_professor_tres_sequencia: int = 0.2
    soft_parallel_bancas: int = 0.4

PENALTY_WEIGHTS = PenaltyWeights()


def parse_time_value(value: str) -> float:
    try:
        hour, minute = value.split(":")
        return int(hour) + int(minute) / 60.0
    except Exception:
        return 0.0


def parse_slot_interval(slot: Slot) -> tuple[float, float]:
    start, end = 0.0, 0.0
    parts = slot.horario.split("-")
    if len(parts) == 2:
        start = parse_time_value(parts[0].strip())
        end = parse_time_value(parts[1].strip())
    return start, end


def is_night_slot(slot: Slot) -> bool:
    start, _ = parse_slot_interval(slot)
    return 18.0 <= start < 23.0


def slot_gap_hours(first: Slot, second: Slot) -> float:
    _, end = parse_slot_interval(first)
    start, _ = parse_slot_interval(second)
    return start - end


def parse_multi_value(value: object) -> list[str]:
    # Converte celulas como "Segunda;Terca" em lista de valores limpos.
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def find_input_workbook() -> Path:
    # Procura especificamente pela planilha de dados reais das bancas
    # Se não encontrar, usa o primeiro .xlsx do diretório como fallback
    target_file = "populacaoinicial.xlsx"
    target_path = Path(target_file)
    
    if target_path.exists():
        return target_path
    
    # Fallback: usa o primeiro .xlsx do diretório que não seja o resultado
    candidates = sorted(
        path for path in Path(".").glob("*.xlsx") if path.name != RESULT_FILE
    )
    if not candidates:
        raise FileNotFoundError("Nenhum arquivo .xlsx foi encontrado no diretorio do projeto.")
    return candidates[0]


def load_professor_availability(workbook_path: Path) -> dict[str, set[Slot]]:
    # Le a aba principal da planilha e monta a disponibilidade por professor.
    df = pd.read_excel(workbook_path, sheet_name=0)
    required_columns = {"Professor", "Dia", "Horario"}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            "A planilha de disponibilidade precisa conter as colunas: "
            + ", ".join(sorted(required_columns))
        )

    availability: dict[str, set[Slot]] = {}

    for _, row in df.iterrows():
        professor = str(row["Professor"]).strip()
        dias = parse_multi_value(row["Dia"])
        horarios = parse_multi_value(row["Horario"])

        if not professor or not dias or not horarios:
            continue

        availability.setdefault(professor, set())
        for dia in dias:
            for horario in horarios:
                availability[professor].add(Slot(dia=dia, horario=horario))

    if len(availability) < 3:
        raise ValueError("Sao necessarios pelo menos 3 professores com horarios validos.")

    return availability


def load_presentations(workbook_path: Path, fallback_count: int) -> tuple[list[Presentation], bool]:
    # Tenta localizar uma aba de apresentacoes. Se nao houver, gera alunos ficticios.
    excel = pd.ExcelFile(workbook_path)
    valid_aliases = {
        "Aluno",
        "Aluna",
        "Nome",
        "Discente",
        "Estudante",
    }

    for sheet_name in excel.sheet_names:
        df = pd.read_excel(workbook_path, sheet_name=sheet_name)
        if df.empty:
            continue

        aluno_column = next((column for column in df.columns if column in valid_aliases), None)
        if not aluno_column:
            continue

        orientador_column = next(
            (column for column in df.columns if column.lower() == "orientador"),
            None,
        )

        presentations: list[Presentation] = []
        for _, row in df.iterrows():
            aluno = str(row[aluno_column]).strip() if pd.notna(row[aluno_column]) else ""
            if not aluno:
                continue

            orientador = None
            if orientador_column and pd.notna(row[orientador_column]):
                orientador = str(row[orientador_column]).strip() or None

            presentations.append(Presentation(aluno=aluno, orientador=orientador))

        if presentations:
            return presentations, False

    generated = [
        Presentation(aluno=f"Aluno {index:02d}") for index in range(1, fallback_count + 1)
    ]
    return generated, True


def build_candidate_lists(
    availability: dict[str, set[Slot]],
) -> tuple[list[Slot], dict[Slot, list[str]], int]:
    # Descobre quais slots existem e quais professores podem participar em cada um deles.
    all_slots = sorted({slot for slots in availability.values() for slot in slots})
    candidates_by_slot = {
        slot: sorted([professor for professor, slots in availability.items() if slot in slots])
        for slot in all_slots
    }
    max_simultaneous = max((len(candidates) for candidates in candidates_by_slot.values()), default=0)
    minimum_required = 3 if max_simultaneous >= 3 else 2
    feasible_slots = [slot for slot in all_slots if len(candidates_by_slot[slot]) >= minimum_required]

    if not feasible_slots:
        raise ValueError("Nenhum horario possui professores suficientes para montar a banca.")

    return feasible_slots, candidates_by_slot, max_simultaneous


def choose_professors_for_slot(
    rng: random.Random,
    slot: Slot,
    candidates_by_slot: dict[Slot, list[str]],
    all_professors: list[str],
    orientador: str | None = None,
    workload: Counter[str] | None = None,
) -> tuple[str, str, str]:
    # Escolhe dois titulares e um suplente para um slot.
    # Quando nao existe terceiro professor simultaneo, o suplente e buscado fora do slot.
    candidates = [name for name in candidates_by_slot[slot] if name != orientador]
    if len(candidates) < 2:
        raise ValueError(
            f"O horario {slot.dia} {slot.horario} nao possui professores suficientes para a banca."
        )

    def sort_by_load(names: list[str]) -> list[str]:
        if workload is None:
            return names.copy()
        return sorted(names, key=lambda name: (workload.get(name, 0), rng.random()))

    selected_pool = sort_by_load(candidates)
    titulares = selected_pool[:2]
    suplente_candidates = [name for name in candidates if name not in titulares]

    if suplente_candidates:
        suplente = sort_by_load(suplente_candidates)[0]
    else:
        suplente_pool = [
            professor
            for professor in all_professors
            if professor not in titulares and professor != orientador
        ]
        if not suplente_pool:
            raise ValueError("Nao foi possivel selecionar um suplente valido.")
        suplente = sort_by_load(suplente_pool)[0]

    return titulares[0], titulares[1], suplente


def create_individual(
    presentations: list[Presentation],
    feasible_slots: list[Slot],
    candidates_by_slot: dict[Slot, list[str]],
    all_professors: list[str],
    rng: random.Random,
) -> list[Assignment]:
    # Cria um individuo da populacao: uma agenda completa com todas as apresentacoes.
    # Permitimos usar o mesmo slot varias vezes, desde que os professores nao sejam repetidos no mesmo horario.
    if not feasible_slots:
        raise ValueError("Nao ha horarios viaveis para agendar apresentacoes.")

    workload: Counter[str] = Counter()
    selected_slots = [rng.choice(feasible_slots) for _ in presentations]
    individual: list[Assignment] = []

    for presentation, slot in zip(presentations, selected_slots):
        professor1, professor2, suplente = choose_professors_for_slot(
            rng,
            slot,
            candidates_by_slot,
            all_professors,
            presentation.orientador,
            workload=workload,
        )
        individual.append(
            Assignment(
                apresentacao=presentation,
                slot=slot,
                professor1=professor1,
                professor2=professor2,
                suplente=suplente,
            )
        )
        workload[professor1] += 1
        workload[professor2] += 1
        workload[suplente] += 1

    return individual


def repair_individual(
    individual: list[Assignment],
    feasible_slots: list[Slot],
    candidates_by_slot: dict[Slot, list[str]],
    all_professors: list[str],
    rng: random.Random,
) -> list[Assignment]:
    # Corrige filhos gerados pelo crossover/mutacao para garantir que cada apresentacao tenha um slot valido.
    available_slots = [slot for slot in feasible_slots if len(candidates_by_slot.get(slot, [])) >= 2]
    if not available_slots:
        raise ValueError("Nao ha horarios com professores suficientes para montar uma banca.")

    workload: Counter[str] = Counter()
    for assignment in individual:
        workload[assignment.professor1] += 1
        workload[assignment.professor2] += 1
        workload[assignment.suplente] += 1

    repaired: list[Assignment] = []

    for assignment in individual:
        slot = assignment.slot
        if len(candidates_by_slot.get(slot, [])) < 2:
            slot = rng.choice(available_slots)

        professor1, professor2, suplente = choose_professors_for_slot(
            rng,
            slot,
            candidates_by_slot,
            all_professors,
            assignment.apresentacao.orientador,
            workload=workload,
        )

        repaired.append(
            Assignment(
                apresentacao=assignment.apresentacao,
                slot=slot,
                professor1=professor1,
                professor2=professor2,
                suplente=suplente,
            )
        )
        workload[professor1] += 1
        workload[professor2] += 1
        workload[suplente] += 1

    return repaired


def crossover(
    parent1: list[Assignment],
    parent2: list[Assignment],
    feasible_slots: list[Slot],
    candidates_by_slot: dict[Slot, list[str]],
    all_professors: list[str],
    rng: random.Random,
) -> tuple[list[Assignment], list[Assignment]]:
    # Combina dois pais cortando a agenda em um ponto aleatorio.
    cut = rng.randint(1, len(parent1) - 1)
    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]
    return (
        repair_individual(child1, feasible_slots, candidates_by_slot, all_professors, rng),
        repair_individual(child2, feasible_slots, candidates_by_slot, all_professors, rng),
    )


def mutate(
    individual: list[Assignment],
    feasible_slots: list[Slot],
    candidates_by_slot: dict[Slot, list[str]],
    all_professors: list[str],
    rng: random.Random,
) -> list[Assignment]:
    # Faz pequenas alteracoes em slots ou professores para aumentar diversidade genetica.
    mutated = [
        Assignment(
            apresentacao=item.apresentacao,
            slot=item.slot,
            professor1=item.professor1,
            professor2=item.professor2,
            suplente=item.suplente,
        )
        for item in individual
    ]

    for index, assignment in enumerate(mutated):
        if rng.random() >= MUTATION_RATE:
            continue

        workload: Counter[str] = Counter()
        for idx, item in enumerate(mutated):
            if idx == index:
                continue
            workload[item.professor1] += 1
            workload[item.professor2] += 1
            workload[item.suplente] += 1

        if rng.random() < 0.45:
            free_slots = [slot for slot in feasible_slots if slot != assignment.slot]
            if free_slots:
                new_slot = rng.choice(free_slots)
                professor1, professor2, suplente = choose_professors_for_slot(
                    rng,
                    new_slot,
                    candidates_by_slot,
                    all_professors,
                    assignment.apresentacao.orientador,
                    workload=workload,
                )
                mutated[index] = Assignment(
                    apresentacao=assignment.apresentacao,
                    slot=new_slot,
                    professor1=professor1,
                    professor2=professor2,
                    suplente=suplente,
                )
        else:
            professor1, professor2, suplente = choose_professors_for_slot(
                rng,
                assignment.slot,
                candidates_by_slot,
                all_professors,
                assignment.apresentacao.orientador,
                workload=workload,
            )
            mutated[index] = Assignment(
                apresentacao=assignment.apresentacao,
                slot=assignment.slot,
                professor1=professor1,
                professor2=professor2,
                suplente=suplente,
            )

    return repair_individual(mutated, feasible_slots, candidates_by_slot, all_professors, rng)


def is_extreme_slot(slot: Slot) -> bool:
    # Penaliza horarios considerados menos confortaveis.
    return slot.horario in {"07:00-09:00", "19:00-21:00"}


def compute_penalty(
    individual: list[Assignment],
    availability: dict[str, set[Slot]],
) -> int:
    # Avalia uma agenda inteira somando penalidades.
    # Quanto menor a penalidade, melhor a solucao.
    penalty = 0
    professor_slot_usage: Counter[tuple[str, Slot]] = Counter()
    assignments_by_professor: dict[str, list[Slot]] = {}
    assignments_by_slot: dict[Slot, list[Assignment]] = {}

    for assignment in individual:
        professores = [assignment.professor1, assignment.professor2, assignment.suplente]

        # Penalidade Grave: banca deve ter 3 papeis distintos.
        if len(set(professores)) < 3:
            penalty += PENALTY_WEIGHTS.hard_banca_incompleta

        for professor, papel in [
            (assignment.professor1, "titular"),
            (assignment.professor2, "titular"),
            (assignment.suplente, "suplente"),
        ]:
            if assignment.slot not in availability.get(professor, set()):
                if papel == "suplente":
                    penalty += PENALTY_WEIGHTS.soft_suplente_fora_do_horario
                else:
                    penalty += PENALTY_WEIGHTS.hard_professor_indisponivel_titular
            professor_slot_usage[(professor, assignment.slot)] += 1
            assignments_by_professor.setdefault(professor, []).append(assignment.slot)

        assignments_by_slot.setdefault(assignment.slot, []).append(assignment)

        # Penalidade Média: orientador não deve participar da própria banca.
        if assignment.apresentacao.orientador and assignment.apresentacao.orientador in professores:
            penalty += PENALTY_WEIGHTS.soft_orientador_na_banca

        # Penalidade Média: priorizar horário noturno.
        if not is_night_slot(assignment.slot):
            penalty += PENALTY_WEIGHTS.soft_prioridade_noite

    # Penalidade Grave: mesmo professor em mais de uma banca no mesmo horário.
    for count in professor_slot_usage.values():
        if count > 1:
            penalty += (
                PENALTY_WEIGHTS.hard_professor_em_duas_bancas_no_mesmo_horario * (count - 1)
            )

    # Penalidades super baixa: intervalo entre bancas e sequencia de bancos para cada professor.
    for professor, slots in assignments_by_professor.items():
        ordered_slots = sorted(slots, key=lambda slot: parse_slot_interval(slot)[0])
        consecutive_count = 1
        previous_slot = None

        for current_slot in ordered_slots:
            if previous_slot is None:
                previous_slot = current_slot
                continue

            gap = slot_gap_hours(previous_slot, current_slot)
            if gap < 0.5:
                penalty += PENALTY_WEIGHTS.soft_professor_sem_intervalo
                consecutive_count += 1
            else:
                consecutive_count = 1

            if consecutive_count > 2:
                penalty += PENALTY_WEIGHTS.soft_professor_tres_sequencia

            previous_slot = current_slot

    # Penalidade super baixa: duas bancas no mesmo horário, mesmo com professores/alunos distintos.
    for slot, assignments in assignments_by_slot.items():
        if len(assignments) > 1:
            penalty += (len(assignments) - 1) * PENALTY_WEIGHTS.soft_parallel_bancas

    return penalty


def compute_fitness(
    individual: list[Assignment],
    availability: dict[str, set[Slot]],
) -> int:
    # O algoritmo genetico continua maximizando fitness.
    # Para usar penalidade, convertemos fitness em negativo da penalidade.
    return -compute_penalty(individual, availability)


def tournament_selection(
    population: list[list[Assignment]],
    scores: list[int],
    rng: random.Random,
) -> list[Assignment]:
    # Seleciona o melhor individuo dentro de um pequeno torneio aleatorio.
    tournament_indexes = rng.sample(range(len(population)), min(TOURNAMENT_SIZE, len(population)))
    best_index = max(tournament_indexes, key=lambda index: scores[index])
    return population[best_index]


def evolve_schedule(
    presentations: list[Presentation],
    availability: dict[str, set[Slot]],
    feasible_slots: list[Slot],
    candidates_by_slot: dict[Slot, list[str]],
    seed: int,
) -> tuple[list[Assignment], int, int, list[RankedSolution]]:
    # Executa o ciclo do algoritmo genetico:
    # 1. cria populacao inicial
    # 2. avalia fitness
    # 3. seleciona pais
    # 4. aplica crossover e mutacao
    # 5. preserva o melhor individuo (elitismo)
    rng = random.Random(seed)
    all_professors = sorted(availability)
    population = [
        create_individual(presentations, feasible_slots, candidates_by_slot, all_professors, rng)
        for _ in range(POPULATION_SIZE)
    ]
    scores = [compute_fitness(individual, availability) for individual in population]

    best_score = max(scores)
    generations_without_improvement = 0

    for _ in range(GENERATIONS):
        elite_index = max(range(len(population)), key=lambda index: scores[index])
        next_population = [population[elite_index]]

        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, scores, rng)
            parent2 = tournament_selection(population, scores, rng)
            child1, child2 = crossover(
                parent1,
                parent2,
                feasible_slots,
                candidates_by_slot,
                all_professors,
                rng,
            )
            next_population.append(
                mutate(child1, feasible_slots, candidates_by_slot, all_professors, rng)
            )
            if len(next_population) < POPULATION_SIZE:
                next_population.append(
                    mutate(child2, feasible_slots, candidates_by_slot, all_professors, rng)
                )

        population = next_population
        scores = [compute_fitness(individual, availability) for individual in population]

        current_best = max(scores)
        if current_best > best_score:
            best_score = current_best
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if best_score == 0:
            break
        if generations_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    penalties = [compute_penalty(individual, availability) for individual in population]
    ranked_indexes = sorted(
        range(len(population)),
        key=lambda index: (scores[index], -penalties[index]),
        reverse=True,
    )
    top_solutions: list[RankedSolution] = []

    for posicao, index in enumerate(ranked_indexes[:TOP_SOLUTIONS], start=1):
        individual = population[index]
        top_solutions.append(
            RankedSolution(
                posicao=posicao,
                fitness=scores[index],
                penalty=penalties[index],
                assignments=individual,
            )
        )

    best_solution = top_solutions[0]
    return (
        best_solution.assignments,
        best_solution.fitness,
        best_solution.penalty,
        top_solutions,
    )


def run_multiple_searches(
    presentations: list[Presentation],
    availability: dict[str, set[Slot]],
    feasible_slots: list[Slot],
    candidates_by_slot: dict[Slot, list[str]],
) -> tuple[list[Assignment], int, int, list[RankedSolution], list[RunSummary], int, int]:
    # Executa varias tentativas com sementes diferentes e preserva a melhor solucao global.
    global_best_assignments: list[Assignment] | None = None
    global_best_fitness: int | None = None
    global_best_penalty: int | None = None
    global_best_seed: int | None = None
    global_best_execution: int | None = None
    global_ranked_solutions: list[RankedSolution] = []
    run_summaries: list[RunSummary] = []

    for execution in range(1, MULTI_RUN_ATTEMPTS + 1):
        seed = RANDOM_SEED + execution - 1
        assignments, fitness, penalty, ranked_solutions = evolve_schedule(
            presentations,
            availability,
            feasible_slots,
            candidates_by_slot,
            seed,
        )
        run_summaries.append(
            RunSummary(
                execution=execution,
                seed=seed,
                best_fitness=fitness,
                best_penalty=penalty,
            )
        )

        global_ranked_solutions.extend(ranked_solutions)

        if global_best_fitness is None or fitness > global_best_fitness:
            global_best_assignments = assignments
            global_best_fitness = fitness
            global_best_penalty = penalty
            global_best_seed = seed
            global_best_execution = execution

            if fitness == 0:
                break

    global_ranked_solutions = sorted(
        global_ranked_solutions,
        key=lambda solution: (solution.fitness, -solution.penalty),
        reverse=True,
    )[:TOP_SOLUTIONS]

    global_ranked_solutions = [
        RankedSolution(
            posicao=index + 1,
            fitness=solution.fitness,
            penalty=solution.penalty,
            assignments=solution.assignments,
        )
        for index, solution in enumerate(global_ranked_solutions)
    ]

    if (
        global_best_assignments is None
        or global_best_fitness is None
        or global_best_penalty is None
        or global_best_seed is None
        or global_best_execution is None
    ):
        raise ValueError("Nenhuma solucao foi encontrada nas execucoes automaticas.")

    return (
        global_best_assignments,
        global_best_fitness,
        global_best_penalty,
        global_ranked_solutions,
        run_summaries,
        global_best_seed,
        global_best_execution,
    )


def build_assignment_penalty_details(
    assignments: list[Assignment],
    availability: dict[str, set[Slot]],
) -> list[dict[str, object]]:
    # Calcula a perda de pontos diretamente atribuivel a cada banca.
    professor_slot_usage: Counter[tuple[str, Slot]] = Counter()
    assignments_by_professor: dict[str, list[tuple[int, Slot]]] = {}
    assignments_by_slot: dict[Slot, list[int]] = {}

    ordered_assignments = sorted(
        assignments,
        key=lambda item: (item.slot.dia, item.slot.horario, item.apresentacao.aluno),
    )

    for index, assignment in enumerate(ordered_assignments):
        for professor in [assignment.professor1, assignment.professor2, assignment.suplente]:
            professor_slot_usage[(professor, assignment.slot)] += 1
            assignments_by_professor.setdefault(professor, []).append((index, assignment.slot))
        assignments_by_slot.setdefault(assignment.slot, []).append(index)

    professor_interval_penalties: dict[int, int] = {}
    professor_sequence_penalties: dict[int, int] = {}

    for professor, slots in assignments_by_professor.items():
        ordered_slots = sorted(slots, key=lambda pair: parse_slot_interval(pair[1])[0])
        consecutive_count = 1
        previous_slot = None

        for assignment_index, current_slot in ordered_slots:
            if previous_slot is None:
                previous_slot = current_slot
                continue

            gap = slot_gap_hours(previous_slot, current_slot)
            if gap < 0.5:
                professor_interval_penalties[assignment_index] = (
                    professor_interval_penalties.get(assignment_index, 0) + 1
                )
                consecutive_count += 1
            else:
                consecutive_count = 1

            if consecutive_count > 2:
                professor_sequence_penalties[assignment_index] = (
                    professor_sequence_penalties.get(assignment_index, 0) + 1
                )

            previous_slot = current_slot

    parallel_penalties: dict[int, int] = {}
    for slot, indexes in assignments_by_slot.items():
        if len(indexes) > 1:
            for index in indexes[1:]:
                parallel_penalties[index] = parallel_penalties.get(index, 0) + 1

    details: list[dict[str, object]] = []
    for index, assignment in enumerate(ordered_assignments):
        row_penalty = 0.0
        reasons: list[str] = []
        professores = [assignment.professor1, assignment.professor2, assignment.suplente]

        if len(set(professores)) < 3:
            row_penalty += PENALTY_WEIGHTS.hard_banca_incompleta
            reasons.append("Banca incompleta ou professor repetido")

        for professor, papel in [
            (assignment.professor1, "Professor 1"),
            (assignment.professor2, "Professor 2"),
            (assignment.suplente, "Suplente"),
        ]:
            if assignment.slot not in availability.get(professor, set()):
                if papel == "Suplente":
                    row_penalty += PENALTY_WEIGHTS.soft_suplente_fora_do_horario
                    reasons.append("Suplente fora do horario")
                else:
                    row_penalty += PENALTY_WEIGHTS.hard_professor_indisponivel_titular
                    reasons.append(f"{papel} indisponivel no horario")

            if professor_slot_usage[(professor, assignment.slot)] > 1:
                extra = professor_slot_usage[(professor, assignment.slot)] - 1
                row_penalty += (
                    PENALTY_WEIGHTS.hard_professor_em_duas_bancas_no_mesmo_horario * extra
                )
                reasons.append(f"{professor} em mais de uma banca no mesmo horario")

        if assignment.apresentacao.orientador and assignment.apresentacao.orientador in professores:
            row_penalty += PENALTY_WEIGHTS.soft_orientador_na_banca
            reasons.append("Orientador participa da propria banca")

        if not is_night_slot(assignment.slot):
            row_penalty += PENALTY_WEIGHTS.soft_prioridade_noite
            reasons.append("Priorizar horario noturno")

        interval_count = professor_interval_penalties.get(index, 0)
        if interval_count > 0:
            row_penalty += interval_count * PENALTY_WEIGHTS.soft_professor_sem_intervalo
            reasons.append("Intervalo inferior a 30 minutos")

        sequence_count = professor_sequence_penalties.get(index, 0)
        if sequence_count > 0:
            row_penalty += sequence_count * PENALTY_WEIGHTS.soft_professor_tres_sequencia
            reasons.append("Mais de duas bancas em sequencia")

        parallel_count = parallel_penalties.get(index, 0)
        if parallel_count > 0:
            row_penalty += parallel_count * PENALTY_WEIGHTS.soft_parallel_bancas
            reasons.append("Bancas paralelas no mesmo horario")

        details.append(
            {
                "assignment": assignment,
                "penalty": row_penalty,
                "fitness": -row_penalty,
                "reasons": "Sem perda nesta linha" if not reasons else "; ".join(dict.fromkeys(reasons)),
            }
        )

    return details


def assignments_to_dataframe(
    assignments: Iterable[Assignment],
    availability: dict[str, set[Slot]] | None = None,
) -> pd.DataFrame:
    # Converte a agenda em tabela; quando possivel, acrescenta a perda por linha.
    assignment_list = list(assignments)
    rows = []

    if availability is None:
        ordered_assignments = sorted(
            assignment_list,
            key=lambda item: (item.slot.dia, item.slot.horario, item.apresentacao.aluno),
        )
        for assignment in ordered_assignments:
            rows.append(
                {
                    "Aluno": assignment.apresentacao.aluno,
                    "Orientador": assignment.apresentacao.orientador or "",
                    "Dia": assignment.slot.dia,
                    "Horario": assignment.slot.horario,
                    "Professor 1": assignment.professor1,
                    "Professor 2": assignment.professor2,
                    "Suplente": assignment.suplente,
                }
            )
        return pd.DataFrame(rows)

    for detail in build_assignment_penalty_details(assignment_list, availability):
        assignment = detail["assignment"]
        rows.append(
            {
                "Aluno": assignment.apresentacao.aluno,
                "Orientador": assignment.apresentacao.orientador or "",
                "Dia": assignment.slot.dia,
                "Horario": assignment.slot.horario,
                "Professor 1": assignment.professor1,
                "Professor 2": assignment.professor2,
                "Suplente": assignment.suplente,
                "Penalidade da linha": detail["penalty"],
                "Fitness da linha": detail["fitness"],
                "Motivos": detail["reasons"],
            }
        )
    return pd.DataFrame(rows)


def build_load_dataframe(assignments: Iterable[Assignment]) -> pd.DataFrame:
    # Resume quantas bancas cada professor recebeu.
    counter: Counter[str] = Counter()
    for assignment in assignments:
        counter[assignment.professor1] += 1
        counter[assignment.professor2] += 1
        counter[assignment.suplente] += 1

    rows = [{"Professor": professor, "Total em bancas": total} for professor, total in sorted(counter.items())]
    return pd.DataFrame(rows)


def export_results(
    workbook_path: Path,
    assignments: list[Assignment],
    fitness: int,
    penalty: int,
    used_demo_presentations: bool,
    max_simultaneous: int,
) -> Path:
    # Salva o resultado final em um arquivo Excel com agenda, carga e resumo.
    agenda_df = assignments_to_dataframe(assignments)
    carga_df = build_load_dataframe(assignments)
    resumo_df = pd.DataFrame(
        [
            {"Indicador": "Arquivo analisado", "Valor": workbook_path.name},
            {"Indicador": "Penalidade final", "Valor": penalty},
            {"Indicador": "Fitness final", "Valor": fitness},
            {
                "Indicador": "Modo demonstracao",
                "Valor": "Sim - alunos de teste gerados automaticamente" if used_demo_presentations else "Nao",
            },
            {"Indicador": "Total de apresentacoes", "Valor": len(assignments)},
            {
                "Indicador": "Layout atual da disponibilidade",
                "Valor": "Aceito via colunas Professor, Dia e Horario com ; separando multiplos valores",
            },
            {
                "Indicador": "Maximo de professores simultaneos por slot",
                "Valor": max_simultaneous,
            },
            {
                "Indicador": "Melhoria sugerida",
                "Valor": "Separar apresentacoes em outra aba e normalizar um horario por linha",
            },
        ]
    )

    result_path = workbook_path.with_name(RESULT_FILE)
    with pd.ExcelWriter(result_path, engine="openpyxl") as writer:
        agenda_df.to_excel(writer, sheet_name="Agenda Final", index=False)
        carga_df.to_excel(writer, sheet_name="Carga Professores", index=False)
        resumo_df.to_excel(writer, sheet_name="Resumo", index=False)

    return result_path


def export_template_if_needed(workbook_path: Path, used_demo_presentations: bool) -> Path | None:
    # Gera um modelo simples de alunos quando a planilha original nao traz apresentacoes.
    if not used_demo_presentations:
        return None

    template_path = workbook_path.with_name(TEMPLATE_FILE)
    template_df = pd.DataFrame(
        [
            {"Aluno": "Aluno 01", "Orientador": "Professor Exemplo"},
            {"Aluno": "Aluno 02", "Orientador": "Professor Exemplo"},
            {"Aluno": "Aluno 03", "Orientador": "Professor Exemplo"},
        ]
    )
    template_df.to_csv(template_path, index=False, encoding="utf-8-sig")
    return template_path


def build_dashboard(
    workbook_path: Path,
    ranked_solutions: list[RankedSolution],
    best_penalty: int,
    best_fitness: int,
    total_presentations: int,
    availability: dict[str, set[Slot]],
    best_seed: int,
    best_execution: int,
) -> Path:
    # Cria um HTML simples para visualizar o ranking das melhores 30 solucoes.
    cards_html = []
    for solution in ranked_solutions:
        agenda_html = assignments_to_dataframe(
            solution.assignments,
            availability=availability,
        ).to_html(index=False, classes="agenda")
        cards_html.append(
            f"""
            <details {"open" if solution.posicao == 1 else ""} class="solution-card">
              <summary>
                <span class="rank">#{solution.posicao}</span>
                <span class="fitness">Fitness: {solution.fitness}</span>
                <span class="penalty">Penalidade: {solution.penalty}</span>
              </summary>
              <div class="table-wrapper">{agenda_html}</div>
            </details>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard das Bancas de TCC</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --card: #ffffff;
      --text: #1e293b;
      --muted: #64748b;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
      --border: #dbe4f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: linear-gradient(180deg, #eff6ff 0%, var(--bg) 45%);
      color: var(--text);
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 2rem;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .stat {{
      background: var(--accent-soft);
      border-radius: 14px;
      padding: 16px;
    }}
    .stat strong {{
      display: block;
      font-size: 1.5rem;
      margin-top: 6px;
    }}
    .section-title {{
      margin: 28px 0 14px;
      font-size: 1.2rem;
    }}
    .solution-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      margin-bottom: 14px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }}
    .solution-card summary {{
      list-style: none;
      cursor: pointer;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      padding: 16px 18px;
      background: #fcfffe;
      border-bottom: 1px solid var(--border);
    }}
    .solution-card summary::-webkit-details-marker {{
      display: none;
    }}
    .rank {{
      background: var(--accent);
      color: white;
      border-radius: 999px;
      padding: 6px 12px;
      font-weight: 700;
    }}
    .fitness, .penalty {{
      font-weight: 600;
    }}
    .table-note {{
      color: var(--muted);
      margin: 0 0 14px;
      line-height: 1.45;
    }}
    .summary-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      margin: 0 0 22px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }}
    .table-wrapper {{
      padding: 18px;
      overflow-x: auto;
    }}
    .agenda td:last-child, .agenda th:last-child {{
      min-width: 320px;
    }}
    .agenda td:nth-last-child(2), .agenda th:nth-last-child(2) {{
      min-width: 120px;
    }}
    .agenda td:nth-last-child(3), .agenda th:nth-last-child(3) {{
      min-width: 140px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-size: 0.95rem;
      vertical-align: top;
    }}
    th {{
      background: #f8fafc;
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>Dashboard das 30 Melhores Bancas</h1>
      <p class="subtitle">Arquivo analisado: {escape(workbook_path.name)}. Ranking ordenado da melhor para a pior solucao pelo fitness.</p>
      <div class="stats">
        <div class="stat">Melhor fitness<strong>{best_fitness}</strong></div>
        <div class="stat">Menor penalidade<strong>{best_penalty}</strong></div>
        <div class="stat">Apresentacoes por solucao<strong>{total_presentations}</strong></div>
        <div class="stat">Solucoes exibidas<strong>{len(ranked_solutions)}</strong></div>
      </div>
    </section>
    <h2 class="section-title">Ranking completo</h2>
    <p class="table-note">As colunas de linha mostram a perda diretamente atribuida a cada banca. A penalidade global de distribuicao desigual entre professores continua no total da solucao.</p>
    {''.join(cards_html)}
  </div>
</body>
</html>
"""

    dashboard_path = workbook_path.with_name(DASHBOARD_FILE)
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


def main() -> None:
    # Fluxo principal:
    # 1. localiza o arquivo
    # 2. le disponibilidade e apresentacoes
    # 3. executa o algoritmo genetico
    # 4. exporta a melhor agenda encontrada
    workbook_path = find_input_workbook()
    availability = load_professor_availability(workbook_path)
    feasible_slots, candidates_by_slot, max_simultaneous = build_candidate_lists(availability)

    fallback_count = min(10, len(feasible_slots))
    presentations, used_demo_presentations = load_presentations(workbook_path, fallback_count)

    if not feasible_slots:
        raise ValueError(
            "Nao ha horarios viaveis para agendar apresentacoes."
        )

    (
        best_schedule,
        best_fitness,
        best_penalty,
        ranked_solutions,
        run_summaries,
        best_seed,
        best_execution,
    ) = run_multiple_searches(
        presentations,
        availability,
        feasible_slots,
        candidates_by_slot,
    )
    result_path = export_results(
        workbook_path,
        best_schedule,
        best_fitness,
        best_penalty,
        used_demo_presentations,
        max_simultaneous,
    )
    template_path = export_template_if_needed(workbook_path, used_demo_presentations)
    dashboard_path = build_dashboard(
        workbook_path,
        ranked_solutions,
        best_penalty,
        best_fitness,
        len(best_schedule),
        availability,
        best_seed,
        best_execution,
    )
    webbrowser.open(dashboard_path.resolve().as_uri())

    print(f"Arquivo analisado: {workbook_path.name}")
    print(f"Apresentacoes agendadas: {len(best_schedule)}")
    print(f"Melhor resultado obtido na tentativa: {best_execution}")
    print(f"Fitness final: {best_fitness}")
    print(f"Resultado salvo em: {result_path}")
    print(f"Dashboard salva em: {dashboard_path}")
    if template_path:
        print(
            "Nao foi encontrada uma lista de apresentacoes na planilha. "
            f"Foi criado um modelo em: {template_path}"
        )
    if max_simultaneous < 3:
        print(
            "Aviso: os dados atuais nao permitem 3 professores no mesmo horario. "
            "O algoritmo gerou bancas validando 2 titulares simultaneos e suplente em melhor esforco."
        )


if __name__ == "__main__":
    main()
