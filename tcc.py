from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from html import escape
from pathlib import Path
import random
import re

import pandas as pd


# Parâmetros do algoritmo genético.
POPULATION_SIZE_FACTOR = 2
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.18
CONFLICT_MUTATION_RATE = 0.70
STABLE_GENE_MUTATION_RATE = 0.04
GENERATIONS = 1000
EARLY_STOPPING_PATIENCE = 250
RANDOM_SEED = 42
BENCHMARK_PERCENTAGES = tuple(range(0, 81, 5))

# Pesos da função de fitness.
PROFESSOR_CONFLICT_PENALTY = 1000
PARALLEL_BOARD_PENALTY = 1

INPUT_DIR = Path("População Inicial")
OUTPUT_DIR = Path("População Final")

DAY_ORDER = {
    "Segunda": 1,
    "Terça": 2,
    "Quarta": 3,
    "Quinta": 4,
    "Sexta": 5,
}

SLOT_PREFIX_DAYS = {
    "S": "Segunda",
    "T": "Terça",
    "Q": "Quarta",
    "QI": "Quinta",
    "SX": "Sexta",
}

SHIFT_STARTS = {
    "T": "13:30",
    "N": "19:00",
}


@dataclass(frozen=True)
class Slot:
    code: str
    day: str
    period: str
    interval: str


@dataclass(frozen=True)
class Professor:
    code: str
    name: str
    availability: frozenset[str]


@dataclass(frozen=True)
class Board:
    number: int
    academic: str
    advisor_code: str
    professor1_code: str
    professor2_code: str
    viable_slots: tuple[int, ...]

    @property
    def professor_codes(self) -> tuple[str, str, str]:
        return (
            self.advisor_code,
            self.professor1_code,
            self.professor2_code,
        )


@dataclass(frozen=True)
class Evaluation:
    fitness: int
    penalty: int
    professor_conflicts: int
    parallel_board_excess: int


@dataclass(frozen=True)
class BenchmarkResult:
    weeks: int
    restriction: int
    successful_runs: int
    failed_runs: int
    average_fitness: float | None
    best_fitness: int | None
    average_professor_conflicts: float | None
    average_parallel_excess: float | None
    average_generations: float | None
    average_viable_slots_per_board: float | None
    boards_without_common_slot: int


def run_benchmark_attempt(
    base_boards: list[Board],
    professors: dict[str, Professor],
    base_slots: list[Slot],
    weeks: int,
    restriction: int,
    seed: int,
) -> tuple[int | None, int | None, int | None, int | None, float, int]:
    (
        restricted_boards,
        expanded_slots,
        average_viable_slots,
        boards_without_common_slot,
    ) = build_restricted_scenario(
        base_boards,
        professors,
        base_slots,
        weeks,
        restriction,
        seed,
    )
    if restricted_boards is None:
        return (
            None,
            None,
            None,
            None,
            average_viable_slots,
            boards_without_common_slot,
        )

    (
        _individual,
        evaluation,
        generations_run,
        _population_size,
        _history,
    ) = evolve(
        restricted_boards,
        expanded_slots,
        seed,
    )
    return (
        evaluation.fitness,
        evaluation.professor_conflicts,
        evaluation.parallel_board_excess,
        generations_run,
        average_viable_slots,
        0,
    )


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def parse_time(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def format_time(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def parse_slot_code(column: object) -> Slot | None:
    code = str(column).strip().upper()
    match = re.fullmatch(r"([A-Z]+)([1-5])", code)
    if not match:
        return None

    prefix_with_shift, number_text = match.groups()
    shift = prefix_with_shift[-1]
    day_prefix = prefix_with_shift[:-1]
    if shift not in SHIFT_STARTS or day_prefix not in SLOT_PREFIX_DAYS:
        return None

    number = int(number_text)
    if shift == "N" and number > 4:
        return None

    start = parse_time(SHIFT_STARTS[shift]) + (number - 1) * 60
    end = start + 60
    return Slot(
        code=code,
        day=SLOT_PREFIX_DAYS[day_prefix],
        period="Tarde" if shift == "T" else "Noite",
        interval=f"{format_time(start)}-{format_time(end)}",
    )


def slot_sort_key(slot: Slot) -> tuple[int, int]:
    start = parse_time(slot.interval.split("-")[0])
    return DAY_ORDER[slot.day], start


def is_available(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in {
        "",
        "0",
        "n",
        "não",
        "nao",
        "false",
        "f",
    }


def find_sheet(
    excel: pd.ExcelFile,
    required_columns: set[str],
) -> tuple[str, pd.DataFrame]:
    for sheet_name in excel.sheet_names:
        frame = pd.read_excel(excel, sheet_name=sheet_name)
        if required_columns.issubset(frame.columns):
            return sheet_name, frame
    raise ValueError(
        "Nenhuma aba contém as colunas obrigatórias: "
        + ", ".join(sorted(required_columns))
    )


def load_problem(
    workbook_path: Path,
) -> tuple[list[Board], dict[str, Professor], list[Slot]]:
    excel = pd.ExcelFile(workbook_path)

    _, professor_frame = find_sheet(excel, {"ID", "Professor"})
    slot_columns = {
        column: slot
        for column in professor_frame.columns
        if (slot := parse_slot_code(column)) is not None
    }
    if len(slot_columns) != 45:
        raise ValueError(
            f"A aba Professores deve conter 45 horários válidos; "
            f"foram encontrados {len(slot_columns)}."
        )

    slots = sorted(set(slot_columns.values()), key=slot_sort_key)
    slot_index_by_code = {
        slot.code: index
        for index, slot in enumerate(slots)
    }

    professors: dict[str, Professor] = {}
    for row_number, row in professor_frame.iterrows():
        code = normalize_code(row["ID"])
        name = str(row["Professor"]).strip() if pd.notna(row["Professor"]) else ""
        if not code or not name:
            continue
        if code in professors:
            raise ValueError(f"Código de professor duplicado: {code}.")

        availability = frozenset(
            slot.code
            for column, slot in slot_columns.items()
            if is_available(row[column])
        )
        professors[code] = Professor(
            code=code,
            name=name,
            availability=availability,
        )

    if not professors:
        raise ValueError("Nenhum professor válido foi encontrado.")

    board_columns = {
        "Nº",
        "Acadêmico",
        "Professor 1 (Presidente)",
        "Professor 2",
        "Professor 3",
    }
    _, board_frame = find_sheet(excel, board_columns)
    boards: list[Board] = []

    for _, row in board_frame.iterrows():
        number = int(row["Nº"])
        academic = str(row["Acadêmico"]).strip()
        codes = (
            normalize_code(row["Professor 1 (Presidente)"]),
            normalize_code(row["Professor 2"]),
            normalize_code(row["Professor 3"]),
        )

        if any(not code for code in codes):
            raise ValueError(f"A banca {number} possui código de professor vazio.")
        if len(set(codes)) != 3:
            raise ValueError(
                f"A banca {number} deve possuir três professores distintos; "
                f"foram informados {codes}."
            )

        unknown = [code for code in codes if code not in professors]
        if unknown:
            raise ValueError(
                f"A banca {number} referencia códigos inexistentes: "
                + ", ".join(unknown)
            )

        common_availability = set(professors[codes[0]].availability)
        common_availability.intersection_update(professors[codes[1]].availability)
        common_availability.intersection_update(professors[codes[2]].availability)
        viable_slots = tuple(
            slot_index_by_code[code]
            for code in common_availability
        )
        if not viable_slots:
            raise ValueError(
                f"A banca {number} não possui horário em que os três professores "
                "estejam simultaneamente disponíveis."
            )

        boards.append(
            Board(
                number=number,
                academic=academic,
                advisor_code=codes[0],
                professor1_code=codes[1],
                professor2_code=codes[2],
                viable_slots=viable_slots,
            )
        )

    if not boards:
        raise ValueError("Nenhuma banca válida foi encontrada.")

    return boards, professors, slots


def build_restricted_scenario(
    base_boards: list[Board],
    professors: dict[str, Professor],
    base_slots: list[Slot],
    weeks: int,
    restriction: int,
    seed: int,
) -> tuple[list[Board] | None, list[Slot], float, int]:
    expanded_slots: list[Slot] = []
    base_code_by_index: dict[int, str] = {}
    for week in range(1, weeks + 1):
        for base_slot in base_slots:
            slot_index = len(expanded_slots)
            expanded_slots.append(
                Slot(
                    code=f"S{week}-{base_slot.code}",
                    day=f"Semana {week} - {base_slot.day}",
                    period=base_slot.period,
                    interval=base_slot.interval,
                )
            )
            base_code_by_index[slot_index] = base_slot.code

    available_cells = [
        (professor_code, slot_index)
        for professor_code, professor in professors.items()
        for slot_index, base_code in base_code_by_index.items()
        if base_code in professor.availability
    ]
    blocked_count = round(len(available_cells) * restriction / 100)
    rng = random.Random(seed)
    rng.shuffle(available_cells)
    blocked_cells = set(available_cells[:blocked_count])

    available_by_professor: dict[str, set[int]] = {
        professor_code: {
            slot_index
            for slot_index, base_code in base_code_by_index.items()
            if base_code in professor.availability
            and (professor_code, slot_index) not in blocked_cells
        }
        for professor_code, professor in professors.items()
    }

    restricted_boards: list[Board] = []
    boards_without_common_slot = 0
    total_viable_slots = 0
    for board in base_boards:
        common_slots = set(available_by_professor[board.advisor_code])
        common_slots.intersection_update(
            available_by_professor[board.professor1_code]
        )
        common_slots.intersection_update(
            available_by_professor[board.professor2_code]
        )
        if not common_slots:
            boards_without_common_slot += 1
            continue
        viable_slots = tuple(sorted(common_slots))
        total_viable_slots += len(viable_slots)
        restricted_boards.append(
            Board(
                number=board.number,
                academic=board.academic,
                advisor_code=board.advisor_code,
                professor1_code=board.professor1_code,
                professor2_code=board.professor2_code,
                viable_slots=viable_slots,
            )
        )

    average_viable_slots = (
        total_viable_slots / len(base_boards)
        if base_boards
        else 0.0
    )
    if boards_without_common_slot:
        return (
            None,
            expanded_slots,
            average_viable_slots,
            boards_without_common_slot,
        )
    return (
        restricted_boards,
        expanded_slots,
        average_viable_slots,
        0,
    )


def evaluate(individual: list[int], boards: list[Board]) -> Evaluation:
    professor_slot_usage: Counter[tuple[str, int]] = Counter()
    slot_usage: Counter[int] = Counter(individual)

    for board, slot_index in zip(boards, individual):
        for professor_code in board.professor_codes:
            professor_slot_usage[(professor_code, slot_index)] += 1

    professor_conflicts = sum(
        count - 1
        for count in professor_slot_usage.values()
        if count > 1
    )
    parallel_board_excess = sum(
        count - 1
        for count in slot_usage.values()
        if count > 1
    )
    penalty = (
        professor_conflicts * PROFESSOR_CONFLICT_PENALTY
        + parallel_board_excess * PARALLEL_BOARD_PENALTY
    )
    return Evaluation(
        fitness=-penalty,
        penalty=penalty,
        professor_conflicts=professor_conflicts,
        parallel_board_excess=parallel_board_excess,
    )


def create_individual(
    boards: list[Board],
    rng: random.Random,
) -> list[int]:
    # A população inicial não recebe qualquer conhecimento sobre conflitos.
    # Cada banca é colocada uniformemente em um de seus horários viáveis.
    return [
        rng.choice(board.viable_slots)
        for board in boards
    ]


def tournament_selection(
    population: list[list[int]],
    evaluations: list[Evaluation],
    rng: random.Random,
) -> list[int]:
    indexes = rng.sample(
        range(len(population)),
        min(TOURNAMENT_SIZE, len(population)),
    )
    best_index = max(indexes, key=lambda index: evaluations[index].fitness)
    return population[best_index]


def crossover(
    parent1: list[int],
    parent2: list[int],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    cut = rng.randint(1, len(parent1) - 1)
    return (
        parent1[:cut] + parent2[cut:],
        parent2[:cut] + parent1[cut:],
    )


def mutate(
    individual: list[int],
    boards: list[Board],
    rng: random.Random,
) -> list[int]:
    mutated = individual.copy()
    professor_slot_usage: Counter[tuple[str, int]] = Counter()
    for board, slot_index in zip(boards, individual):
        for code in board.professor_codes:
            professor_slot_usage[(code, slot_index)] += 1

    for index, board in enumerate(boards):
        slot_index = individual[index]
        has_conflict = any(
            professor_slot_usage[(code, slot_index)] > 1
            for code in board.professor_codes
        )
        mutation_rate = (
            CONFLICT_MUTATION_RATE
            if has_conflict
            else STABLE_GENE_MUTATION_RATE
        )
        if rng.random() >= mutation_rate:
            continue
        alternatives = [
            candidate_slot
            for candidate_slot in board.viable_slots
            if candidate_slot != mutated[index]
        ]
        if alternatives:
            mutated[index] = rng.choice(alternatives)
    return mutated


def theoretical_minimum_penalty(
    boards: list[Board],
    slot_count: int,
) -> int:
    # Com mais bancas que horários, pelo menos n-h bancas serão paralelas.
    return max(0, len(boards) - slot_count) * PARALLEL_BOARD_PENALTY


def evolve(
    boards: list[Board],
    slots: list[Slot],
    seed: int,
) -> tuple[list[int], Evaluation, int, int, list[dict[str, int]]]:
    rng = random.Random(seed)
    population_size = POPULATION_SIZE_FACTOR * len(boards)
    if population_size % 2 != 0:
        population_size += 1
    population = [
        create_individual(boards, rng)
        for _ in range(population_size)
    ]
    evaluations = [evaluate(individual, boards) for individual in population]
    best_fitness = max(item.fitness for item in evaluations)
    generations_without_improvement = 0
    generations_run = 0
    target_fitness = -theoretical_minimum_penalty(boards, len(slots))
    history = [
        {
            "generation": 0,
            "best_fitness": best_fitness,
            "average_fitness": int(
                sum(item.fitness for item in evaluations) / len(evaluations)
            ),
        }
    ]

    for generation in range(1, GENERATIONS + 1):
        generations_run = generation
        elite_index = max(
            range(len(population)),
            key=lambda index: evaluations[index].fitness,
        )
        next_population = [population[elite_index].copy()]

        while len(next_population) < population_size:
            parent1 = tournament_selection(population, evaluations, rng)
            parent2 = tournament_selection(population, evaluations, rng)
            child1, child2 = crossover(parent1, parent2, rng)
            for child in (child1, child2):
                child = mutate(child, boards, rng)
                next_population.append(child)
                if len(next_population) == population_size:
                    break

        population = next_population
        evaluations = [evaluate(individual, boards) for individual in population]
        current_best = max(item.fitness for item in evaluations)
        history.append(
            {
                "generation": generation,
                "best_fitness": current_best,
                "average_fitness": int(
                    sum(item.fitness for item in evaluations) / len(evaluations)
                ),
            }
        )
        if current_best > best_fitness:
            best_fitness = current_best
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if best_fitness == target_fitness:
            break
        if generations_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    best_index = max(
        range(len(population)),
        key=lambda index: evaluations[index].fitness,
    )
    return (
        population[best_index],
        evaluations[best_index],
        generations_run,
        population_size,
        history,
    )


def build_conflict_details(
    individual: list[int],
    boards: list[Board],
) -> tuple[Counter[tuple[str, int]], Counter[int]]:
    professor_usage: Counter[tuple[str, int]] = Counter()
    slot_usage: Counter[int] = Counter(individual)
    for board, slot_index in zip(boards, individual):
        for code in board.professor_codes:
            professor_usage[(code, slot_index)] += 1
    return professor_usage, slot_usage


def export_result(
    workbook_path: Path,
    boards: list[Board],
    professors: dict[str, Professor],
    slots: list[Slot],
    individual: list[int],
    result: Evaluation,
    generations_run: int,
    population_size: int,
    history: list[dict[str, int]],
) -> Path:
    professor_usage, slot_usage = build_conflict_details(individual, boards)
    rows = []

    for board, slot_index in zip(boards, individual):
        slot = slots[slot_index]
        conflicting_professors = [
            professors[code].name
            for code in board.professor_codes
            if professor_usage[(code, slot_index)] > 1
        ]
        rows.append(
            {
                "Nº": board.number,
                "Acadêmico": board.academic,
                "Código Orientador": board.advisor_code,
                "Orientador": professors[board.advisor_code].name,
                "Código Professor 1": board.professor1_code,
                "Professor 1": professors[board.professor1_code].name,
                "Código Professor 2": board.professor2_code,
                "Professor 2": professors[board.professor2_code].name,
                "Código Horário": slot.code,
                "Dia": slot.day,
                "Período": slot.period,
                "Horário": slot.interval,
                "Bancas neste horário": slot_usage[slot_index],
                "Conflito de professor": "Sim" if conflicting_professors else "Não",
                "Professores em conflito": "; ".join(conflicting_professors),
            }
        )

    schedule = pd.DataFrame(rows)
    schedule["_ordem_dia"] = schedule["Dia"].map(DAY_ORDER)
    schedule["_ordem_hora"] = schedule["Horário"].str[:5]
    schedule = schedule.sort_values(
        ["_ordem_dia", "_ordem_hora", "Nº"],
        kind="stable",
    ).drop(columns=["_ordem_dia", "_ordem_hora"])

    summary = pd.DataFrame(
        [
            {"Indicador": "Arquivo analisado", "Valor": workbook_path.name},
            {"Indicador": "Percentual de restrição artificial", "Valor": "0%"},
            {"Indicador": "Total de bancas", "Valor": len(boards)},
            {"Indicador": "Total de horários", "Valor": len(slots)},
            {"Indicador": "Fitness final", "Valor": result.fitness},
            {"Indicador": "Penalidade final", "Valor": result.penalty},
            {
                "Indicador": "Conflitos de professor",
                "Valor": result.professor_conflicts,
            },
            {
                "Indicador": "Bancas paralelas excedentes",
                "Valor": result.parallel_board_excess,
            },
            {
                "Indicador": "Penalidade por conflito de professor",
                "Valor": PROFESSOR_CONFLICT_PENALTY,
            },
            {
                "Indicador": "Penalidade por banca paralela excedente",
                "Valor": PARALLEL_BOARD_PENALTY,
            },
            {
                "Indicador": "Gerações executadas",
                "Valor": generations_run,
            },
            {
                "Indicador": "Tamanho da população",
                "Valor": population_size,
            },
            {
                "Indicador": "Regra do tamanho da população",
                "Valor": f"{POPULATION_SIZE_FACTOR} x total de bancas",
            },
            {
                "Indicador": "Melhor fitness da população inicial",
                "Valor": history[0]["best_fitness"],
            },
            {
                "Indicador": "Semente aleatória",
                "Valor": RANDOM_SEED,
            },
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"resultado_{workbook_path.stem}_0restricao.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        schedule.to_excel(writer, sheet_name="Agenda Final", index=False)
        summary.to_excel(writer, sheet_name="Resumo", index=False)
        pd.DataFrame(history).rename(
            columns={
                "generation": "Geração",
                "best_fitness": "Melhor fitness",
                "average_fitness": "Fitness médio",
            }
        ).to_excel(writer, sheet_name="Evolução", index=False)

        agenda_sheet = writer.book["Agenda Final"]
        agenda_sheet.freeze_panes = "A2"
        agenda_sheet.auto_filter.ref = agenda_sheet.dimensions

    return output_path


def export_dashboard(
    workbook_path: Path,
    boards: list[Board],
    professors: dict[str, Professor],
    slots: list[Slot],
    individual: list[int],
    result: Evaluation,
    generations_run: int,
    population_size: int,
    history: list[dict[str, int]],
) -> Path:
    professor_usage, slot_usage = build_conflict_details(individual, boards)
    scheduled = sorted(
        zip(boards, individual),
        key=lambda item: (
            slot_sort_key(slots[item[1]]),
            item[0].number,
        ),
    )

    rows = []
    for board, slot_index in scheduled:
        slot = slots[slot_index]
        conflicting_professors = [
            professors[code].name
            for code in board.professor_codes
            if professor_usage[(code, slot_index)] > 1
        ]
        conflict_text = (
            "; ".join(conflicting_professors)
            if conflicting_professors
            else "Sem conflito"
        )
        conflict_class = "conflict" if conflicting_professors else "ok"
        rows.append(
            f"""
            <tr>
              <td>{board.number}</td>
              <td>{escape(board.academic)}</td>
              <td><strong>{escape(slot.code)}</strong><br>{escape(slot.day)} · {escape(slot.period)}</td>
              <td>{escape(slot.interval)}</td>
              <td>{escape(professors[board.advisor_code].name)}</td>
              <td>{escape(professors[board.professor1_code].name)}</td>
              <td>{escape(professors[board.professor2_code].name)}</td>
              <td>{slot_usage[slot_index]}</td>
              <td><span class="status {conflict_class}">{escape(conflict_text)}</span></td>
            </tr>
            """
        )

    occupancy = Counter(slot_usage.values())
    occupancy_text = ", ".join(
        f"{count} horário(s) com {boards_in_slot} banca(s)"
        for boards_in_slot, count in sorted(occupancy.items())
    )
    evolution_rows = "".join(
        f"<tr><td>{item['generation']}</td>"
        f"<td>{item['best_fitness']}</td>"
        f"<td>{item['average_fitness']}</td></tr>"
        for item in history
    )
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resultado das Bancas — {escape(workbook_path.stem)}</title>
  <style>
    :root {{
      --bg: #f3f6fb;
      --card: #fff;
      --text: #172033;
      --muted: #64748b;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
      --border: #dbe4f0;
      --danger: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    main {{ max-width: 1500px; margin: auto; padding: 30px 20px 50px; }}
    .hero, .table-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, .06);
    }}
    .hero {{ padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
    .subtitle {{ margin: 0; color: var(--muted); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .stat {{ padding: 16px; background: #f8fafc; border-radius: 12px; }}
    .stat span {{ display: block; color: var(--muted); font-size: .86rem; }}
    .stat strong {{ display: block; margin-top: 5px; font-size: 1.45rem; }}
    .note {{ margin: 18px 0 0; color: var(--muted); }}
    .table-card {{ margin-top: 22px; overflow: hidden; }}
    .evolution-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .evolution-note {{
      padding: 16px;
      border-radius: 12px;
      background: #f8fafc;
      color: var(--muted);
    }}
    .table-header {{ padding: 18px 20px; border-bottom: 1px solid var(--border); }}
    .table-header h2 {{ margin: 0; font-size: 1.15rem; }}
    .table-wrap {{ overflow: auto; max-height: 72vh; }}
    table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #eaf2f8;
      font-size: .85rem;
    }}
    tr:hover td {{ background: #f8fafc; }}
    .status {{
      display: inline-block;
      padding: 5px 9px;
      border-radius: 999px;
      font-size: .82rem;
      font-weight: 700;
    }}
    .status.ok {{ color: var(--accent); background: var(--accent-soft); }}
    .status.conflict {{ color: var(--danger); background: #fee2e2; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Agenda genética das bancas — 0% de restrição artificial</h1>
      <p class="subtitle">Arquivo analisado: {escape(workbook_path.name)}</p>
      <div class="stats">
        <div class="stat"><span>Fitness final</span><strong>{result.fitness}</strong></div>
        <div class="stat"><span>Penalidade total</span><strong>{result.penalty}</strong></div>
        <div class="stat"><span>Conflitos de professor</span><strong>{result.professor_conflicts}</strong></div>
        <div class="stat"><span>Bancas paralelas excedentes</span><strong>{result.parallel_board_excess}</strong></div>
        <div class="stat"><span>Total de bancas</span><strong>{len(boards)}</strong></div>
        <div class="stat"><span>Tamanho da população</span><strong>{population_size}</strong></div>
        <div class="stat"><span>Gerações executadas</span><strong>{generations_run}</strong></div>
      </div>
      <p class="note"><strong>Ocupação:</strong> {escape(occupancy_text)}. Cada banca excedente no mesmo horário reduz o fitness em 1; conflito de professor reduz em 1000.</p>
      <div class="evolution-grid">
        <div class="evolution-note"><strong>População inicial aleatória</strong><br>Melhor fitness inicial: {history[0]['best_fitness']}.</div>
        <div class="evolution-note"><strong>Objetivo evolutivo</strong><br>Fitness ótimo teórico: {-theoretical_minimum_penalty(boards, len(slots))}.</div>
      </div>
    </section>
    <section class="table-card">
      <div class="table-header"><h2>Evolução do fitness por geração</h2></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Geração</th><th>Melhor fitness</th><th>Fitness médio</th></tr></thead>
          <tbody>{evolution_rows}</tbody>
        </table>
      </div>
    </section>
    <section class="table-card">
      <div class="table-header"><h2>Agenda completa</h2></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Nº</th><th>Acadêmico</th><th>Janela</th><th>Horário</th>
              <th>Orientador</th><th>Professor 1</th><th>Professor 2</th>
              <th>Bancas simultâneas</th><th>Conflito</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dashboard_path = (
        OUTPUT_DIR
        / f"dashboard_{workbook_path.stem}_0restricao.html"
    )
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


def run_restriction_benchmark(
    workbook_path: Path,
    runs_per_level: int,
    workers: int = 1,
) -> list[BenchmarkResult]:
    base_boards, professors, base_slots = load_problem(workbook_path)
    results: list[BenchmarkResult] = []
    executor = (
        ProcessPoolExecutor(max_workers=workers)
        if workers > 1
        else None
    )
    try:
        for weeks in (1, 2):
            print(f"\nBenchmark com {weeks} semana(s)")
            for restriction in BENCHMARK_PERCENTAGES:
                fitness_values: list[int] = []
                conflict_values: list[int] = []
                parallel_values: list[int] = []
                generation_values: list[int] = []
                viable_slot_values: list[float] = []
                failed_runs = 0
                maximum_boards_without_slot = 0

                seeds = [
                    RANDOM_SEED
                    + weeks * 1_000_000
                    + restriction * 1000
                    + run_index
                    for run_index in range(runs_per_level)
                ]
                if executor is None:
                    attempt_results = [
                        run_benchmark_attempt(
                            base_boards,
                            professors,
                            base_slots,
                            weeks,
                            restriction,
                            seed,
                        )
                        for seed in seeds
                    ]
                else:
                    futures = [
                        executor.submit(
                            run_benchmark_attempt,
                            base_boards,
                            professors,
                            base_slots,
                            weeks,
                            restriction,
                            seed,
                        )
                        for seed in seeds
                    ]
                    attempt_results = [
                        future.result()
                        for future in futures
                    ]

                for (
                    fitness,
                    conflicts,
                    parallel_excess,
                    generations_run,
                    average_viable_slots,
                    boards_without_common_slot,
                ) in attempt_results:
                    viable_slot_values.append(average_viable_slots)
                    maximum_boards_without_slot = max(
                        maximum_boards_without_slot,
                        boards_without_common_slot,
                    )
                    if fitness is None:
                        failed_runs += 1
                        continue
                    fitness_values.append(fitness)
                    conflict_values.append(conflicts or 0)
                    parallel_values.append(parallel_excess or 0)
                    generation_values.append(generations_run or 0)

                successful_runs = len(fitness_values)
                result = BenchmarkResult(
                    weeks=weeks,
                    restriction=restriction,
                    successful_runs=successful_runs,
                    failed_runs=failed_runs,
                    average_fitness=(
                        sum(fitness_values) / successful_runs
                        if successful_runs
                        else None
                    ),
                    best_fitness=max(fitness_values) if fitness_values else None,
                    average_professor_conflicts=(
                        sum(conflict_values) / successful_runs
                        if successful_runs
                        else None
                    ),
                    average_parallel_excess=(
                        sum(parallel_values) / successful_runs
                        if successful_runs
                        else None
                    ),
                    average_generations=(
                        sum(generation_values) / successful_runs
                        if successful_runs
                        else None
                    ),
                    average_viable_slots_per_board=(
                        sum(viable_slot_values) / len(viable_slot_values)
                        if viable_slot_values
                        else None
                    ),
                    boards_without_common_slot=maximum_boards_without_slot,
                )
                results.append(result)

                if result.average_fitness is None:
                    print(
                        f"  {restriction:>2}%: inviável; "
                        f"até {maximum_boards_without_slot} banca(s) sem horário comum"
                    )
                else:
                    print(
                        f"  {restriction:>2}%: fitness médio "
                        f"{result.average_fitness:.1f}; "
                        f"conflitos médios "
                        f"{result.average_professor_conflicts:.1f}; "
                        f"falhas {failed_runs}/{runs_per_level}"
                    )
    finally:
        if executor is not None:
            executor.shutdown()

    return results


def export_restriction_dashboard(
    workbook_path: Path,
    results: list[BenchmarkResult],
    runs_per_level: int,
) -> tuple[Path, Path]:
    rows = []
    for result in results:
        status = (
            "Inviável"
            if result.average_fitness is None
            else "Concluído"
        )
        status_class = "failure" if status == "Inviável" else "success"
        average_fitness = (
            "—"
            if result.average_fitness is None
            else f"{result.average_fitness:.1f}"
        )
        best_fitness = (
            "—"
            if result.best_fitness is None
            else str(result.best_fitness)
        )
        average_conflicts = (
            "—"
            if result.average_professor_conflicts is None
            else f"{result.average_professor_conflicts:.1f}"
        )
        average_parallel = (
            "—"
            if result.average_parallel_excess is None
            else f"{result.average_parallel_excess:.1f}"
        )
        average_generations = (
            "—"
            if result.average_generations is None
            else f"{result.average_generations:.1f}"
        )
        viable_slots = (
            "—"
            if result.average_viable_slots_per_board is None
            else f"{result.average_viable_slots_per_board:.1f}"
        )
        rows.append(
            f"""
            <tr>
              <td>{result.weeks}</td>
              <td>{result.restriction}%</td>
              <td><span class="badge {status_class}">{status}</span></td>
              <td>{result.successful_runs}</td>
              <td>{result.failed_runs}</td>
              <td>{average_fitness}</td>
              <td>{best_fitness}</td>
              <td>{average_conflicts}</td>
              <td>{average_parallel}</td>
              <td>{average_generations}</td>
              <td>{viable_slots}</td>
              <td>{result.boards_without_common_slot}</td>
            </tr>
            """
        )

    panels = []
    for weeks in (1, 2):
        scenario_results = [
            result for result in results if result.weeks == weeks
        ]
        valid_values = [
            abs(result.average_fitness)
            for result in scenario_results
            if result.average_fitness is not None
        ]
        maximum = max(valid_values, default=1)
        bars = []
        for result in scenario_results:
            if result.average_fitness is None:
                width = 100
                value = "Inviável"
                bar_class = "bar failure"
            else:
                width = max(
                    2,
                    round(abs(result.average_fitness) / maximum * 100),
                )
                value = f"{result.average_fitness:.1f}"
                bar_class = "bar"
            bars.append(
                f"""
                <div class="bar-row">
                  <span>{result.restriction}%</span>
                  <div class="track">
                    <div class="{bar_class}" style="width:{width}%"></div>
                  </div>
                  <strong>{value}</strong>
                </div>
                """
            )
        panels.append(
            f"""
            <section class="panel">
              <h2>{weeks} semana{'s' if weeks > 1 else ''}</h2>
              <p>{45 * weeks} janelas independentes.</p>
              {''.join(bars)}
            </section>
            """
        )

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard de restrições — {escape(workbook_path.stem)}</title>
  <style>
    :root {{
      --bg:#f3f6fb; --card:#fff; --text:#172033; --muted:#64748b;
      --accent:#0f766e; --border:#dbe4f0; --danger:#b91c1c;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; background:var(--bg); color:var(--text);
      font-family:"Segoe UI",Arial,sans-serif;
    }}
    main {{ max-width:1500px; margin:auto; padding:28px 20px 50px; }}
    .hero,.panel,.table-card {{
      background:var(--card); border:1px solid var(--border);
      border-radius:16px; box-shadow:0 10px 30px rgba(15,23,42,.06);
    }}
    .hero {{ padding:24px; }}
    h1,h2 {{ margin:0; }}
    .hero p,.panel p {{ color:var(--muted); }}
    .grid {{
      display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
      gap:18px; margin-top:20px;
    }}
    .panel {{ padding:20px; }}
    .bar-row {{
      display:grid; grid-template-columns:45px 1fr 90px;
      align-items:center; gap:10px; margin-top:10px;
    }}
    .track {{
      height:14px; background:#e8eef6; border-radius:999px;
      overflow:hidden;
    }}
    .bar {{ height:100%; background:var(--accent); }}
    .bar.failure {{ background:var(--danger); }}
    .table-card {{ margin-top:20px; overflow:hidden; }}
    .table-wrap {{ overflow:auto; max-height:70vh; }}
    table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
    th,td {{
      padding:10px 12px; border-bottom:1px solid var(--border);
      text-align:left;
    }}
    th {{ position:sticky; top:0; background:#eaf2f8; z-index:2; }}
    .badge {{
      display:inline-block; padding:4px 8px; border-radius:999px;
      font-size:.8rem; font-weight:700;
    }}
    .badge.success {{ color:var(--accent); background:#ccfbf1; }}
    .badge.failure {{ color:var(--danger); background:#fee2e2; }}
    @media(max-width:850px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Restrições artificiais: uma versus duas semanas</h1>
      <p><strong>Base:</strong> {escape(workbook_path.name)}. Bloqueio de
      células professor–horário de 0% a 80%, em passos de 5%.
      Cada ponto contém {runs_per_level} execução(ões).</p>
      <p>Uma execução é marcada como inviável quando ao menos uma banca não
      possui nenhuma janela comum aos seus três professores fixos.</p>
    </section>
    <div class="grid">{''.join(panels)}</div>
    <section class="table-card">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Semanas</th><th>Restrição</th><th>Status</th>
              <th>Sucessos</th><th>Falhas</th><th>Fitness médio</th>
              <th>Melhor fitness</th><th>Conflitos médios</th>
              <th>Paralelas excedentes</th><th>Gerações médias</th>
              <th>Janelas viáveis/banca</th><th>Bancas sem janela</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = (
        OUTPUT_DIR
        / f"dashboard_restricoes_{workbook_path.stem}.html"
    )
    html_path.write_text(html, encoding="utf-8")

    data_path = OUTPUT_DIR / f"resultados_restricoes_{workbook_path.stem}.xlsx"
    data_rows = [
        {
            "Semanas": result.weeks,
            "Restrição (%)": result.restriction,
            "Execuções concluídas": result.successful_runs,
            "Falhas": result.failed_runs,
            "Fitness médio": result.average_fitness,
            "Melhor fitness": result.best_fitness,
            "Conflitos médios": result.average_professor_conflicts,
            "Bancas paralelas excedentes": result.average_parallel_excess,
            "Gerações médias": result.average_generations,
            "Janelas viáveis médias por banca": (
                result.average_viable_slots_per_board
            ),
            "Máximo de bancas sem horário comum": (
                result.boards_without_common_slot
            ),
        }
        for result in results
    ]
    pd.DataFrame(data_rows).to_excel(data_path, index=False)
    return html_path, data_path


def benchmark_results_to_rows(
    semester: str,
    results: list[BenchmarkResult],
) -> list[dict[str, object]]:
    return [
        {
            "Semestre": semester,
            "Semanas": result.weeks,
            "Restrição (%)": result.restriction,
            "Execuções concluídas": result.successful_runs,
            "Falhas": result.failed_runs,
            "Fitness médio": result.average_fitness,
            "Melhor fitness": result.best_fitness,
            "Conflitos médios": result.average_professor_conflicts,
            "Bancas paralelas excedentes": result.average_parallel_excess,
            "Gerações médias": result.average_generations,
            "Janelas viáveis médias por banca": (
                result.average_viable_slots_per_board
            ),
            "Máximo de bancas sem horário comum": (
                result.boards_without_common_slot
            ),
        }
        for result in results
    ]


def export_combined_restriction_dashboard(
    all_results: dict[str, list[BenchmarkResult]],
    runs_per_level: int,
) -> tuple[Path, Path]:
    semesters = list(all_results)
    lookup = {
        (semester, result.weeks, result.restriction): result
        for semester, results in all_results.items()
        for result in results
    }

    summary_cards = []
    for semester in semesters:
        for weeks in (1, 2):
            scenario = [
                result
                for result in all_results[semester]
                if result.weeks == weeks
            ]
            first_infeasible = next(
                (
                    result.restriction
                    for result in scenario
                    if result.successful_runs == 0
                ),
                None,
            )
            first_partial = next(
                (
                    result.restriction
                    for result in scenario
                    if result.failed_runs > 0
                ),
                None,
            )
            last_without_conflicts = max(
                (
                    result.restriction
                    for result in scenario
                    if result.best_fitness is not None
                    and abs(result.best_fitness)
                    < PROFESSOR_CONFLICT_PENALTY
                ),
                default=None,
            )
            summary_cards.append(
                f"""
                <div class="summary-card">
                  <strong>{escape(semester)} · {weeks} semana{'s' if weeks > 1 else ''}</strong>
                  <span>Melhor resultado sem conflito até:
                    <b>{last_without_conflicts}%</b>
                  </span>
                  <span>Primeiro nível inviável:
                    <b>{str(first_infeasible) + '%' if first_infeasible is not None else 'não ocorreu'}</b>
                  </span>
                  <span>Primeira falha entre repetições:
                    <b>{str(first_partial) + '%' if first_partial is not None else 'não ocorreu'}</b>
                  </span>
                </div>
                """
            )

    header_cells = "".join(
        f"<th>{escape(semester)}<br><small>{weeks} sem.</small></th>"
        for semester in semesters
        for weeks in (1, 2)
    )
    matrix_rows = []
    for restriction in BENCHMARK_PERCENTAGES:
        cells = []
        for semester in semesters:
            for weeks in (1, 2):
                result = lookup[(semester, weeks, restriction)]
                if result.average_fitness is None:
                    cells.append(
                        f"""
                        <td class="result-cell infeasible">
                          <strong>Inviável</strong>
                          <small>{result.boards_without_common_slot} banca(s) sem janela</small>
                        </td>
                        """
                    )
                    continue

                best_fitness = result.best_fitness
                if (
                    best_fitness is not None
                    and abs(best_fitness) < PROFESSOR_CONFLICT_PENALTY
                ):
                    cell_class = "clean"
                elif best_fitness is not None and best_fitness > -10_000:
                    cell_class = "warning"
                else:
                    cell_class = "critical"
                cells.append(
                    f"""
                    <td class="result-cell {cell_class}">
                      <strong>{best_fitness}</strong>
                      <small>melhor fitness</small>
                      <small>{result.successful_runs}/{runs_per_level} viáveis</small>
                    </td>
                    """
                )
        matrix_rows.append(
            f"<tr><th>{restriction}%</th>{''.join(cells)}</tr>"
        )

    legend = """
      <span><i class="dot clean"></i> melhor execução sem conflito de professor</span>
      <span><i class="dot warning"></i> abaixo do ótimo, até -9999</span>
      <span><i class="dot critical"></i> fitness igual ou inferior a -10000</span>
      <span><i class="dot infeasible"></i> banca sem horário comum</span>
    """
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Comparativo geral de restrições</title>
  <style>
    :root {{
      --bg:#f3f6fb; --card:#fff; --text:#172033; --muted:#64748b;
      --border:#dbe4f0; --clean:#dcfce7; --clean-text:#166534;
      --warning:#fef3c7; --warning-text:#92400e;
      --critical:#fee2e2; --critical-text:#b91c1c;
      --infeasible:#e2e8f0; --infeasible-text:#475569;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; background:var(--bg); color:var(--text);
      font-family:"Segoe UI",Arial,sans-serif;
    }}
    main {{ max-width:1380px; margin:auto; padding:26px 18px 42px; }}
    .hero,.matrix-card {{
      background:var(--card); border:1px solid var(--border);
      border-radius:16px; box-shadow:0 10px 30px rgba(15,23,42,.06);
    }}
    .hero {{ padding:22px; }}
    h1 {{ margin:0 0 8px; font-size:1.65rem; }}
    .subtitle {{ margin:0; color:var(--muted); }}
    .summaries {{
      display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
      gap:10px; margin-top:18px;
    }}
    .summary-card {{
      background:#f8fafc; border-radius:11px; padding:12px;
      display:grid; gap:5px;
    }}
    .summary-card span {{ color:var(--muted); font-size:.86rem; }}
    .legend {{
      display:flex; flex-wrap:wrap; gap:14px; margin-top:16px;
      color:var(--muted); font-size:.86rem;
    }}
    .legend span {{ display:flex; align-items:center; gap:6px; }}
    .dot {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
    .dot.clean,.result-cell.clean {{ background:var(--clean); color:var(--clean-text); }}
    .dot.warning,.result-cell.warning {{ background:var(--warning); color:var(--warning-text); }}
    .dot.critical,.result-cell.critical {{ background:var(--critical); color:var(--critical-text); }}
    .dot.infeasible,.result-cell.infeasible {{ background:var(--infeasible); color:var(--infeasible-text); }}
    .matrix-card {{ margin-top:18px; overflow:hidden; }}
    .table-wrap {{ overflow:auto; }}
    table {{ width:100%; border-collapse:separate; border-spacing:0; }}
    th,td {{
      padding:9px 10px; border-right:1px solid var(--border);
      border-bottom:1px solid var(--border); text-align:center;
    }}
    thead th {{ background:#eaf2f8; position:sticky; top:0; z-index:2; }}
    tbody th {{ background:#f8fafc; position:sticky; left:0; z-index:1; }}
    th small {{ color:var(--muted); font-weight:500; }}
    .result-cell {{ min-width:125px; }}
    .result-cell strong,.result-cell small {{ display:block; }}
    .result-cell small {{ margin-top:3px; font-size:.75rem; }}
    .note {{ padding:12px 16px; color:var(--muted); font-size:.85rem; }}
    @media(max-width:900px) {{
      .summaries {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Comparativo geral de restrições das bancas</h1>
      <p class="subtitle">Três semestres, uma e duas semanas, bloqueios de
      0% a 80%. Cada célula mostra o melhor fitness encontrado entre
      {runs_per_level} execução(ões).</p>
      <div class="summaries">{''.join(summary_cards)}</div>
      <div class="legend">{legend}</div>
    </section>
    <section class="matrix-card">
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Restrição</th>{header_cells}</tr>
          </thead>
          <tbody>{''.join(matrix_rows)}</tbody>
        </table>
      </div>
      <div class="note">A dashboard apresenta o melhor resultado das repetições.
      Fitness mais próximo de zero é melhor. A penalidade
      é composta por 1000 pontos por conflito de professor e 1 ponto por
      banca paralela excedente. Médias e conflitos médios permanecem
      disponíveis no arquivo Excel consolidado.</div>
    </section>
  </main>
</body>
</html>
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUTPUT_DIR / "dashboard_restricoes_todos_semestres.html"
    html_path.write_text(html, encoding="utf-8")

    data_rows = [
        row
        for semester, results in all_results.items()
        for row in benchmark_results_to_rows(semester, results)
    ]
    data_path = OUTPUT_DIR / "resultados_restricoes_todos_semestres.xlsx"
    pd.DataFrame(data_rows).to_excel(data_path, index=False)
    return html_path, data_path


def load_benchmark_results(path: Path) -> list[BenchmarkResult]:
    frame = pd.read_excel(path)
    results = []
    for _, row in frame.iterrows():
        results.append(
            BenchmarkResult(
                weeks=int(row["Semanas"]),
                restriction=int(row["Restrição (%)"]),
                successful_runs=int(row["Execuções concluídas"]),
                failed_runs=int(row["Falhas"]),
                average_fitness=(
                    None
                    if pd.isna(row["Fitness médio"])
                    else float(row["Fitness médio"])
                ),
                best_fitness=(
                    None
                    if pd.isna(row["Melhor fitness"])
                    else int(row["Melhor fitness"])
                ),
                average_professor_conflicts=(
                    None
                    if pd.isna(row["Conflitos médios"])
                    else float(row["Conflitos médios"])
                ),
                average_parallel_excess=(
                    None
                    if pd.isna(row["Bancas paralelas excedentes"])
                    else float(row["Bancas paralelas excedentes"])
                ),
                average_generations=(
                    None
                    if pd.isna(row["Gerações médias"])
                    else float(row["Gerações médias"])
                ),
                average_viable_slots_per_board=(
                    None
                    if pd.isna(row["Janelas viáveis médias por banca"])
                    else float(row["Janelas viáveis médias por banca"])
                ),
                boards_without_common_slot=int(
                    row["Máximo de bancas sem horário comum"]
                ),
            )
        )
    return results


def benchmark_cache_matches(
    path: Path,
    runs_per_level: int,
) -> bool:
    if not path.exists():
        return False
    results = load_benchmark_results(path)
    return (
        len(results) == 34
        and all(
            result.successful_runs + result.failed_runs == runs_per_level
            for result in results
        )
    )


def list_workbooks() -> list[Path]:
    return sorted(
        path
        for path in INPUT_DIR.glob("*.xlsx")
        if not path.name.startswith("~$")
    )


def resolve_workbook(selection: str | None) -> Path:
    candidates = list_workbooks()
    if not candidates:
        raise FileNotFoundError(
            f"Nenhuma planilha foi encontrada em {INPUT_DIR}."
        )
    if selection is None:
        return candidates[0]

    direct = Path(selection)
    if direct.exists():
        return direct
    inside_input = INPUT_DIR / selection
    if inside_input.exists():
        return inside_input
    matches = [
        path
        for path in candidates
        if path.name.casefold() == selection.casefold()
    ]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Planilha não encontrada: {selection}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Agenda bancas reais com professores fixos, evoluindo somente "
            "os horários."
        )
    )
    parser.add_argument(
        "-p",
        "--planilha",
        help="Nome ou caminho da planilha de população inicial.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Semente pseudoaleatória.",
    )
    parser.add_argument(
        "--dashboard-restricoes",
        action="store_true",
        help=(
            "Executa o benchmark de 0%% a 80%% de restrição para uma e "
            "duas semanas."
        ),
    )
    parser.add_argument(
        "--execucoes",
        type=int,
        default=1,
        help="Quantidade de execuções independentes por nível de restrição.",
    )
    parser.add_argument(
        "--todos-semestres",
        action="store_true",
        help="Executa e consolida o benchmark de todas as planilhas.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Quantidade de processos paralelos para o benchmark.",
    )
    args = parser.parse_args()

    if args.todos_semestres:
        if args.execucoes < 1:
            raise ValueError("--execucoes deve ser maior ou igual a 1.")
        if args.workers < 1:
            raise ValueError("--workers deve ser maior ou igual a 1.")
        all_results: dict[str, list[BenchmarkResult]] = {}
        for path in list_workbooks():
            semester = path.stem.replace("bancas_", "").replace(
                "_academicos",
                "",
            )
            cached_path = (
                OUTPUT_DIR
                / f"resultados_restricoes_{path.stem}.xlsx"
            )
            if benchmark_cache_matches(cached_path, args.execucoes):
                print(f"Reutilizando resultados: {cached_path.name}")
                all_results[semester] = load_benchmark_results(cached_path)
                continue
            print(f"\nProcessando semestre {semester}")
            results = run_restriction_benchmark(
                path,
                args.execucoes,
                args.workers,
            )
            export_restriction_dashboard(path, results, args.execucoes)
            all_results[semester] = results

        dashboard_path, data_path = export_combined_restriction_dashboard(
            all_results,
            args.execucoes,
        )
        print(f"\nDashboard geral: {dashboard_path}")
        print(f"Dados consolidados: {data_path}")
        return

    workbook_path = resolve_workbook(args.planilha)
    if args.dashboard_restricoes:
        if args.execucoes < 1:
            raise ValueError("--execucoes deve ser maior ou igual a 1.")
        if args.workers < 1:
            raise ValueError("--workers deve ser maior ou igual a 1.")
        results = run_restriction_benchmark(
            workbook_path,
            args.execucoes,
            args.workers,
        )
        dashboard_path, data_path = export_restriction_dashboard(
            workbook_path,
            results,
            args.execucoes,
        )
        print(f"\nDashboard: {dashboard_path}")
        print(f"Dados consolidados: {data_path}")
        return

    boards, professors, slots = load_problem(workbook_path)
    individual, result, generations_run, population_size, history = evolve(
        boards,
        slots,
        args.seed,
    )
    output_path = export_result(
        workbook_path,
        boards,
        professors,
        slots,
        individual,
        result,
        generations_run,
        population_size,
        history,
    )
    dashboard_path = export_dashboard(
        workbook_path,
        boards,
        professors,
        slots,
        individual,
        result,
        generations_run,
        population_size,
        history,
    )

    print(f"Planilha: {workbook_path}")
    print(f"Bancas carregadas: {len(boards)}")
    print(f"Professores carregados: {len(professors)}")
    print(f"Horários disponíveis na grade: {len(slots)}")
    print(
        f"Tamanho da população: {population_size} "
        f"({POPULATION_SIZE_FACTOR} x {len(boards)} bancas)"
    )
    print(f"Fitness final: {result.fitness}")
    print(f"Penalidade final: {result.penalty}")
    print(f"Conflitos de professor: {result.professor_conflicts}")
    print(f"Bancas paralelas excedentes: {result.parallel_board_excess}")
    print(f"Gerações executadas: {generations_run}")
    print(f"Resultado: {output_path}")
    print(f"Dashboard: {dashboard_path}")


if __name__ == "__main__":
    main()
