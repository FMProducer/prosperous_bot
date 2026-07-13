# Промпт для Jules (jules.google) — Инвентарь репозитория

> Использование: вставить в jules.google для анализа https://github.com/FMProducer/prosperous_bot
> Результат: REPOSITORY_INVENTORY.md в корне репозитория

---

## Task: Complete Repository Inventory & System Identification

**Repository:** https://github.com/FMProducer/prosperous_bot

### Goal

Produce a comprehensive inventory document (`REPOSITORY_INVENTORY.md`) committed to the repo root. The document must answer three questions:
1. How many distinct systems, bots, robots, and tools exist in this repository?
2. What is each one's purpose, architecture, and status (active/archived/legacy)?
3. How do they relate to each other — shared code, dependencies, execution hierarchy?

### Methodology

#### Phase 1 — Full Structure Discovery

```bash
find . -maxdepth 3 -type d | grep -v __pycache__ | grep -v .git | grep -v node_modules | grep -v venv
find . -name "*.py" | grep -v __pycache__ | grep -v .git | grep -v venv | wc -l
find . -name "*.json" | grep -v __pycache__ | grep -v .git | grep -v venv | wc -l
find . -name "*.md" | grep -v .git | wc -l
find . -name "*.bat" -o -name "*.sh" -o -name "*.ps1" | wc -l
```

Read ALL top-level files.

#### Phase 2 — Per-Directory Deep Scan

For EVERY directory at depth 1 (excluding .git, __pycache__, venv):
1. List all files
2. Read first 40-60 lines of every .py file
3. Read every .json config
4. Read every .md doc
5. Read every .bat/.sh/.ps1 script
6. Classify: active system, library, test suite, utility, archive, external dependency, documentation, data/state

#### Phase 3 — System Identification

For each system: Name, Purpose, Entry points, Dependencies, External services, State files, Config files, Status (active/experimental/archived/legacy).

#### Phase 4 — Bot & Process Inventory

Search for: pm2 references, subprocess calls, if __name__ blocks, asyncio.run, Docker files, .bat scripts.

For each bot: Name, Type (Real/Paper/Service), Ticker, Module, Status.

#### Phase 5 — Configuration Analysis

Read EVERY config.json. Extract: trading parameters, safety mechanisms, bot lists, portfolio allocation.

#### Phase 6 — Trading Strategy Identification

Read full files for any trading logic. Document: strategy, entry/exit rules, position sizing, risk management.

#### Phase 7 — Test Coverage Map

List ALL test files with what they test and approximate test count.

#### Phase 8 — Evolution Timeline

Read CHANGELOG.md, ROADMAP files. Construct timeline.

### Output Format

```markdown
# Repository Inventory — FMProducer/prosperous_bot

## Executive Summary
## Repository at a Glance (table)
## Directory Map
## Systems Inventory (per system: location, purpose, components, status)
## Bot Process Registry (table)
## Configuration Reference
## Test Coverage
## Evolution Timeline
## Legacy & Archives
## Cross-System Dependencies
## Appendix: Complete File Inventory
```

### Rules

1. DO NOT read or expose API keys or secrets
2. Count precisely — no "approximately"
3. Read large files fully for core logic
4. Distinguish active from legacy
5. Document the WHY, not just WHAT
6. If unsure, say so explicitly
