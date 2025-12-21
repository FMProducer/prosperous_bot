# ChangeLog

## [Unreleased]

### Added
- Создан новый набор тестов `tests/test_identity_v2.py` для проверки идентичности вычислений PnL и вознаграждений, а также для проверки соответствия размерностей тензоров и переименования атрибутов.

### Changed
- В `model.py` исправлена ошибка, из-за которой `value` и `advantage` не деквантовались при `return_components=True`.
- В `configs/alpha_seed_404_v12.py` абсолютные пути заменены на относительные с использованием `pathlib` для улучшения переносимости.
- В `validate_ensemble_prod_q.py` доработана логика `EnsembleAgent` для корректной обработки `conflict_cooldown_bars` и `disable_cross_close`.
