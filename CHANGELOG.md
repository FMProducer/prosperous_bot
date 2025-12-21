# ChangeLog

## v12-RC1 (Release Candidate)

### Added
- **Quantization-Aware Training (QAT):** Внедрена поддержка QAT в `model.py` для повышения производительности инференса на CPU.
- **Ensemble Agent Logic:** Добавлена новая логика для ансамбля, включая параметры `conflict_cooldown_bars` и `disable_cross_close` для более гибкого управления моделями.
- **New Verification Tests:** Добавлен `test_identity_v2.py` для проверки ключевых аспектов новой архитектуры, включая QAT и API агента.

### Changed
- **Cross-Platform Paths:** Абсолютные пути в файлах конфигурации заменены на динамические с использованием `pathlib`, что улучшило кросс-платформенность.
- **CPU Inference Optimization:** Оптимизировано использование потоков CPU для ускорения процесса инференса.
- **Dequantization Fix:** В `model.py` исправлена ошибка, при которой компоненты `value` и `advantage` не проходили деквантование.

### Removed
- **Legacy Tests:** Удален устаревший тест `test_train.py`, несовместимый с новой архитектурой.

## [Unreleased]
