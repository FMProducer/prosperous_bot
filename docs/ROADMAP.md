# Roadmap (aktualizirovannaya)
**C\u0435l P0:** +3 000 000 USDT <= 7 mes pri **Max DD < 20%**  
**KPI:** Sharpe >= 1.5; Profit Factor >= 1.3

## Etapy (ISO-8601 UTC)
| Shag | Zadacha | Moduli | Kriterii priyomki | Dedlayn (UTC) |
|---|---|---|---|---|
| 1 | Dovesti backtester (pozicii/PNL/force_close); sinkhronizirovat' edinyy konfig | `rebalance_backtester.py`, `unified_config*.json` | Testy PASS; PnL != 0 pri dvizhenii; sovpadenie sim/real >= 95% | 2025-09-15 |
| 2 | Optuna po porogu rebalansa, min notional, delta-neitral'nym dolyam, intervaly | `rebalance_optimizer*.py` | Valid: Sharpe >= 1.5; PF >= 1.3; komissii <= 0.2%/den' | 2025-09-30 |
| 3 | HybridStrategy (Rule+ML); rasshirit' fichi (ob'yomy, OI, funding) | `strategy.py`, `ml_model.py`, `signal_generator.py` | WinRate >= 55%; Max DD < 15% v testakh; +0.1 k F1 vs rule-only | 2025-10-30 |
| 4 | Avtomatizirovat' pereobuchenie i otchety po ML | `signal_bot.py`, `ml_model.py` | Ezhenedel'nyy retrain; stabil'nyy Sharpe >= 1.5 na 30-dn okne | 2025-11-30 |
| 5 | Risk-menedzhment: CB urovnya DD, tral-portfelya, limity plech | `rebalance_engine.py` (risk layer) | Max DD < 20% v stress-testakh; Recovery Factor > 1.5 | 2025-12-31 |
| 6 | Monitoring/alerty (Grafana/Telegram), ezhednevnye daydzhesty KPI | `exchange_gate.py`, `telegram_bot` | Alerty < 1 min; ezhednevnyy otchet; "tikhikh" sboev net | 2026-01-15 |
| 7 | Masshtabirovanie kapitala, smart-ordera | vsya strategiya | KPI stabil'ny; itog >= +3M USDT | 2026-04-01 |

## Definition of Ready
- Tsel' i vliyanie na KPI/riski opisany.
- Dannye/period zafiksirovany; metriki/dopuski opredeleny.

## Definition of Done
- Zelyonyy CI (`pytest --cov`, smouk-bektest).
- Otchety v `reports/` prilozeny; dokumentatsiya obnovlena.
- Riski i plan otkata zadokumentirovany.

## Riski i smyagchenie
- Rost Max DD -> Safe-Mode, snizhenie plech/chastoty rebalansa.
- Spayki funding -> fil'try/limity; pauza novykh vhodov.
- Korr-sdvig -> peresmot pary long/short i ATR-porogov.

## Primechaniya
- Daty v UTC; opovescheniya — America/Phoenix.
- Vse izmeneniya idut cherez PR i chek-listy (`CHECKLISTS.md`).
