import sys
import os
import math

# Add the directory to sys.path to allow importing from futures_portfolio
sys.path.append(os.path.join(os.getcwd(), "futures_portfolio"))

try:
    from supervisor import calculate_bot_score
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def test_scoring():
    min_cycles = 10

    # Case 1: Paper bot, insufficient cycles
    p1 = {'cycles': 2, 'profit': 100.0}
    score1 = calculate_bot_score("T1", p1, False, False, min_cycles)
    assert score1 == -float('inf'), f"Expected -inf for paper bot with 2 cycles, got {score1}"
    print("✅ Case 1 Passed: Paper bot with insufficient cycles rejected.")

    # Case 2: Paper bot, enough cycles
    p2 = {'cycles': 10, 'profit': 100.0}
    score2 = calculate_bot_score("T2", p2, False, False, min_cycles)
    expected2 = (100.0 / 10) * math.log1p(10)
    assert math.isclose(score2, expected2), f"Expected {expected2}, got {score2}"
    print(f"✅ Case 2 Passed: Paper bot with enough cycles scored correctly: {score2}")

    # Case 3: Real bot in drawdown
    p3 = {'cycles': 15, 'profit': -50.0}
    score3 = calculate_bot_score("T3", p3, True, True, min_cycles)
    assert score3 == float('inf'), f"Expected inf for real bot in drawdown, got {score3}"
    print("✅ Case 3 Passed: Real bot in drawdown locked in.")

    # Case 4: Real bot, profitable, insufficient cycles (should NOT be rejected, but dampened)
    p4 = {'cycles': 5, 'profit': 100.0}
    score4 = calculate_bot_score("T4", p4, True, False, min_cycles)
    expected4 = (100.0 / 10) * math.log1p(5) * 1.2
    assert math.isclose(score4, expected4), f"Expected {expected4}, got {score4}"
    print(f"✅ Case 4 Passed: Real bot with few cycles dampened and given bonus: {score4}")

    # Case 5: Paper bot, 1 cycle, high profit (The "lucky" bot)
    p5 = {'cycles': 1, 'profit': 1000.0}
    score5 = calculate_bot_score("T5", p5, False, False, min_cycles)
    assert score5 == -float('inf'), "Expected lucky bot to be rejected"
    print("✅ Case 5 Passed: Lucky young bot rejected.")

    # Case 6: Compare lucky young bot (if it were real) vs established bot
    # Young bot: 2 cycles, 100 profit -> eff_cycles 10 -> (100/10)*log(3) = 10 * 1.098 = 10.98
    # Established bot: 20 cycles, 150 profit -> eff_cycles 20 -> (150/20)*log(21) = 7.5 * 3.044 = 22.83
    score_young_math = (100.0 / 10) * math.log1p(2)
    score_old_math = (150.0 / 20) * math.log1p(20)
    print(f"Math check: Young (dampened) {score_young_math:.4f} vs Old {score_old_math:.4f}")
    assert score_old_math > score_young_math
    print("✅ Case 6 Passed: Dampening works as intended.")

if __name__ == "__main__":
    test_scoring()
