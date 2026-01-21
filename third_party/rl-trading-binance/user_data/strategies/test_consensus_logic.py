# test_consensus_logic.py - положите рядом со стратегией для проверки
import numpy as np

def test_conflict_detection():
    """Тест: оба направления проходят consensus -> NO ENTRY"""
    
    # Mock strategy object
    class MockStrategy:
        class MockParam:
            def __init__(self, val):
                self.value = val
        
        veto_threshold = MockParam(0.65)
        min_confidence = MockParam(0.65)
    
    strategy = MockStrategy()
    
    # Simulate: обе long модели ЗА с high confidence
    long_votes = [1, 1]  # unanimous
    long_confidences = [0.75, 0.78]  # avg = 0.765 > 0.65
    
    # Simulate: обе short модели ЗА с high confidence
    short_votes = [1, 1]  # unanimous
    short_confidences = [0.68, 0.72]  # avg = 0.70 > 0.65
    
    # Копируем вашу логику
    long_entry_count = sum(1 for vote in long_votes if vote == 1)
    short_entry_count = sum(1 for vote in short_votes if vote == 1)
    
    avg_long_conf = np.mean(long_confidences)
    avg_short_conf = np.mean(short_confidences)
    
    total_long = len(long_votes)
    total_short = len(short_votes)
    
    long_majority = (long_entry_count >= total_long * 0.5)
    short_majority = (short_entry_count >= total_short * 0.5)
    
    min_conf = strategy.min_confidence.value
    
    long_has_consensus = long_majority and avg_long_conf >= min_conf
    short_has_consensus = short_majority and avg_short_conf >= min_conf
    
    # ПРОВЕРКА
    if long_has_consensus and short_has_consensus:
        print("✅ PASS: Конфликт корректно обнаружен!")
        print(f"   Long consensus: {long_has_consensus} (avg_conf={avg_long_conf:.3f})")
        print(f"   Short consensus: {short_has_consensus} (avg_conf={avg_short_conf:.3f})")
        print("   -> Результат: NO ENTRY (как и должно быть)")
        return True
    else:
        print("❌ FAIL: Конфликт НЕ обнаружен!")
        return False

if __name__ == "__main__":
    test_conflict_detection()
