import json, time, sys, os
sys.path.insert(0, r'C:\Python\Prosperous_Bot\futures_portfolio')

# Use the results from our successful run
results = [
    {"net_change": -2.62, "max_spurt": 13.33, "trend": 1.69, "cycles": 154, "symbol": "PORTALUSDT", "funding": -0.5946},
    {"net_change": 4.81, "max_spurt": 13.04, "trend": 4.26, "cycles": 112, "symbol": "STGUSDT", "funding": -0.1293},
    {"net_change": -2.13, "max_spurt": 16.39, "trend": 2.10, "cycles": 101, "symbol": "NFPUSDT", "funding": -0.1028},
    {"net_change": -3.50, "max_spurt": 15.30, "trend": 3.58, "cycles": 97, "symbol": "HEIUSDT", "funding": 0.0050},
    {"net_change": 13.25, "max_spurt": 9.08, "trend": 18.24, "cycles": 72, "symbol": "WLDUSDT", "funding": -0.0051},
    {"net_change": 3.36, "max_spurt": 7.83, "trend": 5.40, "cycles": 62, "symbol": "FETUSDT", "funding": -0.0022},
    {"net_change": -3.55, "max_spurt": 9.73, "trend": 6.48, "cycles": 54, "symbol": "MEMEUSDT", "funding": -0.0604},
    {"net_change": -2.61, "max_spurt": 7.28, "trend": 5.74, "cycles": 45, "symbol": "IDUSDT", "funding": -0.2048},
    {"net_change": -0.86, "max_spurt": 5.31, "trend": 2.02, "cycles": 42, "symbol": "XLMUSDT", "funding": -0.0246},
    {"net_change": 0.56, "max_spurt": 3.74, "trend": 1.41, "cycles": 39, "symbol": "INJUSDT", "funding": -0.0375},
    {"net_change": -0.56, "max_spurt": 3.36, "trend": 1.48, "cycles": 37, "symbol": "VTHOUSDT", "funding": -0.8398},
    {"net_change": 0.11, "max_spurt": 4.78, "trend": 0.29, "cycles": 36, "symbol": "DYDXUSDT", "funding": 0.0100},
    {"net_change": 1.64, "max_spurt": 3.93, "trend": 4.74, "cycles": 34, "symbol": "EPICUSDT", "funding": 0.0050},
    {"net_change": 2.27, "max_spurt": 3.99, "trend": 6.86, "cycles": 33, "symbol": "ALGOUSDT", "funding": 0.0084},
    {"net_change": 2.22, "max_spurt": 4.92, "trend": 7.18, "cycles": 30, "symbol": "IOUSDT", "funding": -0.0880},
    {"net_change": 2.47, "max_spurt": 4.37, "trend": 8.62, "cycles": 28, "symbol": "RENDERUSDT", "funding": -0.0022},
    {"net_change": -1.06, "max_spurt": 1.92, "trend": 4.04, "cycles": 26, "symbol": "GRASSUSDT", "funding": 0.0050},
    {"net_change": -1.87, "max_spurt": 3.36, "trend": 7.44, "cycles": 25, "symbol": "JTOUSDT", "funding": 0.0050},
    {"net_change": -1.04, "max_spurt": 2.41, "trend": 4.09, "cycles": 25, "symbol": "VVVUSDT", "funding": -0.0037},
    {"net_change": -1.80, "max_spurt": 3.50, "trend": 7.30, "cycles": 24, "symbol": "NEARUSDT", "funding": 0.0100},
    {"net_change": 1.67, "max_spurt": 2.88, "trend": 7.22, "cycles": 23, "symbol": "ZECUSDT", "funding": -0.0025},
    {"net_change": -1.83, "max_spurt": 2.97, "trend": 8.30, "cycles": 22, "symbol": "HBARUSDT", "funding": 0.0044},
    {"net_change": 1.32, "max_spurt": 3.59, "trend": 6.05, "cycles": 21, "symbol": "SEIUSDT", "funding": 0.0039},
    {"net_change": 1.71, "max_spurt": 2.91, "trend": 7.83, "cycles": 21, "symbol": "VIRTUALUSDT", "funding": -0.0043},
]

cache = {"timestamp": time.time(), "results": results}
with open("scan_cache.json", "w") as f:
    json.dump(cache, f, indent=2)
print("Cache saved OK")
