# Script to manually launch a specific list of tickers in PAPER mode via PM2
# Bypasses the standard Supervisor selection for experimental tracking.

$tickers = @(
    "SIRENUSDT", "FILUSDT", "STRKUSDT", "ONDOUSDT", "GALAUSDT", 
    "TIAUSDT", "NOTUSDT", "TSTUSDT", "NILUSDT", "MOVRUSDT", 
    "BIOUSDT", "DUSDT", "DOGSUSDT", "TONUSDT", "PENDLEUSDT", 
    "JTOUSDT", "ICPUSDT", "ARBUSDT", "1000LUNCUSDT", "DASHUSDT"
)

$currentDir = Get-Location
$pythonExe = "python" # Uses the python from your active .venv

Write-Host "🚀 Launching manual experimental swarm (20 bots)..." -ForegroundColor Cyan

foreach ($t in $tickers) {
    $short = $t.Replace("USDT", "").ToLower()
    $name = "bot-$short"
    
    Write-Host "Starting $name for $t..."
    
    # We use cmd /c to ensure PM2 receives arguments correctly on Windows/PowerShell
    $cmd = "pm2 start main.py --name $name --cwd ""$currentDir"" --interpreter ""$pythonExe"" -- --config config.json --ticker $t --paper"
    cmd /c $cmd
}

Write-Host "✅ All bots requested. Check status with 'pm2 list'" -ForegroundColor Green
