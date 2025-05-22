
# Список исполняемых файлов для блокировки (UDP)
$appsToBlock = @(
    "C:\Windows\System32\DiagTrackRunner.exe",
    "C:\Windows\System32\Speech_OneCore\Common\SpeechRuntime.exe",
    "C:\Program Files\Windows Media Player\wmplayer.exe",
    "C:\Windows\System32\wmpnetwk.exe",
    "C:\Windows\System32\msra.exe",
    "C:\Windows\System32\p2phost.exe",
    "C:\Windows\System32\svchost.exe"
)

# Добавление GameBar.exe, если найден в WindowsApps
$gamebarPaths = Get-ChildItem "C:\Program Files\WindowsApps" -Directory -Filter "Microsoft.XboxGamingOverlay*" -ErrorAction SilentlyContinue
foreach ($dir in $gamebarPaths) {
    $gamebarExe = Join-Path $dir.FullName "GameBar.exe"
    if (Test-Path $gamebarExe) {
        $appsToBlock += $gamebarExe
    }
}

foreach ($app in $appsToBlock) {
    if (Test-Path $app) {
        New-NetFirewallRule `
            -DisplayName ("BLOCK OUT UDP - " + ($app.Split('\')[-1])) `
            -Direction Outbound `
            -Program $app `
            -Protocol UDP `
            -Action Block `
            -Profile Any `
            -Enabled True `
            -Description ("Block outbound UDP for " + $app)
    } else {
        Write-Warning ("File not found: " + $app + " - skipped.")
    }
}
