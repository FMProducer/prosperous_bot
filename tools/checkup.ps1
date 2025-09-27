
Write-Host "`n🔍 Проверка среды VS Code + Python..." -ForegroundColor Cyan

# 1. Проверка активного окружения
if ($env:VIRTUAL_ENV) {
    Write-Host "✅ Активировано виртуальное окружение: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "❌ Виртуальное окружение неактивно!" -ForegroundColor Red
}

# 2. Проверка доступности pytest
Write-Host "`n🔎 Проверка наличия pytest..."
try {
    pytest --version
    Write-Host "✅ Pytest доступен." -ForegroundColor Green
} catch {
    Write-Host "❌ Pytest не установлен или не найден в PATH." -ForegroundColor Red
}

# 3. Проверка подключения к интернету
Write-Host "`n🌐 Проверка интернет-соединения (ping 8.8.8.8)..."
if (Test-Connection -Count 1 -ComputerName 8.8.8.8 -Quiet) {
    Write-Host "✅ Интернет доступен." -ForegroundColor Green
} else {
    Write-Host "❌ Нет доступа к интернету!" -ForegroundColor Red
}

# 4. Проверка разрешения DNS
Write-Host "`n🌐 Проверка DNS (ping ya.ru)..."
if (Test-Connection -Count 1 -ComputerName ya.ru -Quiet) {
    Write-Host "✅ DNS работает." -ForegroundColor Green
} else {
    Write-Host "❌ Ошибка разрешения DNS!" -ForegroundColor Red
}

# 5. Проверка разрешения запуска задач в VS Code (файл tasks.json)
Write-Host "`n📄 Проверка tasks.json..."
$taskFile = Join-Path $PWD ".vscode\tasks.json"
if (Test-Path $taskFile) {
    Write-Host "✅ Найден tasks.json: $taskFile" -ForegroundColor Green
} else {
    Write-Host "❌ Файл tasks.json не найден!" -ForegroundColor Red
}

Write-Host "`n✅ Диагностика завершена." -ForegroundColor Cyan
