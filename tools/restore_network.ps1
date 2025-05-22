
# 1. Удаление всех пользовательских правил брандмауэра, связанных с DHCP и svchost.exe
Get-NetFirewallRule | Where-Object {
    ($_.DisplayName -match "DHCP") -or
    ($_.DisplayName -match "svchost") -or
    ($_.DisplayName -match "svchost.exe")
} | Remove-NetFirewallRule

# 2. Сброс настроек сети
ipconfig /release
ipconfig /flushdns
ipconfig /renew

# 3. Сброс TCP/IP и Winsock
netsh int ip reset
netsh winsock reset

# 4. Перезапуск службы DHCP-клиента
Restart-Service dhcp -ErrorAction SilentlyContinue

# 5. Принудительное включение DHCP для Ethernet
Set-NetIPInterface -InterfaceAlias "Ethernet" -Dhcp Enabled -ErrorAction SilentlyContinue
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ResetServerAddresses -ErrorAction SilentlyContinue

Write-Host "`nСеть сброшена. Перезагрузите ПК при необходимости." -ForegroundColor Cyan
