# register_task.ps1 — Registra una tarea DIARIA de Windows que corre run_scheduled.py
# (pronostico operativo + verificacion). Asi la bitacora madura sola con el tiempo.
#
# Uso (en PowerShell, dentro de habs_forecast/):
#   powershell -ExecutionPolicy Bypass -File register_task.ps1
# Para quitarla:
#   schtasks /delete /tn "HABs_Forecast_Diario" /f
#
# Ajusta $Python y $Hora si hace falta. No requiere admin (tarea a nivel de usuario).

$Python = "C:\Python313\python.exe"
$Script = Join-Path $PSScriptRoot "run_scheduled.py"
$Hora   = "06:00"
$Nombre = "HABs_Forecast_Diario"

# Corre el script en su propia carpeta (para que encuentre config.py y los artefactos)
$Accion = "cmd /c cd /d `"$PSScriptRoot`" && `"$Python`" `"$Script`""

schtasks /create /tn $Nombre /tr $Accion /sc DAILY /st $Hora /f
Write-Host "Tarea '$Nombre' registrada: corre $Script todos los dias a las $Hora."
Write-Host "Ver estado:  schtasks /query /tn $Nombre"
Write-Host "Quitar:      schtasks /delete /tn $Nombre /f"
