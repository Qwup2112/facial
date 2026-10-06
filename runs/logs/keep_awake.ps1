# Keep Windows from idle-sleeping (AC sleep timer: 10 min) while the training queue and the eval
# step run. Uses a power request (ES_SYSTEM_REQUIRED) that is released when this script exits;
# power settings are not changed and the screen may still turn off.
param([string]$Pids)   # comma-separated, waited for in order, e.g. "11212,35316"
$status = Join-Path $PSScriptRoot "queue_status.txt"

function Log($msg) {
    $line = "$(Get-Date -Format 'HH:mm') $msg`r`n"
    for ($i = 0; $i -lt 10; $i++) { try { [IO.File]::AppendAllText($status, $line); return } catch { Start-Sleep 2 } }
}

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648          # 0x80000000
$ES_CONTINUOUS_SYSTEM = [uint32]2147483649   # 0x80000000 | ES_SYSTEM_REQUIRED (0x1)

if ([Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS_SYSTEM) -eq 0) { Log "KEEP-AWAKE FAILED"; exit 1 }
Log "KEEP-AWAKE on: no idle sleep until pids $Pids exit"
foreach ($p in $Pids.Split(',')) {
    $proc = Get-Process -Id ([int]$p) -ErrorAction SilentlyContinue
    if ($proc) { $proc.WaitForExit() }
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Log "KEEP-AWAKE off"
