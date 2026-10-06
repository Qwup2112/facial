# Let one training process finish after its queue was stopped: wait for it, then write the
# labeled predictions (test, val) like train_queue.ps1 does, and log the pause.
param([int]$TrainPid, [string]$Log, [string]$Name)
Set-Location "D:\Download\Data-20260919T021318Z-1-001 (2)\Data-20260919T021318Z-1-001\facial emotion"
$env:PYTHONUTF8 = "1"
$status = Join-Path (Get-Location) "runs\logs\queue_status.txt"

function Log($msg) {
    $line = "$(Get-Date -Format 'HH:mm') $msg`r`n"
    for ($i = 0; $i -lt 10; $i++) { try { [IO.File]::AppendAllText($status, $line); return } catch { Start-Sleep 2 } }
}

$p = Get-Process -Id $TrainPid -ErrorAction SilentlyContinue
if ($p) { $p.WaitForExit() }
$hit = Select-String -Path $Log -Pattern '^run dir: (.+)$' | Select-Object -First 1
$run = if ($hit) { $hit.Matches[0].Groups[1].Value.Trim() } else { "" }
if (-not $run -or -not (Test-Path "$run\curves.png")) { Log "END   $Name did not finish (see $Log); PAUSED"; exit 1 }
Log "END   $Name finished ($run)"
foreach ($sp in "test", "val") { cmd /c "python -u -m src.predict --run $run --split $sp >> $Log 2>&1" }
Log "PREDICT done $Name ($run)"
Log "PAUSED by the user after the first run; the other 8 runs start when the user says continue"
