# After the training queue: evaluate every final run on val and, once, on test (hflip TTA reported
# separately; softmax outputs saved for src.ensemble), then aggregate and the val and test ensembles.
# Runs only on an otherwise idle machine, so the ms/image timings are not disturbed by training.
param([int]$QueuePid)
Set-Location "D:\Download\Data-20260919T021318Z-1-001 (2)\Data-20260919T021318Z-1-001\facial emotion"
$env:PYTHONUTF8 = "1"
$status = Join-Path (Get-Location) "runs\logs\queue_status.txt"

function Log($msg) {
    $line = "$(Get-Date -Format 'HH:mm') $msg`r`n"
    for ($i = 0; $i -lt 10; $i++) { try { [IO.File]::AppendAllText($status, $line); return } catch { Start-Sleep 2 } }
}
function Busy {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'src\.(train|predict|evaluate)' }).Count -gt 0
}

# wait for the training queue, then for any leftover python job
if ($QueuePid) {   # 0 / omitted: no queue to wait for (PID 0 is the System Idle process)
    $q = Get-Process -Id $QueuePid -ErrorAction SilentlyContinue
    if ($q) { $q.WaitForExit() }
}
while (Busy) { Start-Sleep 60 }

# touch the test set only for a complete, final set of runs
$failed = Get-Content $status | Select-Object -Skip (Select-String -Path $status -Pattern 'QUEUE start' | Select-Object -Last 1).LineNumber |
    Select-String 'exit=[1-9]|DRYRUN FAILED|SKIPPED'
if ($failed) { Log "EVAL on hold: a training job failed ($(@($failed)[0].Line.Trim()))"; exit 1 }
$runs = @(& python -m src.aggregate --list-final)
if ($runs.Count -ne 9) { Log "EVAL on hold: $($runs.Count) final runs, expected 9 (3 models x 3 seeds)"; exit 1 }

Log "EVAL start (final runs from src.aggregate --list-final)"
foreach ($r in $runs) {
    if (Test-Path "$r\test_metrics.json") { Log "EVAL skip $r (already evaluated)"; continue }
    cmd /c "python -u -m src.evaluate --run $r --split val --tta --no-timing >> runs\logs\evaluate.log 2>&1"
    $valCode = $LASTEXITCODE
    cmd /c "python -u -m src.evaluate --run $r --tta >> runs\logs\evaluate.log 2>&1"
    if ($LASTEXITCODE -ne 0 -or $valCode -ne 0) { Log "EVAL FAILED $r (runs\logs\evaluate.log)" } else { Log "EVAL done $r" }
}
cmd /c "python -u -m src.aggregate >> runs\logs\evaluate.log 2>&1"
Log "AGGREGATE exit=$LASTEXITCODE -> reports\results_summary.md"
cmd /c "python -u -m src.ensemble --split val >> runs\logs\evaluate.log 2>&1"
Log "ENSEMBLE val exit=$LASTEXITCODE -> reports\ensemble_val.md"
# the ensembles were fixed before any test result (all final runs + each model's seeds, +/- TTA)
cmd /c "python -u -m src.ensemble --split test >> runs\logs\evaluate.log 2>&1"
Log "ENSEMBLE test exit=$LASTEXITCODE -> reports\ensemble_test.md"
