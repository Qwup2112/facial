# Live training dashboard (read-only). Closing this window does NOT stop training.
# Reads runs/<model>_s<seed>_*/log.csv (one file per run), so it does not depend on who launched a run.
param([switch]$Once)
Set-Location "D:\Download\Data-20260919T021318Z-1-001 (2)\Data-20260919T021318Z-1-001\facial emotion"
$Host.UI.RawUI.WindowTitle = "FER training - live progress"
try { $Host.UI.RawUI.WindowSize = New-Object Management.Automation.Host.Size(100, 34) } catch {}

$runs = @("cnn_scratch_s42", "resnet18_s42", "mobilenet_v2_s42",
          "cnn_scratch_s123", "resnet18_s123", "mobilenet_v2_s123",
          "cnn_scratch_s2024", "resnet18_s2024", "mobilenet_v2_s2024")
$total = @{ cnn_scratch = 40; resnet18 = 23; mobilenet_v2 = 23 }   # epochs per run (configs)
$inv = [Globalization.CultureInfo]::InvariantCulture

function Render {
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("FER training - 3 models x 3 seeds        " + (Get-Date -Format "HH:mm:ss") + "   (refresh every 15 s)")
    $lines.Add("Closing this window only stops watching; training keeps running.")
    $lines.Add("")
    $lines.Add(("{0,-20} {1,-12} {2,-7} {3,-28} {4,-8} {5}" -f "run", "status", "epoch", "progress", "val F1", "best val F1"))
    $lines.Add(("-" * 92))
    $remain = 0
    foreach ($r in $runs) {
        $model = $r -replace '_s\d+$', ''
        # latest run directory of this model+seed (by log.csv write time)
        $dir = Get-ChildItem "runs" -Directory -Filter "${r}_*" -ErrorAction SilentlyContinue |
            Where-Object { Test-Path "$($_.FullName)\log.csv" } |
            Sort-Object { (Get-Item "$($_.FullName)\log.csv").LastWriteTime } | Select-Object -Last 1
        if (-not $dir) { $lines.Add(("{0,-20} {1,-12}" -f $r, "not started")); continue }
        $rows = @(Import-Csv "$($dir.FullName)\log.csv")
        $ep = 0; $sec = 0.0; $last = "-"; $best = -1.0
        foreach ($x in $rows) {
            $ep = [int]$x.epoch; $sec = [double]::Parse($x.epoch_time, $inv)
            $f = [double]::Parse($x.val_macro_f1, $inv); $last = $f.ToString("0.0000", $inv)
            if ($f -gt $best) { $best = $f }
        }
        $bestS = if ($best -ge 0) { $best.ToString("0.0000", $inv) } else { "-" }
        $age = ((Get-Date) - (Get-Item "$($dir.FullName)\log.csv").LastWriteTime).TotalSeconds
        if (Test-Path "$($dir.FullName)\curves.png") {
            $st = if ($ep -lt $total[$model]) { "done(early)" } else { "done" }
        } elseif ($age -lt [math]::Max(3 * $sec, 300)) {
            $st = "running"; $remain += ($total[$model] - $ep) * $sec
        } else { $st = "stopped" }
        $n = [math]::Floor(25 * $ep / $total[$model]); if ($st -like "done*") { $n = 25 }
        $bar = "[" + ("#" * $n) + ("." * (25 - $n)) + "]"
        $lines.Add(("{0,-20} {1,-12} {2,-7} {3,-28} {4,-8} {5}" -f $r, $st, "$ep/$($total[$model])", $bar, $last, $bestS))
    }
    $lines.Add("")
    if ($remain -gt 0) {
        $lines.Add(("Current run(s): about {0} min left (at most; early stopping can end sooner)" -f [math]::Ceiling($remain / 60)))
    }
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'src\.(train|predict)' })
    if ($procs.Count -eq 0) { $lines.Add("Training processes: none running") }
    foreach ($p in $procs) {
        $c = $p.CommandLine -replace '^.*-m (src\.\S+)\s+', '$1 '
        $lines.Add("Running now: pid $($p.ProcessId)  $c")
    }
    try {
        $g = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits) -split ',\s*'
        $lines.Add(("GPU: {0}% busy | VRAM {1}/{2} MiB | {3} C" -f $g[0], $g[1], $g[2], $g[3]))
    } catch {}
    return $lines
}

if ($Once) { Render; return }
while ($true) {
    $screen = Render
    Clear-Host
    $screen | ForEach-Object { Write-Host $_ }
    Start-Sleep -Seconds 15
}
