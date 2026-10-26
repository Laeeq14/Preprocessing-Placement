# Week 5 — Stable bandwidth, 3 repeated seeds per BW level
$bws   = @(256, 500, 1000, 2000, 5000)
$seeds = @(1, 2, 3)

foreach ($bw in $bws) {
    foreach ($seed in $seeds) {
        Write-Host ">>> Week 5: bw=$bw seed=$seed" -ForegroundColor Cyan
        py scripts/run_experiment.py --mode all --bw $bw --n 50 --seed $seed
        if ($LASTEXITCODE -ne 0) { Write-Host "ERROR" -ForegroundColor Red; exit 1 }
    }
}

Write-Host ">>> Week 6: High -> Low" -ForegroundColor Cyan
py scripts/run_experiment.py --mode adaptive --bw 5000 --n 25
py scripts/run_experiment.py --mode adaptive --bw 256  --n 25

Write-Host ">>> Week 6: Low -> High" -ForegroundColor Cyan
py scripts/run_experiment.py --mode adaptive --bw 256  --n 25
py scripts/run_experiment.py --mode adaptive --bw 5000 --n 25

Write-Host "Done! Regenerating plots..." -ForegroundColor Green
py scripts/analyze_results.py --input results/results.csv --plots
