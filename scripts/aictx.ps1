param([string]$Base = "origin/main")
python tools/context_pack.py --base $Base
Write-Host "Context pack created under .\context\"
