<#
  run.ps1 - Windows helper for the Legal Case Similarity project.

    ./run.ps1 build       # build all ML artifacts + init the DB
    ./run.ps1 evaluate    # benchmark against the expert pairs
    ./run.ps1 test        # pytest + vitest
    ./run.ps1 demo        # build, then start the API and the UI in two windows
    ./run.ps1 backend     # just the API  (:8000)
    ./run.ps1 frontend    # just the UI   (:5173)
#>
param([Parameter(Position = 0)][string]$cmd = "help")

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

switch ($cmd) {
  "build"    { & $py scripts/build_all.py }
  "evaluate" { & $py scripts/evaluate.py }
  "backend"  { $env:WARM_ON_STARTUP = "1"; & $py -m uvicorn app.main:app --reload --app-dir backend }
  "frontend" { Push-Location frontend; npm run dev; Pop-Location }
  "test" {
    & $py -m pytest
    Push-Location frontend; npm run test; Pop-Location
  }
  "demo" {
    & $py scripts/build_all.py
    & $py scripts/evaluate.py
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$env:WARM_ON_STARTUP='1'; $py -m uvicorn app.main:app --reload --app-dir backend"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"
    Write-Host "API -> http://localhost:8000/docs    UI -> http://localhost:5173"
    Write-Host "Follow DEMO.md."
  }
  default {
    Write-Host "usage: ./run.ps1 {build|evaluate|test|demo|backend|frontend}"
  }
}
