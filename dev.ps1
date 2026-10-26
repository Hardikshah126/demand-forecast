<#
 Developer commands for Windows PowerShell (mirror of the Makefile).
 Usage:  ./dev.ps1 <target>      e.g.  ./dev.ps1 setup ; ./dev.ps1 all-data ; ./dev.ps1 serve
#>
param([Parameter(Mandatory = $true, Position = 0)][string]$Target)
$ErrorActionPreference = "Stop"
$py  = ".\.venv\Scripts\python.exe"
$dfs = ".\.venv\Scripts\dfs.exe"

function Run([scriptblock]$b) { & $b; if ($LASTEXITCODE -ne 0) { throw "command failed ($LASTEXITCODE)" } }

switch ($Target) {
    "setup"            { Run { python -m venv .venv }; Run { & $py -m pip install -U pip }; Run { & $py -m pip install -e ".[dev]" } }
    "db-up"            { Run { docker compose up -d --wait } }
    "db-init"          { Run { & $dfs db-init } }
    "download"         { Run { & $dfs download } }
    "ingest"           { Run { & $dfs ingest } }
    "quality"          { Run { & $dfs quality } }
    "train"            { Run { & $dfs train } }
    "serve"            { Run { & $dfs serve --reload } }
    "all-data"         { foreach ($t in "db-up","db-init","download","ingest","quality","train") { & $PSCommandPath $t } }
    "test"             { Run { & $py -m pytest } }
    "lint"             { Run { & $py -m ruff check dfs tests }; Push-Location frontend; try { Run { npm run lint } } finally { Pop-Location } }
    "frontend-install" { Push-Location frontend; try { Run { npm ci } } finally { Pop-Location } }
    "frontend-dev"     { Push-Location frontend; try { Run { npm run dev } } finally { Pop-Location } }
    "frontend-build"   { Push-Location frontend; try { Run { npm run build } } finally { Pop-Location } }
    default            { throw "Unknown target '$Target'. See the Makefile for the list of targets." }
}
