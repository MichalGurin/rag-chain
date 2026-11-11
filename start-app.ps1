$ErrorActionPreference = "Stop"

$composeFile = Join-Path $PSScriptRoot "app\docker-compose.yml"
$composeArgs = @("-f", $composeFile)
$frontendUrl = "http://localhost:5173"
$apiHealthUrl = "http://localhost:8000/health"

function Invoke-Compose {
    param([string[]]$ComposeArguments)

    & docker compose @composeArgs @ComposeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed: $($ComposeArguments -join ' ')"
    }
}

function Wait-ForEndpoint {
    param(
        [string]$Url,
        [string]$Name,
        [int]$TimeoutSeconds = 300
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                Write-Host "$Name is ready."
                return
            }
        } catch {
            Start-Sleep -Seconds 3
        }
    }

    throw "$Name did not become ready at $Url within $TimeoutSeconds seconds."
}

try {
    Set-Location $PSScriptRoot

    & docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop is not running. Start Docker Desktop and run this launcher again."
    }

    if (-not (Test-Path (Join-Path $PSScriptRoot ".env"))) {
        Copy-Item (Join-Path $PSScriptRoot ".env.example") (Join-Path $PSScriptRoot ".env")
        Write-Host "Created .env from .env.example."
    }

    Write-Host "Building and starting the application..."
    Invoke-Compose -ComposeArguments @("up", "--build", "--detach")

    Write-Host "Waiting for Ollama..."
    $deadline = (Get-Date).AddMinutes(5)
    $modelList = $null
    while ((Get-Date) -lt $deadline) {
        $modelList = & docker compose @composeArgs exec -T ollama ollama list 2>$null
        if ($LASTEXITCODE -eq 0) {
            break
        }
        Start-Sleep -Seconds 5
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Ollama did not become ready. Check Docker Desktop and the Ollama container logs."
    }

    $requiredModels = @(
        @{ Name = "llama3.2:1b"; Pattern = '(?m)^llama3\.2:1b\s' },
        @{ Name = "nomic-embed-text"; Pattern = '(?m)^nomic-embed-text(?::latest)?\s' }
    )
    foreach ($model in $requiredModels) {
        if (($modelList -join "`n") -notmatch $model.Pattern) {
            Write-Host "Downloading required model $($model.Name) (first launch may take a while)..."
            Invoke-Compose -ComposeArguments @("exec", "-T", "ollama", "ollama", "pull", $model.Name)
        } else {
            Write-Host "Model $($model.Name) is already installed."
        }
    }

    Wait-ForEndpoint -Url $apiHealthUrl -Name "API"
    Wait-ForEndpoint -Url $frontendUrl -Name "Frontend"

    Write-Host "Opening $frontendUrl"
    Start-Process $frontendUrl
    Write-Host "Application is running. Stop it with: docker compose -f app/docker-compose.yml down"
} catch {
    Write-Host "Startup failed: $_" -ForegroundColor Red
    exit 1
}
