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

function Get-DotEnvValue {
    param(
        [string]$Name,
        [string]$DefaultValue
    )

    $pattern = '^\s*' + [regex]::Escape($Name) + '\s*=\s*(.*?)\s*$'
    $value = $null
    foreach ($line in Get-Content (Join-Path $PSScriptRoot ".env")) {
        if ($line -match $pattern) {
            $value = $Matches[1].Trim()
        }
    }

    if (
        $value -and
        $value.Length -ge 2 -and
        (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        )
    ) {
        $value = $value.Substring(1, $value.Length - 2)
    }

    if ([string]::IsNullOrWhiteSpace($value)) {
        return $DefaultValue
    }
    return $value
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
    Invoke-Compose -ComposeArguments @("wait", "ollama-model-init")

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
        (Get-DotEnvValue -Name "PRIMARY_MODEL" -DefaultValue "llama3.2:1b"),
        (Get-DotEnvValue -Name "FALLBACK_MODEL" -DefaultValue "llama3.2:1b"),
        (Get-DotEnvValue -Name "EMBEDDING_MODEL" -DefaultValue "nomic-embed-text")
    )
    $installedModels = @(
        $modelList |
            Select-Object -Skip 1 |
            ForEach-Object { ($_ -join " ").Trim() -split '\s+' | Select-Object -First 1 }
    )
    foreach ($model in ($requiredModels | Select-Object -Unique)) {
        if ([string]::IsNullOrWhiteSpace($model)) {
            throw "Model names in .env must not be empty."
        }
        $latestModel = if ($model.Contains(":")) { $model } else { "$($model):latest" }
        if (($installedModels -contains $model) -or ($installedModels -contains $latestModel)) {
            Write-Host "Model $model is already installed."
        } else {
            Write-Host "Downloading required model $model (first launch may take a while)..."
            Invoke-Compose -ComposeArguments @("exec", "-T", "ollama", "ollama", "pull", $model)
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
