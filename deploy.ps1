[CmdletBinding()]
param(
    [ValidateSet("up", "down", "restart", "logs", "status")]
    [string]$Action = "up",
    [switch]$Seed
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ComposeFile = Join-Path $ProjectRoot "docker-compose.yml"
$EnvironmentFile = Join-Path $ProjectRoot ".env.docker"
$EnvironmentTemplate = Join-Path $ProjectRoot ".env.docker.example"

function New-HexSecret {
    $bytes = New-Object byte[] 32
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $generator.GetBytes($bytes)
    }
    finally {
        $generator.Dispose()
    }
    return -join ($bytes | ForEach-Object { $_.ToString("x2") })
}

function Get-EnvironmentValue {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Test-Path -LiteralPath $EnvironmentFile)) {
        return ""
    }
    $content = [IO.File]::ReadAllText($EnvironmentFile)
    $match = [regex]::Match($content, "(?m)^$([regex]::Escape($Name))=(.*)$")
    if (-not $match.Success) {
        return ""
    }
    return $match.Groups[1].Value.Trim().Trim('"').Trim("'")
}

function Set-EnvironmentValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value
    )

    if ($Value -match "[`r`n]") {
        throw "Environment value for $Name cannot contain a newline."
    }
    $content = [IO.File]::ReadAllText($EnvironmentFile)
    $pattern = "(?m)^$([regex]::Escape($Name))=.*$"
    $replacement = "$Name=$Value"
    $regex = [regex]::new($pattern)
    if ($regex.IsMatch($content)) {
        $content = $regex.Replace(
            $content,
            [Text.RegularExpressions.MatchEvaluator] { param($match) $replacement },
            1
        )
    }
    else {
        $content = $content.TrimEnd("`r", "`n") + "`n$replacement`n"
    }
    [IO.File]::WriteAllText($EnvironmentFile, $content, [Text.UTF8Encoding]::new($false))
}

function Initialize-EnvironmentFile {
    if (-not (Test-Path -LiteralPath $EnvironmentFile)) {
        Copy-Item -LiteralPath $EnvironmentTemplate -Destination $EnvironmentFile
        Write-Host "Created .env.docker from the committed template."
    }

    foreach ($name in @(
        "MYSQL_APP_PASSWORD",
        "MYSQL_ROOT_PASSWORD",
        "PII_ENCRYPTION_KEY",
        "LANGFUSE_HASH_SALT"
    )) {
        if ((Get-EnvironmentValue $name) -eq "GENERATE_ON_FIRST_RUN") {
            Set-EnvironmentValue -Name $name -Value (New-HexSecret)
        }
    }

    if ($Action -in @("up", "restart")) {
        $apiKey = Get-EnvironmentValue "QWEN_API_KEY"
        if (-not $apiKey -and $env:QWEN_API_KEY) {
            Set-EnvironmentValue -Name "QWEN_API_KEY" -Value $env:QWEN_API_KEY
            $apiKey = $env:QWEN_API_KEY
        }
        if (-not $apiKey) {
            $secureKey = Read-Host "Enter QWEN_API_KEY (input is hidden)" -AsSecureString
            $credential = [PSCredential]::new("qwen", $secureKey)
            $apiKey = $credential.GetNetworkCredential().Password
            if ($apiKey) {
                Set-EnvironmentValue -Name "QWEN_API_KEY" -Value $apiKey
            }
        }
        if (-not $apiKey) {
            throw "QWEN_API_KEY is required. Set it in .env.docker or in the current environment."
        }
    }
}

function Assert-DockerReady {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker is not installed or is not available in PATH."
    }
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Engine is not running. Start Docker Desktop or the Docker service and retry."
    }
    & docker compose version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose v2 is required."
    }
}

function Invoke-Compose {
    param(
        [Parameter(Mandatory = $true, Position = 0, ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & docker compose --env-file $EnvironmentFile --file $ComposeFile @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed: $($Arguments -join ' ')"
    }
}

Initialize-EnvironmentFile
Assert-DockerReady

switch ($Action) {
    "up" {
        Invoke-Compose @("config", "--quiet")
        Invoke-Compose @("up", "--detach", "--build", "--remove-orphans", "--wait", "--wait-timeout", "240")
        if ($Seed) {
            Invoke-Compose @("exec", "--no-TTY", "backend", "python", "scripts/seed_demo_business_data.py", "--apply")
        }
        $port = Get-EnvironmentValue "HTTP_PORT"
        if (-not $port) { $port = "80" }
        $url = if ($port -eq "80") { "http://localhost" } else { "http://localhost:$port" }
        Write-Host "Smart CS Agent is ready: $url"
    }
    "restart" {
        Invoke-Compose @("up", "--detach", "--build", "--remove-orphans", "--wait", "--wait-timeout", "240")
        Write-Host "Smart CS Agent has been rebuilt and restarted."
    }
    "down" {
        Invoke-Compose @("down", "--remove-orphans")
        Write-Host "Containers stopped. Persistent volumes were kept."
    }
    "logs" {
        Invoke-Compose @("logs", "--follow", "--tail", "200")
    }
    "status" {
        Invoke-Compose @("ps")
    }
}
