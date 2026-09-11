param(
  [int]$Port = 8090
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$binary = Join-Path $repo 'integrations\Freebuff2API\Freebuff2API.exe'
$credentials = Join-Path $env:USERPROFILE '.config\manicode\credentials.json'

if (-not (Test-Path -LiteralPath $binary)) {
  throw "找不到 Freebuff2API.exe，請先執行: go build -o integrations\Freebuff2API\Freebuff2API.exe ."
}
if (-not (Test-Path -LiteralPath $credentials)) {
  throw "找不到 Freebuff CLI 登入檔。請先在終端機執行 freebuff login，完成登入後再執行此腳本。"
}

# Read credentials locally and pass only the token count to the console.
# Tokens are never printed or written into the TokenRouter config.
$credentialObject = Get-Content -LiteralPath $credentials -Raw | ConvertFrom-Json
$tokens = @()
foreach ($property in $credentialObject.PSObject.Properties) {
  if ($null -ne $property.Value -and $property.Value.authToken) {
    $tokens += [string]$property.Value.authToken
  }
}
if ($credentialObject.authToken) {
  $tokens += [string]$credentialObject.authToken
}
$tokens = @($tokens | Where-Object { $_ } | Select-Object -Unique)
if ($tokens.Count -eq 0) {
  throw "登入檔內沒有 authToken。請重新執行 freebuff login。"
}

$env:LISTEN_ADDR = "127.0.0.1:$Port"
$env:UPSTREAM_BASE_URL = 'https://www.codebuff.com'
$env:AUTH_TOKENS = ($tokens -join ',')
$env:ROTATION_INTERVAL = '6h'
$env:REQUEST_TIMEOUT = '15m'

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
  $env:AUTH_TOKENS = ''
  Write-Output "Freebuff2API 已在 http://127.0.0.1:$Port/v1 監聽。"
  exit 0
}

$logOut = Join-Path $repo 'freebuff2api-runtime.log'
$logErr = Join-Path $repo 'freebuff2api-runtime.err'
Start-Process -FilePath $binary -WorkingDirectory (Split-Path -Parent $binary) -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr
$env:AUTH_TOKENS = ''
Write-Output "Freebuff2API 已啟動: http://127.0.0.1:$Port/v1（已載入 $($tokens.Count) 個本機登入 token）"
