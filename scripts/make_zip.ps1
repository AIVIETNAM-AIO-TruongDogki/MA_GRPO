$ProgressPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$sourceDir = "c:\Users\ASUS\AI_LLMs\Long_text\MA-GRPO"
$zipPath = "c:\Users\ASUS\AI_LLMs\Long_text\MA-GRPO.zip"
$tempStaging = "c:\Users\ASUS\AI_LLMs\Long_text\staging_zip"

if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
if (Test-Path $tempStaging) { Remove-Item $tempStaging -Recurse -Force }

New-Item -ItemType Directory -Path $tempStaging -Force | Out-Null

# Copy only needed folders
Copy-Item (Join-Path $sourceDir "configs") -Destination $tempStaging -Recurse -Force
Copy-Item (Join-Path $sourceDir "data") -Destination $tempStaging -Recurse -Force
Copy-Item (Join-Path $sourceDir "scripts") -Destination $tempStaging -Recurse -Force
Copy-Item (Join-Path $sourceDir "src") -Destination $tempStaging -Recurse -Force
Copy-Item (Join-Path $sourceDir "requirements.txt") -Destination $tempStaging -Force
Copy-Item (Join-Path $sourceDir "README.md") -Destination $tempStaging -Force

# Remove any __pycache__ inside staging
Get-ChildItem -Path $tempStaging -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force

[System.IO.Compression.ZipFile]::CreateFromDirectory($tempStaging, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)
Remove-Item $tempStaging -Recurse -Force

$item = Get-Item $zipPath
Write-Host "Archive created successfully at: $zipPath"
Write-Host "Archive size: $([math]::Round($item.Length / 1KB, 2)) KB"