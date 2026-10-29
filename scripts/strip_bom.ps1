$path = "c:\Users\ASUS\AI_LLMs\Long_text\MA-GRPO\data\chapters.jsonl"
$text = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($path, $text, $utf8NoBom)

$bytes = [System.IO.File]::ReadAllBytes($path)[0..2]
Write-Host "New first 3 bytes: $bytes"

