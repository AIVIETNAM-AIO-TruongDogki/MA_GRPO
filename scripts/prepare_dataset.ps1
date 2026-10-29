$inputFile = "c:\Users\ASUS\AI_LLMs\Long_text\Dataset\tell_me_a_story_data\tell-me-a-story-test.jsonl"
$outputDir = "c:\Users\ASUS\AI_LLMs\Long_text\MA-GRPO\data"
$outputFile = Join-Path $outputDir "chapters.jsonl"

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$lines = Get-Content $inputFile -Encoding UTF8
$records = [System.Collections.Generic.List[string]]::new()

foreach ($line in $lines) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $item = $line | ConvertFrom-Json
    $premise = $item.inputs
    $cleanPremise = $premise -replace ',', ''
    $targetWords = 1400

    if ($cleanPremise -match 'between\s+(\d+)\s*(?:and|to|–|-)\s*(\d+)\s*words') {
        $targetWords = [int](([int]$matches[1] + [int]$matches[2]) / 2)
    } elseif ($cleanPremise -match '(\d+)\s*(?:–|-)\s*(\d+)\s*words') {
        $targetWords = [int](([int]$matches[1] + [int]$matches[2]) / 2)
    } elseif ($cleanPremise -match '(?:around|about|at least|over|make the story)\s+(\d+)\s*words') {
        $targetWords = [int]$matches[1]
    } elseif ($cleanPremise -match '(\d{3,5})\s*words') {
        $targetWords = [int]$matches[1]
    }

    # Ensure targetWords is in a realistic story range (at least 1000 words)
    if ($targetWords -lt 900) {
        $targetWords = 1200
    }

    # Dynamic scene word budgets that add up to targetWords
    $s1 = [int]($targetWords * 0.25)
    $s2 = [int]($targetWords * 0.45)
    $s3 = $targetWords - $s1 - $s2

    $prompt = @"
You are an elite literary novelist. Write the complete narrative prose for this chapter.
Total Target Length: Approximately $targetWords words.

[CONTEXT SUMMARY]:
$premise

[CHAPTER GOAL]:
Introduce the protagonists, world setting, and inciting tension.

[SCENE BREAKDOWN & WORD BUDGETS]:
- Scene 1: Opening exposition and world details (~$s1 words)
- Scene 2: Inciting tension and primary dialogue (~$s2 words)
- Scene 3: Climax of the chapter goal (~$s3 words)

CRITICAL WRITING INSTRUCTIONS:
1. Test-Time Reasoning (Chain-of-Thought):
   You MUST begin your response with a <think>...</think> block containing your step-by-step narrative plan:
   - Scene-by-scene pacing: Plan word distribution to strictly match each scene's target words (~$s1, ~$s2, ~$s3 words).
   - Narrative transition: Hook directly into the scene continuation.
   - Tone & Sensory palette: Outline sensory imagery and avoid literal phrasing from the context summary.
2. Chapter Prose:
   Immediately after </think>, output ONLY the final narrative prose.
   - Seamlessly flow between scenes according to the budgets.
   - DO NOT include scene labels, section titles, author notes, or meta-commentary outside <think>.
Begin with <think> now:
"@

    $contextSummary = if ($premise.Length -gt 300) { $premise.Substring(0, 300) } else { $premise }
    $rec = [PSCustomObject]@{
        prompt = $prompt
        target_words = $targetWords
        context_summary = $contextSummary
        prev_chapter_tail = ""
    }
    $records.Add(($rec | ConvertTo-Json -Compress))
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($outputFile, $records, $utf8NoBom)
Write-Host "Successfully generated $($records.Count) clean training records at $outputFile"
