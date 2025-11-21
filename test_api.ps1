# Test API Script for Tone Converter

Write-Host "Testing Tone Converter API..." -ForegroundColor Cyan
Write-Host ""

# Test 1: Polite Mode
Write-Host "Test 1: Polite Mode" -ForegroundColor Yellow
$body1 = @{
    text = "I need this done right now"
    mode = "polite"
} | ConvertTo-Json

try {
    $response1 = Invoke-RestMethod -Uri "http://localhost:5000/api/convert" -Method Post -ContentType "application/json" -Body $body1
    Write-Host "Original: $($response1.original_text)" -ForegroundColor White
    Write-Host "Converted: $($response1.converted_text)" -ForegroundColor Green
    Write-Host "Alternatives: $($response1.alternatives.Count)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

# Test 2: Formal Mode
Write-Host "Test 2: Formal Mode" -ForegroundColor Yellow
$body2 = @{
    text = "Hey, I can't make it to the meeting today"
    mode = "formal"
} | ConvertTo-Json

try {
    $response2 = Invoke-RestMethod -Uri "http://localhost:5000/api/convert" -Method Post -ContentType "application/json" -Body $body2
    Write-Host "Original: $($response2.original_text)" -ForegroundColor White
    Write-Host "Converted: $($response2.converted_text)" -ForegroundColor Green
    Write-Host "Alternatives: $($response2.alternatives.Count)" -ForegroundColor Cyan
    Write-Host ""
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

# Test 3: Professional Mode
Write-Host "Test 3: Professional Mode" -ForegroundColor Yellow
$body3 = @{
    text = "We need to fix this problem quickly"
    mode = "professional"
} | ConvertTo-Json

try {
    $response3 = Invoke-RestMethod -Uri "http://localhost:5000/api/convert" -Method Post -ContentType "application/json" -Body $body3
    Write-Host "Original: $($response3.original_text)" -ForegroundColor White
    Write-Host "Converted: $($response3.converted_text)" -ForegroundColor Green
    Write-Host "Confidence: $($response3.dataset_info.confidence)" -ForegroundColor Magenta
    Write-Host "Processing Time: $($response3.dataset_info.processing_time)" -ForegroundColor Magenta
    Write-Host "Alternatives: $($response3.alternatives.Count)" -ForegroundColor Cyan
    
    if ($response3.alternatives.Count -gt 0) {
        Write-Host "`nAlternative Suggestions:" -ForegroundColor Yellow
        foreach ($alt in $response3.alternatives) {
            Write-Host "  - $($alt.mode): $($alt.text)" -ForegroundColor Gray
        }
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host "`nAll tests completed!" -ForegroundColor Cyan
