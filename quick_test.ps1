# Simple API Test
Write-Host "Testing API at http://localhost:5000/api/convert" -ForegroundColor Cyan

$body = @{
    text = "I need help with this task"
    mode = "polite"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "http://localhost:5000/api/convert" -Method Post -ContentType "application/json" -Body $body
    
    Write-Host "`n✅ SUCCESS!" -ForegroundColor Green
    Write-Host "Original: $($response.original_text)" -ForegroundColor White
    Write-Host "Converted: $($response.converted_text)" -ForegroundColor Cyan
    Write-Host "Alternatives: $($response.alternatives.Count)" -ForegroundColor Yellow
    
    if ($response.alternatives.Count -gt 0) {
        Write-Host "`nAlternative Suggestions:" -ForegroundColor Yellow
        foreach ($alt in $response.alternatives) {
            Write-Host "  $($alt.mode): $($alt.text)" -ForegroundColor Gray
        }
    }
    
    Write-Host "`nProcessing Details:" -ForegroundColor Magenta
    Write-Host "  Model: $($response.dataset_info.model)" -ForegroundColor Gray
    Write-Host "  Confidence: $($response.dataset_info.confidence)" -ForegroundColor Gray
    Write-Host "  Time: $($response.dataset_info.processing_time)" -ForegroundColor Gray
    
} catch {
    Write-Host "`n❌ ERROR: $_" -ForegroundColor Red
    Write-Host "Make sure the server is running on localhost:5000" -ForegroundColor Yellow
}
