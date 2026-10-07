# Run after buffalo_l.zip is downloaded to models/
$zip = "C:\projetos\facego\models\buffalo_l.zip"
$out = "C:\projetos\facego\models\buffalo_l_extracted"

Write-Host "Extracting $zip..."
Expand-Archive -Path $zip -DestinationPath $out -Force

# Find the ONNX files
$files = Get-ChildItem $out -Recurse -Filter "*.onnx"
Write-Host "Found models:"
$files | ForEach-Object { Write-Host "  $($_.Name) - $([math]::Round($_.Length/1MB,1)) MB" }

# Copy to models/
foreach ($f in $files) {
    $dest = switch -Wildcard ($f.Name) {
        "det_*"    { "C:\projetos\facego\models\scrfd.onnx" }
        "w600k*"   { "C:\projetos\facego\models\arcface.onnx" }
        "buffalo*" { $null }
        default    { $null }
    }
    if ($dest) {
        Copy-Item $f.FullName $dest -Force
        Write-Host "Copied $($f.Name) -> $dest"
    }
}

# Also copy onnxruntime.dll next to binary if not there
$dll = "C:\projetos\facego\onnxruntime.dll"
if (Test-Path $dll) {
    Write-Host "onnxruntime.dll already in place"
} else {
    Write-Host "WARNING: onnxruntime.dll missing from C:\projetos\facego\"
}

Write-Host "Done. Models ready."
