Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap 1000,600
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.Clear([System.Drawing.Color]::WhiteSmoke)

$titleFont = New-Object System.Drawing.Font('Segoe UI',24,[System.Drawing.FontStyle]::Bold)
$bodyFont = New-Object System.Drawing.Font('Segoe UI',14,[System.Drawing.FontStyle]::Regular)

$graphics.DrawString('Healthcare Patient Survival Prediction (ML)', $titleFont, [System.Drawing.Brushes]::MidnightBlue, 40, 30)
$graphics.DrawRectangle([System.Drawing.Pens]::SlateGray, 40, 100, 300, 160)
$graphics.DrawString('Manual Inputs', $bodyFont, [System.Drawing.Brushes]::Black, 60, 110)
$graphics.DrawRectangle([System.Drawing.Pens]::SlateGray, 380, 100, 560, 160)
$graphics.DrawString('Risk Gauge + Prediction', $bodyFont, [System.Drawing.Brushes]::Black, 400, 110)
$graphics.DrawString('Upload patient CSV data', $bodyFont, [System.Drawing.Brushes]::DimGray, 40, 320)
$graphics.DrawString('Medical explanation & charts', $bodyFont, [System.Drawing.Brushes]::DimGray, 40, 360)
$graphics.DrawString('Mock layout generated automatically for README screenshots.', $bodyFont, [System.Drawing.Brushes]::DarkGreen, 40, 520)

$outputPath = Join-Path $PSScriptRoot '..\notebooks\evaluation\app_mock.png'
$graphics.Dispose()
$bmp.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()

