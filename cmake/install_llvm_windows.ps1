# PowerShell script to install LLVM/Clang 18 on Windows
Write-Host 'Installing LLVM/Clang 18 on Windows...'

# Check if Chocolatey is available
if (Get-Command choco -ErrorAction SilentlyContinue) {
    Write-Host 'Using Chocolatey to install LLVM 18...'
    choco install llvm --version=18.1.8 -y
    if ($LASTEXITCODE -eq 0) {
        Write-Host 'LLVM 18 installed successfully via Chocolatey.'
        exit 0
    } else {
        Write-Host 'Chocolatey installation failed, trying direct download...'
    }
}

# Direct download and installation
Write-Host 'Downloading LLVM 18 from GitHub releases...'
$url = 'https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/LLVM-18.1.8-win64.exe'
$installer = Join-Path $env:TEMP 'LLVM-18.1.8-win64.exe'

try {
    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    Write-Host 'Download completed. Installing LLVM 18...'
    
    # Run installer silently
    Start-Process -FilePath $installer -ArgumentList '/S' -Wait -PassThru
    
    # Add LLVM to PATH if not already there
    $llvmPath = 'C:\Program Files\LLVM\bin'
    if (Test-Path $llvmPath) {
        $currentPath = [Environment]::GetEnvironmentVariable('PATH', 'Machine')
        if ($currentPath -notlike "*$llvmPath*") {
            Write-Host 'Adding LLVM to system PATH...'
            [Environment]::SetEnvironmentVariable('PATH', "$currentPath;$llvmPath", 'Machine')
        }
        Write-Host 'LLVM 18 installation completed successfully.'
        Remove-Item $installer -Force
        exit 0
    } else {
        Write-Error 'LLVM installation failed - LLVM directory not found.'
        exit 1
    }
} catch {
    Write-Error "Installation failed: $($_.Exception.Message)"
    if (Test-Path $installer) { Remove-Item $installer -Force }
    exit 1
}