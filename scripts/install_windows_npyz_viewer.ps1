$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Resolve-Path (Join-Path $ScriptDir "..")

$AppName = "npyz-viewer"
$ProgId = "NPYZViewer.File"
$OldProgId = "NPZNPYViewer.File"
$ExeDir = Join-Path $env:LOCALAPPDATA "npyz-viewer"
$ExePath = Join-Path $ExeDir "npyz-viewer.exe"
$OldExeDir = Join-Path $env:LOCALAPPDATA "npz_npy_viewer"

Write-Host "[1/4] Building release binary..."
& cargo build --release --bin npyz-viewer --manifest-path (Join-Path $ProjectDir "Cargo.toml")

$SrcExe = Join-Path $ProjectDir "target\\release\\npyz-viewer.exe"
if (!(Test-Path $SrcExe)) {
    throw "Release executable not found: $SrcExe"
}

Write-Host "[2/4] Replacing old install and installing binary to $ExePath ..."
if (Test-Path $OldExeDir) {
    Remove-Item -Recurse -Force $OldExeDir
}
New-Item -ItemType Directory -Force -Path $ExeDir | Out-Null
Copy-Item -Force $SrcExe $ExePath

Write-Host "[3/4] Registering file association in HKCU..."
if (Test-Path "HKCU:\Software\Classes\$OldProgId") {
    Remove-Item -Recurse -Force "HKCU:\Software\Classes\$OldProgId"
}
New-Item -Path "HKCU:\Software\Classes\$ProgId" -Force | Out-Null
Set-ItemProperty -Path "HKCU:\Software\Classes\$ProgId" -Name "(default)" -Value $AppName

New-Item -Path "HKCU:\Software\Classes\$ProgId\DefaultIcon" -Force | Out-Null
Set-ItemProperty -Path "HKCU:\Software\Classes\$ProgId\DefaultIcon" -Name "(default)" -Value ('"{0}",0' -f $ExePath)

New-Item -Path "HKCU:\Software\Classes\$ProgId\shell\open\command" -Force | Out-Null
Set-ItemProperty -Path "HKCU:\Software\Classes\$ProgId\shell\open\command" -Name "(default)" -Value ('"{0}" "%1"' -f $ExePath)

foreach ($ext in @('.npy', '.npz')) {
    New-Item -Path "HKCU:\Software\Classes\$ext" -Force | Out-Null
    Set-ItemProperty -Path "HKCU:\Software\Classes\$ext" -Name "(default)" -Value $ProgId
}

Write-Host "[4/4] Notifying shell..."
$signature = @"
using System;
using System.Runtime.InteropServices;
public static class NativeMethods {
    [DllImport("shell32.dll")]
    public static extern void SHChangeNotify(int wEventId, uint uFlags, IntPtr dwItem1, IntPtr dwItem2);
}
"@
Add-Type -TypeDefinition $signature -ErrorAction SilentlyContinue | Out-Null
[NativeMethods]::SHChangeNotify(0x08000000, 0, [IntPtr]::Zero, [IntPtr]::Zero)

Write-Host "Done."
Write-Host "If Explorer does not update immediately, sign out/sign in once."
