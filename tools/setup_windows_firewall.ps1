# setup_windows_firewall.ps1 — open Windows Defender Firewall for the
# perception service stack on the GPU host. Inbound rules only.
#
# Per SoftwareDocs/GPU_Service_Host_Architecture_Plan.md §4.9:
#   Inbound on Windows:
#     UDP 5588  — Gaze service (gaze_runner.py request-reply)
#     UDP 5589  — VLM service (vlm_service.py request-reply)
#     TCP 5590  — VLM overlay JPEG push (panel/Linux dials in)
#   No inbound rule needed for TCP 5591 — Windows is the *client* of the
#   Linux frame relay.
#
# Run from an *elevated* PowerShell:
#     powershell -ExecutionPolicy Bypass -File tools/setup_windows_firewall.ps1
#
# Idempotent: existing rules with the same DisplayName are skipped.

#Requires -RunAsAdministrator

$rules = @(
    @{ Name = "BCI_GazeService_UDP5588"; Proto = "UDP"; Port = 5588; Desc = "Gaze service request-reply" },
    @{ Name = "BCI_VLMService_UDP5589";  Proto = "UDP"; Port = 5589; Desc = "VLM service request-reply" },
    @{ Name = "BCI_VLMOverlay_TCP5590";  Proto = "TCP"; Port = 5590; Desc = "VLM overlay JPEG push" }
)

foreach ($r in $rules) {
    $existing = Get-NetFirewallRule -DisplayName $r.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "[skip] $($r.Name) already present"
        continue
    }
    New-NetFirewallRule `
        -DisplayName $r.Name `
        -Description $r.Desc `
        -Direction Inbound `
        -Action Allow `
        -Protocol $r.Proto `
        -LocalPort $r.Port `
        -Profile Any | Out-Null
    Write-Host "[added] $($r.Name) ($($r.Proto)/$($r.Port))"
}
Write-Host "Done."
