"""Native Windows dialogs the hub opens for the page (the page runs on this PC: server.LOCAL_CLIENTS): the folder
picker of "Import folder" (the browser's own folder upload dialog would carry Edge's "Upload from mobile" button and
read every file through the page; the hub reads the folder itself). A PowerShell child shows the dialog (Windows
PowerShell is on every Windows PC); with SPELLCAST_NO_EXPLORER=1 (tests) nothing opens and None comes back.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

from . import exports

# The modern file dialog used as a folder picker (Windows Forms' FolderBrowserDialog is the old tree on .NET
# Framework): the artist steps into the folder and presses Open; the "file" named is never checked, its folder is
# the answer. The owner form, topmost and invisible, brings the dialog in front of the browser.
_SCRIPT = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true; $owner.ShowInTaskbar = $false; $owner.Opacity = 0
$owner.Size = New-Object System.Drawing.Size(1, 1); $owner.StartPosition = 'CenterScreen'
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Title = $env:SPELLCAST_DIALOG_TITLE
$d.FileName = 'Choose this folder'
$d.Filter = 'Folder|*.none'
$d.CheckFileExists = $false; $d.CheckPathExists = $true; $d.ValidateNames = $false
$d.DereferenceLinks = $true; $d.Multiselect = $false
$owner.Show(); $owner.Activate()
$ok = $d.ShowDialog($owner)
$owner.Close()
if ($ok -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write([System.IO.Path]::GetDirectoryName($d.FileName)) }
"""
TIMEOUT = 900.0         # the artist may take a while


def pick_folder(title: str = "Choose the folder with the props' images, then press Open") -> Optional[str]:
    """The folder the artist picks (its path), or None (cancelled, no dialog possible, SPELLCAST_NO_EXPLORER=1)."""
    if not exports.explorer_allowed() or os.name != "nt":
        return None
    env = {**os.environ, "SPELLCAST_DIALOG_TITLE": title}
    try:
        run = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-STA", "-ExecutionPolicy", "Bypass",
                              "-Command", _SCRIPT], capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env=env, timeout=TIMEOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    chosen = (run.stdout or "").strip()
    return chosen if run.returncode == 0 and chosen and os.path.isdir(chosen) else None
