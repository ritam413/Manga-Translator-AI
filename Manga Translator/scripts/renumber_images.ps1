param (
    [Parameter(Mandatory=$false)]
    [string]$Directory = ".",
    
    [Parameter(Mandatory=$false)]
    [string[]]$Extensions = @(".jpg", ".jpeg", ".png", ".webp")
)

# Normalize extensions to start with a dot
$normalizedExtensions = $Extensions | ForEach-Object { if ($_ -notlike ".*") { ".$_" } else { $_ } }

# Resolve root directory to an absolute path
$rootPath = (Resolve-Path -LiteralPath $Directory).ProviderPath

# Determine python executable path relative to this script
$projectRoot = (Resolve-Path -LiteralPath "$PSScriptRoot\..").ProviderPath
$pythonCmd = "python"
$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
}

# Helper function to process a single directory
function Renumber-Directory {
    param (
        [string]$Path,
        [string[]]$ExtensionsList,
        [string]$PythonExe
    )
    
    Push-Location -LiteralPath $Path
    try {
        # Get all files matching the extensions
        $files = Get-ChildItem -File | Where-Object { $ExtensionsList -contains $_.Extension }

        # Extract the numerical part of the filename and sort them
        $sortedFiles = $files | Where-Object { $_.BaseName -match '\d+' } | Sort-Object {
            # Find the last sequence of digits in the base name
            if ($_.BaseName -match '(\d+)(?!.*\d)') {
                [int]$matches[1]
            } else {
                0
            }
        }

        if ($sortedFiles.Count -eq 0) {
            return
        }

        Write-Host "Processing: $Path"

        # Step 1: Convert all files to temporary names in PNG format
        $tempFiles = @()
        $tempPrefix = "temp_renumber_$(Get-Random)_"
        $counter = 1
        
        foreach ($file in $sortedFiles) {
            $tempName = "$tempPrefix$counter.png"
            $tempPath = Join-Path $Path $tempName
            
            # Format paths for Python (replace backslashes with forward slashes to avoid escaping issues)
            $srcPathEscaped = $file.FullName.Replace('\', '/')
            $destPathEscaped = $tempPath.Replace('\', '/')
            
            $pythonCode = "from PIL import Image; Image.open('$srcPathEscaped').save('$destPathEscaped', 'PNG')"
            & $PythonExe -c $pythonCode
            
            if (Test-Path $tempPath) {
                # Remove original file after successful conversion
                Remove-Item -LiteralPath $file.FullName -Force
                $tempFiles += Get-Item -LiteralPath $tempPath
            } else {
                Write-Error "Failed to convert file $($file.Name) to PNG format."
            }
            $counter++
        }

        # Step 2: Rename temp files to the final sequential numbers (1.png to n.png)
        $counter = 1
        foreach ($tempFile in $tempFiles) {
            $newName = "$counter.png"
            Rename-Item -LiteralPath $tempFile.FullName -NewName $newName
            Write-Host "  Converted & Renamed $($tempFile.Name.Replace($tempPrefix, '')) -> $newName"
            $counter++
        }

        Write-Host "Successfully converted and renumbered $($tempFiles.Count) files in: $Path`n"
    }
    catch {
        Write-Error "Failed to process directory $($Path): $_"
    }
    finally {
        Pop-Location
    }
}

# Collect all directories to process (root + recursive subdirectories)
$targetDirs = [System.Collections.Generic.List[string]]::new()
$targetDirs.Add($rootPath)

$subDirs = Get-ChildItem -LiteralPath $rootPath -Directory -Recurse
if ($subDirs) {
    foreach ($subDir in $subDirs) {
        $targetDirs.Add($subDir.FullName)
    }
}

# Process each directory
$processedCount = 0
foreach ($dir in $targetDirs) {
    $hasFiles = Get-ChildItem -LiteralPath $dir -File | Where-Object { $normalizedExtensions -contains $_.Extension }
    if ($hasFiles) {
        Renumber-Directory -Path $dir -ExtensionsList $normalizedExtensions -PythonExe $pythonCmd
        $processedCount++
    }
}

if ($processedCount -eq 0) {
    Write-Host "No matching files found in any directories."
}
