param(
    [Parameter(Mandatory=$true)]
    [string]$DatasetRoot
)

$ErrorActionPreference = "Stop"

$folders = @(
    "tfrecord_training",
    "tfrecord_validation",
    "tfrecord_testing",
    "train",
    "train\lidar",
    "train\annos",
    "val",
    "val\lidar",
    "val\annos",
    "test",
    "test\lidar",
    "test\annos"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $DatasetRoot $folder) | Out-Null
}

Write-Host "Waymo layout prepared at: $DatasetRoot"

