# SGSMA 2026 Competition Day Runbook

Use this when the two live test workbooks arrive.

## Input

Save the two organizer workbooks somewhere local, for example:

```powershell
C:\Users\walla\Downloads\Test1.xlsx
C:\Users\walla\Downloads\Test2.xlsx
```

Each workbook should contain the bus sheets with a blank `label` column.

## Generate the submission folder

From the repository root:

```powershell
python scripts\prepare_competition_day_submission.py `
  --test1 C:\Users\walla\Downloads\Test1.xlsx `
  --test2 C:\Users\walla\Downloads\Test2.xlsx
```

The script writes:

```text
output\competition_day\J_Mayorga_SGSMA2026\
  J_Mayorga_Results_Test1.xlsx
  J_Mayorga_Results_Test2.xlsx
```

Upload the folder `J_Mayorga_SGSMA2026` to the Dropbox submission link.
Diagnostics are kept outside the upload folder under `output\competition_day\_diagnostics\`.

## What the script does

- Reads each bus sheet from each workbook.
- Validates that the expected 8 PMU buses are present: 2, 5, 6, 10, 19, 22, 29, and 39.
- Validates required PMU signal columns before running inference.
- Preserves the original workbook sheets, row order, timestamps, and columns.
- Fills only the `label` column with the model's predicted event labels `0` through `8`.
- Normalizes `BUS*_FREQ` to `BUS*_Freq` only in the temporary inference CSVs, matching the trained model.
- Adds or updates the `Evaluation Metrics` sheet.
- Reports hidden-label metrics as `N/A - hidden ground truth labels unavailable`, because the test labels are not provided.

## Expected runtime

The production model uses 30-second windows. For two approximately 25-minute test workbooks, expect several minutes of CPU-only inference, depending on the machine.
