# E03 Jacobian audit

Analytic ANDES sparse blocks were converted without densifying for storage and saved as NPZ:

- `fx` shape=(220, 220), nnz=290, density=0.00599174, Frobenius norm=1192.25
- `fy` shape=(220, 479), nnz=170, density=0.00161321, Frobenius norm=75.6461
- `gx` shape=(479, 220), nnz=280, density=0.00265705, Frobenius norm=141.173
- `gy` shape=(479, 479), nnz=1566, density=0.00682528, Frobenius norm=2950.33

`G_z` numerical rank at 1e-10 relative threshold: 479/479; smallest singular value=0.00209934; condition=562870.

The selected measurement columns are independently compared with central finite differences in `e03_jacobian_checks.csv`; the largest recorded VI error is 4.09968e-09.
