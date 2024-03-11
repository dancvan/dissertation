Two scripts in this directory are designed to inform a reasonable estimate of the refractive index of AlGaAs with varying Aluminum compositions. For comparison we look at interpolated values of the extreme cases with no aluminum (GaAs), and no gallium (AlAs).

adachi\_1985.py uses a highly informed estimation method to compute the AlGaAs index with considerations of high aluminum composition

n\_interp\_lambda.py interpolates data given from the "Handbook on optical constants of semiconductors" or (hoocos) that is useful for interpolating with scipy's spline method to grab relevant estimates for 1064nm wavelength light.
