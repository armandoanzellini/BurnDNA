import numpy as np
import pandas as pd
import pytest

# pyproject.toml puts FinalCode on pythonpath, so modules import by bare name.
from DNAPredict import (AmIPO4area_logit, AmIPO4area_thresh, AmIPO4area_qty,
                        read_spc_dir, best_ATRFTIR_spec, best_spectrum)
import DNAPredict
import SpecCalcMod as scm


# ---------------------------------------------------------------------------
# DNAPredict: model functions
# ---------------------------------------------------------------------------
def test_dna_predict_logit_bounds():
    """Logit probabilities stay within [0, 1] and increase with the ratio."""
    prob_low = AmIPO4area_logit(0.05)
    prob_high = AmIPO4area_logit(0.30)

    assert 0.0 <= prob_low <= 1.0
    assert 0.0 <= prob_high <= 1.0
    assert prob_high > prob_low


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.20, "Match Possible"),
        (0.165, "Match Possible"),  # threshold is inclusive (>=)
        (0.10, "No Match Possible"),
    ],
)
def test_dna_predict_threshold(value, expected):
    """Threshold classification for STR match possibility."""
    assert AmIPO4area_thresh(value) == expected


def test_dna_predict_qty_calculation():
    """Exponential regression quantity estimate is a positive float."""
    predicted_qty = AmIPO4area_qty(0.35)
    assert isinstance(predicted_qty, (float, np.floating))
    assert predicted_qty > 0


# ---------------------------------------------------------------------------
# SpecCalcMod: the module DNAPredict depends on (pure functions only)
# ---------------------------------------------------------------------------
def test_wavenumber_conversion():
    """10 micron = 1000 cm-1 and 1000 nm = 10000 cm-1."""
    assert scm.wavenumber(10, unit='micron') == pytest.approx(1000)
    assert scm.wavenumber(1000, unit='nanometer') == pytest.approx(10000)


def test_delta13c_roundtrip():
    """VPDB ratio gives 0 permil, and del13C / invdel13C are inverses."""
    assert scm.del13C(0.01123720) == pytest.approx(0.0)
    assert scm.invdel13C(scm.del13C(0.0110)) == pytest.approx(0.0110)


def test_k_constant_oscillation_roundtrip():
    """oscillation() recovers the peak position used to derive k."""
    k = scm.k_constant(31, 16, 960)
    assert scm.oscillation(31, 16, k) == 960


def test_isotope_shift_lowers_frequency():
    """A heavier oxygen isotope shifts the PO4 peak to lower wavenumber."""
    assert scm.shift(31, 16, 31, 18, 960) < 960


def test_baseline_arpls_shape():
    """arPLS baseline has the input length, is finite, and sits below a peak."""
    x = np.linspace(0, 1, 200)
    y = 0.5 * x + np.exp(-((x - 0.5) ** 2) / 0.002)  # ramp plus one peak
    base = scm.baseline_arPLS(y)

    assert base.shape == y.shape
    assert np.all(np.isfinite(base))
    assert base[100] < y[100]  # baseline does not follow the peak apex


# ---------------------------------------------------------------------------
# Synthetic ATR-FTIR spectra (no instrument data needed)
# ---------------------------------------------------------------------------
def synthetic_spectrum(amide_amp, po4_amp=1.0, name="synthetic"):
    """Bone-like spectrum: PO4 band at 1035 cm-1 (sigma 40) and amide I band at
    1650 cm-1 (sigma 25), on a descending wavenumber axis like instrument output."""
    x = np.linspace(4000.0, 650.0, 6950)
    y = (po4_amp * np.exp(-0.5 * ((x - 1035) / 40) ** 2)
         + amide_amp * np.exp(-0.5 * ((x - 1650) / 25) ** 2))
    return pd.Series(y, index=x, name=name)


def amipo4_area(spectrum):
    """amIPO4area ratio of a synthetic spectrum, the way DNAPredict calls it."""
    df = spectrum.to_frame()
    df["Wavelength"] = df.index
    return scm.amIPO4area(df, "Wavelength", spectrum.name, smooth=True, plot=False)


def expected_ratio(amide_amp, po4_amp=1.0):
    """Analytic area ratio of the two Gaussians (equal-sigma-window cuts are
    small, so this holds to a few percent)."""
    return (amide_amp * 25) / (po4_amp * 40)


@pytest.mark.parametrize("amide_amp", [0.1, 0.25, 0.5])
def test_amipo4area_matches_analytic_ratio(amide_amp):
    """Area ratio recovers the known band-area ratio of a synthetic spectrum."""
    ratio = amipo4_area(synthetic_spectrum(amide_amp))
    assert ratio == pytest.approx(expected_ratio(amide_amp), rel=0.1)


def test_amipo4area_scales_linearly_with_amide_band():
    """Doubling the amide I band doubles the ratio."""
    low = amipo4_area(synthetic_spectrum(0.2))
    high = amipo4_area(synthetic_spectrum(0.4))
    assert high / low == pytest.approx(2.0, rel=0.02)


def test_threshold_separates_preserved_and_degraded_spectra():
    """Collagen-rich spectrum clears the 0.165 threshold, collagen-poor does not."""
    preserved = amipo4_area(synthetic_spectrum(0.5))
    degraded = amipo4_area(synthetic_spectrum(0.1))

    assert AmIPO4area_thresh(preserved) == "Match Possible"
    assert AmIPO4area_thresh(degraded) == "No Match Possible"


@pytest.fixture
def fake_spc(monkeypatch):
    """Replace read_spc with an in-memory lookup so no .spc files are needed.
    pyspectra returns an unnamed Series indexed by wavenumber, so do the same."""
    spectra = {
        "data/sample rep 1.spc": synthetic_spectrum(0.3, po4_amp=0.8, name=None),
        "data/sample rep 2.spc": synthetic_spectrum(0.3, po4_amp=1.0, name=None),
    }
    monkeypatch.setattr(DNAPredict, "read_spc", lambda path: spectra[path])
    return list(spectra)


def test_read_spc_dir_names_columns_after_files(fake_spc):
    """Each replicate becomes a column named after its file, on a shared axis."""
    df = read_spc_dir(fake_spc)

    assert list(df.columns) == ["sample rep 1", "sample rep 2"]
    assert df.shape[0] == 6950
    assert np.isfinite(df.to_numpy()).all()


def test_best_atrftir_spec_picks_strongest_phosphate_replicate(fake_spc):
    """The replicate with the largest peak below 1200 cm-1 is selected."""
    best = best_ATRFTIR_spec(fake_spc)

    assert best.name == "sample rep 2"
    assert best.max() == pytest.approx(1.0, rel=1e-3)


def test_best_spectrum_pipeline_without_figures(fake_spc):
    """best_spectrum runs end to end and returns one ratio per sample."""
    indices, spectra = best_spectrum({"sample": fake_spc},
                                     base=False, plot=False, outdir=None)

    assert list(indices.index) == ["sample"]
    assert indices.loc["sample", "AmIPO4area"] == pytest.approx(
        expected_ratio(0.3), rel=0.1)
    assert "Wavelength" in spectra.columns
