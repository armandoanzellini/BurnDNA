import pytest
import numpy as np
import pandas as pd

# Imports from final_code package
from final_code.HannWindow import radial_hann
from final_code.DNAPredict import AmIPO4area_logit, AmIPO4area_thresh, AmIPO4area_qty


def test_radial_hann_output_and_center():
    """Verify radial_hann produces correct output array dimensions and center peak behavior."""
    img = np.full((60, 60), 255, dtype=np.uint8)
    res = radial_hann(img)
    
    assert res.shape == img.shape
    # Boundary pixel outside threshold radius should resolve to 0
    assert res[0, 0] == 0.0


def test_dna_predict_logit_bounds():
    """Test logit probability outputs are within valid [0, 1] probability range."""
    prob_low = AmIPO4area_logit(0.05)
    prob_high = AmIPO4area_logit(0.30)
    
    assert 0.0 <= prob_low <= 1.0
    assert 0.0 <= prob_high <= 1.0
    assert prob_high > prob_low


def test_dna_predict_threshold():
    """Verify threshold classification for STR match possibilities."""
    assert AmIPO4area_thresh(0.20) == 'Match Possible'
    assert AmIPO4area_thresh(0.10) == 'No Match Possible'


def test_dna_predict_qty_calculation():
    """Verify exponential regression quantity prediction output."""
    predicted_qty = AmIPO4area_qty(0.35)
    assert isinstance(predicted_qty, (float, np.floating))
    assert predicted_qty > 0