from __future__ import annotations

import numpy as np
import torch

from cfd_operator.data.airfrans import parse_airfrans_simulation_name
from cfd_operator.losses.composite import pressure_to_cp


def test_parse_airfrans_simulation_name_for_naca4() -> None:
    parsed = parse_airfrans_simulation_name("airFoil2D_SST_34.0_5.0_5.454_3.799_13.179")
    assert parsed["geometry_id"] == "naca-4-5.454000-3.799000-13.179000"
    geometry_params = parsed["geometry_params"]
    assert np.allclose(geometry_params, np.asarray([4.0, 5.454, 3.799, 13.179, 0.0], dtype=np.float32))
    assert parsed["aoa_deg"] == 5.0


def test_pressure_to_cp_with_reference() -> None:
    pressure = torch.tensor([[[0.5], [1.0]]], dtype=torch.float32)
    cp_reference = torch.tensor([[[0.0, 0.5]]], dtype=torch.float32)
    cp = pressure_to_cp(pressure=pressure, cp_reference=cp_reference)
    expected = torch.tensor([[[1.0], [2.0]]], dtype=torch.float32)
    assert torch.allclose(cp, expected)
