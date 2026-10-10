# Third-party notices and standards

ChannelHub's research-only license applies to ChannelHub project code. It does not replace licenses for separately installed dependencies or grant rights over the referenced standards.

| Component | Use | Upstream |
|---|---|---|
| NVIDIA Sionna PHY 2.2.0 | CDL and TDL channel generation | [Sionna](https://github.com/NVlabs/sionna), Apache-2.0 |
| NVIDIA Sionna RT 2.2.0 | Radio ray tracing | [Sionna RT](https://github.com/NVlabs/sionna-rt), Apache-2.0 |
| Mitsuba / Dr.Jit | Sionna RT execution | [Mitsuba](https://github.com/mitsuba-renderer/mitsuba3), [Dr.Jit](https://github.com/mitsuba-renderer/drjit) |
| NumPy | Numerical arrays and exports | [NumPy](https://numpy.org/) |
| PyTorch | Sionna PHY execution | [PyTorch](https://pytorch.org/) |
| FastAPI, Starlette, Pydantic, Uvicorn | Local web service and configuration | Upstream package licenses apply |

These dependencies are installed from their upstream distributions; their source, wheels, models and example scene assets are not vendored into this repository. Preserve their individual notices when redistributing an environment under separately obtained permission.

Transmit sequence generation in `src/channelhub/ref_signals/` implements public equations and tables from 3GPP TS 38.211: Gold sequences, low-PAPR base sequences, SRS, CSI-RS and SS/PBCH reference signals. SRS bandwidth rows follow V15.7.0 Table 6.4.1.4.3-1. The public configuration supports Comb-2 and Comb-4. The CSI-RS grid wrapper is limited to single-port row 2; the SS/PBCH wrapper omits encoded PBCH payloads.

Normative references:

- [3GPP specification 38.211](https://www.3gpp.org/dynareport/38211.htm)
- [3GPP technical report 38.901](https://www.3gpp.org/dynareport/38901.htm)
- [Sionna documentation](https://nvlabs.github.io/sionna/)

The supplied `reflector.xml` is an original geometric plane example. It contains no imported city or site data.
