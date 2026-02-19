from pytest import mark, param
from torch import Tensor, tensor
from torch.testing import assert_close

from torchjd.aggregation import (
    IMTLG,
    MGDA,
    Aggregator,
    AlignedMTL,
    AlignedMTLWeighting,
    ConFIG,
    Constant,
    ConstantWeighting,
    DualProj,
    DualProjWeighting,
    GradDrop,
    IMTLGWeighting,
    Krum,
    KrumWeighting,
    Mean,
    MeanWeighting,
    MGDAWeighting,
    PCGrad,
    PCGradWeighting,
    Random,
    RandomWeighting,
    Sum,
    SumWeighting,
    TrimmedMean,
    UPGrad,
    UPGradWeighting,
    Weighting,
)

J_base = tensor([[-4.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
J_Krum = tensor(
    [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [75.0, -666.0, 23],  # adversarial row
        [1.0, 2.0, 3.0],
        [2.0, 0.0, 1.0],
    ],
)
J_TrimmedMean = tensor(
    [
        [1e11, 3.0],
        [1.0, -1e11],
        [-1e10, 1e10],
        [2.0, 2.0],
    ],
)

AGGREGATOR_PARAMETRIZATIONS = [
    (AlignedMTL(), J_base, tensor([0.2133, 0.9673, 0.9673])),
    (ConFIG(), J_base, tensor([0.1588, 2.0706, 2.0706])),
    (Constant(tensor([1.0, 2.0])), J_base, tensor([8.0, 3.0, 3.0])),
    (DualProj(), J_base, tensor([0.5563, 1.1109, 1.1109])),
    (GradDrop(), J_base, tensor([6.0, 2.0, 2.0])),
    (IMTLG(), J_base, tensor([0.0767, 1.0000, 1.0000])),
    (Krum(n_byzantine=1, n_selected=4), J_Krum, tensor([1.2500, 0.7500, 1.5000])),
    (Mean(), J_base, tensor([1.0, 1.0, 1.0])),
    (MGDA(), J_base, tensor([0.0, 1.0, 1.0])),
    (PCGrad(), J_base, tensor([0.5848, 3.8012, 3.8012])),
    (Random(), J_base, tensor([-2.6229, 1.0000, 1.0000])),
    (Sum(), J_base, tensor([2.0, 2.0, 2.0])),
    (TrimmedMean(trim_number=1), J_TrimmedMean, tensor([1.5000, 2.5000])),
    (UPGrad(), J_base, tensor([0.2929, 1.9004, 1.9004])),
]

G_base = J_base @ J_base.T
G_Krum = J_Krum @ J_Krum.T

WEIGHTING_PARAMETRIZATIONS = [
    (AlignedMTLWeighting(), G_base, tensor([0.5591, 0.4083])),
    (ConstantWeighting(tensor([1.0, 2.0])), G_base, tensor([1.0, 2.0])),
    (DualProjWeighting(), G_base, tensor([0.6109, 0.5000])),
    (IMTLGWeighting(), G_base, tensor([0.5923, 0.4077])),
    (KrumWeighting(1, 4), G_Krum, tensor([0.2500, 0.2500, 0.0000, 0.2500, 0.2500])),
    (MeanWeighting(), G_base, tensor([0.5000, 0.5000])),
    (MGDAWeighting(), G_base, tensor([0.6000, 0.4000])),
    (PCGradWeighting(), G_base, tensor([2.2222, 1.5789])),
    (RandomWeighting(), G_base, tensor([0.8623, 0.1377])),
    (SumWeighting(), G_base, tensor([1.0, 1.0])),
    (UPGradWeighting(), G_base, tensor([1.1109, 0.7894])),
]

try:
    from torchjd.aggregation import CAGrad, CAGradWeighting

    AGGREGATOR_PARAMETRIZATIONS.append((CAGrad(c=0.5), J_base, tensor([0.1835, 1.2041, 1.2041])))
    WEIGHTING_PARAMETRIZATIONS.append((CAGradWeighting(c=0.5), G_base, tensor([0.7041, 0.5000])))
except ImportError:
    pass

try:
    from torchjd.aggregation import NashMTL

    AGGREGATOR_PARAMETRIZATIONS.append(
        param(
            NashMTL(n_tasks=2),
            J_base,
            tensor([0.0542, 0.7061, 0.7061]),
            marks=mark.filterwarnings("ignore::UserWarning"),
        ),
    )

except ImportError:
    pass


@mark.parametrize(["A", "J", "expected_output"], AGGREGATOR_PARAMETRIZATIONS)
def test_aggregator_output(A: Aggregator, J: Tensor, expected_output: Tensor):
    """Test that the output values of an aggregator are fixed (on cpu)."""

    assert_close(A(J), expected_output, rtol=0, atol=1e-4)


@mark.parametrize(["W", "G", "expected_output"], WEIGHTING_PARAMETRIZATIONS)
def test_weighting_output(W: Weighting, G: Tensor, expected_output: Tensor):
    """Test that the output values of a weighting are fixed (on cpu)."""

    assert_close(W(G), expected_output, rtol=0, atol=1e-4)
