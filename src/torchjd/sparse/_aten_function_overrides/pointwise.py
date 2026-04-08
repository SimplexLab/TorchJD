from torch.ops import aten  # type: ignore

from torchjd.sparse._sparse_latticed_tensor import SparseLatticedTensor, impl

# pointwise functions applied to one Tensor with `0.0 → 0`
_POINTWISE_FUNCTIONS = [
    aten.abs.default,
    aten.absolute.default,
    aten.asin.default,
    aten.asinh.default,
    aten.atan.default,
    aten.atanh.default,
    aten.ceil.default,
    aten.erf.default,
    aten.erfinv.default,
    aten.expm1.default,
    aten.fix.default,
    aten.floor.default,
    aten.hardtanh.default,
    aten.leaky_relu.default,
    aten.log1p.default,
    aten.neg.default,
    aten.negative.default,
    aten.positive.default,
    aten.relu.default,
    aten.round.default,
    aten.sgn.default,
    aten.sign.default,
    aten.sin.default,
    aten.sinh.default,
    aten.sqrt.default,
    aten.square.default,
    aten.tan.default,
    aten.tanh.default,
    aten.trunc.default,
]

_IN_PLACE_POINTWISE_FUNCTIONS = [
    aten.abs_.default,
    aten.absolute_.default,
    aten.asin_.default,
    aten.asinh_.default,
    aten.atan_.default,
    aten.atanh_.default,
    aten.ceil_.default,
    aten.erf_.default,
    aten.erfinv_.default,
    aten.expm1_.default,
    aten.fix_.default,
    aten.floor_.default,
    aten.hardtanh_.default,
    aten.leaky_relu_.default,
    aten.log1p_.default,
    aten.neg_.default,
    aten.negative_.default,
    aten.relu_.default,
    aten.round_.default,
    aten.sgn_.default,
    aten.sign_.default,
    aten.sin_.default,
    aten.sinh_.default,
    aten.sqrt_.default,
    aten.square_.default,
    aten.tan_.default,
    aten.tanh_.default,
    aten.trunc_.default,
]


def _override_pointwise(op):
    @impl(op)
    def func_(t: SparseLatticedTensor) -> SparseLatticedTensor:
        assert isinstance(t, SparseLatticedTensor)
        return SparseLatticedTensor(op(t.physical), t.basis, t.offset, t.shape)

    return func_


def _override_inplace_pointwise(op):
    @impl(op)
    def func_(t: SparseLatticedTensor) -> SparseLatticedTensor:
        assert isinstance(t, SparseLatticedTensor)
        op(t.physical)
        return t


for pointwise_func in _POINTWISE_FUNCTIONS:
    _override_pointwise(pointwise_func)

for pointwise_func in _IN_PLACE_POINTWISE_FUNCTIONS:
    _override_inplace_pointwise(pointwise_func)


@impl(aten.pow.Tensor_Scalar)
def pow_Tensor_Scalar(t: SparseLatticedTensor, exponent: float) -> SparseLatticedTensor:
    assert isinstance(t, SparseLatticedTensor)

    if exponent <= 0.0:
        # Need to densify because we don't have pow(0.0, exponent) = 0.0
        return aten.pow.Tensor_Scalar(t.to_dense(), exponent)

    new_physical = aten.pow.Tensor_Scalar(t.physical, exponent)
    return SparseLatticedTensor(new_physical, t.basis, t.offset, t.shape)


# Somehow there's no pow_.Tensor_Scalar and pow_.Scalar takes tensor and scalar.
@impl(aten.pow_.Scalar)
def pow__Scalar(t: SparseLatticedTensor, exponent: float) -> SparseLatticedTensor:
    assert isinstance(t, SparseLatticedTensor)

    if exponent <= 0.0:
        # Need to densify because we don't have pow(0.0, exponent) = 0.0
        # Note sure if it's even possible to densify in-place, so let's just raise an error.
        raise ValueError(f"in-place pow with an exponent of {exponent} (<= 0) is not supported.")

    aten.pow_.Scalar(t.physical, exponent)
    return t


@impl(aten.div.Scalar)
def div_Scalar(t: SparseLatticedTensor, divisor: float) -> SparseLatticedTensor:
    assert isinstance(t, SparseLatticedTensor)

    new_physical = aten.div.Scalar(t.physical, divisor)
    return SparseLatticedTensor(new_physical, t.basis, t.offset, t.shape)
