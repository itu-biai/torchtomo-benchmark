"""Autograd for projectors that come from another library.

An external projector exposes project_raw and backproject_raw. These two functions
make each one the other's gradient, so a projector swapped in for torchtomo's
still trains as a matched pair.
"""

import torch


class _Project(torch.autograd.Function):
    """A x, with the library's backprojection as the gradient."""

    @staticmethod
    def forward(ctx, image, projector):
        ctx.projector = projector
        return projector.project_raw(image)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.projector.backproject_raw(gradient.contiguous()), None


class _Backproject(torch.autograd.Function):
    """A^T y, with the library's forward projection as the gradient."""

    @staticmethod
    def forward(ctx, sinogram, projector):
        ctx.projector = projector
        return projector.backproject_raw(sinogram)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.projector.project_raw(gradient.contiguous()), None


class MatchedPairMixin:
    """forward, backward and adjoint on top of project_raw and backproject_raw.

    Mixed into a torchtomo ParallelBeam or FanBeam subclass, so the circle mask
    and angle list stay exactly as torchtomo defines them.
    """

    def masked(self, image):
        return image * self.circle_mask.view(1, 1, self.img_size, self.img_size) if self.circle else image

    def forward(self, x):
        return _Project.apply(self.masked(x).contiguous(), self)

    def backward(self, sinogram):
        # forward() projects the masked image, so the matching adjoint masks too.
        return self.masked(_Backproject.apply(sinogram.contiguous(), self))

    def adjoint(self, sinogram):
        return self.backward(sinogram)
