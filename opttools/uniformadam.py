'''

                                    ORIGINAL LICENSE                                 
---------------------------------------------------------------------------------------
 Copyright (c) 2021 Baptiste Nicolet <baptiste.nicolet@epfl.ch>, All rights reserved.
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to the author of this software, without
imposing a separate written license agreement for such Enhancements, then you
hereby grant the following license: a non-exclusive, royalty-free perpetual
license to install, use, modify, prepare derivative works, incorporate into
other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.
---------------------------------------------------------------------------------------

code adapted from: https://github.com/rgl-epfl/large-steps-pytorch

'''
import torch

class UniformAdam(torch.optim.Optimizer):
    """
    Variant of Adam with uniform scaling by the second moment.

    Instead of dividing each component by the square root of its second moment,
    we divide all of them by the max.
    """
    def __init__(self, params, lr=0.1, betas=(0.9,0.999)):
        defaults = dict(lr=lr, betas=betas)
        super(UniformAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(UniformAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state)==0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                g1.mul_(b1).add_(grad, alpha=1-b1)
                g2.mul_(b2).add_(grad.square(), alpha=1-b2)
                m1 = g1 / (1-(b1**state["step"]))
                m2 = g2 / (1-(b2**state["step"]))
                # This is the only modification we make to the original Adam algorithm
                gr = m1 / (1e-8 + m2.sqrt().max())
                p.data.sub_(gr, alpha=lr)
