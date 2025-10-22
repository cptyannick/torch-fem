from torchfem.materials import Material


class AcousticMaterial(Material):
    def __init__(self, rho: float, c: float):
        self.rho = rho
        self.c = c
        self.is_vectorized = True

    def vectorize(self, size):
        return self

    def step(self, H_inc, F, stress, state, de0, lengths, iteration):
        pass
