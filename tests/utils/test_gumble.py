import torch


def gumbel_rsample(shape):
    one = torch.tensor(1.0)
    zero = torch.tensor(0.0)
    gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample
    return gumbel(shape)


def test_gumble():
    shape = (16, 16)
    gumbel = gumbel_rsample(shape) * 0.01
    print(gumbel)

    normal = torch.randn(shape) * 0.01
    print(normal)


if __name__ == "__main__":
    test_gumble()
