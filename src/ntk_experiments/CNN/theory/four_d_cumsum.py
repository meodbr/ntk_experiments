import torch
import torch.nn.functional as F

def absolute_to_relative(x):
    """
    x[a1, a2, b1, b2]
    -> out[a1-b1, a2-b2, b1, b2]

    with difference indices shifted to be nonnegative.
    """
    H, W, H2, W2 = x.shape
    assert H == H2 and W == W2

    out = torch.zeros(
        2 * H - 1,
        2 * W - 1,
        H,
        W,
        dtype=x.dtype,
        device=x.device,
    )

    a1 = torch.arange(H, device=x.device)
    a2 = torch.arange(W, device=x.device)
    b1 = torch.arange(H, device=x.device)
    b2 = torch.arange(W, device=x.device)

    A1, A2, B1, B2 = torch.meshgrid(a1, a2, b1, b2, indexing="ij")

    D1 = A1 - B1 + (H - 1)
    D2 = A2 - B2 + (W - 1)

    out[D1, D2, B1, B2] = x

    return out

def patches_sum_4d(x_rel, k1, k2):
    """
    output[a1, a2, b1, b2] = sum over x_rel[a1, a2, b1:b1+k1, b2:b2+k2]
    """
    # x_rel_cum = F.pad(x_rel, (k1, 0, k2, 0), mode="constant", value=0)
    x_rel_cum = F.pad(x_rel, (1, 0, 1, 0), mode="constant", value=0)
    # x_rel_cum = x_rel
    x_rel_cum = x_rel_cum.cumsum(dim=-2).cumsum(dim=-1)




    output = (
        x_rel_cum[:, :, k1:, k2:]
        - x_rel_cum[:, :, :-k1, k2:]
        - x_rel_cum[:, :, k1:, :-k2]
        + x_rel_cum[:, :, :-k1, :-k2]
    )

    return output

def relative_to_absolute(x_rel, k1, k2):
    """
    x_rel[a1-b1 + H-1, a2-b2 + W-1, b1, b2]
    -> out[a1, a2, b1, b2]

    with difference indices shifted to be nonnegative.
    """
    H_shift, W_shift, H2, W2 = x_rel.shape
    H = (H_shift + 1) // 2
    W = (W_shift + 1) // 2

    out = torch.zeros(
        H - k1 + 1,
        W - k2 + 1,
        H - k1 + 1,
        W - k2 + 1,
        dtype=x_rel.dtype,
        device=x_rel.device,
    )

    a1 = torch.arange(H - k1 + 1, device=x_rel.device)
    a2 = torch.arange(W - k2 + 1, device=x_rel.device)
    b1 = torch.arange(H - k1 + 1, device=x_rel.device)
    b2 = torch.arange(W - k2 + 1, device=x_rel.device)

    A1, A2, B1, B2 = torch.meshgrid(a1, a2, b1, b2, indexing="ij")

    D1 = A1 - B1 + (H - 1)
    D2 = A2 - B2 + (W - 1)

    out[A1, A2, B1, B2] = x_rel[D1, D2, B1, B2]

    return out

def naive_patches_sum_4d(x_rel, k1, k2):
    """
    output[a1, a2, b1, b2] = sum over x_rel[a1, a2, b1:b1+k1, b2:b2+k2]
    """
    H_shift, W_shift, H2, W2 = x_rel.shape
    H = (H_shift + 1) // 2
    W = (W_shift + 1) // 2

    output = torch.zeros(
        H - k1 + 1,
        W - k2 + 1,
        H - k1 + 1,
        W - k2 + 1,
        dtype=x_rel.dtype,
        device=x_rel.device,
    )

    for a1 in range(H - k1 + 1):
        for a2 in range(W - k2 + 1):
            for b1 in range(H - k1 + 1):
                for b2 in range(W - k2 + 1):
                    D1 = a1 - b1 + (H - 1)
                    D2 = a2 - b2 + (W - 1)
                    output[a1, a2, b1, b2] = x_rel[D1, D2, b1:b1+k1, b2:b2+k2].sum()

    return output

def naive_full_patches_sum_4d(x, k1, k2):
    """
    output[a1, a2, b1, b2] = \sum_{p1=0}^{k1-1} \sum_{p2=0}^{k2-1} x[a1+p1, a2+p2, b1+p1, b2+p2]
    """
    H, W, H2, W2 = x.shape
    assert H == H2 and W == W2

    output = torch.zeros(
        H - k1 + 1,
        W - k2 + 1,
        H - k1 + 1,
        W - k2 + 1,
        dtype=x.dtype,
        device=x.device,
    )

    for a1 in range(H - k1 + 1):
        for a2 in range(W - k2 + 1):
            for b1 in range(H - k1 + 1):
                for b2 in range(W - k2 + 1):
                    s = 0.0
                    for p1 in range(k1):
                        for p2 in range(k2):
                            s += x[a1 + p1, a2 + p2, b1+p1, b2+p2]
                    output[a1, a2, b1, b2] = s

    return output


def patches_sum_4d_cumsum_method(x, k1, k2):
    """
    output[a1, a2, b1, b2] = \sum_{p1=0}^{k1-1} \sum_{p2=0}^{k2-1} x[a1+p1, a2+p2, b1+p1, b2+p2]
    """
    H, W, H2, W2 = x.shape
    assert H == H2 and W == W2

    x_rel = absolute_to_relative(x)
    x_rel_patches_sum = patches_sum_4d(x_rel, k1=k1, k2=k2)
    output_reconstructed = relative_to_absolute(x_rel_patches_sum, k1=k1, k2=k2)

    return output_reconstructed


if __name__ == "__main__":
    H, W = 5, 5
    x = torch.randn(H, W, H, W)  # Random inputs

    x_rel = absolute_to_relative(x)
    x_rel_patches_sum = patches_sum_4d(x_rel, k1=3, k2=3)
    output_reconstructed = relative_to_absolute(x_rel_patches_sum, k1=3, k2=3)

    x_reconstructed_naive = naive_full_patches_sum_4d(x, k1=3, k2=3)
    x_rel_patches_sum_naive = naive_patches_sum_4d(x_rel, k1=3, k2=3)


    print("x shape:", x.shape)
    print("x_rel shape:", x_rel.shape)
    print("x_rel_patches_sum shape:", x_rel_patches_sum.shape)
    print("output_reconstructed shape:", output_reconstructed.shape)
    print("x_reconstructed_naive shape:", x_reconstructed_naive.shape)
    print("x_rel_patches_sum_naive shape:", x_rel_patches_sum_naive.shape)

    a1, a2, b1, b2 = 1, 2, 3, 4

    print(f"x[{a1}, {a2}, :, :] = {x[a1, a2, :, :]}")
    print("========================================")
    print(f"x[{a1}, {a2}, {b1}, {b2}] = {x[a1, a2, b1, b2]}")
    print(f"x_rel[{a1}-{b1} + {H-1}, {a2}-{b2} + {W-1}, {b1}, {b2}] = {x_rel[a1-b1 + H-1, a2-b2 + W-1, b1, b2]}")
    print(f"x_rel[{a1-b1} + {H-1}, {a2-b2} + {W-1}, :, :] = {x_rel[a1-b1 + H-1, a2-b2 + W-1, :, :]}")
    print(f"x_rel_patches_sum[{a1-b1} + {H-1}, {a2-b2} + {W-1}, :, :] = {x_rel_patches_sum[a1-b1 + H-1, a2-b2 + W-1, :, :]}")
    print("========================================")
    print(f"output_reconstructed[{a1}, {a2}, :, :] = {output_reconstructed[a1, a2, :, :]}")

    print("========================================")
    print(f"distance btwn outpu_reconstructed and x_reconstructed_naive: {torch.linalg.norm(output_reconstructed - x_reconstructed_naive)}")
    print("3 random examples of comparision\n")
    for _ in range(3):
        a1 = torch.randint(0, H-3+1, (1,)).item()
        a2 = torch.randint(0, W-3+1, (1,)).item()
        b1 = torch.randint(0, H-3+1, (1,)).item()
        b2 = torch.randint(0, W-3+1, (1,)).item()
        
        print(f"output_reconstructed[{a1}, {a2}, {b1}, {b2}] = {output_reconstructed[a1, a2, b1, b2]}")
        print(f"x_reconstructed_naive[{a1}, {a2}, {b1}, {b2}] = {x_reconstructed_naive[a1, a2, b1, b2]}")
        print(f"distance btwn output_reconstructed[{a1}, {a2}, {b1}, {b2}] and x_reconstructed_naive[{a1}, {a2}, {b1}, {b2}]: {torch.abs(output_reconstructed[a1, a2, b1, b2] - x_reconstructed_naive[a1, a2, b1, b2])}")
        print("========================================")   

    print(f"distance btwn x_rel_patches_sum and x_rel_patches_sum_naive: {torch.linalg.norm(x_rel_patches_sum - x_rel_patches_sum_naive)}")

