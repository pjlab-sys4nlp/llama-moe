def estimate_moe_param(
    vocab_size,
    hidden_size,
    num_hidden_layers,
    intermediate_size,
    num_experts,
    num_selects,
):
    """
    Llama Structure with SwiGLU MLP
    MoE split on intermediate_size without weight noise in 2-layer gate
    """

    emb = vocab_size * hidden_size
    lm_head = vocab_size * hidden_size
    final_norm = hidden_size

    self_attn = hidden_size * hidden_size * 4
    mlp = hidden_size * intermediate_size * 3
    input_norm = hidden_size
    post_attn_norm = hidden_size

    dense_one_layer = self_attn + mlp + input_norm + post_attn_norm
    dense_params = emb + lm_head + final_norm + dense_one_layer * num_hidden_layers

    gate = hidden_size * num_experts + num_experts * num_selects
    moe_one_layer = self_attn + mlp + input_norm + post_attn_norm + gate
    moe_total_params = emb + lm_head + final_norm + moe_one_layer * num_hidden_layers

    moe_one_act_layer = (
        self_attn
        + (mlp / num_experts * num_selects)
        + input_norm
        + post_attn_norm
        + gate
    )
    moe_act_params = emb + lm_head + final_norm + moe_one_act_layer * num_hidden_layers

    return {
        "dense_params": dense_params,
        "moe_total_params": moe_total_params,
        "moe_act_params": moe_act_params,
    }


if __name__ == "__main__":
    # 3B
    res_3B = estimate_moe_param(32000, 3200, 26, 8640, 16, 4)
    print("3B", res_3B)

    # 7B
    res_7B = estimate_moe_param(32000, 4096, 32, 11008, 16, 4)
    print("7B", res_7B)

    # 13B
    res_13B = estimate_moe_param(32000, 5120, 40, 13824, 16, 4)
    print("13B", res_13B)

    # 3B upcycling
    for num_experts in range(1, 9):
        res_3B_up = estimate_moe_param(
            32000, 3200, 26, 8640 * num_experts, num_experts, 1
        )
        print(f"3B upcycling {num_experts} experts", res_3B_up)
