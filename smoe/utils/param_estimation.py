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
    res_7B = estimate_moe_param(32000, 4096, 32, 11008, 16, 2)
    print("7B (2/16)", res_7B)
    res_7B = estimate_moe_param(32000, 4096, 32, 11008, 16, 4)
    print("7B", res_7B)

    # 13B
    res_13B = estimate_moe_param(32000, 5120, 40, 13824, 16, 2)
    print("13B (2/16)", res_13B)
    res_13B = estimate_moe_param(32000, 5120, 40, 13824, 16, 4)
    print("13B", res_13B)

    # 3B upcycling
    for num_experts in range(1, 9):
        res_3B_up = estimate_moe_param(
            32000, 3200, 26, 8640 * num_experts, num_experts, 1
        )
        print(f"3B upcycling {num_experts} experts", res_3B_up)

    """
    3B {'dense_params': 3426473600, 'moe_total_params': 3427806464, 'moe_act_params': 1810398464.0}
    7B (2/16) {'dense_params': 6738415616, 'moe_total_params': 6740513792, 'moe_act_params': 2953057280.0}
    7B {'dense_params': 6738415616, 'moe_total_params': 6740514816, 'moe_act_params': 3494123520.0}
    13B (2/16) {'dense_params': 13015864320, 'moe_total_params': 13019142400, 'moe_act_params': 5587360000.0}
    13B {'dense_params': 13015864320, 'moe_total_params': 13019143680, 'moe_act_params': 6649044480.0}
    3B upcycling 1 experts {'dense_params': 3426473600, 'moe_total_params': 3426556826, 'moe_act_params': 3426556826.0}
    3B upcycling 2 experts {'dense_params': 5583017600, 'moe_total_params': 5583184052, 'moe_act_params': 3426640052.0}
    3B upcycling 3 experts {'dense_params': 7739561600, 'moe_total_params': 7739811278, 'moe_act_params': 3426723278.0}
    3B upcycling 4 experts {'dense_params': 9896105600, 'moe_total_params': 9896438504, 'moe_act_params': 3426806504.0}
    3B upcycling 5 experts {'dense_params': 12052649600, 'moe_total_params': 12053065730, 'moe_act_params': 3426889730.0}
    3B upcycling 6 experts {'dense_params': 14209193600, 'moe_total_params': 14209692956, 'moe_act_params': 3426972956.0}
    3B upcycling 7 experts {'dense_params': 16365737600, 'moe_total_params': 16366320182, 'moe_act_params': 3427056182.0}
    3B upcycling 8 experts {'dense_params': 18522281600, 'moe_total_params': 18522947408, 'moe_act_params': 3427139408.0}
    """
