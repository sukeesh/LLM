import torch
import torch.nn.functional as F

if __name__ == "__main__":
    torch.manual_seed(9)  # For reproducibility
    
    # Get sentence input from user
    sentence = input("Enter a sentence: ")
    
    # Split the sentence into words
    words = sentence.split()
    sorted_words = sorted(words)
    dc = {sw: i for i, sw in enumerate(sorted_words)}
    s_idx = [dc[w] for w in words]

    t_sentence = torch.tensor(s_idx)
    len_words = len(words)
    embed = torch.nn.Embedding(len_words, 16)
    embedded_sentence = embed(t_sentence).detach()  # shape: [N, 16]

    d_q, d_k, d_v = 24, 24, 28
    d = embedded_sentence.shape[1]
    print(f"Embedded sentence dim 1 shape: {d}")

    # Single-head attention for reference
    W_query = torch.nn.Parameter(torch.rand(d_q, d))
    W_key = torch.nn.Parameter(torch.rand(d_k, d))
    W_value = torch.nn.Parameter(torch.rand(d_v, d))

    keys = (W_key.matmul(embedded_sentence.T)).T   # shape: [6, d_k]
    query = (W_query.matmul(embedded_sentence.T)).T  # shape: [6, d_q]
    value = (W_value.matmul(embedded_sentence.T)).T  # shape: [6, d_v]

    x_3 = embedded_sentence[3]  # shape: [16]
    q_3 = W_query.matmul(x_3)   # shape: [24]
    omega_3 = q_3.matmul(keys.T)  # q_3: [24], keys.T: [d_k, 6] -> result: [6]
    print(f"omega_3 shape {omega_3.shape}\n")

    attention_weights_3 = F.softmax(omega_3 / (d_k ** 0.5), dim=0)
    context_vector_3 = attention_weights_3.matmul(value)  # shape: [d_v]

    # --- Multi-head Attention ---
    h = 3  # number of heads

    # Weight matrices for each head, each with shape: [h, projection_dim, d]
    multihead_W_q = torch.nn.Parameter(torch.rand(h, d_q, d))
    multihead_W_k = torch.nn.Parameter(torch.rand(h, d_k, d))
    multihead_W_v = torch.nn.Parameter(torch.rand(h, d_v, d))

    # We want to compute the projections over all tokens.
    # First, expand the embedded sentence from [6, 16] to [h, 6, 16]
    stacked_embedded_sentence = embedded_sentence.unsqueeze(0).repeat(h, 1, 1)  # shape: [h, 6, 16]
    # To use batch matrix multiplication, transpose last two dims: [h, 16, 6]
    stacked_embedded_sentence_T = stacked_embedded_sentence.transpose(1, 2)  # shape: [h, 16, 6]

    # Compute projected queries, keys, and values for each head:
    # Resulting shapes after bmm are [h, projection_dim, 6]; then we transpose to [h, 6, projection_dim]
    multihead_query = torch.bmm(multihead_W_q, stacked_embedded_sentence_T).transpose(1, 2)  # shape: [h, 6, 24]
    multihead_keys  = torch.bmm(multihead_W_k, stacked_embedded_sentence_T).transpose(1, 2)  # shape: [h, 6, 24]
    multihead_value = torch.bmm(multihead_W_v, stacked_embedded_sentence_T).transpose(1, 2)  # shape: [h, 6, 28]

    def print_attention_matrix(attention_weights, words):
        """
        Display a single attention matrix with word labels and heat map.
        """
        # Print column headers (words)
        print("\n" + " " * 8, end="")
        for word in words:
            print(f"{word:>10}", end="")
        print("\n" + "-" * (10 * len(words) + 8))

        # Print each row with row header (word) and attention weights
        for i, word in enumerate(words):
            print(f"{word:<8}", end="")
            for j in range(len(words)):
                value = attention_weights[i, j].item()
                # Create grayscale background using ANSI escape codes
                intensity = min(int(value * 255), 255)
                print(f"\033[48;2;{intensity};{intensity};{intensity}m{value:10.5f}\033[0m", end="")
            print()  # New line after each row

    # Replace the loop and add output collection
    outs = []  # List to collect outputs from each head
    for i in range(h):
        print(f"\n=== Attention Head {i+1} ===")
        attention_weights_i = F.softmax(multihead_query[i].matmul(multihead_keys[i].T) / (d_k ** 0.5), dim=1)
        print_attention_matrix(attention_weights_i, words)
        # Calculate output for each head
        out_i = attention_weights_i.matmul(multihead_value[i])  # shape: [6, d_v]
        outs.append(out_i)

    # Concatenate the head outputs along the feature (last) dimension
    multihead_output = torch.cat(outs, dim=1)
    print(f"Multi-head output shape: {multihead_output.shape}")

    # print(multihead_output[0])