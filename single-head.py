import torch
import torch.nn.functional as F

if __name__ == "__main__":
    torch.manual_seed(9)

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

    # If d_v is more, more richer context is retained.
    d_q, d_k, d_v = 24, 24, 28

    d = embedded_sentence.shape[1]
    print(f"Embedded sentence dim 1 shape: {d}")

    # Single-head attention for reference
    W_query = torch.nn.Parameter(torch.rand(d_q, d))
    W_key = torch.nn.Parameter(torch.rand(d_k, d))
    W_value = torch.nn.Parameter(torch.rand(d_v, d))

    keys = (W_key.matmul(embedded_sentence.T)).T  # shape: [6, d_k]
    query = (W_query.matmul(embedded_sentence.T)).T  # shape: [6, d_q]
    value = (W_value.matmul(embedded_sentence.T)).T  # shape: [6, d_v]

    x_3 = embedded_sentence[3]  # shape: [16]
    q_3 = W_query.matmul(x_3)  # shape: [24]
    omega_3 = q_3.matmul(keys.T)  # q_3: [24], keys.T: [d_k, 6] -> result: [6]
    print(f"omega_3 shape {omega_3.shape}\n")

    attention_weights_3 = F.softmax(omega_3 / (d_k ** 0.5), dim=0)
    context_vector_3 = attention_weights_3.matmul(value)  # shape: [d_v]