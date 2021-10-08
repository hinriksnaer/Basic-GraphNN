import torch
import torch.nn.functional as F
import torch.nn as nn


class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        #HW
        node_feats = self.projection(node_feats)

        assert torch.all(torch.eq(torch.bmm(adj_matrix, node_feats),adj_matrix @ node_feats))

        # adj matrix consists of connection A_ij + identity matrix
        # A(HW)
        node_feats = adj_matrix @ node_feats
        # division
        node_feats = node_feats / num_neighbours
        return node_feats

class GATLayer(nn.Module):

    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2, dropout=0.6):
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0
            self.c_out = c_out //num_heads
        
        self.projection = nn.Linear(c_in, self.c_out*num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * self.c_out))
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        #apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        # [batch, nodes, heads, c_out] c_out is c_out per head 
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1) # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        # [batch, node, node, node_feat]
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        # [batch, nodes, heads, c_out]
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats

if __name__ == "__main__":
    node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    adj_matrix = torch.Tensor([[[1, 1, 0, 0],
                                [1, 1, 1, 1],
                                [0, 1, 1, 1],
                                [0, 1, 1, 1]]])

    print("Node features:\n", node_feats)
    print("\nAdjacency matrix:\n", adj_matrix)

    layer = GATLayer(2, 16, num_heads=2)
    #layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
    #layer.projection.bias.data = torch.Tensor([0., 0.])
    #layer.a.data = torch.Tensor([[-0.2, 0.3], [0.1, -0.1]])

    with torch.no_grad():
        out_feats = layer(node_feats, adj_matrix, print_attn_probs=True)

    print("Adjacency matrix", adj_matrix)
    print("Input features", node_feats)
    print("Output features", out_feats)



