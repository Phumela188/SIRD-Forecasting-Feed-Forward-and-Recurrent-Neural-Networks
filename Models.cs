
namespace Assignment2
{
        /// NN1: FFNN with one hidden layer.
        /// Input:  (patterns, 4) S,I,R,D for week w
        /// Output: (patterns, 4)  predicted S,I,R,D for week w+1
        /// ReLU on the hidden layer, linear (identity) on the output layer since S,I,R,D
        /// are normalised to [0,1] but are a regression target, not a probability -- a plain
        /// linear output avoids squashing predictions towards the middle of the range.
        public class FFNN : nn.Module<Tensor, Tensor>
        {
            private readonly nn.Module<Tensor, Tensor> lin1;
            private readonly nn.Module<Tensor, Tensor> lin2;

            public FFNN(int nrInputs, int nrHidden, int nrOutputs) : base(nameof(FFNN))
            {
                lin1 = nn.Linear(nrInputs, nrHidden);
                lin2 = nn.Linear(nrHidden, nrOutputs);
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                var y = nn.functional.relu(lin1.forward(input));
                var o = lin2.forward(y); // linear output for regression
                return o;
            }
        }

        
        /// NN2: Elman RNN layer followed by a linear layer.
        /// Input:  (B, P, 4)
        /// Output: (B, P, 4)
        /// TorchSharp's RNN module already applies tanh internally to its outputs, so no extra
        /// activation is applied before the RNN output is passed to the linear layer.
        public class ElmanNN : nn.Module<Tensor, Tensor>
        {
            private readonly nn.Module<Tensor, Tensor?, (Tensor, Tensor)> rnn1;
            private readonly nn.Module<Tensor, Tensor> lin2;

            public ElmanNN(int nrInputs, int nrHidden, int nrOutputs) : base(nameof(ElmanNN))
            {
                rnn1 = nn.RNN(nrInputs, nrHidden, numLayers: 1, batchFirst: true);
                lin2 = nn.Linear(nrHidden, nrOutputs);
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                var (y, _) = rnn1.forward(input, null);
                var o = lin2.forward(y); // linear output for regression
                return o;
            }
        }

        /// NN3: LSTM layer followed by a linear layer.
        public class LSTMNN : nn.Module<Tensor, Tensor>
        {
            private readonly nn.Module<Tensor, (Tensor, Tensor)?, (Tensor, Tensor, Tensor)> lstm1;
            private readonly nn.Module<Tensor, Tensor> lin2;

            public LSTMNN(int nrInputs, int nrHidden, int nrOutputs) : base(nameof(LSTMNN))
            {
                lstm1 = nn.LSTM(nrInputs, nrHidden, numLayers: 1, batchFirst: true);
                lin2 = nn.Linear(nrHidden, nrOutputs);
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                var (y, _, _) = lstm1.forward(input, null);
                var o = lin2.forward(y);
                return o;
            }
        }

        /// NN4: GRU layer followed by a linear layer.
        public class GRUNN : nn.Module<Tensor, Tensor>
        {
            private readonly nn.Module<Tensor, Tensor?, (Tensor, Tensor)> gru1;
            private readonly nn.Module<Tensor, Tensor> lin2;

            public GRUNN(int nrInputs, int nrHidden, int nrOutputs) : base(nameof(GRUNN))
            {
                gru1 = nn.GRU(nrInputs, nrHidden, numLayers: 1, batchFirst: true);
                lin2 = nn.Linear(nrHidden, nrOutputs);
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                var (y, _) = gru1.forward(input, null);
                var o = lin2.forward(y);
                return o;
            }
        }

        /// NN5: the same input sequence is fed independently into an Elman layer, an LSTM layer
        /// and a GRU layer. The three (B, P, NrHidden) outputs are summed element-wise and the
        /// sum is passed through a single linear output layer.
        /// All three recurrent layers must therefore share the same hidden size so the sum is valid.
        public class CombinedNN : nn.Module<Tensor, Tensor>
        {
            private readonly nn.Module<Tensor, Tensor?, (Tensor, Tensor)> rnn1;
            private readonly nn.Module<Tensor, (Tensor, Tensor)?, (Tensor, Tensor, Tensor)> lstm1;
            private readonly nn.Module<Tensor, Tensor?, (Tensor, Tensor)> gru1;
            private readonly nn.Module<Tensor, Tensor> lin2;

            public CombinedNN(int nrInputs, int nrHidden, int nrOutputs) : base(nameof(CombinedNN))
            {
                rnn1 = nn.RNN(nrInputs, nrHidden, numLayers: 1, batchFirst: true);
                lstm1 = nn.LSTM(nrInputs, nrHidden, numLayers: 1, batchFirst: true);
                gru1 = nn.GRU(nrInputs, nrHidden, numLayers: 1, batchFirst: true);
                lin2 = nn.Linear(nrHidden, nrOutputs);
                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                var (yElman, _) = rnn1.forward(input, null);
                var (yLstm, _, _) = lstm1.forward(input, null);
                var (yGru, _) = gru1.forward(input, null);

                var summed = yElman + yLstm + yGru; // element-wise sum, shape (B, P, NrHidden)
                var o = lin2.forward(summed);
                return o;
            }
        }
}
    
