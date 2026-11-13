using System;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;

namespace Assignment2
{
    public class Program
    {
        const string DataFolder =
            @"C:\Users\phume\Advanced Programming\Assignment2";

        const int NrHidden = 32;
        const float Eta = 0.001f;
        const int NrEpochsFFNN = 2000;
        const int NrEpochsRNN = 2000;

        static void Main(string[] args)
        {
            Console.WriteLine("Loading and preprocessing data...");

            var data = new DataReader(DataFolder);

            Console.WriteLine("Data loaded.");

            // NN1: FFNN

            Console.WriteLine("\n=== Training NN1 (FFNN) ===");

            var nn1 = new FFNN(
                DataReader.NrVars,
                NrHidden,
                DataReader.NrVars);

            TrainFFNN(
                nn1,
                data,
                NrEpochsFFNN,
                "NN1_FFNN");


            // NN2: Elman


            Console.WriteLine("\n=== Training NN2 (Elman) ===");

            var nn2 = new ElmanNN(
                DataReader.NrVars,
                NrHidden,
                DataReader.NrVars);

            TrainSequenceNN(
                nn2,
                data,
                NrEpochsRNN,
                "NN2_Elman");


            // NN3: LSTM

            Console.WriteLine("\n=== Training NN3 (LSTM) ===");

            var nn3 = new LSTMNN(
                DataReader.NrVars,
                NrHidden,
                DataReader.NrVars);

            TrainSequenceNN(
                nn3,
                data,
                NrEpochsRNN,
                "NN3_LSTM");


            // NN4: GRU


            Console.WriteLine("\n=== Training NN4 (GRU) ===");

            var nn4 = new GRUNN(
                DataReader.NrVars,
                NrHidden,
                DataReader.NrVars);

            TrainSequenceNN(
                nn4,
                data,
                NrEpochsRNN,
                "NN4_GRU");


            // NN5: Combined

            Console.WriteLine(
                "\n=== Training NN5 (Elman+LSTM+GRU combined) ===");

            var nn5 = new CombinedNN(
                DataReader.NrVars,
                NrHidden,
                DataReader.NrVars);

            TrainSequenceNN(
                nn5,
                data,
                NrEpochsRNN,
                "NN5_Combined");


            Console.WriteLine(
                "\nAll models trained. See *_results.csv for MSE summaries " +
                "and *_predictions_seqN.csv for per-sequence graphing data.");
        }


        // NN1 TRAINING
        // Plain (patterns, 4) tensors


        static void TrainFFNN(
            FFNN model,
            DataReader data,
            int epochs,
            string name)
        {
            var loss = nn.MSELoss();

            var optimizer =
                torch.optim.Adam(model.parameters(), Eta);

            for (int g = 0; g < epochs; g++)
            {
                var output = model.forward(data.z);

                var error =
                    loss.forward(output, data.t);

                model.zero_grad();

                error.backward();

                optimizer.step();

                if (g % 100 == 0)
                {
                    Console.WriteLine(
                        $"{g}\ttrain MSE = {error.item<float>():F6}");
                }
            }


            // ============================================================
            // Evaluation
            // ============================================================

            model.eval();

            var testOutput =
                model.forward(data.ztest);

            var testError =
                loss.forward(testOutput, data.ttest);

            var trainOutput =
                model.forward(data.z);

            var trainError =
                loss.forward(trainOutput, data.t);


            Console.WriteLine(
                $"{name}: final train MSE = " +
                $"{trainError.item<float>():F6}, " +
                $"test MSE = " +
                $"{testError.item<float>():F6}");


            // ============================================================
            // Save results
            // ============================================================

            File.WriteAllText(
                $"{name}_results.csv",
                $"TrainMSE,TestMSE\n" +
                $"{trainError.item<float>()}," +
                $"{testError.item<float>()}\n");


            model.train();

            model.save($"{name}.mc");
        }


        // NN2-NN5 TRAINING
        // Sequence tensors: (B, P, 4)

        static void TrainSequenceNN(
            nn.Module<Tensor, Tensor> model,
            DataReader data,
            int epochs,
            string name)
        {
            var loss = nn.MSELoss();

            var optimizer =
                torch.optim.Adam(model.parameters(), Eta);


            for (int g = 0; g < epochs; g++)
            {
                var output =
                    model.forward(data.zseq);

                var error =
                    loss.forward(output, data.tseq);

                model.zero_grad();

                error.backward();

                optimizer.step();


                if (g % 100 == 0)
                {
                    Console.WriteLine(
                        $"{g}\ttrain MSE = {error.item<float>():F6}");
                }

                System.GC.Collect();
            }



            // Evaluation

            model.eval();

            var testOutput =
                model.forward(data.zseqtest);

            var testError =
                loss.forward(
                    testOutput,
                    data.tseqtest);


            var trainOutput =
                model.forward(data.zseq);

            var trainError =
                loss.forward(
                    trainOutput,
                    data.tseq);


            Console.WriteLine(
                $"{name}: final train MSE = " +
                $"{trainError.item<float>():F6}, " +
                $"test MSE = " +
                $"{testError.item<float>():F6}");


            // Save MSE results


            File.WriteAllText(
                $"{name}_results.csv",
                $"TrainMSE,TestMSE\n" +
                $"{trainError.item<float>()}," +
                $"{testError.item<float>()}\n");


            // Export first 4 test sequences

            for (int seq = 0; seq < 4; seq++)
            {
                ExportSequenceForGraphing(
                    testOutput,
                    data.tseqtest,
                    data,
                    seq,
                    $"{name}_predictions_seq{seq}.csv");
            }


            // Save model

            model.train();

            model.save($"{name}.mc");
        }


        // EXPORT SEQUENCE DATA FOR GRAPHING

        static void ExportSequenceForGraphing(
            Tensor predicted,
            Tensor expected,
            DataReader data,
            int seqIndex,
            string filename)
        {
            using (var sw = new StreamWriter(filename))
            {
                // If you have multiple variables, write headers or log step-by-step
                sw.WriteLine("Step,Predicted,Target");

                int numSteps = (int)predicted.shape[1];

                for (int i = 0; i < numSteps; i++)
                {
                    // Assuming variable 0 is tracked, or you can loop over NrVars if needed. 
                    // Indexing down to individual scalar elements [seqIndex, step, variableIndex]
                    float predVal = data.Denormalise(predicted[seqIndex, i, 0].item<float>(), 0);
                    float targVal = data.Denormalise(expected[seqIndex, i, 0].item<float>(), 0);

                    sw.WriteLine($"{i},{predVal},{targVal}");
                }
            }
        }
    }
}