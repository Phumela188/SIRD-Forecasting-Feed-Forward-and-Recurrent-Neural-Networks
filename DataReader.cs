using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;
using TorchSharp;
using static TorchSharp.torch;

namespace Assignment2
{
    
    /// Reads SIRD1.csv ... SIRD1000.csv.
    ///
    /// Each CSV file should contain:
    /// S,I,R,D
    /// followed by 50 weekly rows.
    ///
    /// The first 800 files are used for training.
    /// The remaining 200 files are used for testing.
    ///
    /// The data is normalised using min-max normalisation.
    /// The minimum and maximum values are calculated ONLY
    /// from the training files to prevent test-data leakage.
    public class DataReader
    {
        
        // DATASET CONSTANTS

        public const int NrFiles = 1000;

        public const int NrTrainFiles = 800;

        public const int NrTestFiles = 200;

        public const int WeeksPerFile = 50;

        public const int NrVars = 4; // S, I, R, D

        // 50 weeks gives us 49 input/target pairs:
        //
        // Week 1  -> Week 2
        // Week 2  -> Week 3
        // ...
        // Week 49 -> Week 50
        //
        public const int P = WeeksPerFile - 1;


        // FFNN TENSORS
        
        // Shape:
        //
        // Training:
        // 800 files × 49 patterns = 39,200 patterns
        //
        // z:
        //     (39200, 4)
        //
        // t:
        //     (39200, 4)
        //
        // Test:
        //     200 files × 49 patterns = 9,800 patterns
        //
        // ztest:
        //     (9800, 4)
        //
        // ttest:
        //     (9800, 4)
        //

        public Tensor z;

        public Tensor t;

        public Tensor ztest;

        public Tensor ttest;


        // RECURRENT NEURAL NETWORK TENSORS
        // The dimensions represent:
        //
        // B = number of sequences/files
        // P = number of weeks/patterns
        // 4 = S, I, R, D
        //

        public Tensor zseq;

        public Tensor tseq;

        public Tensor zseqtest;

        public Tensor tseqtest;


        
        // NORMALISATION PARAMETERS

        public float[] MinVals = new float[NrVars];

        public float[] MaxVals = new float[NrVars];


        // DATA FOLDER
        

        private readonly string dataFolder;


        // CONSTRUCTOR
       

        public DataReader(string dataFolder)
        {
            this.dataFolder = dataFolder;

            Console.WriteLine("Reading SIRD data files...");

            // --------------------------------------------------------
            // raw[file, week, variable]
            //
            // variable:
            //
            // 0 = S
            // 1 = I
            // 2 = R
            // 3 = D
            // --------------------------------------------------------

            float[,,] raw =
                new float[NrFiles, WeeksPerFile, NrVars];


            // READ ALL CSV FILES

            for (int f = 0; f < NrFiles; f++)
            {
                string path =
                    Path.Combine(
                        dataFolder,
                        $"SIRD{f + 1}.csv");


                // Check that the file exists before attempting
                // to read it.
                if (!File.Exists(path))
                {
                    throw new FileNotFoundException(
                        $"Could not find SIRD data file: {path}");
                }


                string[] lines =
                    File.ReadAllLines(path);


                

                if (lines.Length < WeeksPerFile + 1)
                {
                    throw new Exception(
                        $"File {path} does not contain " +
                        $"{WeeksPerFile} data rows.");
                }


                
                // READ THE 50 WEEKS
                

                for (int w = 0; w < WeeksPerFile; w++)
                {
                    string line =
                        lines[w + 1];


                    // Split the CSV row into columns.
                    string[] parts =
                        line.Split(',');


                    
                    // S,I,R,D
                   

                    if (parts.Length < NrVars)
                    {
                        throw new Exception(
                            $"Invalid data in {path}, " +
                            $"week {w + 1}. " +
                            $"Expected {NrVars} values but found " +
                            $"{parts.Length}.");
                    }


                    // CONVERT CSV VALUES TO FLOAT

                    for (int v = 0; v < NrVars; v++)
                    {
                        try
                        {
                            

                            raw[f, w, v] =
                                float.Parse(
                                    parts[v].Trim(),
                                    CultureInfo.InvariantCulture);
                        }
                        catch (FormatException)
                        {
                            

                            throw new FormatException(
                                $"Could not convert value to a number.\n" +
                                $"File: {path}\n" +
                                $"Week: {w + 1}\n" +
                                $"Variable: {v}\n" +
                                $"Value: '{parts[v]}'\n" +
                                $"Full line: {line}");
                        }
                    }
                }


                // Display progress every 100 files.

                if ((f + 1) % 100 == 0)
                {
                    Console.WriteLine(
                        $"Loaded {f + 1}/{NrFiles} files...");
                }
            }


            Console.WriteLine(
                "All SIRD files loaded successfully.");


            // COMPUTE NORMALISATION PARAMETERS

            ComputeNormalisation(raw);


            // NORMALISE DATA

            Normalise(raw);


            
            // BUILD FFNN DATA

            BuildFFNNTensors(raw);


            // BUILD RNN DATA

            BuildSequenceTensors(raw);


            Console.WriteLine(
                "Data preprocessing completed.");
        }


        // COMPUTE NORMALISATION

        

        private void ComputeNormalisation(float[,,] raw)
        {
            for (int v = 0; v < NrVars; v++)
            {
                float min =
                    float.MaxValue;

                float max =
                    float.MinValue;


                // Only use training files.

                for (int f = 0;
                     f < NrTrainFiles;
                     f++)
                {
                    for (int w = 0;
                         w < WeeksPerFile;
                         w++)
                    {
                        float val =
                            raw[f, w, v];


                        if (val < min)
                        {
                            min = val;
                        }


                        if (val > max)
                        {
                            max = val;
                        }
                    }
                }


                MinVals[v] = min;

                MaxVals[v] = max;
            }


            // Display the normalisation values.

            Console.WriteLine(
                "\nNormalisation parameters:");

            Console.WriteLine(
                $"S: Min = {MinVals[0]}, Max = {MaxVals[0]}");

            Console.WriteLine(
                $"I: Min = {MinVals[1]}, Max = {MaxVals[1]}");

            Console.WriteLine(
                $"R: Min = {MinVals[2]}, Max = {MaxVals[2]}");

            Console.WriteLine(
                $"D: Min = {MinVals[3]}, Max = {MaxVals[3]}");
        }


        
        // NORMALISE DATA

       
        private void Normalise(float[,,] raw)
        {
            for (int f = 0;
                 f < NrFiles;
                 f++)
            {
                for (int w = 0;
                     w < WeeksPerFile;
                     w++)
                {
                    for (int v = 0;
                         v < NrVars;
                         v++)
                    {
                        float range =
                            MaxVals[v] - MinVals[v];


                        if (range > 1e-8f)
                        {
                            raw[f, w, v] =
                                (raw[f, w, v] - MinVals[v])
                                / range;
                        }
                        else
                        {
                            // If min == max, there is no variation
                            // in that variable.

                            raw[f, w, v] = 0f;
                        }
                    }
                }
            }
        }


        // DENORMALISE

       

        public float Denormalise(
            float value,
            int variableIndex)
        {
            float range =
                MaxVals[variableIndex]
                - MinVals[variableIndex];


            return
                value * range
                + MinVals[variableIndex];
        }


        // ============================================================
        // BUILD FFNN TENSORS
        // ============================================================

       

        private void BuildFFNNTensors(
            float[,,] raw)
        {
            // Training arrays
            

            float[,] zArr =
                new float[
                    NrTrainFiles * P,
                    NrVars];


            float[,] tArr =
                new float[
                    NrTrainFiles * P,
                    NrVars];


            // Test arrays
           

            float[,] ztArr =
                new float[
                    NrTestFiles * P,
                    NrVars];


            float[,] ttArr =
                new float[
                    NrTestFiles * P,
                    NrVars];


            // TRAINING DATA

            int idx = 0;


            for (int f = 0;
                 f < NrTrainFiles;
                 f++)
            {
                for (int w = 0;
                     w < P;
                     w++)
                {
                    for (int v = 0;
                         v < NrVars;
                         v++)
                    {
                        // Input = current week

                        zArr[idx, v] =
                            raw[f, w, v];


                        // Target = next week

                        tArr[idx, v] =
                            raw[f, w + 1, v];
                    }


                    idx++;
                }
            }


            // TEST DATA

            idx = 0;


            for (int f = NrTrainFiles;
                 f < NrFiles;
                 f++)
            {
                for (int w = 0;
                     w < P;
                     w++)
                {
                    for (int v = 0;
                         v < NrVars;
                         v++)
                    {
                        // Input = current week

                        ztArr[idx, v] =
                            raw[f, w, v];


                        // Target = next week

                        ttArr[idx, v] =
                            raw[f, w + 1, v];
                    }


                    idx++;
                }
            }


            // CONVERT ARRAYS TO TORCH TENSORS

            z =
                torch.from_array(zArr)
                .to_type(ScalarType.Float32);


            t =
                torch.from_array(tArr)
                .to_type(ScalarType.Float32);


            ztest =
                torch.from_array(ztArr)
                .to_type(ScalarType.Float32);


            ttest =
                torch.from_array(ttArr)
                .to_type(ScalarType.Float32);


            Console.WriteLine(
                $"FFNN training tensor shape: {z.shape}");

            Console.WriteLine(
                $"FFNN target tensor shape: {t.shape}");

            Console.WriteLine(
                $"FFNN test tensor shape: {ztest.shape}");

            Console.WriteLine(
                $"FFNN test target tensor shape: {ttest.shape}");
        }


        // BUILD SEQUENCE TENSORS
       

        /// <summary>
       
        private void BuildSequenceTensors(
            float[,,] raw)
        {
            // Training sequences

            float[,,] zArr =
                new float[
                    NrTrainFiles,
                    P,
                    NrVars];


            float[,,] tArr =
                new float[
                    NrTrainFiles,
                    P,
                    NrVars];


            
            // Test sequences

            float[,,] ztArr =
                new float[
                    NrTestFiles,
                    P,
                    NrVars];


            float[,,] ttArr =
                new float[
                    NrTestFiles,
                    P,
                    NrVars];


            // TRAINING SEQUENCES

            for (int f = 0;
                 f < NrTrainFiles;
                 f++)
            {
                for (int w = 0;
                     w < P;
                     w++)
                {
                    for (int v = 0;
                         v < NrVars;
                         v++)
                    {
                        // Current week

                        zArr[f, w, v] =
                            raw[f, w, v];


                        // Next week

                        tArr[f, w, v] =
                            raw[f, w + 1, v];
                    }
                }
            }


            // TEST SEQUENCES

            for (int f = 0;
                 f < NrTestFiles;
                 f++)
            {
                for (int w = 0;
                     w < P;
                     w++)
                {
                    for (int v = 0;
                         v < NrVars;
                         v++)
                    {
                        // Test files start at index 800.

                        ztArr[f, w, v] =
                            raw[
                                f + NrTrainFiles,
                                w,
                                v];


                        ttArr[f, w, v] =
                            raw[
                                f + NrTrainFiles,
                                w + 1,
                                v];
                    }
                }
            }


            
            // CONVERT TO TORCH TENSORS

            zseq =
                torch.from_array(zArr)
                .to_type(ScalarType.Float32);


            tseq =
                torch.from_array(tArr)
                .to_type(ScalarType.Float32);


            zseqtest =
                torch.from_array(ztArr)
                .to_type(ScalarType.Float32);


            tseqtest =
                torch.from_array(ttArr)
                .to_type(ScalarType.Float32);


            Console.WriteLine(
                $"RNN training input shape: {zseq.shape}");

            Console.WriteLine(
                $"RNN training target shape: {tseq.shape}");

            Console.WriteLine(
                $"RNN test input shape: {zseqtest.shape}");

            Console.WriteLine(
                $"RNN test target shape: {tseqtest.shape}");
        }
    }
}