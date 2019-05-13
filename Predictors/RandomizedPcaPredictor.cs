using System;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers.PCA;

namespace MLNetEventAnomaly.Predictors
{
    public class RandomizedPcaPredictor
    {
        private const string modelPath = "data/RandomizedPcaPredictorModel.zip";
        private readonly MLContext _ml;
        private PredictionFunction<RandomizedPcaPredictorData, RandomizedPcaPredictorPrediction> _predictionFunction;

        public RandomizedPcaPredictor()
        {
            _ml = new MLContext(seed: 1, conc: 1);
        }
        
        public void Train()
        {
            var pipeline = _ml.Transforms.Conversion.ConvertType("DayOfWeek")
                .Append(_ml.Transforms.Conversion.ConvertType("Hour"))
                .Append(_ml.Transforms.Concatenate("Features", "DayOfWeek", "Hour"))
                .Append(new RandomizedPcaTrainer(_ml, "Features", rank: 2, oversampling: 10));

            var data = File
                .ReadAllLines("training_data/maindoorstates.csv")
                .Select(row => row.Split(';'))
                .Select(row => new { Date = DateTime.Parse(row[0], null, DateTimeStyles.AssumeUniversal), Value = int.Parse(row[1]) })
                .Select(row => new RandomizedPcaPredictorData((int) row.Date.DayOfWeek, row.Date.Hour) { Label = row.Value > 0 })
                .ToList();

            var transformer = pipeline.Fit(_ml.CreateStreamingDataView(data));

            using (var file = File.OpenWrite(modelPath))
                transformer.SaveTo(_ml, file);
        }

        public void Initialize()
        {
            using (var file = File.OpenRead(modelPath))
            {
                var model = TransformerChain.LoadFrom(_ml, file);
                _predictionFunction = model.MakePredictionFunction<RandomizedPcaPredictorData, RandomizedPcaPredictorPrediction>(_ml);
            }          
        }

        public RandomizedPcaPredictorPrediction Predict(DateTime dateTime) => 
            _predictionFunction.Predict(new RandomizedPcaPredictorData((int)dateTime.DayOfWeek, dateTime.Hour));
    }
 }